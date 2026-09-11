"""A bounded pinned host cache of routed experts, backed by disk.

The offload cache is two levels today: a GPU slot cache in front of host banks that
hold *every* expert, pinned. That makes host RAM the ceiling on model size. This adds
a third level underneath -- a fixed-size pool of pinned host slots filled on demand
from :class:`~freetoken.moe.disk_store.GgufExpertStore` -- so the host banks no longer
have to be complete.

Two things shape the design, both measured rather than assumed (see the trace replay
in the RDNA bring-up notes):

* **Locality is strong.** A fifth of the experts resident absorbs about three quarters
  of lookups, so a small pool does most of the work of a complete one.
* **The cost is latency, not bandwidth.** Routing for layer L is not known until L's
  own router runs, so its reads cannot be prefetched across layers; a token pays
  (layers that stall) x (one read latency), not (bytes / bandwidth). Total traffic is
  small -- tens of MB per token. That is why :meth:`ensure` issues a whole layer's
  misses concurrently: what matters is collapsing a layer's reads into one round trip,
  not streaming rate. ``os.preadv`` drops the GIL, so the pool overlaps for real.

Slots touched by the in-flight :meth:`ensure` are never chosen as eviction victims,
so a row cannot be overwritten while the GPU is still copying out of it.

Eviction is LRU here because this tier runs on the host, where an ordered dict is
free. The device-side port cannot copy that: the existing GPU slot cache evicts by
``argmin`` over a vector as wide as the pool, which is affordable for a 4000-slot
GPU cache and not for a host pool that is deliberately larger. CLOCK is the natural
replacement -- one reference bit and a rotating hand, O(1), no reduction -- and it
was measured against this trace before being committed to. Miss rate, replaying the
real 899-step routing trace (40 layers x 256 experts, top-8):

    resident    5%     10%    20%    30%    44%    60%    80%
    LRU       63.8%  42.2%  23.9%  13.6%   5.7%   2.0%   0.8%
    CLOCK     64.0%  43.3%  25.0%  14.3%   6.1%   2.1%   0.8%
    RANDOM    69.7%  50.6%  30.4%  19.0%   9.6%   3.9%   0.8%

CLOCK costs at most +1.1 points anywhere in the range and +0.4 at a realistic
operating point; RANDOM costs +3.9, so the reference bit is doing the work, not the
pool size. A device-side host tier should use CLOCK.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

from freetoken.moe.disk_store import GgufExpertStore
from freetoken.utils import init_logger

logger = init_logger(__name__)

MISS = -1


@dataclass
class HostTierStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    reads: int = 0
    stalled_ensures: int = 0
    ensures: int = 0

    def as_dict(self) -> dict[str, float]:
        touched = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "reads": self.reads,
            "miss_rate": (self.misses / touched) if touched else 0.0,
            # The number that actually predicts added latency: how often a call had to
            # touch disk at all, not how many rows it moved.
            "stall_rate": (self.stalled_ensures / self.ensures) if self.ensures else 0.0,
        }


class HostExpertCache:
    """``capacity`` experts held pinned, the rest read from disk on demand.

    One pinned pool per bank, each ``[capacity, rows, row_bytes]`` uint8 and shaped
    exactly like the complete host bank's per-expert row block, so whatever reads the
    full bank today can read a slot of this instead.
    """

    def __init__(
        self,
        store: GgufExpertStore,
        num_experts: int,
        capacity: int,
        *,
        pin: bool = True,
        workers: int = 8,
    ):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.store = store
        self.num_experts = int(num_experts)
        self.capacity = int(capacity)
        self.stats = HostTierStats()

        self.banks: dict[str, torch.Tensor] = {}
        for name in store.banks:
            rows, row_bytes = store.row_shape(name)
            t = torch.empty((self.capacity, rows, row_bytes), dtype=torch.uint8)
            if pin and torch.cuda.is_available():
                from freetoken.kernel.pinned import alloc_pinned_tensor

                t = alloc_pinned_tensor(self.capacity, rows, row_bytes, dtype=torch.uint8)
            self.banks[name] = t

        # flat id (layer * num_experts + expert) -> slot, in LRU order.
        self._lru: OrderedDict[int, int] = OrderedDict()
        self._free: list[int] = list(range(self.capacity))
        self._id_of_slot: list[int] = [MISS] * self.capacity
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers)) if workers > 1 else None

    # ---- lookup ---------------------------------------------------------------

    def _fid(self, layer: int, expert: int) -> int:
        return layer * self.num_experts + int(expert)

    def slot_of(self, layer: int, expert: int) -> int:
        """The slot holding this expert, or ``MISS``. Does not change recency."""
        return self._lru.get(self._fid(layer, expert), MISS)

    @property
    def resident(self) -> int:
        return len(self._lru)

    # ---- admission ------------------------------------------------------------

    def _claim_slot(self, protected: set[int]) -> int:
        if self._free:
            return self._free.pop()
        for fid, slot in self._lru.items():           # least recent first
            if slot not in protected:
                del self._lru[fid]
                self._id_of_slot[slot] = MISS
                self.stats.evictions += 1
                return slot
        raise RuntimeError(
            f"every one of {self.capacity} slots is in use by the current step; "
            "raise the host cache capacity"
        )

    def ensure(self, layer: int, expert_ids: Sequence[int] | Iterable[int]) -> list[int]:
        """Make these experts resident and return their slots, in the same order.

        Misses for the call are read concurrently -- one round trip for the layer
        rather than one per expert.
        """
        ids = [int(e) for e in expert_ids]
        self.stats.ensures += 1
        slots: list[int] = []
        protected: set[int] = set()
        missing: list[tuple[int, int]] = []           # (position, expert)

        for pos, e in enumerate(ids):
            fid = self._fid(layer, e)
            slot = self._lru.get(fid, MISS)
            if slot != MISS:
                self._lru.move_to_end(fid)
                self.stats.hits += 1
                slots.append(slot)
                protected.add(slot)
            else:
                self.stats.misses += 1
                slots.append(MISS)
                missing.append((pos, e))

        if not missing:
            return slots

        self.stats.stalled_ensures += 1
        # Claim every victim before reading, so no two misses race for one slot and
        # nothing in flight for this call can be chosen.
        claims: list[tuple[int, int, int]] = []       # (position, expert, slot)
        for pos, e in missing:
            slot = self._claim_slot(protected)
            protected.add(slot)
            claims.append((pos, e, slot))

        # One job per (expert, bank), not per expert: a layer's reads are latency-
        # bound and independent, so every one of them should be in flight at once.
        # Serialising the banks within an expert triples the round trips for nothing.
        jobs = [(e, slot, name) for _pos, e, slot in claims for name in self.banks]

        def fill(job: tuple[int, int, str]) -> None:
            expert, slot, name = job
            self.store.read_expert(name, layer, expert, self.banks[name][slot].numpy())

        if self._pool is not None and len(jobs) > 1:
            list(self._pool.map(fill, jobs))
        else:
            for job in jobs:
                fill(job)

        for pos, e, slot in claims:
            fid = self._fid(layer, e)
            self._lru[fid] = slot
            self._id_of_slot[slot] = fid
            slots[pos] = slot
            self.stats.reads += len(self.banks)
        return slots

    def read_into(self, layer: int, experts: Sequence[int], dst: dict) -> None:
        """Read these experts straight into ``dst[bank][i]``, bypassing the pool.

        Prefill streams a whole layer. Admitting all of it would evict everything the
        pool holds for decode and leave only the last layer behind, so prefill reads
        *through* the tier without disturbing residency. Reads are issued concurrently
        for the same reason :meth:`ensure` does -- the cost is round trips, not bytes.
        """
        jobs = [
            (i, int(e), name)
            for i, e in enumerate(experts)
            for name in self.banks
        ]
        if not jobs:
            return

        def fill(job: tuple[int, int, str]) -> None:
            i, expert, name = job
            self.store.read_expert(name, layer, expert, dst[name][i].numpy())

        if self._pool is not None and len(jobs) > 1:
            list(self._pool.map(fill, jobs))
        else:
            for job in jobs:
                fill(job)
        self.stats.reads += len(jobs)

    def admit_free(self, layer: int, experts: Sequence[int]) -> list[tuple[int, int]]:
        """Admit as many of ``experts`` as there are FREE slots, never evicting.

        Prefill touches every expert of every layer, so admitting it with eviction
        would churn the pool decode just warmed -- on every turn of a conversation.
        Filling only free slots gets the opposite of both bad outcomes: a cold pool is
        warmed by the first prefill (which otherwise reads 100% from disk however big
        the pool is), and a warm pool is left exactly as decode left it.

        Returns the ``(expert, slot)`` pairs admitted, in claim order.
        """
        claims: list[tuple[int, int]] = []
        for e in experts:
            if not self._free:
                break
            claims.append((int(e), self._free.pop()))
        if not claims:
            return []

        jobs = [(e, slot, name) for e, slot in claims for name in self.banks]

        def fill(job: tuple[int, int, str]) -> None:
            expert, slot, name = job
            self.store.read_expert(name, layer, expert, self.banks[name][slot].numpy())

        if self._pool is not None and len(jobs) > 1:
            list(self._pool.map(fill, jobs))
        else:
            for job in jobs:
                fill(job)

        for e, slot in claims:
            fid = self._fid(layer, e)
            self._lru[fid] = slot
            self._id_of_slot[slot] = fid
        self.stats.reads += len(jobs)
        return claims

    def residency_split(self, layer: int, num_experts: int) -> tuple[list[int], list[int], list[int]]:
        """``(hit_experts, hit_slots, missing_experts)`` for a whole layer.

        Lookups do not change recency: this is a query about what prefill can take
        from the pool, not a use of those entries.
        """
        hit_e: list[int] = []
        hit_s: list[int] = []
        miss: list[int] = []
        for e in range(num_experts):
            slot = self._lru.get(self._fid(layer, e), MISS)
            if slot == MISS:
                miss.append(e)
            else:
                hit_e.append(e)
                hit_s.append(slot)
        return hit_e, hit_s, miss

    # ---- lifecycle ------------------------------------------------------------

    def reset(self) -> None:
        self._lru.clear()
        self._free = list(range(self.capacity))
        self._id_of_slot = [MISS] * self.capacity

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> "HostExpertCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
