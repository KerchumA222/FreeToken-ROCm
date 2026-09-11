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
