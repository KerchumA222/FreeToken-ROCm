"""Disk tier for GGUF routed experts: a bounded pinned host cache that reads its misses straight from the GGUF files.

The offload cache normally holds every expert in pinned host banks, so host RAM bounds the model size.
A GGUF needs no repacked spill file to lift that bound: expert ``e`` of ``blk.N.ffn_{gate,up,down}_exps`` is the contiguous byte range
``offset + e * rows * row_bytes`` of the file that holds the tensor, and a bank row is those bytes (gate then up for ``gate_up``).
``GgufExpertStore`` resolves and reads those ranges; ``HostExpertTier`` keeps a bounded LRU of them in pinned memory, one pool per
slot layout (layers keep their own ggml types, so pools of different row widths cannot share slots).

Reads bypass the page cache by default (``O_DIRECT`` with a page-aligned bounce buffer, since GGUF aligns tensors to 32 bytes only):
the point of the tier is to run models bigger than RAM, where page-cached expert bytes only push out what the process needs.
``FREETOKEN_DISK_TIER_IO=buffered`` reads through the page cache and drops the range afterwards; filesystems without O_DIRECT fall back to it.
"""

from __future__ import annotations

import mmap
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_BLK = 4096


@dataclass(frozen=True)
class _Part:
    """One tensor's share of an expert row block: ``nbytes`` per expert, expert ``e`` at ``offset + e * nbytes`` of ``path``."""

    path: str
    offset: int
    nbytes: int


def gguf_expert_parts(model_path: str, num_layers: int, *, first_layer: int = 0) -> dict[str, list[list[_Part]]]:
    """Per bank, per layer: the file ranges that make up one expert's row block, in bank row order."""
    from freetoken.models.gguf.reader import gguf_tensor_index

    index = gguf_tensor_index(model_path)
    parts: dict[str, list[list[_Part]]] = {"gate_up": [], "down": []}
    for layer in range(num_layers):
        blk = f"blk.{first_layer + layer}"
        fused = f"{blk}.ffn_gate_up_exps.weight"
        names = {
            "gate_up": [fused] if fused in index else [f"{blk}.ffn_gate_exps.weight", f"{blk}.ffn_up_exps.weight"],
            "down": [f"{blk}.ffn_down_exps.weight"],
        }
        for bank, tensors in names.items():
            row = []
            for name in tensors:
                info = index[name]
                experts = info.ne[-1]
                if info.nbytes % experts:
                    raise ValueError(f"{name}: {info.nbytes} bytes do not split into {experts} experts")
                row.append(_Part(info.path, info.offset, info.nbytes // experts))
            parts[bank].append(row)
    return parts


class _Reader:
    """Positional reads of byte ranges, page-cache-avoiding where the filesystem allows it."""

    def __init__(self, mode: str | None = None):
        mode = (mode or os.environ.get("FREETOKEN_DISK_TIER_IO", "direct")).strip().lower()
        if mode not in ("direct", "buffered"):
            raise ValueError(f"FREETOKEN_DISK_TIER_IO must be direct or buffered, got {mode!r}")
        self.mode = mode
        self._fds: dict[str, tuple[int, bool]] = {}
        self._lock = threading.Lock()
        self._local = threading.local()

    def _fd(self, path: str) -> tuple[int, bool]:
        got = self._fds.get(path)
        if got is not None:
            return got
        with self._lock:
            got = self._fds.get(path)
            if got is not None:
                return got
            direct = False
            if self.mode == "direct" and hasattr(os, "O_DIRECT"):
                try:
                    got = (os.open(path, os.O_RDONLY | os.O_DIRECT), True)
                    direct = True
                except OSError as exc:
                    logger.warning(f"disk tier: O_DIRECT unavailable for {path} ({exc}); reading through the page cache")
            if not direct:
                fd = os.open(path, os.O_RDONLY)
                if hasattr(os, "posix_fadvise"):
                    # an expert is read whole and its neighbours are other experts, so readahead only fetches bytes nobody asked for
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
                got = (fd, False)
            self._fds[path] = got
            return got

    def _bounce(self, nbytes: int) -> memoryview:
        buf = getattr(self._local, "buf", None)
        if buf is None or len(buf) < nbytes:
            buf = mmap.mmap(-1, -(-nbytes // _BLK) * _BLK)  # anonymous mmaps are page aligned, as O_DIRECT needs
            self._local.buf = buf
        return memoryview(buf)

    def read(self, path: str, offset: int, dst: memoryview) -> None:
        """Fill ``dst`` with ``len(dst)`` bytes of ``path`` at ``offset``."""
        fd, direct = self._fd(path)
        n = len(dst)
        if not direct:
            got = os.preadv(fd, [dst], offset)
            if got != n:
                raise OSError(f"short read: {got} of {n} bytes at {offset} in {path}")
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, offset, n, os.POSIX_FADV_DONTNEED)
            return
        start = offset - offset % _BLK
        end = -(-(offset + n) // _BLK) * _BLK
        bounce = self._bounce(end - start)
        done = 0
        while start + done < offset + n:
            # the aligned span may run past EOF; a short read there is fine as long as it covers the range
            got = os.preadv(fd, [bounce[done : end - start]], start + done)
            if got <= 0:
                raise OSError(f"short read: {done} of {end - start} bytes at {start} in {path}")
            done += got
        dst[:] = bounce[offset - start : offset - start + n]

    def close(self) -> None:
        with self._lock:
            for fd, _ in self._fds.values():
                os.close(fd)
            self._fds.clear()


class GgufExpertStore:
    """Where each (bank, layer, expert) row block of a GGUF lives, and how to read it."""

    def __init__(self, model_path: str, num_layers: int, *, first_layer: int = 0, io_mode: str | None = None):
        self.model_path = model_path
        self.num_layers = num_layers
        self.parts = gguf_expert_parts(model_path, num_layers, first_layer=first_layer)
        self.banks = tuple(self.parts)
        self._reader = _Reader(io_mode)

    def append_layers(self, other: "GgufExpertStore") -> None:
        """Serve ``other``'s layers after this store's (an MTP draft head's experts, possibly from another file)."""
        if set(other.banks) != set(self.banks):
            raise ValueError(f"expert banks {other.banks} of {other.model_path} do not match {self.banks}")
        for bank in self.banks:
            self.parts[bank].extend(other.parts[bank])
        self.num_layers += other.num_layers
        other.close()

    @property
    def io_mode(self) -> str:
        return self._reader.mode

    def expert_bytes(self, bank: str, layer: int) -> int:
        return sum(p.nbytes for p in self.parts[bank][layer])

    def read_expert(self, bank: str, layer: int, expert: int, dst) -> None:
        """Fill ``dst`` (a writable buffer of at least ``expert_bytes(bank, layer)`` bytes) with expert ``expert``'s row block."""
        mv = memoryview(dst).cast("B")
        at = 0
        for p in self.parts[bank][layer]:
            self._reader.read(p.path, p.offset + expert * p.nbytes, mv[at : at + p.nbytes])
            at += p.nbytes

    def close(self) -> None:
        self._reader.close()


def _byte_view(t: torch.Tensor) -> np.ndarray:
    return t.view(torch.uint8).numpy().reshape(-1)


class HostExpertTier:
    """A bounded pinned host cache of routed experts over a ``GgufExpertStore``.

    ``layouts[g]`` is each bank's ``(dtype, row shape)`` for the layers ``layer_layout`` maps to ``g``; pool ``g`` holds
    ``capacities[g]`` experts of those layers, ``pools[g][k]`` shaped ``[capacity, *row shape]`` like a full host bank, so the offload
    cache's copy plan reads a pool slot exactly as it reads an expert row of a complete bank.
    Each pool is an LRU over ``layer * num_experts + expert``; the slots one ``ensure`` hands out are never its own eviction victims.
    The caller must not let a pool slot be overwritten while a device copy still reads it (the offload cache syncs on the miss count first).
    """

    def __init__(
        self,
        store: GgufExpertStore,
        num_experts: int,
        bank_names: tuple[str, ...],
        layouts: list[tuple[tuple[torch.dtype, tuple[int, ...]], ...]],
        layer_layout: list[int],
        capacities: list[int],
        *,
        pin: bool | None = None,
        workers: int | None = None,
    ):
        assert len(capacities) == len(layouts), (capacities, len(layouts))
        assert set(bank_names) == set(store.banks), (bank_names, store.banks)
        self.store = store
        self.num_experts = num_experts
        self.bank_names = tuple(bank_names)
        self.layer_layout = list(layer_layout)
        self.capacities = [min(int(c), num_experts * self.layer_layout.count(g)) for g, c in enumerate(capacities)]
        if min(self.capacities) < 1:
            raise ValueError(f"every host pool needs at least one slot, got {self.capacities}")
        for layer, g in enumerate(self.layer_layout):
            for k, name in enumerate(self.bank_names):
                dtype, shape = layouts[g][k]
                want = int(np.prod(shape)) * dtype.itemsize
                if store.expert_bytes(name, layer) != want:
                    raise ValueError(f"layer {layer} {name}: {store.expert_bytes(name, layer)} bytes per expert on disk, the bank row is {want}")
        if pin is None:
            pin = torch.cuda.is_available()

        def alloc(n: int, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
            if pin:
                from freetoken.kernel.pinned import alloc_pinned_tensor

                return alloc_pinned_tensor(n, *shape, dtype=dtype)
            return torch.empty((n, *shape), dtype=dtype)

        self.pools = [[alloc(cap, dtype, shape) for dtype, shape in layout] for cap, layout in zip(self.capacities, layouts)]
        # prefill reads its pool misses here, at row == expert id; one buffer per bank serves every layout (the widest fits)
        stage_bytes = [num_experts * max(int(np.prod(layout[k][1])) * layout[k][0].itemsize for layout in layouts) for k in range(len(self.bank_names))]
        self._stage = [alloc(n, torch.uint8, ()) for n in stage_bytes]
        self._stage_views = [
            [s[: num_experts * int(np.prod(shape)) * dtype.itemsize].view(dtype).view(num_experts, *shape) for s, (dtype, shape) in zip(self._stage, layout)]
            for layout in layouts
        ]
        self._lru: list[OrderedDict[int, int]] = [OrderedDict() for _ in layouts]
        self._free: list[list[int]] = [list(range(cap - 1, -1, -1)) for cap in self.capacities]
        workers = int(os.environ.get("FREETOKEN_DISK_TIER_READ_WORKERS", "8")) if workers is None else workers
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ft-disk-tier") if workers > 1 else None
        self.hits = self.misses = self.prefill_reads = 0
        self.host_bytes = sum(t.numel() * t.element_size() for pools in self.pools for t in pools) + sum(s.numel() for s in self._stage)

    def pool(self, layer: int, bank: int) -> torch.Tensor:
        return self.pools[self.layer_layout[layer]][bank]

    def stage(self, layer: int, bank: int) -> torch.Tensor:
        return self._stage_views[self.layer_layout[layer]][bank]

    def _fid(self, layer: int, expert: int) -> int:
        return layer * self.num_experts + expert

    def slot_of(self, layer: int, expert: int) -> int:
        """The pool slot holding this expert, or -1; does not touch recency."""
        return self._lru[self.layer_layout[layer]].get(self._fid(layer, expert), -1)

    def _read(self, jobs: list[tuple[int, int, int, torch.Tensor]]) -> None:
        """Run (layer, expert, bank, destination row) reads, concurrently when there is a pool: a layer's reads are latency bound."""

        def one(job) -> None:
            layer, expert, k, dst = job
            self.store.read_expert(self.bank_names[k], layer, expert, _byte_view(dst))

        if self._pool is not None and len(jobs) > 1:
            list(self._pool.map(one, jobs))
        else:
            for job in jobs:
                one(job)

    def ensure(self, layer: int, experts: list[int]) -> list[int]:
        """Pool slots holding ``experts`` (distinct), reading the misses from disk; most recently used afterwards."""
        g = self.layer_layout[layer]
        lru, free = self._lru[g], self._free[g]
        if len(experts) > self.capacities[g]:
            raise RuntimeError(
                f"disk tier: layer {layer} needs {len(experts)} experts at once but its host pool holds {self.capacities[g]}; raise --moe-host-cache-mem"
            )
        slots, protected, missing = [], set(), []
        for pos, e in enumerate(experts):
            fid = self._fid(layer, e)
            slot = lru.get(fid)
            if slot is None:
                missing.append(pos)
                slots.append(-1)
            else:
                lru.move_to_end(fid)
                protected.add(slot)
                slots.append(slot)
        self.hits += len(experts) - len(missing)
        self.misses += len(missing)
        if not missing:
            return slots
        jobs = []
        for pos in missing:
            if free:
                slot = free.pop()
            else:
                victim = next(f for f, s in lru.items() if s not in protected)
                slot = lru.pop(victim)
            protected.add(slot)
            slots[pos] = slot
            jobs += [(layer, experts[pos], k, self.pools[g][k][slot]) for k in range(len(self.bank_names))]
        try:
            self._read(jobs)
        except BaseException:
            free.extend(slots[pos] for pos in missing)  # the rows are not filled; never hand them out as cached
            raise
        for pos in missing:
            lru[self._fid(layer, experts[pos])] = slots[pos]
        return slots

    def prefill(self, layer: int, experts: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
        """Make ``experts`` readable for a prefill: ``(hits, staged)`` where hits are ``(expert, pool slot)`` and staged experts sit in ``stage`` at row == expert id.

        Pool residents are reused without touching recency (a prefill scan must not reorder the decode LRU); misses fill free pool slots
        first, so a cold pool warms without evicting anything, and the rest are read into the stage."""
        g = self.layer_layout[layer]
        lru, free = self._lru[g], self._free[g]
        hits, claimed, staged, jobs = [], [], [], []
        for e in experts:
            slot = lru.get(self._fid(layer, e))
            if slot is not None:
                hits.append((e, slot))
            elif free:
                slot = free.pop()
                claimed.append((e, slot))
                jobs += [(layer, e, k, self.pools[g][k][slot]) for k in range(len(self.bank_names))]
            else:
                staged.append(e)
                jobs += [(layer, e, k, self.stage(layer, k)[e]) for k in range(len(self.bank_names))]
        try:
            self._read(jobs)
        except BaseException:
            free.extend(slot for _, slot in claimed)
            raise
        for e, slot in claimed:
            lru[self._fid(layer, e)] = slot
            lru.move_to_end(self._fid(layer, e), last=False)
        self.prefill_reads += len(claimed) + len(staged)
        return hits + claimed, staged

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self.store.close()


def split_host_capacity(budget_bytes: int, layer_layout: list[int], slot_bytes: list[int], num_experts: int, min_slots: int) -> list[int]:
    """Slots per pool for a host budget: each layer gets an equal share of the bytes, so a pool's share follows its layer count.

    A pool never gets fewer than ``min_slots`` (the distinct experts one decode step can route a layer to) or more than its layers' experts."""
    num_layers = len(layer_layout)
    caps = []
    for g, per_slot in enumerate(slot_bytes):
        layers = layer_layout.count(g)
        cap = budget_bytes * layers // num_layers // per_slot
        caps.append(int(min(max(cap, min_slots), layers * num_experts)))
    return caps


__all__ = ["GgufExpertStore", "HostExpertTier", "gguf_expert_parts", "split_host_capacity"]
