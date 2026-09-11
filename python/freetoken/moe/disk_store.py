"""Byte-range addressing of the routed-expert banks inside the original GGUF shards.

The offload cache needs every expert resident in pinned host RAM, which makes RAM
the ceiling on model size. A disk tier lifts that ceiling, and for a GGUF checkpoint
it needs no repacked copy on the side: **a host bank row already is a contiguous
byte range of the file we loaded from.**

``load_q4_0_expert_sources`` builds ``gate_up[l]`` as ``[E, 2I, gu_bytes]`` by copying
``ffn_gate_exps.weight`` into the first ``I`` rows and ``ffn_up_exps.weight`` into the
next ``I``, and ``down[l]`` as ``[E, H, dn_bytes]`` straight from ``ffn_down_exps``.
Each of those source tensors is ``[E * rows_per_expert, row_bytes]`` in file order, so
one expert's slice of one tensor is a single contiguous extent. Reading expert ``e``
of a layer is therefore two ``pread``\\s for ``gate_up`` (gate half, up half) and one
for ``down`` -- no spill file, no repack, no extra disk.

The one thing that breaks the identity is *requantization*: a layer whose tensor type
differs from the resolved bank type is converted at load, so its bytes on disk are not
its bytes in the bank. Those (bank, layer) pairs are reported by
:meth:`GgufExpertStore.requantized_layers` and cannot be served from disk.

Reads are buffered rather than ``O_DIRECT``: GGUF aligns tensor data to
``general.alignment`` (32 by default), not to a 4096-byte block, so O_DIRECT would
need read-around into a bounce buffer on every access. Buffered reads also let the
page cache act as a free, *reclaimable* second-level cache -- which is the right
trade when the whole point of the tier is to stop pinning unreclaimable RAM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from freetoken.models.gguf.reader import gguf_shard_paths, iter_gguf_tensors
from freetoken.utils import init_logger

logger = init_logger(__name__)

# Bank name -> the GGUF tensor suffixes that make it up, in destination row order.
# gate_up is two halves so that its rows line up with silu_and_mul's.
_BANK_PARTS: dict[str, tuple[str, ...]] = {
    "gate_up": ("ffn_gate_exps.weight", "ffn_up_exps.weight"),
    "down": ("ffn_down_exps.weight",),
}


@dataclass(frozen=True)
class Extent:
    """One contiguous read: ``nbytes`` at ``offset`` in ``path``, landing at
    ``dst_offset`` in the destination expert row."""

    path: str
    offset: int
    nbytes: int
    dst_offset: int


@dataclass(frozen=True)
class _Part:
    path: str
    data_offset: int
    rows_per_expert: int
    row_bytes: int
    ggml_type: int


class GgufExpertStore:
    """Where each (bank, layer, expert) lives on disk, and how to read it.

    ``expert_bytes(bank)`` is the size of one expert's row block -- the same size the
    pinned host bank allocates per expert -- so a destination buffer is just that many
    bytes and the extents fill it end to end.
    """

    def __init__(self, model_path: str, num_experts: int, bank_types: dict[str, int]):
        self.model_path = model_path
        self.num_experts = int(num_experts)
        self._bank_types = dict(bank_types)
        self._parts: dict[tuple[str, int], _Part] = {}   # (suffix, layer) -> part
        self._fds: dict[str, int] = {}

        wanted = {s for parts in _BANK_PARTS.values() for s in parts}
        for shard in gguf_shard_paths(model_path):
            for t in iter_gguf_tensors(shard):
                if not t.name.startswith("blk."):
                    continue
                suffix = t.name.split(".", 2)[2]
                if suffix not in wanted:
                    continue
                layer = int(t.name.split(".")[1])
                if t.rows % self.num_experts:
                    raise ValueError(
                        f"{t.name}: {t.rows} rows is not a multiple of "
                        f"{self.num_experts} experts"
                    )
                self._parts[(suffix, layer)] = _Part(
                    path=shard,
                    data_offset=int(t.data_offset),
                    rows_per_expert=t.rows // self.num_experts,
                    row_bytes=t.row_bytes,
                    ggml_type=t.ggml_type,
                )

    # ---- geometry -------------------------------------------------------------

    @property
    def banks(self) -> tuple[str, ...]:
        return tuple(b for b in _BANK_PARTS if (b in self._bank_types))

    def layers(self, bank: str) -> tuple[int, ...]:
        head = _BANK_PARTS[bank][0]
        return tuple(sorted(l for (s, l) in self._parts if s == head))

    def row_shape(self, bank: str) -> tuple[int, int]:
        """One expert's row block as the host bank shapes it: ``[rows, row_bytes]``.

        ``gate_up`` stacks its two parts (gate rows then up rows) into a single row
        axis, which is the layout ``silu_and_mul`` expects.
        """
        layer = self.layers(bank)[0]
        parts = [self._part(s, layer) for s in _BANK_PARTS[bank]]
        row_bytes = {p.row_bytes for p in parts}
        if len(row_bytes) != 1:
            raise ValueError(f"bank {bank!r} parts disagree on row_bytes: {row_bytes}")
        return sum(p.rows_per_expert for p in parts), next(iter(row_bytes))

    def expert_bytes(self, bank: str) -> int:
        return sum(
            p.rows_per_expert * p.row_bytes
            for p in (self._part(s, self.layers(bank)[0]) for s in _BANK_PARTS[bank])
        )

    def requantized_layers(self) -> dict[str, tuple[int, ...]]:
        """(bank -> layers) whose on-disk type differs from the resolved bank type.

        Those layers are converted during load, so their file bytes are not their bank
        bytes and they must stay resident (or be re-spilled) rather than read from here.
        An empty dict means the whole model is directly disk-addressable.
        """
        out: dict[str, tuple[int, ...]] = {}
        for bank, parts in _BANK_PARTS.items():
            want = self._bank_types.get(bank)
            if want is None:
                continue
            bad = sorted(
                {
                    layer
                    for suffix in parts
                    for (s, layer), p in self._parts.items()
                    if s == suffix and p.ggml_type != want
                }
            )
            if bad:
                out[bank] = tuple(bad)
        return out

    # ---- addressing -----------------------------------------------------------

    def _part(self, suffix: str, layer: int) -> _Part:
        try:
            return self._parts[(suffix, layer)]
        except KeyError:
            raise KeyError(f"{suffix} missing for layer {layer} in {self.model_path}") from None

    def extents(self, bank: str, layer: int, expert: int) -> tuple[Extent, ...]:
        """The reads that fill one expert's row block, in destination order."""
        if not 0 <= expert < self.num_experts:
            raise IndexError(f"expert {expert} out of range [0, {self.num_experts})")
        out, dst = [], 0
        for suffix in _BANK_PARTS[bank]:
            p = self._part(suffix, layer)
            span = p.rows_per_expert * p.row_bytes
            out.append(
                Extent(
                    path=p.path,
                    offset=p.data_offset + expert * span,
                    nbytes=span,
                    dst_offset=dst,
                )
            )
            dst += span
        return tuple(out)

    # ---- reading --------------------------------------------------------------

    def _fd(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd

    def read_expert(self, bank: str, layer: int, expert: int, dst) -> int:
        """Fill ``dst`` (a writable uint8 buffer of ``expert_bytes(bank)``) and return
        the number of reads issued."""
        mv = memoryview(dst).cast("B")
        ext = self.extents(bank, layer, expert)
        total = sum(e.nbytes for e in ext)
        if len(mv) != total:
            raise ValueError(f"destination is {len(mv)} bytes, expert needs {total}")
        for e in ext:
            got = os.preadv(self._fd(e.path), [mv[e.dst_offset : e.dst_offset + e.nbytes]], e.offset)
            if got != e.nbytes:
                raise OSError(f"short read: {got} of {e.nbytes} at {e.offset} in {e.path}")
        return len(ext)

    def read_layer(self, bank: str, layer: int, dst) -> int:
        """Fill a whole layer's bank -- ``[num_experts, rows, row_bytes]`` -- and return
        the number of reads issued.

        Prefill copies an entire layer at once, so this is the shape the prefill path
        needs and the shape a bounded per-expert pool cannot serve. Each part is one
        contiguous file range covering every expert, but it lands on a *strided* slice
        of the destination (``gate_up``'s gate half is rows ``[0, I)`` of each expert's
        block). ``preadv`` is exactly the right tool: one syscall, one file range, an
        iovec per expert -- so a layer costs one read per part rather than one per
        expert.
        """
        import numpy as np

        arr = np.asarray(dst)
        if arr.dtype != np.uint8:
            raise TypeError(f"destination must be uint8, got {arr.dtype}")
        rows, row_bytes = self.row_shape(bank)
        if arr.shape != (self.num_experts, rows, row_bytes):
            raise ValueError(
                f"destination {arr.shape} != ({self.num_experts}, {rows}, {row_bytes})"
            )
        reads, row0 = 0, 0
        for suffix in _BANK_PARTS[bank]:
            p = self._part(suffix, layer)
            part_rows = p.rows_per_expert
            # One iovec per expert: the file range is contiguous across experts, the
            # destination is not.
            iov = [
                memoryview(arr[e, row0 : row0 + part_rows]).cast("B")
                for e in range(self.num_experts)
            ]
            want = self.num_experts * part_rows * p.row_bytes
            got = os.preadv(self._fd(p.path), iov, p.data_offset)
            if got != want:
                raise OSError(
                    f"short read: {got} of {want} at {p.data_offset} in {p.path}"
                )
            reads += 1
            row0 += part_rows
        return reads

    def layer_bytes(self, bank: str) -> int:
        return self.num_experts * self.expert_bytes(bank)

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self) -> "GgufExpertStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
