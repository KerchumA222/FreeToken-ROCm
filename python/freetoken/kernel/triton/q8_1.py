"""q8_1 activation blocks written by the kernel that produces the activation.

A packed GGUF GEMV multiplies int8 activations: ``quantize_q8_1`` turns its fp16 input
into blocks of 32 int8 with an fp16 scale and sum (``block_q8_1``, 36 bytes), padded to a
multiple of 512 columns. On a decode step that is one extra launch per packed matmul
(~2.8 us each on an RX 6800 inside a graph). A producer whose output only feeds packed
GEMVs can write the blocks itself; see :mod:`freetoken.layers.q8_act`.

Matches the CUDA ``quantize_q8_1`` kernel: ``d = amax / 127``, ``q = roundf(x / d)`` on the
fp16-rounded value, ``sum`` of the same values; blocks past the real columns are zero.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

Q8_1_BLOCK = 32
Q8_1_BYTES = 36


def q8_1_padded(cols: int) -> int:
    return (cols + 511) // 512 * 512


def q8_1_empty(rows: int, cols: int, device) -> torch.Tensor:
    """The int32 [rows, padded / 32 * 9] buffer the GEMV op takes."""
    return torch.empty((rows, q8_1_padded(cols) // 32 * 9), dtype=torch.int32, device=device)


@triton.jit
def q8_1_store(v, q_row_ptr, first_block, n_blocks, NB: tl.constexpr):
    """Quantize ``v`` ([NB * 32] fp32, zero past the real columns) into q8_1 blocks
    ``first_block ..`` of the row at ``q_row_ptr`` (int8 pointer); only the first
    ``n_blocks`` of them are stored."""
    v = v.to(tl.float16).to(tl.float32)
    vb = tl.reshape(v, (NB, 32))
    amax = tl.max(tl.abs(vb), axis=1)
    total = tl.sum(vb, axis=1)
    d = amax / 127.0
    r = vb / tl.where(amax == 0.0, 1.0, d)[:, None]
    q = tl.where(amax[:, None] == 0.0, 0.0, libdevice.round(r))
    blocks = tl.arange(0, NB)
    keep = blocks < n_blocks
    base = (first_block + blocks) * 36
    h = q_row_ptr.to(tl.pointer_type(tl.float16))
    tl.store(h + base // 2, d.to(tl.float16), mask=keep)
    tl.store(h + base // 2 + 1, total.to(tl.float16), mask=keep)
    qs = base[:, None] + 4 + tl.arange(0, 32)[None, :]
    tl.store(q_row_ptr + qs, q.to(tl.int8), mask=keep[:, None])


@triton.jit
def _quantize_rows_kernel(x_ptr, q_ptr, stride_x, stride_q, COLS: tl.constexpr,
                          PADDED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + row * stride_x + offs, mask=offs < COLS, other=0.0).to(tl.float32)
    n_blocks = tl.minimum(BLOCK, PADDED - tile * BLOCK) // 32
    q8_1_store(v, q_ptr + row * stride_q, tile * (BLOCK // 32), n_blocks, BLOCK // 32)


def quantize_rows_q8_1(x: torch.Tensor) -> torch.Tensor:
    """Reference Triton quantizer (tests compare it with the CUDA one)."""
    rows, cols = x.shape
    q = q8_1_empty(rows, cols, x.device)
    padded = q8_1_padded(cols)
    _quantize_rows_kernel[(rows, padded // 512)](
        x, q.view(torch.int8), x.stride(0), q.shape[1] * 4, cols, padded, 512)
    return q


__all__ = ["q8_1_store", "q8_1_empty", "q8_1_padded", "quantize_rows_q8_1"]
