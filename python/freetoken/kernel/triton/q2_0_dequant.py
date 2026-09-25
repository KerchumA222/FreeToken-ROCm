"""Q2_0 / Q2_0_SYM dequantization on the GPU (blocks of 64 two-bit codes and an fp16 scale).

The torch reference (``models/gguf/dequant.py``) runs several fp32 passes; the grouped
MoE prefill dequantizes every routed expert's down projection per chunk, where that
dominated. Same arithmetic: ``(q - 1) * d`` for Q2_0, ``(2q - 3) * d`` for Q2_0_SYM, code
``4*j + s`` held in byte ``j`` at bit ``2*s``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _q2_0_dequant_kernel(raw_ptr, out_ptr, n_blocks, SYM: tl.constexpr, BLOCKS: tl.constexpr):
    pid = tl.program_id(0)
    b = pid * BLOCKS + tl.arange(0, BLOCKS)
    mask_b = b < n_blocks
    base = b.to(tl.int64) * 18
    lo = tl.load(raw_ptr + base, mask=mask_b, other=0).to(tl.uint16)
    hi = tl.load(raw_ptr + base + 1, mask=mask_b, other=0).to(tl.uint16)
    d = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    e = tl.arange(0, 64)
    byte = tl.load(raw_ptr + base[:, None] + 2 + (e // 4)[None, :], mask=mask_b[:, None], other=0)
    q = ((byte >> ((e % 4) * 2)[None, :]) & 3).to(tl.float32)
    v = (2.0 * q - 3.0) if SYM else (q - 1.0)
    out = v * d[:, None]
    tl.store(out_ptr + b.to(tl.int64)[:, None] * 64 + e[None, :], out.to(out_ptr.dtype.element_ty),
             mask=mask_b[:, None])


def dequant_q2_0(raw: torch.Tensor, out_dtype: torch.dtype, sym: bool) -> torch.Tensor:
    """``raw`` uint8 (a whole number of 18-byte blocks) -> flat ``out_dtype``."""
    raw = raw.reshape(-1)
    n_blocks = raw.numel() // 18
    out = torch.empty(n_blocks * 64, dtype=out_dtype, device=raw.device)
    blocks = 64
    _q2_0_dequant_kernel[(triton.cdiv(n_blocks, blocks),)](raw, out, n_blocks, SYM=sym, BLOCKS=blocks)
    return out


__all__ = ["dequant_q2_0"]
