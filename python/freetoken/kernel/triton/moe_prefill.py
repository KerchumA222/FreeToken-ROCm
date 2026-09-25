"""Layout kernels for the grouped MoE prefill (``moe/fused_q4_0.py``).

The grouped gate_up GEMM runs as ``W [n, 2I, H] @ x^T [n, H, c]`` (the batched shape
rocBLAS runs fast on gfx1030), which needs the gathered token rows transposed on the way
in and the result transposed back for the activation. torch does both as strided copies
through its generic elementwise kernel, ~26 GB/s on ROCm: 21% of a 7.5k-token
Qwen3.8-Flash-Next prefill. These do each in one pass through a shared-memory tile.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_t_kernel(x_ptr, idx_ptr, out_ptr, h, cols, stride_x,
                     BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr):
    """out[n, :, c] = x[idx[n, c], :]  (out is [n, h, cols])."""
    n = tl.program_id(0).to(tl.int64)
    c0 = tl.program_id(1) * BLOCK_C
    h0 = tl.program_id(2) * BLOCK_H
    c = c0 + tl.arange(0, BLOCK_C)
    hh = h0 + tl.arange(0, BLOCK_H)
    cm = c < cols
    rows = tl.load(idx_ptr + n * cols + c, mask=cm, other=0).to(tl.int64)
    tile = tl.load(x_ptr + rows[:, None] * stride_x + hh[None, :],
                   mask=cm[:, None] & (hh[None, :] < h), other=0.0)
    tl.store(out_ptr + (n * h + hh[None, :]) * cols + c[:, None], tile,
             mask=cm[:, None] & (hh[None, :] < h))


def gather_transposed(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``x[idx]`` for ``idx`` [n, c], laid out [n, H, c]."""
    n, cols = idx.shape
    h = x.shape[1]
    out = torch.empty((n, h, cols), dtype=x.dtype, device=x.device)
    bc, bh = 64, 64
    _gather_t_kernel[(n, triton.cdiv(cols, bc), triton.cdiv(h, bh))](
        x, idx.contiguous(), out, h, cols, x.stride(0), BLOCK_C=bc, BLOCK_H=bh, num_warps=4)
    return out


@triton.jit
def _silu_mul_t_kernel(gu_ptr, out_ptr, inter, cols, BLOCK_C: tl.constexpr, BLOCK_I: tl.constexpr):
    """out[n * cols + c, i] = silu(gu[n, i, c]) * gu[n, inter + i, c]  (gu is [n, 2I, cols])."""
    n = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    i = tl.program_id(2) * BLOCK_I + tl.arange(0, BLOCK_I)
    m = (i[:, None] < inter) & (c[None, :] < cols)
    base = gu_ptr + n * (2 * inter) * cols
    gate = tl.load(base + i[:, None] * cols + c[None, :], mask=m, other=0.0).to(tl.float32)
    up = tl.load(base + (inter + i[:, None]) * cols + c[None, :], mask=m, other=0.0).to(tl.float32)
    y = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(out_ptr + (n * cols + c[None, :]) * inter + i[:, None], y.to(out_ptr.dtype.element_ty), mask=m)


def silu_and_mul_transposed(gu: torch.Tensor) -> torch.Tensor:
    """``gu`` [n, 2I, c] (gate rows then up rows) -> silu(gate) * up as [n * c, I]."""
    n, two_i, cols = gu.shape
    inter = two_i // 2
    out = torch.empty((n * cols, inter), dtype=gu.dtype, device=gu.device)
    bc, bi = 64, 64
    _silu_mul_t_kernel[(n, triton.cdiv(cols, bc), triton.cdiv(inter, bi))](
        gu, out, inter, cols, BLOCK_C=bc, BLOCK_I=bi, num_warps=4)
    return out


__all__ = ["gather_transposed", "silu_and_mul_transposed"]
