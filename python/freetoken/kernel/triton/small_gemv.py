"""``x @ W.T`` for a handful of rows against a small dense weight (MoE routers).

rocBLAS runs these as large Tensile GEMM tiles on RDNA, and a broadcast multiply-and-sum
materializes an ``[M, N, K]`` temporary. This reads each weight row once for every
activation row, accumulates in fp32, and is graph-capturable.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_ROWS = 4


@triton.jit
def _small_gemv_kernel(
    x_ptr, w_ptr, o_ptr, M, N, K,
    stride_xm, stride_wn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
    tl.store(o_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
             acc.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def small_gemv(x: torch.Tensor, w: torch.Tensor, block_n: int = 4, block_k: int = 256,
               num_warps: int = 2) -> torch.Tensor:
    """``x [M, K] @ w [N, K].T -> [M, N]`` in ``x.dtype``, for ``M <= MAX_ROWS``."""
    M, K = x.shape
    N = w.shape[0]
    assert M <= MAX_ROWS and w.shape[1] == K and x.stride(1) == 1 and w.stride(1) == 1
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    _small_gemv_kernel[(triton.cdiv(N, block_n),)](
        x, w, out, M, N, K, x.stride(0), w.stride(0), out.stride(0),
        BLOCK_M=MAX_ROWS, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps,
    )
    return out


# Bigger weights are left to rocBLAS: the lm_head class of GEMV is bandwidth-bound there too.
_DENSE_MAX_ELEMS = 64 << 20


def dense_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    """``F.linear`` for a dense weight, through :func:`small_gemv` at decode sizes on HIP.

    rocBLAS runs an M<=4 fp16 GEMV as a large Tensile tile on RDNA: 85-540 us for weights
    a few MB wide (Qwen3.8-Flash-Next's QSA indexer, a dense GDN out_proj). This reads
    the weight once in a few us."""
    rows = x.numel() // x.shape[-1] if x.dim() else 0
    if (torch.version.hip is None or not x.is_cuda or rows == 0 or rows > MAX_ROWS
            or w.dim() != 2 or w.dtype != x.dtype or w.stride(1) != 1
            or w.numel() > _DENSE_MAX_ELEMS):
        return torch.nn.functional.linear(x, w, b)
    out = small_gemv(x.reshape(rows, -1).contiguous(), w, block_n=1, block_k=2048, num_warps=2)
    out = out.reshape(*x.shape[:-1], w.shape[0])
    return out + b.to(out.dtype) if b is not None else out


__all__ = ["small_gemv", "dense_linear", "MAX_ROWS"]
