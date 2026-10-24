"""``x @ W.T`` for a handful of rows against a small dense weight (MoE routers).

rocBLAS runs these as large Tensile GEMM tiles on RDNA, and a broadcast multiply-and-sum
materializes an ``[M, N, K]`` temporary. This reads each weight row once for every
activation row, accumulates in fp32, and is graph-capturable.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

MAX_ROWS = 4


@triton.jit
def _small_gemv_kernel(
    x_ptr, w_ptr, o_ptr, side_w_ptr, side_o_ptr, M, N, K,
    stride_xm, stride_wn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_SIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    side_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
        if HAS_SIDE and pid == 0:
            side_w = tl.load(side_w_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            side_acc += tl.sum(x * side_w[None, :], axis=1)
    tl.store(o_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
             acc.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])
    if HAS_SIDE and pid == 0:
        tl.store(side_o_ptr + offs_m, tl.sigmoid(side_acc), mask=mask_m)


def small_gemv(x: torch.Tensor, w: torch.Tensor, block_n: int = 4, block_k: int = 256,
               num_warps: int = 2, side_weight: torch.Tensor | None = None,
               side_output: torch.Tensor | None = None,
               block_m: int | None = None) -> torch.Tensor:
    """``x [M, K] @ w [N, K].T -> [M, N]`` in ``x.dtype``, for ``M <= MAX_ROWS``.

    Optionally calculate ``sigmoid(x @ side_weight)`` into ``side_output`` in the same
    launch. The auxiliary reduction is assigned to the first expert tile only.
    ``block_m`` can reduce the row tile for decode-sized inputs; by default the legacy
    four-row tile is retained.
    """
    M, K = x.shape
    N = w.shape[0]
    assert M <= MAX_ROWS and w.shape[1] == K and x.stride(1) == 1 and w.stride(1) == 1
    if block_m is None:
        block_m = (
            1
            if M == 1 and x.is_cuda and _is_gfx1030(x.device.index)
            else MAX_ROWS
        )
    if block_m not in (1, 2, 4) or block_m < triton.next_power_of_2(M):
        raise ValueError("block_m must be 1, 2 or 4 and cover every input row")
    if (side_weight is None) != (side_output is None):
        raise ValueError("side_weight and side_output must be provided together")
    if side_weight is not None:
        if side_weight.shape != (K,) or side_weight.stride(0) != 1:
            raise ValueError("side_weight must be a contiguous vector matching the input width")
        if side_weight.device != x.device:
            raise ValueError("side_weight must be on the input device")
        if side_output.shape != (M,) or side_output.dtype != torch.float32:
            raise ValueError("side_output must be a float32 vector with one value per input row")
        if side_output.device != x.device or side_output.stride(0) != 1:
            raise ValueError("side_output must be contiguous and on the input device")
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    _small_gemv_kernel[(triton.cdiv(N, block_n),)](
        x, w, out,
        side_weight if side_weight is not None else w,
        side_output if side_output is not None else out,
        M, N, K, x.stride(0), w.stride(0), out.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        HAS_SIDE=side_weight is not None,
        num_warps=num_warps,
    )
    return out


@functools.lru_cache(maxsize=None)
def _is_gfx1030(device_index: int | None) -> bool:
    return (torch.version.hip is not None
            and torch.cuda.get_device_properties(device_index).gcnArchName.startswith("gfx1030"))


def flash_next_gemv_config(rows: int, n: int, k: int, device_index: int | None = None) -> tuple[int, int, int]:
    """Decode configurations measured on the RX 6800; other shapes keep the old launch."""
    if rows == 1 and _is_gfx1030(device_index):
        if (n, k) == (336, 10240):
            return 2, 2048, 4
        if (n, k) == (512, 2560):
            return 2, 1024, 2
        if (n, k) == (640, 2560):
            return 2, 1024, 2
    if rows == 4 and (n, k) == (256, 2560) and _is_gfx1030(device_index):
        return 2, 2048, 4
    return 1, 2048, 2


# Bigger weights are left to rocBLAS: the lm_head class of GEMV is bandwidth-bound there too.
_DENSE_MAX_ELEMS = 64 << 20


def _pick_orientation(x: torch.Tensor, w: torch.Tensor) -> bool:
    from freetoken.env import ENV

    return (ENV.PREFILL_GEMM_PICK and x.is_cuda and w.dim() == 2 and w.dtype == x.dtype
            and not torch.cuda.is_current_stream_capturing())


def dense_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    """``F.linear`` for a dense weight, through :func:`small_gemv` at decode sizes on HIP.

    rocBLAS runs an M<=4 fp16 GEMV as a large Tensile tile on RDNA: 85-540 us for weights
    a few MB wide (Qwen3.8-Flash-Next's QSA indexer, a dense GDN out_proj). This reads
    the weight once in a few us."""
    rows = x.numel() // x.shape[-1] if x.dim() else 0
    if rows > MAX_ROWS and _pick_orientation(x, w):
        from freetoken.kernel.gemm_pick import flipped_linear, pick, row_bucket

        x2 = x.reshape(rows, -1)
        out = pick(("dense", row_bucket(rows), w.shape[0], w.shape[1], w.dtype),
                   (lambda: torch.nn.functional.linear(x2, w), lambda: flipped_linear(x2, w)))
        out = out.reshape(*x.shape[:-1], w.shape[0])
        return out + b.to(out.dtype) if b is not None else out
    if (torch.version.hip is None or not x.is_cuda or rows == 0 or rows > MAX_ROWS
            or w.dim() != 2 or w.dtype != x.dtype or w.stride(1) != 1
            or w.numel() > _DENSE_MAX_ELEMS):
        return torch.nn.functional.linear(x, w, b)
    block_n, block_k, num_warps = flash_next_gemv_config(rows, w.shape[0], w.shape[1], x.device.index)
    out = small_gemv(x.reshape(rows, -1).contiguous(), w,
                     block_n=block_n, block_k=block_k, num_warps=num_warps)
    out = out.reshape(*x.shape[:-1], w.shape[0])
    return out + b.to(out.dtype) if b is not None else out


__all__ = ["small_gemv", "dense_linear", "flash_next_gemv_config", "MAX_ROWS"]
