from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _store_q8_0_kernel(
    k_cache, v_cache, k_scale, v_scale, indices, k, v,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCKS: tl.constexpr, BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    block_id = tl.program_id(2)
    d = block_id * BLOCK + tl.arange(0, BLOCK)
    xk = tl.load(k + token * stride_kt + head * stride_kh + d, d < HEAD_DIM, 0).to(tl.float32)
    xv = tl.load(v + token * stride_vt + head * stride_vh + d, d < HEAD_DIM, 0).to(tl.float32)
    amax_k = tl.max(tl.abs(xk), axis=0)
    amax_v = tl.max(tl.abs(xv), axis=0)
    scale_k = amax_k / 127.0
    scale_v = amax_v / 127.0
    inv_k = tl.where(scale_k > 0.0, 1.0 / scale_k, 0.0)
    inv_v = tl.where(scale_v > 0.0, 1.0 / scale_v, 0.0)
    qk = tl.floor(tl.abs(xk * inv_k) + 0.5) * tl.where(xk < 0.0, -1.0, 1.0)
    qv = tl.floor(tl.abs(xv * inv_v) + 0.5) * tl.where(xv < 0.0, -1.0, 1.0)
    slot = tl.load(indices + token).to(tl.int32)
    row = slot * NUM_HEADS * HEAD_DIM + head * HEAD_DIM + d
    scale_row = slot * NUM_HEADS * BLOCKS + head * BLOCKS + block_id
    tl.store(k_cache + row, qk.to(tl.int8), d < HEAD_DIM)
    tl.store(v_cache + row, qv.to(tl.int8), d < HEAD_DIM)
    tl.store(k_scale + scale_row, scale_k.to(tl.float16))
    tl.store(v_scale + scale_row, scale_v.to(tl.float16))


def store_q8_0_cache(
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Quantize token/head rows with llama.cpp Q8_0 scale and rounding semantics."""
    _, heads, head_dim = k_cache.shape
    input_tokens = k.shape[0]
    if k.ndim == 2 and k.shape[1] == heads * head_dim:
        k = k.view(input_tokens, heads, head_dim)
    if v.ndim == 2 and v.shape[1] == heads * head_dim:
        v = v.view(v.shape[0], heads, head_dim)
    tokens = k.shape[0]
    if k.ndim != 3 or k.shape != (tokens, heads, head_dim):
        raise ValueError(
            f"Q8_0 K input must have shape [tokens, {heads}, {head_dim}] or "
            f"[tokens, {heads * head_dim}], got {tuple(k.shape)}"
        )
    if v.ndim != 3 or v.shape != (tokens, heads, head_dim):
        raise ValueError(
            f"Q8_0 V input must have shape [tokens, {heads}, {head_dim}] or "
            f"[tokens, {heads * head_dim}], got {tuple(v.shape)}"
        )
    if head_dim % 32:
        raise ValueError(f"Q8_0 requires head_dim divisible by 32, got {head_dim}")
    if v.shape != k.shape:
        raise ValueError("Q8_0 K and V inputs must have the same shape")
    if indices.numel() != tokens:
        raise ValueError("Q8_0 cache indices must match the number of input tokens")
    if tokens == 0:
        return
    blocks = head_dim // 32
    _store_q8_0_kernel[(tokens, heads, blocks)](
        k_cache, v_cache, k_scale, v_scale, indices, k, v,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        NUM_HEADS=heads, HEAD_DIM=head_dim, BLOCKS=blocks, BLOCK=32,
        num_warps=1,
    )


__all__ = ["store_q8_0_cache"]
