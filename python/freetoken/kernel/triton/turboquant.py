from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _nearest_centroid(x, midpoints):
    lo = tl.full(x.shape, 0, tl.int32)
    hi = tl.full(x.shape, 15, tl.int32)
    for _ in range(4):
        mid = (lo + hi) // 2
        boundary = tl.load(midpoints + mid)
        above = x >= boundary
        lo = tl.where(above, mid + 1, lo)
        hi = tl.where(above, hi, mid)
    return lo


@triton.jit
def _store_turboquant_kernel(
    k_codes, v_codes, k_norms, v_scales, v_zeros, indices,
    k_rot, k_norm, v, centroids, midpoints,
    stride_kt, stride_kh, stride_vt, stride_vh,
    HEADS: tl.constexpr, D: tl.constexpr, HALF_D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    byte_d = tl.arange(0, BLOCK)
    valid = byte_d < HALF_D
    coord0 = byte_d * 2
    coord1 = coord0 + 1
    x0 = tl.load(k_rot + token * stride_kt + head * stride_kh + coord0, valid, 0).to(tl.float32)
    x1 = tl.load(k_rot + token * stride_kt + head * stride_kh + coord1, valid, 0).to(tl.float32)
    c0 = _nearest_centroid(x0, midpoints)
    c1 = _nearest_centroid(x1, midpoints)
    packed_k = (c0 | (c1 << 4)).to(tl.uint8)

    # Value min/max and quantization use the full head dimension.
    d = tl.arange(0, BLOCK * 2)
    xv = tl.load(v + token * stride_vt + head * stride_vh + d, d < D, 0).to(tl.float32)
    vmin = tl.min(tl.where(d < D, xv, float("inf")), axis=0)
    vmax = tl.max(tl.where(d < D, xv, -float("inf")), axis=0)
    scale = tl.maximum((vmax - vmin) / 15.0, 1.0e-8)
    value0 = tl.load(v + token * stride_vt + head * stride_vh + coord0, valid, 0).to(tl.float32)
    value1 = tl.load(v + token * stride_vt + head * stride_vh + coord1, valid, 0).to(tl.float32)
    q0 = tl.minimum(tl.maximum(tl.floor((value0 - vmin) / scale + 0.5), 0), 15).to(tl.uint8)
    q1 = tl.minimum(tl.maximum(tl.floor((value1 - vmin) / scale + 0.5), 0), 15).to(tl.uint8)
    packed_v = q0 | (q1 << 4)

    slot = tl.load(indices + token).to(tl.int32)
    row = slot * HEADS * HALF_D + head * HALF_D + byte_d
    meta = slot * HEADS + head
    tl.store(k_codes + row, packed_k, valid)
    tl.store(v_codes + row, packed_v, valid)
    tl.store(k_norms + meta, tl.load(k_norm + token * HEADS + head).to(tl.float16))
    tl.store(v_scales + meta, scale.to(tl.float16))
    tl.store(v_zeros + meta, vmin.to(tl.float16))


@triton.jit
def _turboquant_attention_kernel(
    q_ptr, k_codes, v_codes, k_norms, v_scales, v_zeros, centroids,
    o_ptr, indptr, indices, q_to_req, q_pos, sinks,
    stride_qt, stride_qh, stride_ks, stride_kh, stride_vs, stride_vh,
    stride_ot, stride_oh, sm_scale,
    GROUP: tl.constexpr, D: tl.constexpr, HALF_D: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr, HAS_SINKS: tl.constexpr,
):
    q_tok = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // GROUP
    req = tl.load(q_to_req + q_tok)
    kv_start = tl.load(indptr + req)
    kv_end = tl.load(indptr + req + 1)
    kv_len = kv_end - kv_start
    qpos = tl.load(q_pos + q_tok)
    d = tl.arange(0, BLOCK_D)
    mask_d = d < D
    q = tl.load(q_ptr + q_tok * stride_qt + q_head * stride_qh + d, mask_d, 0).to(tl.float32)
    if HAS_SINKS:
        m_i = tl.load(sinks + q_head).to(tl.float32)
        l_i = 1.0
    else:
        m_i = -float("inf")
        l_i = 0.0
    acc = tl.zeros((BLOCK_D,), tl.float32)

    for start in range(0, kv_len, BLOCK_N):
        rows = start + tl.arange(0, BLOCK_N)
        valid_row = rows < kv_len
        causal = rows <= qpos
        if SLIDING_WINDOW > 0:
            causal = causal & (rows + SLIDING_WINDOW > qpos)
        active = valid_row & causal
        slots = tl.load(indices + kv_start + rows, valid_row, 0).to(tl.int32)
        packed_off = d[None, :] // 2
        shifts = (d[None, :] % 2) * 4
        kp = tl.load(
            k_codes + slots[:, None] * stride_ks + kv_head * stride_kh + packed_off,
            mask=active[:, None] & mask_d[None, :], other=0,
        ).to(tl.uint32)
        ki = ((kp >> shifts.to(tl.uint32)) & 15).to(tl.int32)
        kval = tl.load(centroids + ki, mask=mask_d[None, :] & active[:, None], other=0).to(tl.float32)
        knorm = tl.sqrt(tl.sum(kval * kval, axis=1))
        kval = kval / tl.maximum(knorm[:, None], 1.0e-12)
        orig_norm = tl.load(k_norms + slots * (stride_ks // HALF_D) + kv_head)
        scores = tl.sum(q[None, :] * kval, axis=1) * orig_norm.to(tl.float32) * sm_scale
        scores = tl.where(active, scores, -float("inf"))
        row_max = tl.max(scores, axis=0)
        row_max = tl.where(row_max == -float("inf"), -1.0e20, row_max)
        m_new = tl.maximum(m_i, row_max)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        vp = tl.load(
            v_codes + slots[:, None] * stride_vs + kv_head * stride_vh + packed_off,
            mask=active[:, None] & mask_d[None, :], other=0,
        ).to(tl.uint32)
        vi = ((vp >> shifts.to(tl.uint32)) & 15).to(tl.float32)
        vscale = tl.load(v_scales + slots * (stride_vs // HALF_D) + kv_head).to(tl.float32)
        vzero = tl.load(v_zeros + slots * (stride_vs // HALF_D) + kv_head).to(tl.float32)
        v = vi * vscale[:, None] + vzero[:, None]
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
    out = tl.where(l_i == 0.0, 0.0, acc / l_i)
    tl.store(o_ptr + q_tok * stride_ot + q_head * stride_oh + d, out, mask_d)


def hadamard_rotation(head_dim: int, device: torch.device) -> torch.Tensor:
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError(f"TurboQuant requires a power-of-two head_dim, got {head_dim}")
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < head_dim:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    generator = torch.Generator().manual_seed(0)
    signs = torch.randint(0, 2, (head_dim,), generator=generator, dtype=torch.int32)
    signs = signs.to(torch.float32).mul_(2).sub_(1)
    return ((signs[:, None] * h) / (head_dim**0.5)).to(device=device)


def gaussian_centroids(head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    import numpy as np

    sigma = 1.0 / head_dim**0.5
    samples = np.linspace(-6.0, 6.0, 131073, dtype=np.float64)
    weights = np.exp(-0.5 * samples**2)
    centers = np.linspace(-2.7, 2.7, 16, dtype=np.float64)
    for _ in range(64):
        boundaries = (centers[:-1] + centers[1:]) * 0.5
        groups = np.searchsorted(boundaries, samples)
        mass = np.bincount(groups, weights=weights, minlength=16)
        means = np.bincount(groups, weights=weights * samples, minlength=16) / mass
        if np.max(np.abs(means - centers)) < 1e-12:
            centers = means
            break
        centers = means
    for i in range(8):
        magnitude = (centers[15 - i] - centers[i]) * 0.5
        centers[i] = -magnitude
        centers[15 - i] = magnitude
    centers *= sigma
    midpoints = (centers[:-1] + centers[1:]) * 0.5
    return torch.from_numpy(centers.astype(np.float32)), torch.from_numpy(midpoints.astype(np.float32))


def store_turboquant_4bit_nc_cache(
    *, k_cache, v_cache, k_norms, v_scales, v_zeros, indices, k, v, rotation, centroids, midpoints
) -> None:
    tokens, heads, d = k.shape
    if d % 2:
        raise ValueError(f"TurboQuant 4-bit requires even head_dim, got {d}")
    if tokens == 0:
        return
    k32 = k.float()
    norms = torch.linalg.vector_norm(k32, dim=-1).clamp_min(1.0e-12)
    rotated = torch.matmul(k32 / norms.unsqueeze(-1), rotation.T).contiguous()
    block = triton.next_power_of_2(d // 2)
    v = v.contiguous()
    _store_turboquant_kernel[(tokens, heads)](
        k_cache, v_cache, k_norms, v_scales, v_zeros, indices,
        rotated, norms, v, centroids, midpoints,
        rotated.stride(0), rotated.stride(1), v.stride(0), v.stride(1),
        HEADS=heads, D=d, HALF_D=d // 2, BLOCK=block,
        num_warps=4,
    )


def turboquant_4bit_nc_attention(
    *, q, k_cache, v_cache, k_norms, v_scales, v_zeros, centroids,
    indptr, indices, q_to_req, q_positions, sm_scale, rotation,
    sliding_window=None, sinks=None, block_n=16,
):
    tokens, q_heads, d = q.shape
    kv_heads = k_cache.shape[1]
    if d % 2 or d & (d - 1):
        raise ValueError(f"TurboQuant requires a power-of-two even head_dim, got {d}")
    q_rot = torch.matmul(q.float(), rotation.T).contiguous()
    out = torch.empty_like(q)
    _turboquant_attention_kernel[(tokens, q_heads)](
        q_rot, k_cache, v_cache, k_norms, v_scales, v_zeros, centroids,
        out, indptr, indices, q_to_req, q_positions,
        sinks if sinks is not None else q,
        q_rot.stride(0), q_rot.stride(1), k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1), out.stride(0), out.stride(1), sm_scale,
        GROUP=q_heads // kv_heads, D=d, HALF_D=d // 2,
        BLOCK_D=triton.next_power_of_2(d), BLOCK_N=block_n,
        SLIDING_WINDOW=sliding_window or 0, HAS_SINKS=sinks is not None,
        num_warps=8 if d >= 256 else 4, num_stages=1,
    )
    return out


__all__ = [
    "gaussian_centroids",
    "hadamard_rotation",
    "store_turboquant_4bit_nc_cache",
    "turboquant_4bit_nc_attention",
]
