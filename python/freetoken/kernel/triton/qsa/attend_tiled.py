"""Tiled QSA sparse attention for prefill: consecutive queries share their K/V loads.

``qsa_sparse_paged_attention`` runs one program per (query row, KV head) and gathers that
row's own selection, ~2k tokens, in 16-token tiles. That is the right shape for decode.
In a prefill, neighbouring queries select almost the same blocks: on Qwen3.8-Flash-Next
at 7.5k tokens a row selects ~55 of its visible 64-token blocks and 32 consecutive rows
together select ~59, so per-row gathers re-read the same pages over and over (QSA
attention was a quarter of a long prefill's GPU time).

Here ``BQ`` consecutive rows of one request form a tile. The union of their selected
blocks is streamed once per KV head (a block is one KV page, ``page_size == ratio == 64``),
and every row applies its own token mask, built from the exact selection, so the result
is the same attention, not an approximation. The online softmax covers all
``BQ * group_size`` query heads of a tile in one dot (M rows).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_TOKENS = 64


@triton.jit
def _qsa_tiled_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, out_ptr,
    tile_row_ptr, tile_req_ptr, tile_rows_ptr, union_ptr, union_count_ptr, mask_ptr,
    block_table_ptr,
    stride_q_row, stride_q_head, stride_k_block, stride_k_token, stride_k_head,
    stride_v_block, stride_v_token, stride_v_head, stride_out_row, stride_out_head,
    stride_table_req,
    U_MAX: tl.constexpr, BQ: tl.constexpr, GROUP_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1)
    row0 = tl.load(tile_row_ptr + tile).to(tl.int64)
    req = tl.load(tile_req_ptr + tile).to(tl.int64)
    n_rows = tl.load(tile_rows_ptr + tile)
    n_union = tl.load(union_count_ptr + tile)

    m = tl.arange(0, BLOCK_M)
    qi = m // GROUP_SIZE                      # query of the tile
    hi = m % GROUP_SIZE                       # head within the KV group
    row_ok = (qi < n_rows) & (m < BQ * GROUP_SIZE)
    head = kv_head * GROUP_SIZE + hi
    d = tl.arange(0, HEAD_DIM)
    query = tl.load(
        q_ptr + (row0 + qi)[:, None] * stride_q_row + head[:, None] * stride_q_head + d[None, :],
        mask=row_ok[:, None], other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    scale: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634
    n = tl.arange(0, BLOCK_N)
    SUB: tl.constexpr = 64 // BLOCK_N

    for u in range(0, n_union):
        block = tl.load(union_ptr + tile * U_MAX + u)
        page = tl.load(block_table_ptr + req * stride_table_req + block).to(tl.int64)
        for sub in tl.static_range(SUB):
            tok = sub * BLOCK_N + n
            keys = tl.load(k_cache_ptr + page * stride_k_block + tok[None, :] * stride_k_token
                           + kv_head * stride_k_head + d[:, None])
            values = tl.load(v_cache_ptr + page * stride_v_block + tok[:, None] * stride_v_token
                             + kv_head * stride_v_head + d[None, :])
            sel = tl.load(mask_ptr + ((tile * BQ + qi[:, None]) * U_MAX + u) * 64 + tok[None, :],
                          mask=row_ok[:, None], other=0)
            s = tl.dot(query, keys) * scale
            s = tl.where(sel != 0, s, -1.0e20)
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.where(sel != 0, tl.math.exp2(s - m_new[:, None]), 0.0)
            acc = tl.dot(p.to(values.dtype), values, acc=acc * alpha[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

    out = tl.where((l_i > 0)[:, None], acc / tl.maximum(l_i[:, None], 1.0e-20), 0.0)
    tl.store(out_ptr + (row0 + qi)[:, None] * stride_out_row + head[:, None] * stride_out_head
             + d[None, :], out, mask=row_ok[:, None])


def _tiles(cu_seqlens_host: list[int], bq: int):
    """(first row, request, rows) per tile; tiles never cross a request."""
    first, req, rows = [], [], []
    for r in range(len(cu_seqlens_host) - 1):
        lo, hi = cu_seqlens_host[r], cu_seqlens_host[r + 1]
        for s in range(lo, hi, bq):
            first.append(s)
            req.append(r)
            rows.append(min(bq, hi - s))
    return first, req, rows


# Tiles prepared and launched at once: bounds the index bookkeeping (a few [tiles, bq, W]
# int tensors, W ~ 2.1k) to ~100 MB whatever the prefill length.
_TILE_GROUP = 256


def _run_group(q, k_cache, v_cache, out, indices, block_table, first, req, rows, bq, group,
               dim, block_m):
    dev = q.device
    n_tiles = len(first)
    kvh = k_cache.shape[2]
    tile_row = torch.tensor(first, dtype=torch.int32).to(dev, non_blocking=True)
    tile_req = torch.tensor(req, dtype=torch.int32).to(dev, non_blocking=True)
    tile_rows = torch.tensor(rows, dtype=torch.int32).to(dev, non_blocking=True)

    # Rows of each tile, padded to bq with rows whose selection is empty.
    w = indices.shape[1]
    q_in_tile = torch.arange(bq, device=dev)
    real = q_in_tile[None, :] < tile_rows[:, None]
    row_ids = torch.where(real, tile_row[:, None] + q_in_tile[None, :], 0)
    sel = indices.index_select(0, row_ids.reshape(-1).long()).view(n_tiles, bq, w)
    valid = real[:, :, None] & (sel >= 0)

    # Union of 64-token blocks per tile: sort, keep the first of each run, compact.
    big = torch.iinfo(torch.int32).max
    blocks = torch.where(valid, sel // BLOCK_TOKENS, big).view(n_tiles, bq * w)
    blocks, _ = torch.sort(blocks, dim=1)
    keep = torch.ones_like(blocks, dtype=torch.bool)
    keep[:, 1:] = blocks[:, 1:] != blocks[:, :-1]
    keep &= blocks != big
    count = keep.sum(dim=1)
    u_max = max(1, int(count.max().item()))
    slot = torch.cumsum(keep, dim=1) - 1
    union = torch.zeros((n_tiles, u_max), dtype=torch.int32, device=dev)
    dump = torch.zeros((n_tiles, u_max + 1), dtype=torch.int32, device=dev)
    dump.scatter_(1, torch.where(keep, slot, u_max), torch.where(keep, blocks, 0))
    union.copy_(dump[:, :u_max])
    del blocks, keep, slot, dump

    # Token mask [tiles, bq, u_max, 64]: a selected token's block is found in its tile's
    # (sorted) union by one search over global (tile, block) keys.
    shift = 1 << 24
    tiles = torch.arange(n_tiles, device=dev, dtype=torch.int64)
    in_union = torch.arange(u_max, device=dev)[None, :] < count[:, None]
    union_keys = torch.where(in_union, tiles[:, None] * shift + union.long(), tiles[:, None] * shift + (shift - 1))
    tok = torch.where(valid, sel, 0).long()
    keys = tiles[:, None, None] * shift + tok // BLOCK_TOKENS
    pos = torch.searchsorted(union_keys.view(-1), keys.view(-1)).view_as(keys) - tiles[:, None, None] * u_max
    flat = ((tiles[:, None, None] * bq + q_in_tile[None, :, None]) * u_max + pos) * BLOCK_TOKENS + tok % BLOCK_TOKENS
    n_mask = n_tiles * bq * u_max * BLOCK_TOKENS
    mask = torch.zeros(n_mask + 1, dtype=torch.uint8, device=dev)
    mask.index_fill_(0, torch.where(valid, flat, n_mask).view(-1), 1)
    del sel, tok, keys, pos, flat, valid

    _qsa_tiled_kernel[(n_tiles, kvh)](
        q, k_cache, v_cache, out,
        tile_row, tile_req, tile_rows, union, count.to(torch.int32), mask,
        block_table,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), out.stride(0), out.stride(1),
        block_table.stride(0),
        U_MAX=u_max, BQ=bq, GROUP_SIZE=group, HEAD_DIM=dim, BLOCK_M=block_m, BLOCK_N=16,
        num_warps=4, num_stages=1,
    )



def qsa_tiled_attention(
    q: torch.Tensor,             # [T, HQ, D]
    k_cache: torch.Tensor,       # [pages, 64, KVH, D]
    v_cache: torch.Tensor,
    indices: torch.Tensor,       # [T, W] logical token ids, -1 padded
    block_table: torch.Tensor,   # [R, pages per request]
    cu_seqlens_host: list[int],  # request row boundaries, on the host
    out: torch.Tensor | None = None,
    bq: int | None = None,
) -> torch.Tensor:
    t, hq, dim = q.shape
    kvh = k_cache.shape[2]
    group = hq // kvh
    assert k_cache.shape[1] == BLOCK_TOKENS, "tiled QSA assumes one 64-token block per page"
    block_m = 64
    bq = bq or max(1, block_m // group)       # 5 queries x 12 heads for Flash-Next
    assert bq * group <= block_m
    if out is None:
        out = torch.empty_like(q)

    first, req, rows = _tiles(cu_seqlens_host, bq)
    for g in range(0, len(first), _TILE_GROUP):
        _run_group(q, k_cache, v_cache, out, indices, block_table,
                   first[g:g + _TILE_GROUP], req[g:g + _TILE_GROUP], rows[g:g + _TILE_GROUP],
                   bq, group, dim, block_m)
    return out


__all__ = ["qsa_tiled_attention"]


def _union_and_mask(indices, first, req, rows, tq):
    """For tiles of ``tq`` rows: (union [G, U] sorted block ids, count [G], token mask
    [G, tq, U * 64] bool) from the exact per-row selection."""
    dev = indices.device
    n_tiles = len(first)
    tile_row = torch.tensor(first, dtype=torch.int64).to(dev, non_blocking=True)
    tile_rows = torch.tensor(rows, dtype=torch.int64).to(dev, non_blocking=True)
    w = indices.shape[1]
    q_in_tile = torch.arange(tq, device=dev)
    # Rows past a request's end repeat its last row: same selection, same output.
    row_ids = tile_row[:, None] + torch.minimum(q_in_tile[None, :], tile_rows[:, None] - 1)
    sel = indices.index_select(0, row_ids.reshape(-1)).view(n_tiles, tq, w)
    valid = sel >= 0

    big = torch.iinfo(torch.int32).max
    blocks = torch.where(valid, sel // BLOCK_TOKENS, big).view(n_tiles, tq * w)
    blocks, _ = torch.sort(blocks, dim=1)
    keep = torch.ones_like(blocks, dtype=torch.bool)
    keep[:, 1:] = blocks[:, 1:] != blocks[:, :-1]
    keep &= blocks != big
    count = keep.sum(dim=1)
    u_max = max(1, int(count.max().item()))
    slot = torch.cumsum(keep, dim=1) - 1
    dump = torch.zeros((n_tiles, u_max + 1), dtype=torch.int32, device=dev)
    dump.scatter_(1, torch.where(keep, slot, u_max), torch.where(keep, blocks, 0))
    union = dump[:, :u_max].contiguous()
    del blocks, keep, slot, dump

    shift = 1 << 24
    tiles = torch.arange(n_tiles, device=dev, dtype=torch.int64)
    in_union = torch.arange(u_max, device=dev)[None, :] < count[:, None]
    union_keys = tiles[:, None] * shift + torch.where(in_union, union.long(), shift - 1)
    tok = torch.where(valid, sel, 0).long()
    keys = tiles[:, None, None] * shift + tok // BLOCK_TOKENS
    pos = torch.searchsorted(union_keys.view(-1), keys.view(-1)).view_as(keys) - tiles[:, None, None] * u_max
    flat = (tiles[:, None, None] * tq + q_in_tile[None, :, None]) * (u_max * BLOCK_TOKENS) \
        + pos * BLOCK_TOKENS + tok % BLOCK_TOKENS
    n_mask = n_tiles * tq * u_max * BLOCK_TOKENS
    mask = torch.zeros(n_mask + 1, dtype=torch.bool, device=dev)
    mask.index_fill_(0, torch.where(valid, flat, n_mask).view(-1), True)
    return union, count, mask[:n_mask].view(n_tiles, tq, u_max * BLOCK_TOKENS), row_ids


@triton.jit
def _masked_softmax_kernel(s_ptr, mask_ptr, p_ptr, length, kvh_tq_group, tq_group, group, tq,
                           scale, BLOCK: tl.constexpr):
    """One score row -> probabilities: scale, the query's token mask, softmax in fp32.
    Rows are laid out [G, KVH, TQ, GROUP, L]; the mask is [G, TQ, L]."""
    r = tl.program_id(0).to(tl.int64)
    gi = r // kvh_tq_group
    qi = (r // group) % tq
    offs = tl.arange(0, BLOCK)
    inb = offs < length
    s = tl.load(s_ptr + r * length + offs, mask=inb, other=0.0).to(tl.float32) * scale
    keep = tl.load(mask_ptr + (gi * tq + qi) * length + offs, mask=inb, other=0) != 0
    s = tl.where(keep, s, -float("inf"))
    m = tl.max(s, axis=0)
    e = tl.where(keep, tl.exp(s - m), 0.0)
    total = tl.sum(e, axis=0)
    p = tl.where(total > 0, e / total, 0.0)
    tl.store(p_ptr + r * length + offs, p.to(p_ptr.dtype.element_ty), mask=inb)


# Rows per tile of the GEMM formulation, and tiles per batched GEMM.
_GEMM_TQ = 32
_GEMM_GROUP = 8


def qsa_gemm_attention(
    q: torch.Tensor,             # [T, HQ, D]
    k_cache: torch.Tensor,       # [pages, 64, KVH, D]
    v_cache: torch.Tensor,
    indices: torch.Tensor,       # [T, W] logical token ids, -1 padded
    block_table: torch.Tensor,   # [R, pages per request]
    cu_seqlens_host: list[int],
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prefill QSA as batched GEMMs: tiles of ``_GEMM_TQ`` consecutive rows gather the
    union of their selected pages once and run QK^T / softmax / PV through rocBLAS. On
    RDNA2 (no matrix units) Triton's dot runs ~1 TFLOP/s; rocBLAS ~10."""
    t, hq, dim = q.shape
    kvh = k_cache.shape[2]
    group = hq // kvh
    tq = _GEMM_TQ
    if out is None:
        out = torch.empty_like(q)
    scale = dim ** -0.5
    first, req, rows = _tiles(cu_seqlens_host, tq)
    for g0 in range(0, len(first), _GEMM_GROUP):
        f, r, n = first[g0:g0 + _GEMM_GROUP], req[g0:g0 + _GEMM_GROUP], rows[g0:g0 + _GEMM_GROUP]
        union, count, mask, row_ids = _union_and_mask(indices, f, r, n, tq)
        g, u_max = union.shape
        reqs = torch.tensor(r, dtype=torch.int64).to(q.device, non_blocking=True)
        pages = block_table.index_select(0, reqs).gather(1, union.long()).long().view(-1)
        length = u_max * BLOCK_TOKENS
        heads = torch.arange(group, device=q.device)
        for h in range(kvh):
            # Gathers land in GEMM layout directly: permuting a gathered [page, token,
            # head, dim] block ran at ~26 GB/s and cost more than the GEMMs.
            kg = k_cache[:, :, h, :].index_select(0, pages).view(g, length, dim)
            vg = v_cache[:, :, h, :].index_select(0, pages).view(g, length, dim)
            flat = (row_ids[:, :, None] * hq + h * group + heads[None, None, :]).view(-1)
            qg = q.view(t * hq, dim).index_select(0, flat).view(g, tq * group, dim)
            s = torch.bmm(qg, kg.transpose(1, 2))                    # [G, TQ*GROUP, L]
            p = torch.empty_like(s)
            _masked_softmax_kernel[(s.numel() // length,)](
                s, mask, p, length, tq * group, tq * group, group, tq, scale,
                BLOCK=triton.next_power_of_2(length), num_warps=8,
            )
            del s
            o = torch.bmm(p, vg)                                     # [G, TQ*GROUP, D]
            out.view(t * hq, dim).index_copy_(0, flat, o.view(-1, dim))
    return out


__all__ += ["qsa_gemm_attention"]
