"""QSA prefill attention as tiled batched GEMMs.

``qsa_sparse_paged_attention`` runs one program per (query row, KV head) and gathers that
row's own selection in small tiles, the right shape for decode. In a prefill, neighbouring
rows select almost the same pages, so here ``_TILE_ROWS`` consecutive rows of one request
form a tile: the union of their selected KV pages is gathered once per KV head and the tile
runs QK^T / masked softmax / PV as batched GEMMs. Every row keeps its own token mask, built
from the exact selection, so this is the same attention, not an approximation. The point is
the hardware: on RDNA2 (no matrix units) Triton's ``tl.dot`` runs ~10x slower than rocBLAS.

A tile's union can span the whole context, so its scores, probabilities and gathered K/V
grow with it. Tiles run in batches sized by a byte budget (``_GEMM_BYTES``), and a tile
whose own union does not fit the budget falls back to the per-row sparse kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_TOKENS = 64
_TILE_ROWS = 32
# Tiles whose page unions are built together (one host sync for the largest union).
_TILE_GROUP = 8
# Cap on one batched GEMM's scores + probabilities + gathered K/V (an uncapped 8-tile group
# OOMed a 16 GB card at ~69k context).
_GEMM_BYTES = 128 << 20
# A tile's union grows toward the whole visible context as the context grows, while each row still
# selects only ``indices.shape[1]`` tokens: past this many selections' worth of union the GEMMs
# do more work than the per-row kernel saves (RX 6800, Flash-Next: the union is ~1x the selection
# in a first 4k chunk and ~6.6x by 15k).
_MAX_UNION_RATIO = 3.0


def _tiles(cu_seqlens_host: list[int], rows: int):
    """(first row, request, rows) per tile; tiles never cross a request."""
    first, req, count = [], [], []
    for r in range(len(cu_seqlens_host) - 1):
        lo, hi = cu_seqlens_host[r], cu_seqlens_host[r + 1]
        for start in range(lo, hi, rows):
            first.append(start)
            req.append(r)
            count.append(min(rows, hi - start))
    return first, req, count


def _union_and_rows(indices, first, count, page_count):
    """Selected-page membership per tile, and each tile row's selection."""
    dev = indices.device
    n_tiles = len(first)
    tile_row = torch.tensor(first, dtype=torch.int64).to(dev, non_blocking=True)
    tile_rows = torch.tensor(count, dtype=torch.int64).to(dev, non_blocking=True)
    width = indices.shape[1]
    in_tile = torch.arange(_TILE_ROWS, device=dev)
    # Rows past a request's end repeat its last row: same selection, same output.
    row_ids = tile_row[:, None] + torch.minimum(in_tile[None, :], tile_rows[:, None] - 1)
    sel = indices.index_select(0, row_ids.reshape(-1)).view(n_tiles, _TILE_ROWS, width)
    valid = sel >= 0
    # The page-table width bounds the union: mark membership instead of sorting rows * width ids.
    pages = torch.where(valid, sel // BLOCK_TOKENS, page_count).long().view(n_tiles, -1)
    seen = torch.zeros((n_tiles, page_count + 1), dtype=torch.bool, device=dev)
    seen.scatter_(1, pages, True)
    seen = seen[:, :page_count]
    return seen, seen.sum(dim=1), row_ids, sel, valid


def _compact_union(seen, u_max):
    """Ascending logical page ids of each tile's union, zero padded to ``u_max``."""
    n_tiles, page_count = seen.shape
    dev = seen.device
    slot = seen.to(torch.int32).cumsum(dim=1) - 1
    tile_ids = torch.arange(n_tiles, device=dev, dtype=torch.int64)[:, None]
    page_ids = torch.arange(page_count, device=dev, dtype=torch.int32)[None, :]
    union = torch.zeros((n_tiles, u_max), dtype=torch.int32, device=dev)
    union.view(-1).index_copy_(0, (tile_ids * u_max + slot)[seen], page_ids.expand(n_tiles, -1)[seen])
    return union


def _token_mask(sel, valid, union, count):
    """Bool ``[tiles, rows, u_max * 64]``: which gathered tokens each row selected."""
    dev = sel.device
    n_tiles, u_max = union.shape
    in_tile = torch.arange(_TILE_ROWS, device=dev)
    shift = 1 << 24
    tiles = torch.arange(n_tiles, device=dev, dtype=torch.int64)
    in_union = torch.arange(u_max, device=dev)[None, :] < count[:, None]
    # One search over global (tile, page) keys finds each selected token's page in its union.
    union_keys = tiles[:, None] * shift + torch.where(in_union, union.long(), shift - 1)
    tok = torch.where(valid, sel, 0).long()
    keys = tiles[:, None, None] * shift + tok // BLOCK_TOKENS
    pos = torch.searchsorted(union_keys.view(-1), keys.view(-1)).view_as(keys) - tiles[:, None, None] * u_max
    flat = (tiles[:, None, None] * _TILE_ROWS + in_tile[None, :, None]) * (u_max * BLOCK_TOKENS) \
        + pos * BLOCK_TOKENS + tok % BLOCK_TOKENS
    n_mask = n_tiles * _TILE_ROWS * u_max * BLOCK_TOKENS
    mask = torch.zeros(n_mask + 1, dtype=torch.bool, device=dev)
    mask.index_fill_(0, torch.where(valid, flat, n_mask).view(-1), True)
    return mask[:n_mask].view(n_tiles, _TILE_ROWS, u_max * BLOCK_TOKENS)


@triton.jit
def _masked_softmax_kernel(
    s_ptr, mask_ptr, p_ptr, length, tile_heads, group, scale, ROWS: tl.constexpr, BLOCK: tl.constexpr
):
    """One score row to probabilities: scale, mask and softmax in fp32."""
    r = tl.program_id(0).to(tl.int64)
    tile = r // tile_heads
    row = (r // group) % ROWS
    offs = tl.arange(0, BLOCK)
    inb = offs < length
    s = tl.load(s_ptr + r * length + offs, mask=inb, other=0.0).to(tl.float32) * scale
    keep = tl.load(mask_ptr + (tile * ROWS + row) * length + offs, mask=inb, other=0) != 0
    s = tl.where(keep, s, -float("inf"))
    m = tl.max(s, axis=0)
    e = tl.where(keep, tl.exp(s - m), 0.0)
    total = tl.sum(e, axis=0)
    p = tl.where(total > 0, e / total, 0.0)
    tl.store(p_ptr + r * length + offs, p.to(p_ptr.dtype.element_ty), mask=inb)


@triton.jit
def _turboquant_gather_kernel(
    k_cache_ptr, v_cache_ptr, k_norms_ptr, v_scales_ptr, v_zeros_ptr, centroids_ptr,
    pages_ptr, active_ptr, k_out_ptr, v_out_ptr,
    length, u_max, head, dim,
    stride_k_page, stride_k_token, stride_k_head,
    stride_v_page, stride_v_token, stride_v_head,
    stride_meta_page, stride_meta_token, stride_meta_head,
    PAGE_TOKENS: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Gather one KV head's page union and decode its packed K/V into the GEMM layout.

    K stays in the rotated basis the codes live in (same decode as the sparse kernel: the
    codebook vector renormalized, then scaled by the stored source norm); positions no row
    of the tile selected decode to zero, since their unwritten metadata is undefined."""
    row = tl.program_id(0).to(tl.int64)
    tile = row // length
    pos = row % length
    page = tl.load(pages_ptr + tile * u_max + pos // PAGE_TOKENS).to(tl.int64)
    token = pos % PAGE_TOKENS
    d = tl.arange(0, BLOCK_D)
    in_dim = d < dim
    active = tl.load(active_ptr + row)
    meta = page * stride_meta_page + token * stride_meta_token + head * stride_meta_head

    k_byte = tl.load(
        k_cache_ptr + page * stride_k_page + token * stride_k_token + head * stride_k_head + d // 2,
        mask=in_dim, other=0,
    ).to(tl.int32)
    centroid = tl.load(centroids_ptr + ((k_byte >> ((d % 2) * 4)) & 15), mask=in_dim, other=0).to(tl.float32)
    code_norm = tl.maximum(tl.sqrt(tl.sum(centroid * centroid, axis=0)), 1.0e-12)
    key = tl.where(active, centroid * (tl.load(k_norms_ptr + meta).to(tl.float32) / code_norm), 0.0)

    v_byte = tl.load(
        v_cache_ptr + page * stride_v_page + token * stride_v_token + head * stride_v_head + d // 2,
        mask=in_dim, other=0,
    ).to(tl.int32)
    v_code = ((v_byte >> ((d % 2) * 4)) & 15).to(tl.float32)
    value = tl.where(
        active, v_code * tl.load(v_scales_ptr + meta).to(tl.float32) + tl.load(v_zeros_ptr + meta).to(tl.float32), 0.0
    )
    tl.store(k_out_ptr + row * dim + d, key, mask=in_dim)
    tl.store(v_out_ptr + row * dim + d, value, mask=in_dim)


def _tile_bytes(length: int, group: int, dim: int, element_size: int, turboquant: bool) -> int:
    """Estimated scratch of one tile's GEMMs for one KV head: scores and probabilities
    (fp32 under TurboQuant), gathered K and V, and under TurboQuant the fp32 V copy the PV
    product reads. The 3/2 covers the mask, gather and index temporaries."""
    score_bytes = 4 if turboquant else element_size
    total = (2 * _TILE_ROWS * group * score_bytes + 2 * dim * element_size) * length
    if turboquant:
        total += 4 * dim * length
    return total * 3 // 2


def qsa_gemm_attention(
    q: torch.Tensor,             # [T, HQ, D]
    k_cache: torch.Tensor,       # [pages, 64, KVH, D] (D // 2 packed uint8 under TurboQuant)
    v_cache: torch.Tensor,
    indices: torch.Tensor,       # [T, W] logical token ids, -1 padded
    block_table: torch.Tensor,   # [R, pages per request]
    token_to_req: torch.Tensor,  # [T] int32
    cu_seqlens_host: list[int],  # request row boundaries, on the host
    out: torch.Tensor | None = None,
    *,
    k_norms: torch.Tensor | None = None,
    v_scales: torch.Tensor | None = None,
    v_zeros: torch.Tensor | None = None,
    centroids: torch.Tensor | None = None,
    rotation: torch.Tensor | None = None,
    gemm_bytes: int | None = None,
    max_union_ratio: float | None = None,
) -> torch.Tensor:
    """Prefill QSA over tiles of consecutive rows (see the module docstring).

    TurboQuant caches take the same metadata as :func:`qsa_sparse_paged_attention`. Tiles
    whose union does not fit ``gemm_bytes`` (default ``_GEMM_BYTES``) run through that
    kernel, contiguous ranges at a time."""
    from .attend import qsa_sparse_paged_attention

    _, hq, dim = q.shape
    kvh = k_cache.shape[2]
    group = hq // kvh
    turboquant = k_norms is not None
    if k_cache.shape[1] != BLOCK_TOKENS:
        raise ValueError("tiled QSA assumes one 64-token block per page")
    if turboquant and any(x is None for x in (v_scales, v_zeros, centroids, rotation)):
        raise ValueError("TurboQuant QSA GEMM requires all quantization metadata")
    budget = _GEMM_BYTES if gemm_bytes is None else gemm_bytes
    ratio = _MAX_UNION_RATIO if max_union_ratio is None else max_union_ratio
    max_union = ratio * indices.shape[1] if ratio > 0 else float("inf")
    if out is None:
        out = torch.empty_like(q)
    quant = dict(k_norms=k_norms, v_scales=v_scales, v_zeros=v_zeros, centroids=centroids)
    q_gemm = None  # the rotated query, built once a tile actually runs as a GEMM

    def sparse(lo, hi):
        sparse_quant = dict(quant, rotation=rotation) if turboquant else {}
        qsa_sparse_paged_attention(
            q[lo:hi], k_cache, v_cache, indices[lo:hi], block_table, token_to_req[lo:hi], out[lo:hi],
            **sparse_quant,
        )

    first, req, count = _tiles(cu_seqlens_host, _TILE_ROWS)
    pending = None  # [lo, hi) rows waiting for the sparse kernel
    for g0 in range(0, len(first), _TILE_GROUP):
        f, r, n = first[g0:g0 + _TILE_GROUP], req[g0:g0 + _TILE_GROUP], count[g0:g0 + _TILE_GROUP]
        seen, union_count, row_ids, sel, valid = _union_and_rows(indices, f, n, block_table.shape[1])
        tiles = union_count.numel()
        u_max = max(1, int(union_count.max().item()))
        length = u_max * BLOCK_TOKENS
        tile_bytes = _tile_bytes(length, group, dim, q.element_size(), turboquant)
        if tile_bytes > budget or length > max_union:
            pending = (f[0], f[-1] + n[-1]) if pending is None else (pending[0], f[-1] + n[-1])
            continue
        if pending is not None:
            sparse(*pending)
            pending = None
        union = _compact_union(seen, u_max)
        mask = _token_mask(sel, valid, union, union_count)
        reqs = torch.tensor(r, dtype=torch.int64).to(q.device, non_blocking=True)
        pages = block_table.index_select(0, reqs).gather(1, union.long()).long()
        if turboquant and q_gemm is None:
            q_gemm = torch.matmul(q.float(), rotation.T).to(q.dtype).contiguous()
        per = max(1, min(tiles, budget // tile_bytes))
        for s0 in range(0, tiles, per):
            s1 = min(tiles, s0 + per)
            _gemm_tiles(
                q_gemm if turboquant else q, k_cache, v_cache, out, pages[s0:s1].reshape(-1), mask[s0:s1],
                row_ids[s0:s1], kvh, group, length, dim, **quant,
            )
    if pending is not None:
        sparse(*pending)
    return out


def _gemm_tiles(q, k_cache, v_cache, out, pages, mask, row_ids, kvh, group, length, dim, *,
                k_norms=None, v_scales=None, v_zeros=None, centroids=None):
    t, hq, _ = q.shape
    tiles = row_ids.shape[0]
    turboquant = k_norms is not None
    heads = torch.arange(group, device=q.device)
    active = mask.any(dim=1) if turboquant else None
    for h in range(kvh):
        # Gathers land in the GEMM layout directly: permuting a gathered [page, token, head,
        # dim] block ran at ~26 GB/s and cost more than the GEMMs.
        if turboquant:
            kg = torch.empty((tiles, length, dim), dtype=q.dtype, device=q.device)
            vg = torch.empty_like(kg)
            _turboquant_gather_kernel[(tiles * length,)](
                k_cache, v_cache, k_norms, v_scales, v_zeros, centroids,
                pages, active, kg, vg,
                length, length // BLOCK_TOKENS, h, dim,
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
                k_cache.shape[1] * k_cache.shape[2], k_cache.shape[2], 1,
                PAGE_TOKENS=BLOCK_TOKENS, BLOCK_D=triton.next_power_of_2(dim),
                num_warps=4,
            )
        else:
            kg = k_cache[:, :, h, :].index_select(0, pages).view(tiles, length, dim)
            vg = v_cache[:, :, h, :].index_select(0, pages).view(tiles, length, dim)
        flat = (row_ids[:, :, None] * hq + h * group + heads[None, None, :]).view(-1)
        rows = row_ids.shape[1]
        qg = q.view(t * hq, dim).index_select(0, flat).view(tiles, rows * group, dim)
        # Model-dtype scores are rounded to the model dtype; under TurboQuant keep fp32 so the
        # small differences between quantized keys survive into the softmax.
        s = torch.bmm(qg, kg.transpose(1, 2), out_dtype=torch.float32) if turboquant else torch.bmm(qg, kg.transpose(1, 2))
        p = torch.empty_like(s)
        _masked_softmax_kernel[(s.numel() // length,)](
            s, mask, p, length, rows * group, group, dim**-0.5, ROWS=rows,
            BLOCK=triton.next_power_of_2(length), num_warps=8,
        )
        del s
        # The quantized path accumulates PV in fp32 (fp16 operands: an fp32 GEMM runs at a fraction
        # of the fp16 rate on RDNA2); weights below fp16's subnormal range contribute nothing visible.
        o = torch.bmm(p.to(vg.dtype), vg, out_dtype=torch.float32).to(out.dtype) if turboquant else torch.bmm(p, vg)
        out.view(t * hq, dim).index_copy_(0, flat, o.view(-1, dim))


__all__ = ["qsa_gemm_attention"]
