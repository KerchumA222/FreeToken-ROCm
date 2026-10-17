from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="TurboQuant Triton kernels need CUDA")


def _init_tp() -> None:
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _pack_nibbles(values: torch.Tensor) -> torch.Tensor:
    return (values[..., 0::2] | (values[..., 1::2] << 4)).to(torch.uint8)


def _unpack_nibbles(packed: torch.Tensor, dim: int) -> torch.Tensor:
    lo = packed.to(torch.int32) & 15
    hi = packed.to(torch.int32) >> 4
    return torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], dim)


def _reference_attention(
    q,
    k_rot,
    k_norms,
    v_decoded,
    indptr,
    indices,
    q_to_req,
    q_positions,
    rotation,
    sm_scale,
    *,
    sliding_window,
    block_ends,
):
    q_rot = q.float() @ rotation.T
    out = torch.zeros_like(q, dtype=torch.float32)
    group = q.shape[1] // k_rot.shape[1]
    for qi in range(q.shape[0]):
        req = int(q_to_req[qi])
        start, end = int(indptr[req]), int(indptr[req + 1])
        slots = indices[start:end].long()
        positions = torch.arange(end - start, device=q.device)
        mask = positions <= q_positions[qi]
        if int(block_ends[qi]) > 0:
            mask |= positions < block_ends[qi]
        if sliding_window:
            mask &= positions + sliding_window > q_positions[qi]
        for qh in range(q.shape[1]):
            kvh = qh // group
            scores = (
                k_rot[slots, kvh] @ q_rot[qi, qh]
            ) * k_norms[slots, kvh] * sm_scale
            weights = torch.softmax(scores[mask], dim=0)
            out[qi, qh] = weights @ v_decoded[slots[mask], kvh]
    return out.to(q.dtype)


@pytest.mark.parametrize("flattened_input", [False, True])
def test_turboquant_store_and_attention_match_decoded_reference(flattened_input):
    from freetoken.kernel.triton.turboquant import turboquant_4bit_nc_attention
    from freetoken.kvcache.mha_pool import MHAKVCache

    _init_tp()
    torch.manual_seed(83)
    device = torch.device("cuda")
    slots, kv_heads, q_heads, dim = 40, 2, 4, 64
    pool = MHAKVCache(
        num_kv_heads=kv_heads,
        num_layers=1,
        head_dim=dim,
        num_pages=10,
        page_size=4,
        dtype=torch.bfloat16,
        device=device,
        kv_cache_dtype="turboquant_4bit_nc",
    )
    k = torch.randn(slots, kv_heads, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    k_input, v_input = k, v
    if flattened_input:
        width = kv_heads * dim
        # Match qkv.split(): a flattened 2D view with a wider row stride.
        k_rows = torch.empty(slots, width + 7, device=device, dtype=k.dtype)
        v_rows = torch.empty_like(k_rows)
        k_rows[:, :width].copy_(k.view(slots, width))
        v_rows[:, :width].copy_(v.view(slots, width))
        k_input, v_input = k_rows[:, :width], v_rows[:, :width]
        assert not k_input.is_contiguous() and not v_input.is_contiguous()
    out_loc = torch.arange(slots, device=device, dtype=torch.int32)
    pool.store_kv(k_input, v_input, out_loc, layer_id=0)
    torch.cuda.synchronize()

    norms = torch.linalg.vector_norm(k.float(), dim=-1).clamp_min(1.0e-12)
    k_rot = (k.float() / norms.unsqueeze(-1)) @ pool.turboquant_rotation.T
    key_ids = torch.bucketize(k_rot.contiguous(), pool._midpoints, right=True)
    expected_k = _pack_nibbles(key_ids)
    actual_k = pool.k_cache(0).view(slots, kv_heads, dim // 2)
    assert torch.equal(actual_k, expected_k)
    torch.testing.assert_close(
        pool.k_norm_cache(0).float(), norms.to(torch.float16).float()
    )

    v_float = v.float()
    v_min = v_float.amin(dim=-1)
    v_max = v_float.amax(dim=-1)
    v_scale = ((v_max - v_min) / 15.0).clamp_min(1.0e-8)
    v_ids = ((v_float - v_min.unsqueeze(-1)) / v_scale.unsqueeze(-1) + 0.5).floor().clamp(0, 15).to(torch.int32)
    actual_v = pool.v_cache(0).view(slots, kv_heads, dim // 2)
    actual_v_ids = _unpack_nibbles(actual_v, dim).to(torch.int32)
    assert (actual_v_ids - v_ids).abs().max() <= 1
    torch.testing.assert_close(
        pool.v_scale_cache(0).float(), v_scale.to(torch.float16).float()
    )
    torch.testing.assert_close(
        pool.v_zero_cache(0).float(), v_min.to(torch.float16).float()
    )
    restored_v = (
        actual_v_ids.float() * pool.v_scale_cache(0).float().unsqueeze(-1)
        + pool.v_zero_cache(0).float().unsqueeze(-1)
    )
    quantization_bound = (
        v_scale.unsqueeze(-1) * 0.5
        + (v_scale - pool.v_scale_cache(0).float()).abs().unsqueeze(-1) * 15
        + (v_min - pool.v_zero_cache(0).float()).abs().unsqueeze(-1)
        + 1e-3
    )
    assert torch.all((restored_v - v_float).abs() <= quantization_bound)

    decoded_k_ids = _unpack_nibbles(actual_k, dim).long()
    decoded_k_rot = pool.turboquant_centroids[decoded_k_ids]
    decoded_k_rot = decoded_k_rot / decoded_k_rot.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    decoded_v_ids = _unpack_nibbles(actual_v, dim).float()
    decoded_v = (
        decoded_v_ids * pool.v_scale_cache(0).float().unsqueeze(-1)
        + pool.v_zero_cache(0).float().unsqueeze(-1)
    )

    q = torch.randn(3, q_heads, dim, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor([0, 20, 40], device=device, dtype=torch.int32)
    indices = torch.arange(slots, device=device, dtype=torch.int32)
    q_to_req = torch.tensor([0, 0, 1], device=device, dtype=torch.int32)
    q_positions = torch.tensor([1, 19, 19], device=device, dtype=torch.int64)
    block_ends = torch.tensor([20, 0, 0], device=device, dtype=torch.int32)
    sm_scale = dim**-0.5

    actual = turboquant_4bit_nc_attention(
        q=q,
        k_cache=actual_k,
        v_cache=actual_v,
        k_norms=pool.k_norm_cache(0),
        v_scales=pool.v_scale_cache(0),
        v_zeros=pool.v_zero_cache(0),
        centroids=pool.turboquant_centroids,
        indptr=indptr,
        indices=indices,
        q_to_req=q_to_req,
        q_positions=q_positions,
        sm_scale=sm_scale,
        rotation=pool.turboquant_rotation,
        sliding_window=3,
        block_ends=block_ends,
    )
    expected = _reference_attention(
        q,
        decoded_k_rot,
        pool.k_norm_cache(0).float(),
        decoded_v,
        indptr,
        indices,
        q_to_req,
        q_positions,
        pool.turboquant_rotation,
        sm_scale,
        sliding_window=3,
        block_ends=block_ends,
    )
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
def test_triton_backend_routes_turboquant_cache_to_quantized_attention(monkeypatch):
    from types import SimpleNamespace

    from freetoken.attention import AttentionSpec
    from freetoken.attention.triton import TritonAttentionBackend

    device = torch.device("cuda")
    dim, kv_heads, q_heads = 64, 2, 4
    pool = SimpleNamespace(
        device=device,
        dtype=torch.uint8,
        kv_cache_dtype="turboquant_4bit_nc",
        k=torch.zeros(2, 4, kv_heads, dim // 2, device=device, dtype=torch.uint8),
        v=torch.zeros(2, 4, kv_heads, dim // 2, device=device, dtype=torch.uint8),
        k_norms=torch.ones(8, kv_heads, device=device, dtype=torch.float16),
        v_scales=torch.ones(8, kv_heads, device=device, dtype=torch.float16),
        v_zeros=torch.zeros(8, kv_heads, device=device, dtype=torch.float16),
        rotation=torch.eye(dim, device=device),
        centroids=torch.linspace(-1, 1, 16, device=device),
        stored=False,
    )
    pool.store_kv = lambda *args, **kwargs: setattr(pool, "stored", True)
    pool.k_cache = lambda layer_id: pool.k
    pool.v_cache = lambda layer_id: pool.v
    pool.k_norm_cache = lambda layer_id: pool.k_norms
    pool.v_scale_cache = lambda layer_id: pool.v_scales
    pool.v_zero_cache = lambda layer_id: pool.v_zeros
    pool.turboquant_rotation = pool.rotation
    pool.turboquant_centroids = pool.centroids
    ctx = SimpleNamespace(
        kv_cache=pool,
        page_table=torch.tensor([[0, 1]], device=device, dtype=torch.int32),
    )
    monkeypatch.setattr("freetoken.attention.triton.get_global_ctx", lambda: ctx)
    captured = {}

    def fake_turboquant_attention(**kwargs):
        captured.update(kwargs)
        return kwargs["q"].clone()

    monkeypatch.setattr(
        "freetoken.kernel.triton.turboquant.turboquant_4bit_nc_attention",
        fake_turboquant_attention,
    )
    config = SimpleNamespace(
        num_qo_heads=q_heads,
        head_dim=dim,
        kv_cache_group_specs=lambda: [SimpleNamespace(head_dim=dim)],
    )
    backend = TritonAttentionBackend(config)
    batch = SimpleNamespace(
        padded_reqs=[
            SimpleNamespace(extend_len=2, device_len=2, cached_len=0, table_idx=0)
        ],
        positions=torch.tensor([0, 1], device=device, dtype=torch.int64),
        out_loc=torch.tensor([0, 1], device=device, dtype=torch.int32),
    )
    backend.prepare_metadata(batch)
    q = torch.randn(2, q_heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(2, kv_heads, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    actual = backend.forward(
        q, k, v, layer_id=0, batch=batch, attn_spec=AttentionSpec()
    )

    assert pool.stored
    assert actual is not q
    assert captured["k_cache"].shape == (8, kv_heads, dim // 2)
    assert captured["v_cache"].shape == (8, kv_heads, dim // 2)
    assert captured["q_positions"].equal(batch.attn_metadata.q_positions)
    assert captured["rotation"] is pool.rotation
