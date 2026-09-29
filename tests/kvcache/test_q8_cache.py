from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.distributed import set_tp_info, try_get_tp_info


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _q8_0_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = x.float().reshape(*x.shape[:-1], -1, 32)
    scale_fp32 = blocks.abs().amax(dim=-1) / 127.0
    inverse = torch.where(scale_fp32 > 0, 1.0 / scale_fp32, 0.0)
    scaled = blocks * inverse.unsqueeze(-1)
    codes = torch.sign(scaled) * torch.floor(scaled.abs() + 0.5)
    return codes.to(torch.int8).reshape_as(x), scale_fp32.to(torch.float16)


def test_q8_0_reference_uses_llama_rounding_and_zero_blocks():
    x = torch.zeros((1, 1, 64), dtype=torch.float32)
    x[0, 0, :4] = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    codes, scales = _q8_0_reference(x)

    assert codes[0, 0, :4].tolist() == [-127, -64, 64, 127]
    assert torch.count_nonzero(codes[0, 0, 36:]) == 0
    assert scales[0, 0, 1].item() == 0.0


def test_q8_0_pool_accounts_for_scale_bytes_and_rebuilds():
    from freetoken.kvcache.mha_pool import MHAKVCache

    _init_tp()
    pool = MHAKVCache(
        num_kv_heads=2,
        num_layers=3,
        head_dim=64,
        num_pages=5,
        page_size=16,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        layer_ids=(0, 2),
        kv_cache_dtype="q8_0",
    )
    assert pool.k_cache(2).dtype == torch.int8
    assert pool.k_scale_cache(2).dtype == torch.float16
    assert pool.k_scale_cache(2).shape == (5, 16, 2, 2)
    assert pool.unit_bytes()[0] == 2 * 2 * 2 * 64 + 2 * 2 * 2 * 2 * 2

    pool.rebuild(7)
    assert pool.k_cache(2).shape == (7, 16, 2, 64)
    assert pool.v_scale_cache(2).shape == (7, 16, 2, 2)


def test_q8_0_pool_cost_includes_fp16_block_scales():
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import KVCacheGroupSpec
    from freetoken.attention import AttnType

    _init_tp()
    spec = KVCacheGroupSpec(
        name="full", layer_ids=(0, 1), num_kv_heads=4, head_dim=64,
        sliding_window=None, attn_type=AttnType.FULL,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(kv_cache_group_specs=lambda: (spec,)),
        dtype=torch.bfloat16,
        kv_cache_dtype="q8_0",
        tp_info=SimpleNamespace(size=1),
        page_size=16,
    )
    per_page, fixed, page_tokens, reserve = MHAKVCache.kv_cost(config)
    expected_per_token = 2 * 2 * (64 + 2 * 2) * 4
    assert (per_page, fixed, page_tokens, reserve) == (
        expected_per_token * 16, 0, 16, 0
    )


def test_turboquant_4bit_nc_pool_accounts_for_codes_and_metadata():
    from freetoken.kvcache.mha_pool import MHAKVCache

    _init_tp()
    pool = MHAKVCache(
        num_kv_heads=2,
        num_layers=3,
        head_dim=64,
        num_pages=5,
        page_size=16,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        layer_ids=(0, 2),
        kv_cache_dtype="turboquant_4bit_nc",
    )
    assert pool.k_cache(2).dtype == torch.uint8
    assert pool.k_cache(2).shape == (5, 16, 2, 32)
    assert pool.k_norm_cache(2).shape == (5, 16, 2)
    assert pool.v_scale_cache(2).shape == (5, 16, 2)
    assert pool.v_zero_cache(2).shape == (5, 16, 2)
    assert pool.unit_bytes()[0] == 4 * (64 + 6)

    pool.rebuild(7)
    assert pool.k_cache(2).shape == (7, 16, 2, 32)
    assert pool.k_norm_cache(2).shape == (7, 16, 2)


def test_turboquant_centroids_are_ordered_and_midpoints_match():
    from freetoken.kernel.triton.turboquant import gaussian_centroids

    centroids, midpoints = gaussian_centroids(128)
    assert centroids.shape == (16,)
    assert midpoints.shape == (15,)
    assert torch.all(centroids[1:] > centroids[:-1])
    torch.testing.assert_close(midpoints, (centroids[:-1] + centroids[1:]) / 2)
    torch.testing.assert_close(centroids, -centroids.flip(0), atol=1e-6, rtol=1e-5)


def test_turboquant_pool_cost_includes_key_norm_and_value_metadata():
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import KVCacheGroupSpec
    from freetoken.attention import AttnType

    _init_tp()
    spec = KVCacheGroupSpec(
        name="full", layer_ids=(0, 1), num_kv_heads=4, head_dim=64,
        sliding_window=None, attn_type=AttnType.FULL,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(kv_cache_group_specs=lambda: (spec,)),
        dtype=torch.bfloat16,
        kv_cache_dtype="turboquant_4bit_nc",
        tp_info=SimpleNamespace(size=1),
        page_size=16,
    )
    per_page, fixed, page_tokens, reserve = MHAKVCache.kv_cost(config)
    expected_per_token = 2 * 2 * (64 + 6) * 4
    assert (per_page, fixed, page_tokens, reserve) == (
        expected_per_token * 16, 0, 16, 0
    )
