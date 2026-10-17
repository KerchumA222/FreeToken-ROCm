from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kvcache.mha_pool import MHAKVCache


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def test_turboquant_attention_rejects_non_power_of_two_block_size():
    from freetoken.kernel.triton.turboquant import turboquant_4bit_nc_attention

    q = torch.zeros((1, 1, 2), dtype=torch.float16)
    cache = torch.zeros((1, 1, 1), dtype=torch.uint8)
    meta = torch.ones((1, 1), dtype=torch.float16)
    with pytest.raises(ValueError, match="positive power of two"):
        turboquant_4bit_nc_attention(
            q=q,
            k_cache=cache,
            v_cache=cache,
            k_norms=meta,
            v_scales=meta,
            v_zeros=meta,
            centroids=torch.zeros(16),
            indptr=torch.tensor([0, 0], dtype=torch.int32),
            indices=torch.empty(0, dtype=torch.int32),
            q_to_req=torch.zeros(1, dtype=torch.int32),
            q_positions=torch.zeros(1, dtype=torch.int32),
            sm_scale=1.0,
            rotation=torch.eye(2),
            block_n=3,
        )


def test_turboquant_pool_allocation_rebuild_and_accounting():
    _init_tp()
    pool = MHAKVCache(
        num_kv_heads=2,
        num_layers=2,
        head_dim=64,
        num_pages=3,
        page_size=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        kv_cache_dtype="turboquant_4bit_nc",
    )

    assert pool.k_cache(0).shape == (3, 4, 2, 32)
    assert pool.v_cache(0).dtype == torch.uint8
    assert pool.k_norm_cache(0).shape == (12, 2)
    assert pool.v_scale_cache(0).shape == (12, 2)
    assert pool.v_zero_cache(0).shape == (12, 2)
    assert pool.dtype == torch.uint8
    assert pool.unit_bytes() == (2 * 2 * (64 + 6), 0)
    torch.testing.assert_close(
        pool.turboquant_rotation @ pool.turboquant_rotation.T,
        torch.eye(64),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.all(pool.turboquant_centroids[1:] > pool.turboquant_centroids[:-1])

    old_buffer = pool.k_cache(0)
    pool.rebuild(5)
    assert pool.k_cache(0).shape == (5, 4, 2, 32)
    assert pool.k_norm_cache(1).shape == (20, 2)
    assert pool.k_cache(0).data_ptr() != old_buffer.data_ptr()
    assert pool.unit_bytes() == (2 * 2 * (64 + 6), 0)


def test_turboquant_pool_rejects_unsupported_dtype_and_geometry():
    _init_tp()
    args = dict(
        num_kv_heads=2,
        num_layers=1,
        num_pages=2,
        page_size=4,
        device=torch.device("cpu"),
        kv_cache_dtype="turboquant_4bit_nc",
    )
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        MHAKVCache(head_dim=64, dtype=torch.float32, **args)
    with pytest.raises(ValueError, match="power-of-two"):
        MHAKVCache(head_dim=48, dtype=torch.bfloat16, **args)


def test_turboquant_sizing_includes_codes_metadata_and_fixed_rotation():
    from freetoken.kvcache.base import spec_kv_bytes_per_token

    spec = SimpleNamespace(
        num_kv_heads=2,
        head_dim=64,
        num_layers=3,
        mla=False,
        index_head_dim=0,
        num_index_layers=0,
        index_ratio=1,
        is_swa=False,
    )
    config = SimpleNamespace(
        tp_info=SimpleNamespace(size=1),
        kv_cache_dtype="turboquant_4bit_nc",
        page_size=16,
        model_config=SimpleNamespace(kv_cache_group_specs=lambda: [spec]),
    )
    assert spec_kv_bytes_per_token(spec, config) == 3 * 2 * (64 + 6)
    page_bytes, fixed_bytes, page_tokens, reserve = MHAKVCache.kv_cost(config)
    assert page_bytes == 16 * 3 * 2 * (64 + 6)
    assert fixed_bytes == (64 * 64 + 16 + 15) * 4
    assert page_tokens == 16
    assert reserve == 0


def test_turboquant_pool_factory_passes_opt_in_format():
    from freetoken.attention import AttnType
    from freetoken.kvcache import create_kvcache_pool

    _init_tp()
    spec = SimpleNamespace(
        name="full",
        num_kv_heads=2,
        head_dim=64,
        num_layers=2,
        layer_ids=(0, 1),
        sliding_window=None,
        mla=False,
        index_head_dim=0,
        num_index_layers=0,
        index_ratio=1,
        attn_type=AttnType.FULL,
        is_swa=False,
    )
    config = SimpleNamespace(
        has_swa_attention=False,
        has_linear_attention=False,
        num_kv_layers=2,
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        kv_cache_group_specs=lambda: [spec],
    )
    pool = create_kvcache_pool(
        config,
        num_pages=3,
        page_size=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        kv_cache_dtype="turboquant_4bit_nc",
    )
    assert pool.kv_cache_dtype == "turboquant_4bit_nc"
    assert pool.k_cache(0).dtype == torch.uint8

    from freetoken.kvcache import create_kv_pool

    engine_config = SimpleNamespace(
        model_config=config,
        page_size=4,
        max_running_req=2,
        cache_type="naive",
        kv_cache_dtype="turboquant_4bit_nc",
    )
    engine_pool = create_kv_pool(
        engine_config, num_pages=3, dtype=torch.bfloat16, device=torch.device("cpu")
    )
    assert engine_pool.kv_cache_dtype == "turboquant_4bit_nc"
    assert engine_pool.k_cache(0).dtype == torch.uint8
