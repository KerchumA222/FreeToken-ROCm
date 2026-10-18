from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _init_tp() -> None:
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _hybrid_model_config():
    from freetoken.models.config import (
        FullAttentionGroupConfig,
        ModelConfig,
        RotaryConfig,
        SWAAttentionGroupConfig,
    )

    full_rope = RotaryConfig(
        head_dim=512,
        rotary_dim=256,
        max_position=4096,
        base=1_000_000.0,
        scaling=None,
    )
    swa_rope = RotaryConfig(
        head_dim=256,
        rotary_dim=256,
        max_position=4096,
        base=10_000.0,
        scaling=None,
    )
    return ModelConfig(
        num_layers=6,
        num_qo_heads=16,
        num_kv_heads=8,
        head_dim=256,
        hidden_size=4096,
        vocab_size=32000,
        intermediate_size=8192,
        rms_norm_eps=1e-6,
        rotary_config=swa_rope,
        hidden_act="gelu_tanh",
        tie_word_embeddings=True,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=704,
        norm_topk_prob=True,
        model_type="hybrid_swa_moe",
        architectures=["HybridSWAModel"],
        moe_enabled=True,
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=(2, 5),
                num_kv_heads=2,
                head_dim=512,
                rotary_config=full_rope,
                k_eq_v=True,
            ),
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=(0, 1, 3, 4),
                num_kv_heads=8,
                head_dim=256,
                rotary_config=swa_rope,
                sliding_window=1024,
            ),
        ),
    )


def _kv_group_specs():
    from freetoken.models.config import KVCacheGroupSpec

    return (
        KVCacheGroupSpec(
            name="full",
            layer_ids=(2, 5),
            num_kv_heads=2,
            head_dim=512,
            sliding_window=None,
        ),
        KVCacheGroupSpec(
            name="swa",
            layer_ids=(0, 1, 3, 4),
            num_kv_heads=8,
            head_dim=256,
            sliding_window=1024,
        ),
    )


def _patch_tp(monkeypatch) -> None:
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.hybrid_swa_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def test_hybrid_swa_cache_maps_layers_to_physical_groups(monkeypatch):
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    _patch_tp(monkeypatch)

    pool = HybridSWAKVCache(
        groups=_kv_group_specs(),
        num_layers=6,
        num_full_pages=4,
        page_size=16,
        num_swa_tokens=32,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert pool.full_num_tokens == 64
    assert pool.swa_num_tokens == 32
    assert pool.group_of(2) == "full"
    assert pool.group_of(0) == "swa"
    assert pool.is_full_layer(5)
    assert not pool.is_full_layer(4)
    assert pool.k_cache(2).shape == (4, 16, 2, 512)
    assert pool.v_cache(2).shape == (4, 16, 2, 512)
    assert pool.k_cache(0).shape == (32, 1, 8, 256)
    assert pool.v_cache(0).shape == (32, 1, 8, 256)


def test_create_kvcache_pool_uses_hybrid_swa_cache(monkeypatch):
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    _patch_tp(monkeypatch)

    pool = create_kvcache_pool(
        model_config=_hybrid_model_config(),
        num_pages=4,
        page_size=16,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert isinstance(pool, HybridSWAKVCache)
    assert pool.group_of(2) == "full"
    assert pool.group_of(0) == "swa"


def test_adjust_config_resolves_swa_cache_type():
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.engine import _adjust_config
    from freetoken.scheduler.config import SchedulerConfig

    _init_tp()
    # `--cache-type radix` on a SWA model materializes to the global-paged SWA radix cache
    # (SWARadixCache, cross-request reuse).
    radix = SchedulerConfig(
        model_path="/unused",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        cache_type="radix",
    )
    radix.__dict__["model_config"] = _hybrid_model_config()
    _adjust_config(radix)
    assert radix.attention_backend == "triton"
    assert radix.cache_type == "swa_radix"

    # `--cache-type naive` stays 'naive' (NaivePrefixCache, no reuse) on the same paged pool.
    naive = SchedulerConfig(
        model_path="/unused",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        cache_type="naive",
    )
    naive.__dict__["model_config"] = _hybrid_model_config()
    _adjust_config(naive)
    assert naive.attention_backend == "triton"
    assert naive.cache_type == "naive"


def _turbo_specs():
    from freetoken.models.config import KVCacheGroupSpec

    return (
        KVCacheGroupSpec(name="full", layer_ids=(1, 3), num_kv_heads=2, head_dim=64, sliding_window=None),
        KVCacheGroupSpec(name="swa", layer_ids=(0, 2), num_kv_heads=2, head_dim=64, sliding_window=128),
    )


def _turbo_pool(monkeypatch, device="cpu", full_pages=3, page_size=4, swa_tokens=9):
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    _patch_tp(monkeypatch)
    return HybridSWAKVCache(
        groups=_turbo_specs(),
        num_layers=4,
        num_full_pages=full_pages,
        page_size=page_size,
        num_swa_tokens=swa_tokens,
        dtype=torch.bfloat16,
        device=torch.device(device),
        kv_cache_dtype="turboquant_4bit_nc",
    )


def test_turboquant_hybrid_pool_quantizes_only_the_full_group(monkeypatch):
    pool = _turbo_pool(monkeypatch)

    assert [pool.layer_kv_cache_dtype(i) for i in range(4)] == ["model", "turboquant_4bit_nc"] * 2
    assert pool.k_cache(1).shape == (3, 4, 2, 32) and pool.k_cache(1).dtype == torch.uint8
    assert pool.k_norm_cache(1).shape == (12, 2) and pool.v_zero_cache(3).dtype == torch.float16
    assert pool.k_cache(0).shape == (9, 1, 2, 64) and pool.k_cache(0).dtype == torch.bfloat16
    assert pool.k_norm_cache(0) is None and pool.v_scale_cache(2) is None and pool.k_scale_cache(1) is None
    # full: 2 layers x 2 heads x (64 code bytes + 3 fp16 metadata); swa: 2 layers x 2 heads x 64 x 2 slabs x bf16
    assert pool.unit_bytes() == (2 * 2 * (64 + 6), 2 * 2 * 64 * 2 * 2)

    pool.rebuild(num_full_pages=5, num_swa_tokens=11)
    assert pool.k_cache(3).shape == (5, 4, 2, 32) and pool.k_norm_cache(3).shape == (20, 2)
    assert pool.k_cache(2).shape == (11, 1, 2, 64) and pool.k_cache(2).dtype == torch.bfloat16
    assert pool.unit_bytes() == (2 * 2 * (64 + 6), 2 * 2 * 64 * 2 * 2)


def test_turboquant_hybrid_cost_prices_swa_in_the_model_dtype():
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    full, swa = _turbo_specs()
    full = SimpleNamespace(**{**full.__dict__, "num_layers": 2, "mla": False, "index_head_dim": 0, "num_index_layers": 0, "index_ratio": 1, "is_swa": False})
    swa = SimpleNamespace(**{**swa.__dict__, "num_layers": 2, "mla": False, "index_head_dim": 0, "num_index_layers": 0, "index_ratio": 1, "is_swa": True})
    config = SimpleNamespace(
        tp_info=SimpleNamespace(size=1),
        kv_cache_dtype="turboquant_4bit_nc",
        dtype=torch.bfloat16,
        page_size=4,
        cache_type="swa_radix",
        swa_num_pages_override=1000,
        model_config=SimpleNamespace(kv_cache_group_specs=lambda: [full, swa]),
    )
    import freetoken.kvcache.hybrid_swa_pool as hybrid

    floor = 7
    orig = hybrid._swa_pool_floor
    hybrid._swa_pool_floor = lambda _config: floor
    try:
        per_page, fixed, page_tokens, _ = HybridSWAKVCache.kv_cost(config)
    finally:
        hybrid._swa_pool_floor = orig
    assert page_tokens == 4
    assert per_page == 4 * 2 * 2 * (64 + 6) + 8 * 4
    assert fixed == 2 * 2 * 64 * 2 * 2 * (1000 + 1) + (64 * 64 + 16 + 15) * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TurboQuant store is a Triton kernel")
def test_turboquant_hybrid_store_matches_the_mha_pool_and_keeps_swa_exact(monkeypatch):
    from freetoken.kvcache.mha_pool import MHAKVCache

    _init_tp()
    pool = _turbo_pool(monkeypatch, device="cuda")
    ref = MHAKVCache(
        num_kv_heads=2, num_layers=1, head_dim=64, num_pages=3, page_size=4,
        dtype=torch.bfloat16, device=torch.device("cuda"), kv_cache_dtype="turboquant_4bit_nc",
    )
    gen = torch.Generator(device="cuda").manual_seed(0)
    k = torch.randn((5, 128), device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn((5, 128), device="cuda", dtype=torch.bfloat16, generator=gen)
    loc = torch.tensor([1, 2, 5, 6, 11], device="cuda", dtype=torch.int32)

    pool.store_kv(k, v, loc, layer_id=3)
    ref.store_kv(k, v, loc, layer_id=0)
    flat = loc.long()
    assert torch.equal(pool.k_cache(3).view(-1, 2, 32)[flat], ref.k_cache(0).view(-1, 2, 32)[flat])
    assert torch.equal(pool.v_cache(3).view(-1, 2, 32)[flat], ref.v_cache(0).view(-1, 2, 32)[flat])
    for name in ("k_norm_cache", "v_scale_cache", "v_zero_cache"):
        assert torch.equal(getattr(pool, name)(3)[flat], getattr(ref, name)(0)[flat]), name

    pool.alloc_swa(loc)
    pool.store_kv(k, v, loc, layer_id=2)
    swa = pool.translate_loc_from_full_to_swa(loc).long()
    assert torch.equal(pool.k_cache(2).view(-1, 2, 64)[swa], k.view(5, 2, 64))
    assert torch.equal(pool.v_cache(2).view(-1, 2, 64)[swa], v.view(5, 2, 64))
