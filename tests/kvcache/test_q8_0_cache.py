from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kvcache.mha_pool import MHAKVCache


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def test_q8_0_pool_storage_rebuild_and_byte_accounting():
    _init_tp()
    pool = MHAKVCache(
        num_kv_heads=2,
        num_layers=2,
        head_dim=64,
        num_pages=3,
        page_size=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        kv_cache_dtype="q8_0",
    )

    assert pool.k_cache(0).shape == (3, 4, 2, 64)
    assert pool.v_cache(0).dtype == torch.int8
    assert pool.k_scale_cache(0).shape == (12, 2, 2)
    assert pool.v_scale_cache(0).dtype == torch.float16
    assert pool.dtype == torch.int8
    assert pool.unit_bytes() == (2 * 2 * (2 * 64 + 4 * 2), 0)

    old_buffer = pool.k_cache(0)
    pool.rebuild(5)
    assert pool.k_cache(0).shape == (5, 4, 2, 64)
    assert pool.k_scale_cache(1).shape == (20, 2, 2)
    assert pool.k_cache(0).data_ptr() != old_buffer.data_ptr()
    assert pool.unit_bytes() == (2 * 2 * (2 * 64 + 4 * 2), 0)


def test_q8_0_pool_rejects_unsupported_geometry_and_compute_dtype():
    _init_tp()
    args = dict(
        num_kv_heads=2,
        num_layers=1,
        num_pages=2,
        page_size=4,
        device=torch.device("cpu"),
        kv_cache_dtype="q8_0",
    )
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        MHAKVCache(head_dim=64, dtype=torch.float32, **args)
    with pytest.raises(ValueError, match="divisible by 32"):
        MHAKVCache(head_dim=48, dtype=torch.bfloat16, **args)


def test_q8_0_pool_factory_passes_storage_dtype_and_rejects_unknown_format():
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
        attn_type=AttnType.FULL,
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
        kv_cache_dtype="q8_0",
    )
    assert pool.kv_cache_dtype == "q8_0"
    assert pool.k_cache(0).dtype == torch.int8

    from freetoken.kvcache import create_kv_pool

    default_pool = create_kvcache_pool(
        config,
        num_pages=3,
        page_size=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert default_pool.kv_cache_dtype == "model"
    assert default_pool.k_cache(0).dtype == torch.bfloat16

    engine_config = SimpleNamespace(
        model_config=config,
        page_size=4,
        max_running_req=2,
        cache_type="naive",
        kv_cache_dtype="q8_0",
    )
    engine_pool = create_kv_pool(
        engine_config, num_pages=3, dtype=torch.bfloat16, device=torch.device("cpu")
    )
    assert engine_pool.kv_cache_dtype == "q8_0"
    assert engine_pool.k_cache(0).dtype == torch.int8

    default_engine_config = SimpleNamespace(
        model_config=config,
        page_size=4,
        max_running_req=2,
        cache_type="naive",
    )
    default_engine_pool = create_kv_pool(
        default_engine_config,
        num_pages=3,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert default_engine_pool.kv_cache_dtype == "model"
    assert default_engine_pool.k_cache(0).dtype == torch.bfloat16

    with pytest.raises(ValueError, match="unsupported KV cache dtype"):
        create_kvcache_pool(
            config,
            num_pages=3,
            page_size=4,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            kv_cache_dtype="bad",
        )


def test_q8_0_sizing_uses_int8_values_and_fp16_block_scales():
    from freetoken.kvcache.base import spec_kv_bytes_per_token

    spec = SimpleNamespace(
        num_kv_heads=2,
        head_dim=64,
        num_layers=3,
        mla=False,
        index_head_dim=0,
        num_index_layers=0,
        index_ratio=1,
    )
    config = SimpleNamespace(
        tp_info=SimpleNamespace(size=1), kv_cache_dtype="q8_0", page_size=16
    )
    assert spec_kv_bytes_per_token(spec, config) == 3 * 2 * (2 * 64 + 4 * 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Q8_0 writer requires Triton GPU")
def test_q8_0_writer_matches_block_reference_and_scatter_slots():
    from freetoken.kernel.triton.kv_cache import store_q8_0_cache

    device = torch.device("cuda")
    torch.manual_seed(7)
    k = torch.randn((3, 2, 64), device=device, dtype=torch.bfloat16)
    v = torch.randn((3, 2, 64), device=device, dtype=torch.float16)
    k[0, 0, :32] = 0
    # With amax=127 the scale is 1: GGML roundf sends half-steps away from zero.
    k[2, 0, :32] = 0
    k[2, 0, 0] = 127.0
    k[2, 0, 1] = 63.5
    k[2, 0, 2] = -62.5
    v[1, 1, 32:] = 0
    assert torch.count_nonzero(k[0, 0, 32:]) > 0
    assert torch.count_nonzero(v[1, 1, :32]) > 0
    k_cache = torch.full((6, 2, 64), 91, device=device, dtype=torch.int8)
    v_cache = torch.full((6, 2, 64), 91, device=device, dtype=torch.int8)
    k_scale = torch.full((6, 2, 2), -1, device=device, dtype=torch.float16)
    v_scale = torch.full((6, 2, 2), -1, device=device, dtype=torch.float16)
    indices = torch.tensor([4, 1, 3], device=device, dtype=torch.int64)

    store_q8_0_cache(
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        indices=indices,
        k=k,
        v=v,
    )
    torch.cuda.synchronize()

    for token, slot in enumerate(indices.tolist()):
        for head in range(2):
            for block in range(2):
                start = block * 32
                for src, cache, scales in (
                    (k, k_cache, k_scale),
                    (v, v_cache, v_scale),
                ):
                    values = src[token, head, start : start + 32].float()
                    scale = values.abs().max() / 127.0
                    actual_codes = cache[slot, head, start : start + 32]
                    actual_scale = scales[slot, head, block].float()
                    if scale.item() == 0:
                        assert actual_scale == 0
                        assert torch.count_nonzero(actual_codes) == 0
                        restored = actual_codes.float() * actual_scale
                        assert torch.count_nonzero(restored) == 0
                        continue
                    normalized = values * (1.0 / scale)  # GGML and the writer multiply by the reciprocal
                    expected = (
                        torch.sign(normalized) * torch.floor(normalized.abs() + 0.5)
                    ).clamp(-127, 127).to(torch.int8)
                    assert actual_codes.equal(expected)
                    assert torch.allclose(actual_scale, scale.to(torch.float16).float())
                    restored = actual_codes.float() * actual_scale
                    assert (restored - values).abs().max() <= actual_scale / 2 + 0.01

    tie_slot = int(indices[2].item())
    assert k_scale[tie_slot, 0, 0].item() == 1.0
    assert k_cache[tie_slot, 0, :3].tolist() == [127, 64, -63]

    untouched = torch.tensor([0, 2, 5], device=device)
    assert torch.all(k_cache[untouched] == 91)
    assert torch.all(v_cache[untouched] == 91)
    assert torch.all(k_scale[untouched] == -1)
    assert torch.all(v_scale[untouched] == -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Q8_0 pool writer requires Triton GPU")
def test_q8_0_pool_store_kv_writes_scale_accessors():
    _init_tp()
    device = torch.device("cuda")
    pool = MHAKVCache(
        num_kv_heads=2,
        num_layers=1,
        head_dim=64,
        num_pages=2,
        page_size=4,
        dtype=torch.bfloat16,
        device=device,
        kv_cache_dtype="q8_0",
    )
    k = torch.randn(3, 2, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out_loc = torch.tensor([5, 1, 6], device=device, dtype=torch.int32)
    pool.store_kv(k, v, out_loc, layer_id=0)
    torch.cuda.synchronize()

    actual_k = pool.k_cache(0).view(-1, 2, 64)
    actual_v = pool.v_cache(0).view(-1, 2, 64)
    k_scale = pool.k_scale_cache(0)
    v_scale = pool.v_scale_cache(0)
    for token, slot in enumerate(out_loc.tolist()):
        for head in range(2):
            for block in range(2):
                start = 32 * block
                for source, cache, scales in ((k, actual_k, k_scale), (v, actual_v, v_scale)):
                    values = source[token, head, start : start + 32].float()
                    scale = values.abs().max() / 127.0
                    normalized = values * (1.0 / scale)  # GGML and the writer multiply by the reciprocal
                    expected = (
                        torch.sign(normalized) * torch.floor(normalized.abs() + 0.5)
                    ).clamp(-127, 127).to(torch.int8)
                    assert cache[slot, head, start : start + 32].equal(expected)
                    assert torch.allclose(scales[slot, head, block].float(), scale.to(torch.float16).float())


@pytest.mark.parametrize(("is_decode", "num_rows"), [(True, 1), (False, 2)])
def test_q8_0_triton_backend_forwards_scale_caches(monkeypatch, is_decode, num_rows):
    from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata
    import freetoken.kernel.triton.attention as triton_attention

    class _Pool:
        kv_cache_dtype = "q8_0"

        def __init__(self):
            self.k = torch.zeros((4, 2, 64), dtype=torch.int8)
            self.v = torch.zeros_like(self.k)
            self.k_scale = torch.ones((4, 2, 2), dtype=torch.float16)
            self.v_scale = torch.full((4, 2, 2), 2, dtype=torch.float16)

        def store_kv(self, k, v, out_loc, layer_id):
            pass

        def k_cache(self, layer_id):
            return self.k

        def v_cache(self, layer_id):
            return self.v

        def k_scale_cache(self, layer_id):
            return self.k_scale

        def v_scale_cache(self, layer_id):
            return self.v_scale

    pool = _Pool()
    q = torch.zeros((num_rows, 1, 64), dtype=torch.float16)
    k = torch.zeros((num_rows, 2, 64), dtype=torch.float16)
    v = torch.zeros_like(k)
    metadata = TritonMetadata(
        cu_seqlens_q_gpu=torch.tensor([0, num_rows], dtype=torch.int32),
        indptr=torch.tensor([0, 2], dtype=torch.int32),
        indices=torch.tensor([0, 1], dtype=torch.int32),
        q_to_req=torch.zeros(num_rows, dtype=torch.int32),
        q_positions=torch.arange(num_rows, dtype=torch.int64),
        is_decode=is_decode,
        prefix_lens=torch.tensor([1], dtype=torch.int32),
        max_q_len=num_rows,
        attn_logits=torch.empty((1, 1, 8, 64)) if is_decode else None,
        attn_lse=torch.empty((1, 1, 8)) if is_decode else None,
    )
    batch = SimpleNamespace(
        attn_metadata=metadata,
        out_loc=torch.arange(num_rows, dtype=torch.int32),
    )
    backend = object.__new__(TritonAttentionBackend)
    backend.kvcache = pool
    backend.device = torch.device("cpu")
    backend.max_head_dim = 64
    # the attributes upstream's split cap (_split_cap) reads
    backend.num_q_heads = 1
    backend.kv_heads = [2]
    result = object()
    captured = {}

    def capture_call(**kwargs):
        captured.update(kwargs)
        return result

    if is_decode:
        monkeypatch.setattr(triton_attention, "decode_paged_attention", capture_call)
    else:
        monkeypatch.setattr(triton_attention, "extend_paged_attention", capture_call)

    actual = backend.forward(q, k, v, layer_id=0, batch=batch)

    assert actual is result
    assert captured["k_scale_cache"] is pool.k_scale
    assert captured["v_scale_cache"] is pool.v_scale
