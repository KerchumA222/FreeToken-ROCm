"""The MoE disk tier (--moe-disk-tier): GGUF experts read from their files into a bounded pinned host cache.

Factuality only: every expert row the tier hands to the GPU must be byte-identical to the row the complete-bank loader builds, through
pool hits, misses, evictions, prefill staging and mixed per-layer ggml types; and a real model must decode the same greedy tokens with the
tier on (and a host cache forced tiny) as with every expert in host RAM."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info

_ALIGN = 32
Q4_0, Q8_0 = 2, 8
E, H, I, TOP_K = 8, 64, 32, 3


def _init_tp():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, value) -> bytes:
    if isinstance(value, str):
        return _str(key) + struct.pack("<I", 8) + _str(value)
    return _str(key) + struct.pack("<IH", 2, value)  # u16


def _write_gguf(path, kv: dict, tensors) -> None:
    """A minimal GGUF v3 writer: ``tensors`` are (name, ggml type, ne in ggml order, packed bytes)."""
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + b"".join(_kv(k, v) for k, v in kv.items())
    infos, data = b"", b""
    for name, ggml_type, ne, raw in tensors:
        data += b"\0" * (-len(data) % _ALIGN)
        infos += _str(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", n) for n in ne) + struct.pack("<IQ", ggml_type, len(data))
        data += raw
    body = head + infos
    with open(path, "wb") as f:
        f.write(body + b"\0" * (-len(body) % _ALIGN) + data)


def _rows(ggml_type: int, rows: int, k: int, rng) -> bytes:
    """Random blocks with finite fp16 scales."""
    block = {Q4_0: 18, Q8_0: 34}[ggml_type]
    raw = rng.integers(0, 256, (rows, k // 32, block), dtype=np.uint8)
    raw[..., :2] = np.array([np.float16(rng.uniform(0.01, 1.0)).view(np.uint16)], dtype=np.uint16).view(np.uint8)
    return raw.tobytes()


def _clear_caches():
    from freetoken.models.gguf import reader

    for fn in (reader._reader, reader.load_gguf_metadata, reader.gguf_split_paths, reader.gguf_tensor_index):
        fn.cache_clear()


@pytest.fixture
def expert_gguf(tmp_path):
    """Three MoE layers split over two files, layer 1 in Q4_0 and the others in Q8_0; returns the path of split 1 and the layer count."""
    rng = np.random.default_rng(0)
    types = [Q8_0, Q4_0, Q8_0]
    per_split: dict[int, list] = {1: [], 2: []}
    for layer, t in enumerate(types):
        split = 1 if layer < 2 else 2
        per_split[split] += [
            (f"blk.{layer}.ffn_gate_exps.weight", t, (H, I, E), _rows(t, E * I, H, rng)),
            (f"blk.{layer}.ffn_up_exps.weight", t, (H, I, E), _rows(t, E * I, H, rng)),
            (f"blk.{layer}.ffn_down_exps.weight", t, (I, H, E), _rows(t, E * H, I, rng)),
        ]
    paths = []
    for no in (1, 2):
        path = str(tmp_path / f"m-{no:05d}-of-00002.gguf")
        _write_gguf(path, {"general.architecture": "qwen35moe", "split.no": no - 1, "split.count": 2}, per_split[no])
        paths.append(path)
    _clear_caches()
    yield paths[0], len(types)
    _clear_caches()


def _resident(path: str, layers: int):
    from freetoken.models.gguf.experts import load_gguf_expert_banks

    return load_gguf_expert_banks(path, layers, device=torch.device("cpu"))


@pytest.mark.parametrize("io_mode", ["direct", "buffered"])
def test_store_reads_the_rows_the_bank_loader_builds(expert_gguf, io_mode):
    from freetoken.moe.disk_tier import GgufExpertStore

    path, layers = expert_gguf
    banks = _resident(path, layers)
    store = GgufExpertStore(path, layers, io_mode=io_mode)
    try:
        # layer 2 lives in split 2: its parts must address that file
        assert {p.path for p in store.parts["down"][2]} == {path.replace("00001-of", "00002-of")}
        for bank in ("gate_up", "down"):
            for layer in range(layers):
                ref = banks.sources[bank][layer]
                assert store.expert_bytes(bank, layer) == ref[0].numel()
                for e in range(E):
                    dst = np.full(ref[0].numel(), 0xAB, dtype=np.uint8)
                    store.read_expert(bank, layer, e, dst)
                    assert np.array_equal(dst, ref[e].numpy().reshape(-1)), (bank, layer, e)
    finally:
        store.close()


def test_disk_tier_meta_banks_match_the_loaded_banks(expert_gguf):
    from freetoken.models.gguf.experts import load_gguf_expert_banks

    path, layers = expert_gguf
    meta, ref = load_gguf_expert_banks(path, layers, on_disk=True), _resident(path, layers)
    assert meta.on_disk and not ref.on_disk
    assert (meta.layer_tags, meta.row_tail_bytes) == (ref.layer_tags, ref.row_tail_bytes)
    for bank in ("gate_up", "down"):
        for m, r in zip(meta.sources[bank], ref.sources[bank]):
            assert m.is_meta and (m.shape, m.dtype) == (r.shape, r.dtype)


def _tier(path: str, layers: int, capacities, workers: int = 4):
    from freetoken.moe.disk_tier import GgufExpertStore, HostExpertTier
    from freetoken.moe.offload_cache import bank_layouts

    meta = _meta(path, layers)
    keys = bank_layouts(meta.sources, meta.layer_tags)
    unique = list(dict.fromkeys(keys))
    store = GgufExpertStore(path, layers, io_mode="buffered")
    return HostExpertTier(
        store, E, ("gate_up", "down"), [k for _, k in unique], [unique.index(k) for k in keys], capacities, pin=False, workers=workers
    )


def _meta(path: str, layers: int):
    from freetoken.models.gguf.experts import load_gguf_expert_banks

    return load_gguf_expert_banks(path, layers, on_disk=True)


def test_host_tier_lru_serves_exact_rows_through_evictions(expert_gguf):
    path, layers = expert_gguf
    ref = _resident(path, layers).sources
    tier = _tier(path, layers, [TOP_K, TOP_K])  # pools of the Q8_0 layers (0, 2) and the Q4_0 layer (1), one step's worth each
    rng = np.random.default_rng(1)
    try:
        for _ in range(60):
            for layer in range(layers):
                experts = rng.choice(E, TOP_K, replace=False).tolist()
                slots = tier.ensure(layer, experts)
                assert len(set(slots)) == TOP_K
                for e, s in zip(experts, slots):
                    for k, bank in enumerate(("gate_up", "down")):
                        assert torch.equal(tier.pool(layer, k)[s], ref[bank][layer][e]), (layer, e, s)
        assert tier.misses > 0 and tier.hits > 0
        # a repeated step is all hits and keeps its slots
        again = tier.ensure(0, [1, 2, 3])
        before = tier.misses
        assert tier.ensure(0, [1, 2, 3]) == again and tier.misses == before
        with pytest.raises(RuntimeError, match="host pool holds"):
            tier.ensure(1, [0, 1, 2, 3])
    finally:
        tier.close()


def test_host_tier_prefill_stages_misses_without_evicting(expert_gguf):
    path, layers = expert_gguf
    ref = _resident(path, layers).sources
    tier = _tier(path, layers, [4, 4])
    try:
        warm = tier.ensure(0, [5, 6])
        hits, staged = tier.prefill(0, [0, 1, 5, 6, 7])
        # 5 and 6 are pool hits, two misses fill the free slots, the last one is staged
        assert sorted(e for e, _ in hits) == [0, 1, 5, 6] and staged == [7]
        assert dict(hits)[5] == warm[0] and dict(hits)[6] == warm[1]
        for e, s in hits:
            for k, bank in enumerate(("gate_up", "down")):
                assert torch.equal(tier.pool(0, k)[s], ref[bank][0][e])
        for k, bank in enumerate(("gate_up", "down")):
            assert torch.equal(tier.stage(0, k)[7], ref[bank][0][7])
        # the decode residents survive the prefill
        assert tier.slot_of(0, 5) == warm[0] and tier.slot_of(0, 6) == warm[1]
    finally:
        tier.close()


def test_split_host_capacity_follows_layer_counts():
    from freetoken.moe.disk_tier import split_host_capacity

    assert split_host_capacity(3000, [0, 1, 0], [100, 50], 8, 2) == [16, 8]  # capped at the layers' experts
    assert split_host_capacity(600, [0, 1, 0], [100, 50], 64, 2) == [4, 4]
    assert split_host_capacity(1, [0, 1, 0], [100, 50], 64, 3) == [3, 3]


def _cache(path: str, layers: int, host_bytes: int, device: torch.device, cache_size: int):
    from freetoken.moe.disk_tier import GgufExpertStore
    from freetoken.moe.offload_cache import OffloadMoeCache, bank_layouts
    from freetoken.moe.pool_policy import group_layers

    _init_tp()
    meta = _meta(path, layers)
    groups = group_layers("unified", bank_layouts(meta.sources, meta.layer_tags), [1] * layers)
    pool_of_layer = [next(p for p, g in enumerate(groups) if layer in g) for layer in range(layers)]
    sizes = [cache_size] * len(groups)
    cache = OffloadMoeCache(
        num_layers=layers, num_experts=E, cache_size=sum(sizes), device=device, prefill_overlap=False,
        quant_format="gguf", pool_of_layer=pool_of_layer, pool_sizes=sizes,
    )
    cache.attach_disk_tier(GgufExpertStore(path, layers, io_mode="buffered"), host_bytes, TOP_K)
    cache.set_bank_sources(meta.sources, layer_tags=meta.layer_tags, row_tail_bytes=meta.row_tail_bytes)
    return cache


def test_offload_cache_refuses_meta_banks_without_the_tier(expert_gguf):
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    path, layers = expert_gguf
    meta = _meta(path, 1)
    cache = OffloadMoeCache(num_layers=1, num_experts=E, cache_size=2 * E, device=torch.device("cpu"), quant_format="gguf")
    with pytest.raises(ValueError, match="attach_disk_tier"):
        cache.set_bank_sources(meta.sources, layer_tags=meta.layer_tags, row_tail_bytes=meta.row_tail_bytes)
    with pytest.raises(ValueError, match="overlap"):
        OffloadMoeCache(num_layers=1, num_experts=E, cache_size=2 * E, device=torch.device("cpu"), prefill_overlap=True).attach_disk_tier(None, 1, 1)


def _check_prefill(cache, ref, layers: int, rng) -> None:
    for layer in range(layers):
        ids = torch.from_numpy(rng.integers(0, E, (5, TOP_K))).to(torch.int32).to(cache.device)
        views = cache.materialize_routed(layer, ids)
        if cache.device.type == "cuda":
            torch.cuda.synchronize()
        tails = [t for t in cache._pad_bytes[cache.layer_layout[layer]]]
        for e in torch.unique(ids).tolist():
            for (bank, view), tail in zip(zip(("gate_up", "down"), views), tails):
                assert torch.equal(view[e].cpu(), ref[bank][layer][e]), (layer, e, bank)
                if tail and e + 1 < E and e + 1 not in ids:
                    # the next expert's rows are what an MMQ tail of e's last row reads: never stale bytes
                    assert not view[e + 1].view(torch.uint8).reshape(-1)[:tail].any()
        cache.release_prefill_layer(layer)


def test_prefill_stages_the_routed_experts_on_cpu(expert_gguf):
    path, layers = expert_gguf
    ref = _resident(path, layers).sources
    cache = _cache(path, layers, 1, torch.device("cpu"), 2 * E)
    assert cache.host_tier.capacities == [TOP_K, TOP_K]
    rng = np.random.default_rng(2)
    for _ in range(4):
        _check_prefill(cache, ref, layers, rng)
    assert cache.prefill_hit_rows > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("host_bytes", [1, 1 << 30], ids=["tiny-host-cache", "whole-model-host-cache"])
def test_decode_and_prefill_slots_hold_the_resident_rows(expert_gguf, host_bytes):
    path, layers = expert_gguf
    ref = _resident(path, layers).sources
    cache = _cache(path, layers, host_bytes, torch.device("cuda"), E + 2)
    rng = np.random.default_rng(3)

    def decode(steps: int, batch: int = 1) -> None:
        for _ in range(steps):
            for layer in range(layers):
                ids = np.stack([rng.choice(E, TOP_K, replace=False) for _ in range(batch)])
                experts = torch.from_numpy(ids).to(torch.int32).cuda()
                cache.ensure_experts(layer, experts)
                cache.copy_missing()
                torch.cuda.synchronize()
                views = cache.pool_bank_views(layer)
                for e, s in zip(ids.reshape(-1).tolist(), experts.reshape(-1).tolist()):
                    for bank, view in zip(("gate_up", "down"), views):
                        assert torch.equal(view[s].cpu(), ref[bank][layer][e]), (layer, e, s)

    decode(30)
    if host_bytes == 1:
        assert cache.host_tier.misses > cache.host_tier.hits
        with pytest.raises(RuntimeError, match="host pool holds"):
            decode(1, batch=4)  # more distinct experts in one step than the pool can hold
    else:
        decode(1, batch=2)
    _check_prefill(cache, ref, layers, rng)
    decode(30)


def test_disk_tier_config_turns_off_graphs_and_prefill_overlap(expert_gguf):
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_disk_tier_config

    path, _ = expert_gguf

    def config(**kw):
        base = dict(model_path=path, moe_strategy="offload", moe_cpu_layers=None, moe_prefill_overlap=True, moe_prefill_hit_d2d=True, cuda_graph_bs=None, cuda_graph_max_bs=4)
        return SimpleNamespace(**(base | kw))

    def override(cfg):
        return lambda name, value: setattr(cfg, name, value)

    cfg = config()
    _adjust_disk_tier_config(cfg, override(cfg))
    assert (cfg.moe_prefill_overlap, cfg.moe_prefill_hit_d2d, cfg.cuda_graph_bs, cfg.cuda_graph_max_bs) == (False, False, [], 0)
    cfg = config(moe_strategy="hybrid")
    _adjust_disk_tier_config(cfg, override(cfg))
    assert cfg.moe_strategy == "offload"
    for bad in (config(moe_strategy="cpu"), config(moe_cpu_layers="4"), config(moe_strategy="fused"), config(model_path=os.path.dirname(path))):
        with pytest.raises(ValueError, match="disk-tier"):
            _adjust_disk_tier_config(bad, override(bad))


_QWEN36 = os.path.expanduser("~/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-Q4_0exp.gguf")

_RUN = r"""
import json, sys, torch
from freetoken.core import SamplingParams
from freetoken.llm import LLM
path, extra, out = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
llm = LLM(model_path=path, dtype=torch.bfloat16, max_running_req=1, max_seq_len_override=2048, moe_strategy="offload", **extra)
prompts = ["The capital of France is", "Write a haiku about the sea.\n", "def fibonacci(n):"]
res = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True))
tier = llm.engine.moe_offload_cache.host_tier
json.dump({"tokens": [r["token_ids"] for r in res], "tier": None if tier is None else [tier.hits, tier.misses]}, open(out, "w"))
"""


def _greedy(tmp_path, name: str, extra: dict, model: str = _QWEN36) -> dict:
    out = tmp_path / f"{name}.json"
    script = tmp_path / "run.py"
    script.write_text(_RUN)
    extra = {"moe_cache_size": 512} | extra if model == _QWEN36 else extra
    proc = subprocess.run([sys.executable, str(script), model, json.dumps(extra), str(out)], capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(out.read_text())


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isfile(_QWEN36), reason="needs a GPU and the Qwen3.6-35B-A3B GGUF")
def test_qwen36_greedy_decode_is_the_same_with_the_disk_tier(tmp_path):
    """Every expert in pinned host RAM (eager, and the default CUDA-graph + prefill-overlap run) against the disk tier with a host cache of
    one decode step per layer (every miss read from disk) and with a 4 GiB one (hits and misses): the same greedy tokens."""
    eager = _greedy(tmp_path, "eager", {"cuda_graph_max_bs": 0, "moe_prefill_overlap": False})
    graphs = _greedy(tmp_path, "graphs", {})
    tiny = _greedy(tmp_path, "tiny", {"moe_disk_tier": True, "moe_host_cache_mem": 1})
    big = _greedy(tmp_path, "big", {"moe_disk_tier": True, "moe_host_cache_mem": 4 << 30})
    assert all(len(t) == 32 for t in eager["tokens"])
    assert tiny["tier"][0] == 0 and tiny["tier"][1] > 0
    assert big["tier"][0] > 0 and big["tier"][1] > 0
    assert tiny["tokens"] == eager["tokens"]
    assert big["tokens"] == eager["tokens"]
    assert graphs["tokens"] == eager["tokens"]


_FLASH_NEXT_IQ1_M = os.path.expanduser("~/models/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf")


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isfile(_FLASH_NEXT_IQ1_M), reason="needs a GPU and the Flash-Next IQ1_M GGUF")
def test_flash_next_greedy_decode_is_the_same_with_the_disk_tier(tmp_path):
    """Flash-Next's experts come in several ggml types (one host pool per slot layout); its IQ1_M experts (23 GiB) still fit in host RAM,
    so the tier with a tiny host cache can be checked against every expert pinned."""
    cache = {"moe_cache_size": 4096, "kv_reserve_tokens": 4096}
    base = _greedy(tmp_path, "base", cache | {"cuda_graph_max_bs": 0, "moe_prefill_overlap": False}, _FLASH_NEXT_IQ1_M)
    tier = _greedy(tmp_path, "tier", cache | {"moe_disk_tier": True, "moe_host_cache_mem": 1}, _FLASH_NEXT_IQ1_M)
    assert tier["tier"][1] > 0
    assert tier["tokens"] == base["tokens"]
