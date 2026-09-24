"""Hybrid decode's bandwidth-matched fetch split.

Covers the two halves of --moe-hybrid-max-fetch auto: the profile reader that turns
`ft bench bw` kernel bandwidths into a fetch fraction, and the ensure kernel's
per-step integer split (GPU kernel vs CPU reference mirror, and the balance rule).
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.bench_profile import default_profile_path, load_backend_recommendation, load_hybrid_fetch_fraction
from freetoken.moe.offload_cache import OffloadMoeCache

Q = 1 << 16


def _balanced_fetch(num_missing: int, frac_q16: int) -> int:
    """Reference split: F ~ frac * misses, rounded to whichever integer neighbor
    minimizes the slower overlapped side (fetch ~ F*(1-frac), CPU ~ (M-F)*frac)."""
    lo = (num_missing * frac_q16) >> 16
    cost = lambda f: max(f * (Q - frac_q16), (num_missing - f) * frac_q16)  # noqa: E731
    return min(num_missing, lo if cost(lo) <= cost(lo + 1) else lo + 1)


def test_balanced_fetch_tracks_fraction():
    # The split follows fetched : cpu = pcie : (cpu - pcie) up to integer rounding, and
    # never over/under-shoots by more than one expert.
    for frac in (0.1, 0.415, 0.454, 0.7, 1.0):
        q = round(frac * Q)
        for m in range(0, 65):
            f = _balanced_fetch(m, q)
            assert 0 <= f <= m
            assert abs(f - frac * m) <= 1.0
    # ceil would over-fetch here (the regression this rule fixed): 41.5% of 3 misses is
    # 1.24 -> fetching 2 makes the PCIe side ~1.6x slower than balance; keep it at 1.
    assert _balanced_fetch(3, round(0.415 * Q)) == 1
    assert _balanced_fetch(4, round(0.415 * Q)) == 2


def test_load_hybrid_fetch_fraction(tmp_path):
    prof = {
        "gpu": {"name": "FAKE GPU"},
        "dtype_kernels": {
            "bf16": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            # overlapped (contended) pair wins over the standalone numbers when present
            "nvfp4_x": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                        "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
        "workloads": {
            "m": {"kernels": {"ds_fp4": {"cpu_moe_gbs": 80.0, "pcie_gather_gbs": 50.0}}}
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    # standalone fallback: full-contention assumption -> pcie / cpu
    assert load_hybrid_fetch_fraction("bf16", path=str(path)) == pytest.approx(0.4)
    # overlapped pair preferred: pcie_ov / (pcie_ov + cpu_ov)
    assert load_hybrid_fetch_fraction("nvfp4_x", path=str(path)) == pytest.approx(0.25)
    # per-model fallback when there is no per-dtype entry for the format
    assert load_hybrid_fetch_fraction("ds_fp4", path=str(path)) == pytest.approx(0.625)
    assert load_hybrid_fetch_fraction("nvfp4", path=str(path)) is None
    # a profile from different hardware is ignored
    assert load_hybrid_fetch_fraction("bf16", gpu_name="OTHER", path=str(path)) is None


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, verdict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"gpu": {"name": name}, "dtypes": {"bf16": verdict}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "hybrid")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "hybrid"
    assert load_backend_recommendation("bf16", gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "offload")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "offload"


def _hybrid_decode_fake_cache(events, host_tier):
    class Executor:
        def decode_submit(self, layer_id, hidden_states, topk_weights, topk_ids):
            events.append("submit")
            return hidden_states

        def decode_sync(self, pending):
            events.append("sync")
            return torch.zeros_like(pending)

    def ensure(layer_id, topk_ids):
        events.append("ensure")

    def copy_missing():
        events.append("copy")

    return SimpleNamespace(
        cpu_executor=Executor(),
        host_tier=host_tier,
        collect_stats=False,
        ensure_experts_hybrid=ensure,
        copy_missing=copy_missing,
        bank_views=lambda: (),
        alphas_for_slots=lambda layer_id: None,
    )


@pytest.mark.parametrize(
    ("host_tier", "expected"),
    [
        (None, ["ensure", "submit", "copy", "gemm", "sync"]),
        (object(), ["ensure", "copy", "submit", "gemm", "sync"]),
    ],
    ids=["resident", "bounded-host"],
)
def test_hybrid_decode_orders_bounded_copy_before_cpu_admission(
    host_tier, expected, monkeypatch
):
    monkeypatch.setattr("freetoken.layers.moe._HYBRID_OVERLAP", True)
    events = []
    cache = _hybrid_decode_fake_cache(events, host_tier)
    layer = object.__new__(OffloadMoELayer)
    layer.layer_id = 0
    layer._expert_gemm = lambda *args, **kwargs: events.append("gemm") or torch.zeros(
        (1, 4), dtype=torch.bfloat16
    )
    hidden = torch.zeros((1, 4), dtype=torch.bfloat16)
    weights = torch.ones((1, 2), dtype=torch.float32)
    ids = torch.tensor([[0, 1]], dtype=torch.int32)

    layer._decode_hybrid(cache, hidden, weights, ids)

    assert events == expected


@pytest.mark.parametrize("host_tier", [None, object()], ids=["resident", "bounded-host"])
def test_ensure_experts_hybrid_admits_only_with_host_tier(monkeypatch, host_tier):
    events = []
    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache.collect_decode_freq = False
    cache.hybrid_max_fetch = 1
    cache.hybrid_fetch_fraction = 0.0
    cache.host_tier = host_tier
    cache._admit_to_host_tier = lambda layer_id: events.append("admit")

    def ensure_kernel(*args):
        events.append("kernel")

    monkeypatch.setattr("freetoken.moe.offload_kernels.ensure_experts_hybrid", ensure_kernel)
    cache.ensure_experts_hybrid(0, torch.tensor([2], dtype=torch.int32))

    assert events == (["kernel", "admit"] if host_tier is not None else ["kernel"])


def test_hybrid_decode_partitions_multiple_routes_and_merges(monkeypatch):
    """GPU slots and CPU overflow routes each contribute exactly once."""
    monkeypatch.setattr("freetoken.layers.moe._HYBRID_OVERLAP", True)
    seen = {}

    class Executor:
        def decode_submit(self, layer_id, hidden_states, topk_weights, topk_ids):
            seen["cpu_ids"] = topk_ids.clone()
            seen["cpu_weights"] = topk_weights.clone()
            cpu_weight = topk_weights.masked_select(topk_ids >= 0).sum()
            return torch.full_like(hidden_states, cpu_weight)

        def decode_sync(self, pending):
            return pending

    cache = SimpleNamespace(
        cpu_executor=Executor(),
        host_tier=None,
        collect_stats=False,
        ensure_experts_hybrid=lambda layer_id, ids: ids.copy_(
            torch.tensor([[2, -1, 5, -1]], dtype=ids.dtype)
        ),
        copy_missing=lambda: None,
        bank_views=lambda: (),
        alphas_for_slots=lambda layer_id: None,
    )
    layer = object.__new__(OffloadMoELayer)
    layer.layer_id = 0

    def gpu_gemm(cache, hidden_states, weights, slots, **kwargs):
        seen["gpu_slots"] = slots.clone()
        seen["gpu_weights"] = weights.clone()
        return torch.full_like(hidden_states, weights.sum())

    layer._expert_gemm = gpu_gemm
    hidden = torch.zeros((1, 2), dtype=torch.float32)
    weights = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    ids = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)

    output = layer._decode_hybrid(cache, hidden, weights, ids)

    assert torch.equal(seen["cpu_ids"], torch.tensor([[-1, 11, -1, 13]], dtype=torch.int32))
    assert torch.equal(seen["cpu_weights"], weights)
    assert torch.equal(seen["gpu_slots"], torch.tensor([[2, 0, 5, 0]], dtype=torch.int32))
    assert torch.equal(seen["gpu_weights"], torch.tensor([[1.0, 0.0, 3.0, 0.0]]))
    assert torch.equal(output, torch.full_like(hidden, 10.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("policy", ["recency", "frequency", "lowest_id"])
def test_hybrid_fraction_gpu_matches_cpu_reference(policy):
    torch.manual_seed(0)
    num_experts, cache_size, top_k, frac = 32, 40, 8, 0.415

    def make():
        return OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts, hybrid_fetch_fraction=frac,
            hybrid_fetch_policy=policy, hybrid_frequency_warmup=2,
        )

    gpu, ref = make(), make()
    frac_q16 = round(frac * Q)
    for step in range(64):
        ids = torch.randperm(num_experts)[:top_k].to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()  # a CPU ids tensor drives the reference path
        gpu.ensure_experts_hybrid(0, g)
        ref.ensure_experts_hybrid(0, c)
        missing = int(gpu.num_missing_full.item())
        fetched = int(gpu.num_indices.item())
        assert missing == int(ref.num_missing_full.item())
        assert fetched == int(ref.num_indices.item()) == _balanced_fetch(missing, frac_q16)
        # slot rewrites (hit/fetched -> slot, overflow -> -1) and LRU state stay identical
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.id_of_slot.cpu(), ref.id_of_slot.cpu())
        assert torch.equal(gpu.expert_frequency.cpu(), ref.expert_frequency.cpu())
        assert torch.equal(gpu.hybrid_call_count.cpu(), ref.hybrid_call_count.cpu())
        assert (g >= 0).sum().item() == len(set(ids.tolist())) - (missing - fetched)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fixed_cap_unchanged():
    # fraction 0 (no profile / explicit --moe-hybrid-max-fetch) keeps the fixed cap.
    cache = OffloadMoeCache(
        num_layers=1, num_experts=32, cache_size=40, device=torch.device("cuda"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=1,
    )
    ids = torch.arange(8, dtype=torch.int32).cuda()
    cache.ensure_experts_hybrid(0, ids)
    assert int(cache.num_missing_full.item()) == 8
    assert int(cache.num_indices.item()) == 1


def test_hybrid_frequency_learns_per_layer_then_freezes():
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=8,
        cache_size=8,
        device=torch.device("cpu"),
        quant_format="bf16",
        decode_target="hybrid",
        hybrid_max_fetch=1,
        hybrid_fetch_policy="frequency",
        hybrid_frequency_warmup=2,
    )

    # Duplicate routes count as separate occurrences during the warmup.
    for ids in ([1, 1, 1, 2], [2]):
        cache.ensure_experts_hybrid(0, torch.tensor(ids, dtype=torch.int32))
    assert int(cache.hybrid_call_count[0]) == 2
    assert cache.expert_frequency[0].tolist()[:3] == [0, 3, 2]

    # Make both candidates misses and make expert 2 more recent. Frequency still wins.
    cache.slot_for_id.fill_(-1)
    cache.id_of_slot.fill_(-1)
    cache.usage.zero_()
    cache.expert_recency[0, 1] = 1
    cache.expert_recency[0, 2] = 2

    ids = torch.tensor([1, 2], dtype=torch.int32)
    cache.ensure_experts_hybrid(0, ids)
    assert int(cache.src_indices[0]) == 1
    assert int(cache.hybrid_call_count[0]) == 3
    learned = cache.expert_frequency[0].clone()

    # Once learned, later calls do not change the frequency ranking table.
    cache.ensure_experts_hybrid(0, torch.tensor([1, 2], dtype=torch.int32))
    assert torch.equal(cache.expert_frequency[0], learned)

    # The other layer has its own warmup counter and starts learning independently.
    cache.ensure_experts_hybrid(1, torch.tensor([6, 7], dtype=torch.int32))
    assert int(cache.hybrid_call_count[1]) == 1
    assert int(cache.hybrid_call_count[0]) == 4


def test_hybrid_fetch_policy_default_is_not_environment_controlled(monkeypatch):
    monkeypatch.setenv("FREETOKEN_HYBRID_FETCH", "lowest_id")
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=4,
        device=torch.device("cpu"),
        quant_format="bf16",
        decode_target="hybrid",
    )
    assert cache.hybrid_fetch_policy == "recency"


@pytest.mark.parametrize("policy", ["recency", "lowest_id"])
def test_non_frequency_policies_do_not_update_frequency_warmup_counter(policy):
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=4,
        device=torch.device("cpu"),
        quant_format="bf16",
        decode_target="hybrid",
        hybrid_max_fetch=1,
        hybrid_fetch_policy=policy,
    )
    cache.ensure_experts_hybrid(0, torch.tensor([1, 2], dtype=torch.int32))
    assert int(cache.hybrid_call_count[0]) == 0
    assert not torch.any(cache.expert_frequency)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_frequency_warmup_uses_recency_on_gpu_and_cpu():
    kwargs = dict(
        num_layers=1,
        num_experts=4,
        cache_size=4,
        quant_format="bf16",
        decode_target="hybrid",
        hybrid_max_fetch=1,
        hybrid_fetch_policy="frequency",
        hybrid_frequency_warmup=2,
    )
    gpu = OffloadMoeCache(device=torch.device("cuda"), **kwargs)
    ref = OffloadMoeCache(device=torch.device("cpu"), **kwargs)
    for cache in (gpu, ref):
        cache.hybrid_call_count[0] = 1
        cache.expert_recency[0, 3] = 5

    g = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    c = torch.tensor([1, 3], dtype=torch.int32)
    gpu.ensure_experts_hybrid(0, g)
    ref.ensure_experts_hybrid(0, c)
    assert int(gpu.src_indices[0]) == int(ref.src_indices[0]) == 3
    assert torch.equal(g.cpu(), c)
