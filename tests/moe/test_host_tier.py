"""HostExpertCache: a bounded pinned pool that reads misses from disk.

The contract that matters downstream is that a slot's bytes are indistinguishable
from the complete pinned bank's row for that expert, no matter how much thrashing
it took to get there.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from freetoken.moe.disk_store import GgufExpertStore
from freetoken.moe.host_tier import MISS, HostExpertCache
from freetoken.moe.offload_cache import OffloadMoeCache
from tests.moe.conftest import E, L, Q4_0


@pytest.fixture
def store(tiny_gguf):
    path, _ = tiny_gguf
    with GgufExpertStore(path, E, {"gate_up": Q4_0, "down": Q4_0}) as s:
        yield s


def _check(cache, expected_expert, layer, experts, slots):
    for e, slot in zip(experts, slots):
        assert slot != MISS
        for bank in ("gate_up", "down"):
            got = cache.banks[bank][slot].numpy().reshape(-1)
            assert np.array_equal(got, expected_expert(bank, layer, e)), (bank, layer, e)


def test_serves_correct_bytes_when_everything_fits(store, expected_expert):
    with HostExpertCache(store, E, capacity=E * L, pin=False) as c:
        for layer in range(L):
            experts = list(range(E))
            _check(c, expected_expert, layer, experts, c.ensure(layer, experts))
        assert c.stats.evictions == 0
        assert c.stats.misses == E * L
        # A second pass is entirely hits.
        c.stats.hits = 0
        for layer in range(L):
            c.ensure(layer, range(E))
        assert c.stats.hits == E * L


def test_serves_correct_bytes_under_maximum_thrashing(store, expected_expert):
    """Capacity of exactly one request's worth: every call evicts everything."""
    with HostExpertCache(store, E, capacity=2, pin=False) as c:
        for layer in range(L):
            for e in range(E):
                _check(c, expected_expert, layer, [e], c.ensure(layer, [e]))
        assert c.stats.evictions > 0


def test_evicts_least_recently_used(store):
    with HostExpertCache(store, E, capacity=2, pin=False) as c:
        c.ensure(0, [0])
        c.ensure(0, [1])
        c.ensure(0, [0])          # 0 is now the more recent of the two
        c.ensure(0, [2])          # must evict 1, not 0
        assert c.slot_of(0, 0) != MISS
        assert c.slot_of(0, 2) != MISS
        assert c.slot_of(0, 1) == MISS


def test_same_expert_in_different_layers_is_a_different_entry(store, expected_expert):
    with HostExpertCache(store, E, capacity=E * L, pin=False) as c:
        s0 = c.ensure(0, [1])[0]
        s1 = c.ensure(1, [1])[0]
        assert s0 != s1
        _check(c, expected_expert, 0, [1], [s0])
        _check(c, expected_expert, 1, [1], [s1])


def test_slots_in_flight_are_never_evicted(store, expected_expert):
    """Every expert of one call must be simultaneously resident and correct, or the
    GPU could copy out of a row that a later miss in the same call overwrote."""
    with HostExpertCache(store, E, capacity=E, pin=False) as c:
        experts = list(range(E))
        slots = c.ensure(0, experts)
        assert len(set(slots)) == E
        _check(c, expected_expert, 0, experts, slots)


def test_a_request_larger_than_the_pool_is_an_error_not_corruption(store):
    with HostExpertCache(store, E, capacity=2, pin=False) as c:
        with pytest.raises(RuntimeError, match="raise the host cache capacity"):
            c.ensure(0, range(E))


def test_stall_rate_counts_calls_not_rows(store):
    """Added latency tracks how often a call touched disk at all, so that is what
    the stat has to measure."""
    with HostExpertCache(store, E, capacity=E * L, pin=False) as c:
        c.ensure(0, [0, 1])        # one stalled call, two misses
        c.ensure(0, [0, 1])        # one clean call
        s = c.stats.as_dict()
        assert c.stats.ensures == 2
        assert c.stats.stalled_ensures == 1
        assert s["stall_rate"] == 0.5
        assert s["miss_rate"] == 0.5


def test_reset_drops_residency_but_keeps_the_pool(store, expected_expert):
    with HostExpertCache(store, E, capacity=E, pin=False) as c:
        c.ensure(0, [3])
        c.reset()
        assert c.slot_of(0, 3) == MISS
        assert c.resident == 0
        _check(c, expected_expert, 0, [3], c.ensure(0, [3]))


def test_single_threaded_path_matches_the_pooled_one(store, expected_expert):
    with HostExpertCache(store, E, capacity=E * L, pin=False, workers=1) as c:
        experts = list(range(E))
        _check(c, expected_expert, 2, experts, c.ensure(2, experts))


def test_residency_split_reports_what_prefill_can_reuse(store):
    with HostExpertCache(store, E, capacity=E * L, pin=False) as c:
        c.ensure(1, [0, 2])
        hit_e, hit_s, miss = c.residency_split(1, E)
        assert hit_e == [0, 2]
        assert hit_s == [c.slot_of(1, 0), c.slot_of(1, 2)]
        assert miss == [1, 3]
        # a different layer shares nothing
        assert c.residency_split(0, E) == ([], [], list(range(E)))


def test_residency_split_does_not_change_recency(store):
    """It answers what prefill *could* take, which is not a use of those entries --
    if it bumped recency, a prefill scan would reorder the whole decode LRU."""
    with HostExpertCache(store, E, capacity=2, pin=False) as c:
        c.ensure(0, [0])
        c.ensure(0, [1])          # 1 is the most recent
        c.residency_split(0, E)   # must not promote 0
        c.ensure(0, [2])          # evicts the LRU, which should still be 0
        assert c.slot_of(0, 0) == MISS
        assert c.slot_of(0, 1) != MISS


def test_read_into_bypasses_the_pool(store, expected_expert):
    """Prefill reads through the tier without admitting: admitting a whole layer
    would evict everything decode depends on and leave only the last layer."""
    import torch

    with HostExpertCache(store, E, capacity=E * L, pin=False) as c:
        c.ensure(0, [0])
        before = c.resident
        experts = [1, 2, 3]
        dst = {
            name: torch.empty((len(experts), *c.banks[name].shape[1:]), dtype=torch.uint8)
            for name in c.banks
        }
        c.read_into(0, experts, dst)
        assert c.resident == before                      # nothing admitted
        for name in c.banks:
            for i, e in enumerate(experts):
                assert np.array_equal(
                    dst[name][i].numpy().reshape(-1), expected_expert(name, 0, e)
                )
                assert c.slot_of(0, e) == MISS


def test_admit_free_fills_a_cold_pool(store, expected_expert):
    with HostExpertCache(store, E, capacity=E * L, pin=False) as c:
        admitted = c.admit_free(0, list(range(E)))
        assert [e for e, _ in admitted] == list(range(E))
        for e, slot in admitted:
            assert c.slot_of(0, e) == slot
            for bank in ("gate_up", "down"):
                assert np.array_equal(
                    c.banks[bank][slot].numpy().reshape(-1), expected_expert(bank, 0, e)
                )


def test_admit_free_never_evicts(store):
    """A warm pool must survive a prefill scan: admitting with eviction would churn
    what decode just warmed, on every turn of a conversation."""
    with HostExpertCache(store, E, capacity=2, pin=False) as c:
        c.ensure(0, [0, 1])                     # pool is now full
        before = {(0, 0): c.slot_of(0, 0), (0, 1): c.slot_of(0, 1)}
        admitted = c.admit_free(1, list(range(E)))
        assert admitted == []                   # nothing free, so nothing admitted
        assert c.slot_of(0, 0) == before[(0, 0)]
        assert c.slot_of(0, 1) == before[(0, 1)]
        assert c.stats.evictions == 0


def test_admit_free_takes_only_what_is_free(store):
    with HostExpertCache(store, E, capacity=3, pin=False) as c:
        c.ensure(0, [0])                        # 1 used, 2 free
        admitted = c.admit_free(1, [5 % E, 2, 3])
        assert len(admitted) == 2
        assert c.resident == 3


def test_residency_split_restricted_to_the_routed_experts(store, expected_expert):
    """Prefill stages what the chunk routes to, not the whole layer.

    The GPU buffer is indexed by expert id and the GEMM only reads the rows
    ``topk_ids`` names, so an unrouted expert must not be classified -- and above all
    must not be read -- merely because it belongs to the layer. Restricting the split
    is what turns a whole-layer stage into a slice of one.
    """
    with HostExpertCache(store, E, capacity=E, pin=False) as c:
        # make experts 0 and 2 of layer 1 resident, leave the rest cold
        c.ensure(1, [0, 2])

        # whole layer: every expert is classified
        hit_e, hit_s, miss = c.residency_split(1, E)
        assert sorted(hit_e) == [0, 2]
        assert sorted(hit_e + miss) == list(range(E))

        # restricted: only the routed ones, hits and misses alike
        routed = [2, 3]
        hit_e, hit_s, miss = c.residency_split(1, E, routed)
        assert hit_e == [2] and miss == [3]
        assert set(hit_e) | set(miss) == set(routed)
        # and the slot handed back still holds that expert's bytes
        _check(c, expected_expert, 1, hit_e, hit_s)

        # an empty routing asks for nothing
        assert c.residency_split(1, E, []) == ([], [], [])


def test_residency_split_does_not_disturb_recency(store):
    """It is a query about what prefill can take, not a use of those entries."""
    with HostExpertCache(store, E, capacity=2, pin=False) as c:
        c.ensure(0, [0])
        c.ensure(0, [1])          # LRU order: 0 (oldest), 1
        c.residency_split(0, E, [0])   # must not promote expert 0
        c.ensure(0, [2])          # evicts the oldest, which is still expert 0
        hit_e, _hit_s, miss = c.residency_split(0, E, [0, 1, 2])
        assert 0 in miss and sorted(hit_e) == [1, 2]


def test_graph_admission_callback_writes_inference_buffers_from_foreign_thread():
    cache = type("Harness", (), {})()

    class Tier:
        def ensure(self, _layer, experts):
            return [expert + 1000 for expert in experts]

    cache.host_tier = Tier()
    cache._admit_error = None
    with torch.inference_mode():
        bufs = {
            "n": torch.tensor([2], dtype=torch.int64),
            "src": torch.tensor([4, 9], dtype=torch.int32),
            "out": torch.zeros(2, dtype=torch.int32),
        }
    callback = OffloadMoeCache._admit_callback(cache, 3, bufs)
    worker = threading.Thread(target=callback)
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert cache._admit_error is None
    assert bufs["out"].tolist() == [1004, 1009]


def _cpu_host_callback(cache, layer_id, ids):
    import weakref

    from freetoken.moe.cpu_executor import CpuMoeExecutor

    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor._cache_ref = weakref.ref(cache)
    executor.host_tier = cache.host_tier
    executor.num_experts = cache.num_experts
    return executor._host_admit_callback(layer_id, {"ids": ids})


def test_cpu_host_admission_deduplicates_routes_and_preserves_negative_ids():
    cache = type("Harness", (), {"num_experts": 8, "_admit_error": None})()

    class Tier:
        capacity = 3

        def __init__(self):
            self.calls = []

        def ensure(self, layer_id, experts):
            self.calls.append((layer_id, list(experts)))
            return [20 + expert for expert in experts]

    cache.host_tier = Tier()
    ids = torch.tensor([3, 3, -1, 1, -1], dtype=torch.int32)
    _cpu_host_callback(cache, 2, ids)()

    assert cache.host_tier.calls == [(2, [3, 1])]
    assert ids.tolist() == [23, 23, -1, 21, -1]


def test_cpu_host_admission_rejects_out_of_range_raw_id():
    cache = type("Harness", (), {"num_experts": 8, "_admit_error": None})()

    class Tier:
        capacity = 3

        def ensure(self, _layer_id, _experts):
            raise AssertionError("invalid routes must fail before host admission")

    cache.host_tier = Tier()
    ids = torch.tensor([8, -1], dtype=torch.int32)
    _cpu_host_callback(cache, 0, ids)()

    assert ids.tolist() == [-1, -1]
    assert isinstance(cache._admit_error, ValueError)


def test_cpu_host_admission_remaps_same_raw_id_independently_per_layer():
    cache = type("Harness", (), {"num_experts": 8, "_admit_error": None})()

    class Tier:
        capacity = 4

        def ensure(self, layer_id, experts):
            return [1000 * layer_id + 500 + expert for expert in experts]

    cache.host_tier = Tier()
    ids0 = torch.tensor([2], dtype=torch.int32)
    ids1 = torch.tensor([2], dtype=torch.int32)
    _cpu_host_callback(cache, 0, ids0)()
    _cpu_host_callback(cache, 1, ids1)()

    assert ids0.tolist() == [502]
    assert ids1.tolist() == [1502]


def test_cpu_host_admission_parks_first_error_and_poison_ids():
    cache = type("Harness", (), {"num_experts": 8, "_admit_error": None})()

    class Tier:
        capacity = 2

        def ensure(self, _layer_id, _experts):
            assert torch.is_inference_mode_enabled()
            raise OSError("disk read failed")

    cache.host_tier = Tier()
    ids = torch.tensor([4, -1], dtype=torch.int32)
    _cpu_host_callback(cache, 0, ids)()
    first = cache._admit_error
    _cpu_host_callback(cache, 0, torch.tensor([1], dtype=torch.int32))()

    assert ids.tolist() == [-1, -1]
    assert isinstance(first, OSError)
    assert cache._admit_error is first


def test_cpu_native_row_bound_uses_host_capacity_but_keeps_logical_experts():
    from freetoken.moe.cpu_executor import _native_num_experts

    cache = type("Harness", (), {"num_experts": 128})()
    cache.host_tier = type("Tier", (), {"capacity": 7})()

    assert cache.num_experts == 128
    assert _native_num_experts(cache) == 7


def test_prefetched_experts_serve_correct_bytes_without_a_second_read(store, expected_expert):
    with HostExpertCache(store, E, capacity=E * L, pin=False, workers=4) as c:
        c.prefetch(1, [0, 2])
        reads_before = c.stats.reads
        slots = c.ensure(1, [0, 2])
        _check(c, expected_expert, 1, [0, 2], slots)
        assert c.stats.prefetched == 2 and c.stats.prefetch_used == 2
        assert c.stats.reads == reads_before and c.stats.stalled_ensures == 0


def test_prefetch_never_evicts_what_the_same_step_ensures(store, expected_expert):
    with HostExpertCache(store, E, capacity=2, pin=False, workers=4) as c:
        c.ensure(0, [0, 1])
        c.prefetch(1, [3], keep=[(0, 0), (0, 1)])   # no evictable slot left for the guess
        assert c.stats.prefetched == 0
        c.prefetch(1, [3], keep=[(0, 1)])
        assert c.stats.prefetched == 1
        _check(c, expected_expert, 0, [1], c.ensure(0, [1]))


def test_unused_prefetches_return_to_the_lru(store):
    with HostExpertCache(store, E, capacity=4, pin=False, workers=4) as c:
        c.prefetch(0, [1, 2])
        for _ in range(200):                      # well past the sweep horizon
            c.ensure(1, [0])
        c.prefetch(1, [])                         # sweeps
        assert not c._inflight and c.resident == 3


def test_wrong_predictions_do_not_accumulate(store):
    """Guesses a layer's ensure did not use return to the pool once their reads finish,
    so a stream of wrong predictions cannot pin every slot."""
    with HostExpertCache(store, E, capacity=8, pin=False, workers=4) as c:
        for step in range(50):
            layer = step % L
            c.prefetch((layer + 1) % L, range(E))
            c.ensure(layer, [0])
        c._pool.shutdown(wait=True)
        c._pool = ThreadPoolExecutor(max_workers=4)
        c.prefetch(0, [])                 # sweep now that every read has finished
        assert len(c._inflight) <= 2
