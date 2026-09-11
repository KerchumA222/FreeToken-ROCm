"""HostExpertCache: a bounded pinned pool that reads misses from disk.

The contract that matters downstream is that a slot's bytes are indistinguishable
from the complete pinned bank's row for that expert, no matter how much thrashing
it took to get there.
"""
from __future__ import annotations

import numpy as np
import pytest

from freetoken.moe.disk_store import GgufExpertStore
from freetoken.moe.host_tier import MISS, HostExpertCache
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
