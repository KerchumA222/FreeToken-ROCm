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
