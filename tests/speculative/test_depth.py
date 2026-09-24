import itertools

from freetoken.speculative.depth import DepthSelector

# Accepted drafts of successive rounds, as a chain of 3 would score them; a round at depth
# k accepts min(a, k) of them.
_ACCEPT = [3, 1, 0, 2, 3, 1, 0, 3]


def _run(sel: DepthSelector, seconds, rounds: int, accept=None) -> list[int]:
    """Drive the selector; ``seconds[k]`` is the wall time of a round at depth k."""
    accept = accept or itertools.cycle(_ACCEPT)
    chosen = []
    for _ in range(rounds):
        k = sel.choose()
        chosen.append(k)
        sel.record(k, [min(next(accept), k)], seconds[k])
    return chosen


def test_seeds_every_depth_deepest_first():
    sel = DepthSelector(3, seed_rounds=2, settle=0)
    assert _run(sel, {1: 0.10, 2: 0.11, 3: 0.12}, 6) == [3, 3, 2, 2, 1, 1]


def test_a_deep_round_scores_every_shallower_depth():
    sel = DepthSelector(3)
    sel.record(3, [2], 0.1)
    assert [sel.stats[d].tokens for d in (1, 2, 3)] == [2, 3, 3]
    assert sel.stats[1].rounds == 0          # timing only counts at the depth that ran


def test_settles_on_the_fastest_depth():
    # Disk-bound shape: extra rows cost more than the tokens they add.
    sel = DepthSelector(3, seed_rounds=4, settle=0)
    chosen = _run(sel, {1: 0.070, 2: 0.120, 3: 0.170}, 80)
    assert sel.best() == 1
    assert chosen[-20:] == [1] * 20


def test_settles_deep_when_rows_are_cheap():
    sel = DepthSelector(3, seed_rounds=4, settle=0)
    chosen = _run(sel, {1: 0.022, 2: 0.025, 3: 0.028}, 80)
    assert sel.best() == 3
    assert chosen[-20:] == [3] * 20


def test_follows_a_change_in_costs():
    sel = DepthSelector(3, seed_rounds=4, settle=0, probe_every=16, probe_rounds=4, alpha=0.3)
    _run(sel, {1: 0.070, 2: 0.120, 3: 0.170}, 60)
    assert sel.best() == 1
    # The cache warms: deeper rounds stop costing extra reads.
    _run(sel, {1: 0.070, 2: 0.075, 3: 0.080}, 200)
    assert sel.best() == 3


def test_probes_the_stalest_depth_periodically():
    sel = DepthSelector(3, seed_rounds=2, settle=0, probe_every=10, probe_rounds=3)
    chosen = _run(sel, {1: 0.05, 2: 0.10, 3: 0.15}, 60)
    assert {2, 3} <= set(chosen[10:])


def test_drops_rounds_that_swallowed_a_gap():
    sel = DepthSelector(2, seed_rounds=2, settle=0)
    for _ in range(4):
        sel.record(1, [1], 0.1)
    sel.record(1, [1], 5.0)       # a prefill of another request landed in this round
    assert abs(sel.stats[1].seconds - 0.1) < 1e-9


def test_fixed_mode_keeps_the_configured_depth():
    sel = DepthSelector(3, adapt=False)
    assert set(_run(sel, {1: 0.05, 2: 0.10, 3: 0.15}, 20)) == {3}
    assert "fixed k=3" in sel.summary()


def test_ignores_the_cold_rounds_after_a_switch():
    """A deeper depth's first rounds pay reads for experts the shallower one evicted;
    measured from its steady state it wins."""
    sel = DepthSelector(2, seed_rounds=4, settle=3, probe_every=16, probe_rounds=7)
    run = {"k": None, "n": 0}
    accept = itertools.cycle(_ACCEPT)
    for _ in range(300):
        k = sel.choose()
        if k != run["k"]:
            run["k"], run["n"] = k, 0
        run["n"] += 1
        cold = run["n"] <= 3
        seconds = {1: 0.070, 2: 0.300 if cold else 0.075}[k]
        sel.record(k, [min(next(accept), k)], seconds)
    assert sel.best() == 2
