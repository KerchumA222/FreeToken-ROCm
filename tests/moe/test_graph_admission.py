"""The disk tier's graph-safe expert admission.

Exercises the real protocol methods against stand-in device state: a captured
graph stages its miss list, a host-function node fills in the slots the host
tier chose, and the graph resumes with them -- all inside one replay.
"""

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _host_nodes_available():
    from freetoken.moe import graph_host

    return graph_host.available()


needs_host_nodes = pytest.mark.skipif(
    not torch.cuda.is_available() or not _host_nodes_available(),
    reason="no host-function launcher in this GPU runtime",
)


class _Tier:
    """Stands in for HostExpertCache.ensure: slot = expert + 1000."""

    def __init__(self):
        self.calls = []

    def ensure(self, layer_id, experts):
        self.calls.append((layer_id, list(experts)))
        return [e + 1000 for e in experts]


def _harness(width=8):
    obj = type("Harness", (), {})()
    obj.src_indices = torch.zeros(width, dtype=torch.int32, device="cuda")
    obj.num_indices = torch.zeros(1, dtype=torch.int64, device="cuda")
    obj.host_tier = _Tier()
    for name in ("_admit_graph_bufs", "_admit_callback", "_admit_capture",
                 "prepare_graph_admission", "raise_admission_error"):
        setattr(obj, name, getattr(OffloadMoeCache, name).__get__(obj))
    return obj


def _capture(h, misses):
    # device-resident before capture: building a tensor from a list inside capture
    # would stage it through unpinned host memory, which capture rejects
    src_dev = {k: torch.tensor(v, dtype=torch.int32, device="cuda") for k, v in misses.items()}
    seen = {}
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for layer_id, ids in misses.items():
            src = src_dev[layer_id]
            h.num_indices.fill_(len(ids))
            h.src_indices[: src.numel()].copy_(src)
            h._admit_capture(layer_id)
            seen[layer_id] = h.src_indices[: src.numel()].clone()
    return g, seen


@needs_host_nodes
def test_a_replay_takes_the_slots_the_host_tier_chose():
    h = _harness()
    misses = {0: [3, 7, 11], 1: [5]}
    h.prepare_graph_admission(misses)
    g, seen = _capture(h, misses)

    assert h._admit_order == [0, 1]
    assert h.host_tier.calls == [], "the host node must not fire during capture"

    g.replay()
    torch.cuda.synchronize()
    h.raise_admission_error()

    assert h.host_tier.calls == [(0, [3, 7, 11]), (1, [5])]
    assert seen[0].tolist() == [1003, 1007, 1011]
    assert seen[1].tolist() == [1005]


@needs_host_nodes
def test_every_replay_refetches():
    h = _harness()
    h.prepare_graph_admission({0: [4, 9]})
    g, seen = _capture(h, {0: [4, 9]})
    for i in range(3):
        g.replay()
        torch.cuda.synchronize()
        h.raise_admission_error()
        assert seen[0].tolist() == [1004, 1009]
        assert len(h.host_tier.calls) == i + 1


@needs_host_nodes
def test_a_zero_miss_layer_does_not_call_the_tier():
    h = _harness()
    h.prepare_graph_admission([0])
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        h.num_indices.fill_(0)
        h._admit_capture(0)
    g.replay()
    torch.cuda.synchronize()
    h.raise_admission_error()
    assert h.host_tier.calls == []


@needs_host_nodes
def test_a_failing_host_surfaces_on_the_engine_thread_without_hanging():
    h = _harness()
    h.prepare_graph_admission({0: [2]})
    g, _ = _capture(h, {0: [2]})

    class Boom(_Tier):
        def ensure(self, layer_id, experts):
            raise RuntimeError("disk is gone")

    h.host_tier = Boom()
    g.replay()
    torch.cuda.synchronize()  # must return: the node swallowed the failure
    with pytest.raises(RuntimeError, match="expert admission failed"):
        h.raise_admission_error()
    # and the error is consumed, not sticky
    h.raise_admission_error()
