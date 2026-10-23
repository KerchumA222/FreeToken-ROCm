from types import SimpleNamespace

import pytest
import torch

from freetoken.env import ENV
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE, _fused_router, _router_linear
from freetoken.models.qwen4_exp.moe import Qwen4ExpMoE


class _Gate:
    def __init__(self, weight, output=None):
        self.weight = weight
        self.output = output
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.output


def test_router_keeps_upstream_linear_path_by_default_on_hip(monkeypatch):
    monkeypatch.setattr(ENV.QWEN35_ROUTER_GEMV, "value", False)
    monkeypatch.setattr(torch.version, "hip", "7.2")
    output = object()
    gate = _Gate(torch.ones((3, 2)), output)

    result = _router_linear(gate, torch.ones((2, 2)))

    assert result is output
    assert gate.calls == 1


def test_router_broadcast_reduction_is_opt_in_for_small_hip_batches(monkeypatch):
    monkeypatch.setattr(ENV.QWEN35_ROUTER_GEMV, "value", True)
    monkeypatch.setattr(torch.version, "hip", "7.2")
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    gate = _Gate(weight)

    result = _router_linear(gate, x)

    torch.testing.assert_close(result, torch.tensor([[17.0, 39.0], [23.0, 53.0]]))
    assert gate.calls == 0


@pytest.mark.parametrize("hip, rows", [(None, 2), ("7.2", 5)])
def test_router_opt_in_falls_back_outside_supported_case(monkeypatch, hip, rows):
    monkeypatch.setattr(ENV.QWEN35_ROUTER_GEMV, "value", True)
    monkeypatch.setattr(torch.version, "hip", hip)
    output = object()
    gate = _Gate(torch.ones((3, 2)), output)

    result = _router_linear(gate, torch.ones((rows, 2)))

    assert result is output
    assert gate.calls == 1


# (hidden, experts, top_k): Qwen3.6-35B-A3B, Flash-Next 256 / 512 experts
_FUSED_SHAPES = [(2048, 256, 8), (2560, 256, 10), (2560, 512, 10)]


def _hip_gpu():
    return torch.version.hip is not None and torch.cuda.is_available()


def _stub_moe(hidden, experts, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gate = SimpleNamespace(weight=torch.randn((experts, hidden), device="cuda", generator=g, dtype=torch.float16) * 0.05)
    shared = SimpleNamespace(weight=torch.randn((1, hidden), device="cuda", generator=g, dtype=torch.float16) * 0.05)
    return SimpleNamespace(gate=gate, shared_expert_gate=shared)


def _topk(logits, k, bias):
    probs = torch.softmax(logits.float(), dim=-1)
    sel = logits.float() if bias is None else logits.float() + bias
    ids = sel.topk(k, dim=-1).indices
    w = probs.gather(1, ids)
    return ids, w / w.sum(-1, keepdim=True)


@pytest.mark.skipif(not _hip_gpu(), reason="requires an AMD GPU")
@pytest.mark.parametrize("hidden, experts, top_k", _FUSED_SHAPES)
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("route_bias", [False, True])
def test_fused_router_matches_separate_linears(monkeypatch, hidden, experts, top_k, rows, route_bias):
    monkeypatch.setattr(ENV.FUSED_ROUTER, "value", True)
    moe = _stub_moe(hidden, experts, seed=rows)
    torch.manual_seed(rows)
    x = torch.randn((rows, hidden), device="cuda", dtype=torch.float16)

    logits, gate = _fused_router(moe, x)

    ref_logits = torch.nn.functional.linear(x.float(), moe.gate.weight.float())
    ref_gate = torch.sigmoid(torch.nn.functional.linear(x.float(), moe.shared_expert_gate.weight.float()))[:, 0]
    assert logits.dtype == torch.float16 and logits.shape == (rows, experts)
    assert gate.dtype == torch.float32 and gate.shape == (rows,)
    torch.testing.assert_close(logits.float(), ref_logits, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(gate, ref_gate, rtol=1e-3, atol=1e-4)

    bias = None
    if route_bias:
        bias = (torch.arange(experts, device="cuda") % 3 == 0).float() * 0.5
    ids, w = _topk(logits, top_k, bias)
    ref_ids, ref_w = _topk(ref_logits.half(), top_k, bias)
    # fp16 rounding of the logits may swap one near-tie at the cut, never more
    for row in range(rows):
        assert len(set(ids[row].tolist()) ^ set(ref_ids[row].tolist())) <= 2
    if torch.equal(ids, ref_ids):
        torch.testing.assert_close(w, ref_w, rtol=1e-2, atol=1e-3)


@pytest.mark.skipif(not _hip_gpu(), reason="requires an AMD GPU")
def test_fused_router_is_graph_capturable(monkeypatch):
    monkeypatch.setattr(ENV.FUSED_ROUTER, "value", True)
    moe = _stub_moe(2560, 512)
    x = torch.randn((2, 2560), device="cuda", dtype=torch.float16)
    _fused_router(moe, x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits, gate = _fused_router(moe, x)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.testing.assert_close(
        logits.float(), torch.nn.functional.linear(x.float(), moe.gate.weight.float()), rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        gate, torch.sigmoid(x.float() @ moe.shared_expert_gate.weight.float().T)[:, 0], rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("flag, hip, rows", [(False, "7.2", 2), (True, None, 2), (True, "7.2", 5)])
def test_fused_router_declines_outside_its_case(monkeypatch, flag, hip, rows):
    monkeypatch.setattr(ENV.FUSED_ROUTER, "value", flag)
    monkeypatch.setattr(torch.version, "hip", hip)
    moe = SimpleNamespace(gate=SimpleNamespace(weight=torch.ones((3, 2))),
                          shared_expert_gate=SimpleNamespace(weight=torch.ones((1, 2))))

    assert _fused_router(moe, torch.ones((rows, 2))) is None


@pytest.mark.skipif(not _hip_gpu(), reason="requires an AMD GPU")
@pytest.mark.parametrize("cls, hidden, experts", [(Qwen3_5MoE, 2048, 256), (Qwen4ExpMoE, 2560, 512)])
def test_moe_forward_flag_off_is_the_separate_path_and_on_agrees(monkeypatch, cls, hidden, experts):
    moe = object.__new__(cls)
    stub = _stub_moe(hidden, experts)
    calls = []

    def gate_forward(x):
        calls.append("gate")
        return torch.nn.functional.linear(x, stub.gate.weight)

    moe.gate = SimpleNamespace(weight=stub.gate.weight, forward=gate_forward)
    moe.shared_expert_gate = SimpleNamespace(
        weight=stub.shared_expert_gate.weight,
        forward=lambda x: torch.nn.functional.linear(x, stub.shared_expert_gate.weight))
    moe.shared_expert = SimpleNamespace(forward=lambda x: x * 2)
    moe.experts = SimpleNamespace(
        forward=lambda hidden_states, router_logits: hidden_states * router_logits.float().mean(-1, keepdim=True).half())
    x = torch.randn((3, hidden), device="cuda", dtype=torch.float16)

    monkeypatch.setattr(ENV.FUSED_ROUTER, "value", False)
    off = cls.forward(moe, x.clone())
    assert calls == ["gate"]

    monkeypatch.setattr(ENV.FUSED_ROUTER, "value", True)
    on = cls.forward(moe, x.clone())
    assert calls == ["gate"]
    torch.testing.assert_close(on, off, rtol=2e-2, atol=2e-2)
