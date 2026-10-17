import pytest
import torch

from freetoken.env import ENV
from freetoken.models.qwen3_5_moe.moe import _router_linear


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
