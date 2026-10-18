from types import SimpleNamespace

import torch

from freetoken.moe import route_bias


def test_bias_is_zero_without_offload_cache():
    experts = SimpleNamespace(layer_id=0)
    assert route_bias.selection_bias(experts, 1, torch.device("cpu")) is None


def test_decode_and_prefill_biases_are_independent(monkeypatch):
    monkeypatch.setattr(route_bias, "_ROUTE_BIAS", 1.0)
    monkeypatch.setattr(route_bias, "_PREFILL_BIAS", 0.25)
    experts = SimpleNamespace(
        layer_id=0,
        offload_cache=SimpleNamespace(
            slot_for_id=torch.tensor([[-1, -1, 0, 3]], dtype=torch.int32)
        ),
    )

    decode = route_bias.selection_bias(experts, 4, torch.device("cpu"))
    prefill = route_bias.selection_bias(experts, 9, torch.device("cpu"))

    assert decode.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert prefill.tolist() == [0.0, 0.0, 0.25, 0.25]


def test_route_bias_ignores_out_of_range_layer(monkeypatch):
    monkeypatch.setattr(route_bias, "_ROUTE_BIAS", 1.0)
    experts = SimpleNamespace(
        layer_id=2,
        offload_cache=SimpleNamespace(slot_for_id=torch.zeros((2, 4), dtype=torch.int32)),
    )

    assert route_bias.selection_bias(experts, 1, torch.device("cpu")) is None


def test_moe_layer_uses_cache_bias_for_selection_not_weights(monkeypatch):
    from freetoken.layers import moe
    from freetoken.moe import expert_remap, fused

    monkeypatch.setattr(route_bias, "_ROUTE_BIAS", 1.0)
    monkeypatch.setattr(route_bias, "_PREFILL_BIAS", 0.0)
    monkeypatch.setattr(expert_remap, "_PATH", "")
    monkeypatch.setattr(expert_remap, "_table", None)
    experts = SimpleNamespace(
        top_k=2,
        renormalize=False,
        layer_id=0,
        offload_cache=SimpleNamespace(
            slot_for_id=torch.tensor([[-1, -1, 0, -1]], dtype=torch.int32)
        ),
    )
    hidden = torch.zeros((1, 4))
    logits = torch.tensor([[4.0, 3.0, 2.5, 1.0]])
    calls = []

    def reference_router(**kwargs):
        calls.append(kwargs)
        return fused._torch_fused_topk(
            kwargs["gating_output"],
            kwargs["topk"],
            kwargs["renormalize"],
            None,
            kwargs.get("selection_bias"),
        )

    monkeypatch.setattr(moe, "fused_topk", reference_router)
    weights, ids = moe.MoELayer._route_topk(experts, hidden, logits)

    assert ids.tolist() == [[0, 2]]
    torch.testing.assert_close(weights, torch.softmax(logits, dim=-1).gather(1, ids.long()))
    assert len(calls) == 1
    assert calls[0]["selection_bias"].tolist() == [0.0, 0.0, 1.0, 0.0]


def _host_experts(layer_id=1):
    return SimpleNamespace(
        layer_id=layer_id,
        offload_cache=SimpleNamespace(
            slot_for_id=torch.tensor([[0, -1, -1, -1], [-1, 2, -1, -1]], dtype=torch.int32),
            host_tier=SimpleNamespace(
                resident=torch.tensor([[True, False, False, False], [False, True, True, False]])
            ),
        ),
    )


def test_host_bias_adds_to_the_gpu_bias_for_decode_only(monkeypatch):
    monkeypatch.setattr(route_bias, "_ROUTE_BIAS", 1.0)
    monkeypatch.setattr(route_bias, "_PREFILL_BIAS", 0.0)
    monkeypatch.setattr(route_bias, "_HOST_BIAS", 0.25)
    experts = _host_experts()

    assert route_bias.selection_bias(experts, 1, torch.device("cpu")).tolist() == [0.0, 1.25, 0.25, 0.0]
    assert route_bias.selection_bias(experts, 9, torch.device("cpu")) is None


def test_host_bias_alone_and_without_a_disk_tier(monkeypatch):
    monkeypatch.setattr(route_bias, "_ROUTE_BIAS", 0.0)
    monkeypatch.setattr(route_bias, "_HOST_BIAS", 0.5)

    assert route_bias.selection_bias(_host_experts(), 1, torch.device("cpu")).tolist() == [0.0, 0.5, 0.5, 0.0]
    no_tier = SimpleNamespace(layer_id=0, offload_cache=SimpleNamespace(slot_for_id=torch.zeros((1, 4), dtype=torch.int32)))
    assert route_bias.selection_bias(no_tier, 1, torch.device("cpu")) is None
