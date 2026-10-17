from types import SimpleNamespace

import pytest
import torch

from freetoken.layers.moe import MoELayer, OffloadMoELayer
from freetoken.moe import expert_remap


@pytest.fixture
def remap_file(monkeypatch, tmp_path):
    def install(table):
        path = tmp_path / "remap.pt"
        torch.save(table, path)
        monkeypatch.setattr(expert_remap, "_PATH", str(path))
        monkeypatch.setattr(expert_remap, "_table", None)
        return path

    return install


def test_hard_remap_aggregates_collisions_and_keeps_topk_distinct(remap_file):
    remap_file(torch.tensor([[0, 0, 1, 1, 2]], dtype=torch.int64))
    logits = torch.tensor([[3.0, 2.0, 0.0, 1.0, -1.0]])

    weights, ids = expert_remap.route(0, logits, topk=2, renormalize=False)

    source_probs = torch.softmax(logits, dim=-1)
    expected = torch.tensor(
        [[source_probs[0, :2].sum(), source_probs[0, 2:4].sum(), source_probs[0, 4]]]
    )
    assert ids.tolist() == [[0, 1]]
    assert torch.unique(ids).numel() == 2
    torch.testing.assert_close(weights, expected[:, :2])
    torch.testing.assert_close(weights.sum(dim=-1), expected.sum(dim=-1) - expected[:, 2])


def test_soft_remap_conserves_probability_mass(remap_file):
    table = {
        "ids": torch.tensor([[[0, 1], [0, 2], [1, 2], [1, 2]]]),
        "w": torch.tensor([[[0.25, 0.75], [0.5, 0.5], [0.4, 0.6], [0.2, 0.8]]]),
    }
    remap_file(table)
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])

    weights, ids = expert_remap.route(0, logits, topk=3, renormalize=False)

    source_probs = torch.softmax(logits, dim=-1)[0]
    expected = torch.zeros(3)
    for source in range(4):
        for choice in range(2):
            destination = table["ids"][0, source, choice]
            expected[destination] += source_probs[source] * table["w"][0, source, choice]
    expected_ids = torch.argsort(expected, descending=True).unsqueeze(0)
    assert torch.equal(ids, expected_ids.to(torch.int32))
    assert torch.unique(ids).numel() == 3
    torch.testing.assert_close(weights[0].sum(), torch.tensor(1.0))
    torch.testing.assert_close(weights[0], expected.gather(0, expected_ids[0]))


def test_identity_remap_matches_standard_router(remap_file):
    remap_file(torch.arange(5).reshape(1, 5))
    logits = torch.tensor([[4.0, 2.0, 1.0, -1.0, 0.0]])

    weights, ids = expert_remap.route(0, logits, topk=3, renormalize=True)

    expected_ids = torch.topk(logits, 3, dim=-1).indices
    expected_weights = torch.softmax(logits, dim=-1).gather(1, expected_ids)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    assert torch.equal(ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(weights, expected_weights)


def test_moe_layer_uses_opt_in_remap(remap_file):
    remap_file(torch.tensor([[0, 0, 1, 2]], dtype=torch.int64))
    experts = SimpleNamespace(top_k=2, renormalize=False, layer_id=0)
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])

    weights, ids = MoELayer._route_topk(experts, torch.zeros((1, 4)), logits)

    expected = torch.softmax(logits, dim=-1)[0]
    expected_mapped = torch.stack((expected[:2].sum(), expected[2], expected[3]))
    assert ids.tolist() == [[0, 1]]
    torch.testing.assert_close(weights, expected_mapped[:2].unsqueeze(0))


def test_offload_router_matches_resident_remap_before_cache_slot_rewrite(
    remap_file, monkeypatch
):
    from freetoken.moe import route_bias

    remap_file(torch.tensor([[0, 0, 1, 2, 2]], dtype=torch.int64))
    monkeypatch.setattr(route_bias, "_ROUTE_BIAS", 0.0)
    monkeypatch.setattr(route_bias, "_PREFILL_BIAS", 0.0)
    hidden = torch.zeros((1, 4))
    logits = torch.tensor([[3.0, 2.0, 0.0, 1.0, -1.0]])
    captured = {}

    class Cache:
        decode_target = "gpu"
        cpu_executor = None
        slot_for_id = torch.tensor([[4, 3, 1]])

        def is_cpu_layer(self, _layer_id):
            return False

        def ensure_experts(self, layer_id, ids):
            ids.copy_(self.slot_for_id[layer_id, ids.long()])

        def copy_missing(self):
            pass

        def pool_bank_views(self, _layer_id):
            return None

        def alphas_for_slots(self, _layer_id):
            return None

    def expert_gemm(_cache, _hidden, _weights, ids, **_kwargs):
        captured["ids"] = ids.clone()
        return torch.zeros_like(hidden)

    resident = SimpleNamespace(top_k=2, renormalize=False, layer_id=0)
    offload = SimpleNamespace(
        top_k=2,
        renormalize=False,
        layer_id=0,
        offload_cache=Cache(),
        _expert_gemm=expert_gemm,
    )
    resident_weights, resident_ids = MoELayer._route_topk(resident, hidden, logits)
    offload_weights, offload_ids = OffloadMoELayer._route_topk(offload, hidden, logits)

    assert resident_ids.tolist() == [[0, 1]]
    assert torch.equal(offload_ids, resident_ids)
    torch.testing.assert_close(offload_weights, resident_weights)
    resident_ids_before = resident_ids.clone()
    table_ids_before = expert_remap._load(logits.device)[0].clone()

    OffloadMoELayer._decode_routed(offload, hidden, offload_weights, offload_ids)

    assert captured["ids"].tolist() == [[4, 3]]
    assert offload_ids.tolist() == [[4, 3]]
    assert torch.equal(resident_ids, resident_ids_before)
    assert torch.equal(expert_remap._load(logits.device)[0], table_ids_before)


def test_moe_layer_uses_upstream_router_when_remap_is_disabled(monkeypatch):
    from freetoken.layers import moe

    monkeypatch.setattr(expert_remap, "_PATH", "")
    monkeypatch.setattr(expert_remap, "_table", None)
    experts = SimpleNamespace(top_k=2, renormalize=True, layer_id=0)
    hidden = torch.zeros((1, 4))
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    expected = (torch.ones((1, 2)), torch.zeros((1, 2), dtype=torch.int32))
    calls = []

    def standard_router(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(moe, "fused_topk", standard_router)
    actual = MoELayer._route_topk(experts, hidden, logits)

    assert actual is expected
    assert len(calls) == 1
    assert calls[0]["hidden_states"] is hidden
    assert calls[0]["gating_output"] is logits
    assert calls[0]["topk"] == 2
    assert calls[0]["renormalize"] is True
    assert "selection_bias" not in calls[0]


def test_soft_remap_respects_selection_bias_without_changing_weights(remap_file):
    remap_file(
        {
            "ids": torch.tensor([[[0], [1], [2], [3]]]),
            "w": torch.ones((1, 4, 1)),
        }
    )
    logits = torch.tensor([[4.0, 3.0, 2.5, 1.0]])
    bias = torch.tensor([0.0, 0.0, 1.0, 1.0])

    weights, ids = expert_remap.route(0, logits, topk=2, renormalize=False, selection_bias=bias)

    assert ids.tolist() == [[0, 2]]
    torch.testing.assert_close(weights, torch.softmax(logits, dim=-1).gather(1, ids.long()))


def test_moe_layer_combines_cache_bias_with_remap_selection_only(
    remap_file, monkeypatch
):
    from freetoken.moe import route_bias

    remap_file(
        {
            "ids": torch.tensor([[[0], [1], [2], [3]]]),
            "w": torch.ones((1, 4, 1)),
        }
    )
    monkeypatch.setattr(route_bias, "_ROUTE_BIAS", 1.0)
    monkeypatch.setattr(route_bias, "_PREFILL_BIAS", 0.0)
    experts = SimpleNamespace(
        top_k=2,
        renormalize=False,
        layer_id=0,
        offload_cache=SimpleNamespace(
            slot_for_id=torch.tensor([[-1, -1, 0, -1]], dtype=torch.int32)
        ),
    )
    logits = torch.tensor([[4.0, 3.0, 2.5, 1.0]])

    weights, ids = MoELayer._route_topk(experts, torch.zeros((1, 4)), logits)

    assert ids.tolist() == [[0, 2]]
    torch.testing.assert_close(weights, torch.softmax(logits, dim=-1).gather(1, ids.long()))


@pytest.mark.parametrize(
    "table, message",
    [
        (torch.tensor([[0, 1, 4]], dtype=torch.int64), "destination IDs"),
        ({"ids": torch.zeros((1, 2, 2), dtype=torch.int64), "w": torch.ones((1, 2, 2))}, "sum to 1"),
        ({"ids": torch.zeros((1, 2, 2), dtype=torch.int64), "w": torch.tensor([[[1.0, -1.0], [1.0, 0.0]]])}, "nonnegative"),
        ({"ids": torch.zeros((1, 2, 2), dtype=torch.int64), "w": torch.ones((1, 2, 1))}, "share nonempty shape"),
    ],
)
def test_invalid_remap_tables_are_rejected(remap_file, table, message):
    remap_file(table)

    with pytest.raises(ValueError, match=message):
        expert_remap.route(0, torch.zeros((1, 3)), topk=1, renormalize=False)


def test_remap_rejects_too_few_destinations_for_topk(remap_file):
    remap_file(torch.tensor([[0, 0, 0, 1]], dtype=torch.int64))

    with pytest.raises(ValueError, match="only 2 active destinations"):
        expert_remap.route(0, torch.zeros((1, 4)), topk=3, renormalize=False)


def test_remap_leaves_layers_beyond_table_unmapped(remap_file):
    remap_file(torch.arange(3).reshape(1, 3))

    assert expert_remap.route(1, torch.zeros((1, 3)), topk=1, renormalize=False) is None
