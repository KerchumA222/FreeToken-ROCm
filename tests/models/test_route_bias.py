"""FT_ROUTE_BIAS (moe/route_bias.py): resident experts win close top-k calls; chosen experts keep their weights."""

import torch

from freetoken.moe import route_bias as moe


def _weights(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # what fused_topk(renormalize=True) computes from the router logits
    top = logits.float().softmax(-1).topk(k, dim=-1)
    return top.indices, top.values / top.values.sum(-1, keepdim=True)


def test_resident_expert_wins_within_the_margin(monkeypatch):
    monkeypatch.setattr(moe, "_ROUTE_BIAS", 1.0)
    logits = torch.tensor([[4.0, 3.0, 2.5, 1.0]])
    resident = torch.tensor([False, False, True, True])
    out = moe.bias_to_resident(logits, resident, top_k=2)
    ids, w = _weights(out, 2)
    # expert 2 (2.5 + 1.0) beats expert 1 (3.0); expert 3 (1.0 + 1.0) does not beat it
    assert sorted(ids[0].tolist()) == [0, 2]
    ref = torch.tensor([4.0, 2.5]).softmax(-1)
    torch.testing.assert_close(w[0].sort(descending=True).values, ref)


def test_zero_margin_keeps_the_router_choice(monkeypatch):
    monkeypatch.setattr(moe, "_ROUTE_BIAS", 0.0)
    logits = torch.randn(3, 64)
    resident = torch.rand(64) > 0.5
    out = moe.bias_to_resident(logits, resident, top_k=8)
    ids, w = _weights(out, 8)
    ref_ids, ref_w = _weights(logits, 8)
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(w, ref_w)


def test_prefill_batches_use_the_prefill_bias(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(moe, "_ROUTE_BIAS", 1.0)
    monkeypatch.setattr(moe, "_PREFILL_BIAS", 0.0)
    resident = torch.tensor([[-1, -1, 0, 1]], dtype=torch.int32)  # experts 2, 3 cached
    experts = SimpleNamespace(top_k=2, layer_id=0, offload_cache=SimpleNamespace(slot_for_id=resident))
    decode = torch.tensor([[4.0, 3.0, 2.5, 1.0]])
    prefill = decode.expand(9, 4)  # more rows than a decode / verify batch

    assert sorted(_weights(moe.apply(experts, decode), 2)[0][0].tolist()) == [0, 2]
    assert sorted(_weights(moe.apply(experts, prefill), 2)[0][0].tolist()) == [0, 1]  # unbiased
    monkeypatch.setattr(moe, "_PREFILL_BIAS", 1.0)
    assert sorted(_weights(moe.apply(experts, prefill), 2)[0][0].tolist()) == [0, 2]
