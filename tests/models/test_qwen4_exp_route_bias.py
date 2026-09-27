"""FT_ROUTE_BIAS: resident experts win close top-k calls; chosen experts keep their weights."""

import torch

from freetoken.models.qwen4_exp import moe


def _weights(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # what fused_topk(renormalize=True) computes from the router logits
    top = logits.float().softmax(-1).topk(k, dim=-1)
    return top.indices, top.values / top.values.sum(-1, keepdim=True)


def test_resident_expert_wins_within_the_margin(monkeypatch):
    monkeypatch.setattr(moe, "_ROUTE_BIAS", 1.0)
    logits = torch.tensor([[4.0, 3.0, 2.5, 1.0]])
    resident = torch.tensor([False, False, True, True])
    out = moe._bias_to_resident(logits, resident, top_k=2)
    ids, w = _weights(out, 2)
    # expert 2 (2.5 + 1.0) beats expert 1 (3.0); expert 3 (1.0 + 1.0) does not beat it
    assert sorted(ids[0].tolist()) == [0, 2]
    ref = torch.tensor([4.0, 2.5]).softmax(-1)
    torch.testing.assert_close(w[0].sort(descending=True).values, ref)


def test_zero_margin_keeps_the_router_choice(monkeypatch):
    monkeypatch.setattr(moe, "_ROUTE_BIAS", 0.0)
    logits = torch.randn(3, 64)
    resident = torch.rand(64) > 0.5
    out = moe._bias_to_resident(logits, resident, top_k=8)
    ids, w = _weights(out, 8)
    ref_ids, ref_w = _weights(logits, 8)
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(w, ref_w)
