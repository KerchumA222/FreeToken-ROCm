"""FREETOKEN_TOPK_SAMPLER: sampling over torch.topk's candidates draws from the same top-k-first, top-p distribution."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.sample import BatchSamplingArgs, Sampler, sample_top_k_first

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _reference(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """The exact distribution: keep the top_k, renormalize at the temperature, then the top_p prefix, renormalize."""
    vals, idx = torch.sort(logits.double(), descending=True)
    p = torch.softmax(vals[:top_k] / temperature, dim=-1)
    keep = (torch.cumsum(p, 0) - p) < top_p
    p = torch.where(keep, p, torch.zeros_like(p))
    out = torch.zeros_like(logits, dtype=torch.float64)
    out[idx[:top_k]] = p / p.sum()
    return out


@pytest.mark.parametrize("temperature, top_k, top_p", [(0.6, 20, 0.95), (1.0, 5, 1.0), (0.3, 50, 0.5)])
def test_empirical_frequencies_match_the_exact_distribution(temperature, top_k, top_p):
    torch.manual_seed(0)
    V, n = 300, 40000
    logits = torch.randn(V, device="cuda") * 2
    rows = logits.expand(n, V).contiguous()
    t = torch.full((n,), temperature, device="cuda")
    k = torch.full((n,), top_k, device="cuda", dtype=torch.int32)
    p = torch.full((n,), top_p, device="cuda")
    tokens = sample_top_k_first(rows, t, k, p, top_k)
    freq = torch.bincount(tokens, minlength=V).double() / n
    ref = _reference(logits, temperature, top_k, top_p)
    assert (freq[ref == 0] == 0).all(), "sampled a token outside the top-k / top-p set"
    assert (freq - ref).abs().max() < 0.012


def test_rows_with_different_k_and_p_stay_inside_their_own_sets():
    torch.manual_seed(1)
    V = 200
    logits = torch.randn(3, V, device="cuda")
    t = torch.tensor([0.7, 1.0, 0.5], device="cuda")
    k = torch.tensor([1, 7, 40], device="cuda", dtype=torch.int32)
    p = torch.tensor([1.0, 0.8, 0.9], device="cuda")
    for _ in range(200):
        tok = sample_top_k_first(logits, t, k, p, 40)
        for r in range(3):
            assert _reference(logits[r], float(t[r]), int(k[r]), float(p[r]))[tok[r]] > 0
    assert (tok[0] == logits[0].argmax()).item()  # top_k 1 is argmax


def _req(temperature, top_k, top_p=0.95):
    params = SimpleNamespace(temperature=temperature, top_k=top_k, top_p=top_p, presence_penalty=0.0,
                             frequency_penalty=0.0, is_greedy=temperature == 0.0)
    return SimpleNamespace(sampling_params=params, can_decode=True, output_token_counts=None)


def test_prepare_records_max_top_k_only_when_every_sampled_row_sets_one():
    s = Sampler(device=torch.device("cuda"), vocab_size=1000)
    args = s.prepare(SimpleNamespace(reqs=[_req(0.6, 20), _req(0.0, -1), _req(0.8, 7)]))
    assert args.max_top_k == 20  # the greedy row does not count
    assert s.prepare(SimpleNamespace(reqs=[_req(0.6, 20), _req(0.6, -1)])).max_top_k is None
    assert args.first_rows(2).max_top_k == 20


def test_the_flag_routes_sampling_and_keeps_greedy_rows_exact(monkeypatch):
    from freetoken.engine import sample as sample_mod
    from freetoken.env import ENV

    s = Sampler(device=torch.device("cuda"), vocab_size=500)
    torch.manual_seed(2)
    logits = torch.randn(2, 500, device="cuda")
    args = s.prepare(SimpleNamespace(reqs=[_req(0.0, -1), _req(0.6, 20)]))
    calls = []
    real = sample_mod.sample_top_k_first
    monkeypatch.setattr(sample_mod, "sample_top_k_first", lambda *a: calls.append(a[-1]) or real(*a))
    monkeypatch.setattr(ENV.TOPK_SAMPLER, "value", False)
    s.sample(logits, args)
    assert calls == []
    monkeypatch.setattr(ENV.TOPK_SAMPLER, "value", True)
    tok = s.sample(logits, args)
    assert calls == [20]
    assert tok[0] == logits[0].argmax()  # the greedy row
    assert _reference(logits[1], 0.6, 20, 0.95)[tok[1]] > 0
    no_k = s.prepare(SimpleNamespace(reqs=[_req(0.6, -1)]))
    s.sample(logits[:1], no_k)
    assert calls == [20]  # no top_k: the fused kernels
