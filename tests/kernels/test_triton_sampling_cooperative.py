"""The cooperative (several CTAs per row) fused sampling kernels match the one-CTA-per-row launch."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a CUDA or ROCm GPU is required"
)

from freetoken.kernel.triton import sampling as S


def _single(probs, tk, tp, draw):
    kernel = S._topp_fused if tp is not None else S._topk_fused
    return S._fused_launch(probs, kernel, tk, tp, draw, 11, 3, force_single=True)


def _cases(probs, k=40, p=0.9):
    B, dev = probs.size(0), probs.device
    tk, tp = S._topk_target(k, B, dev), S._topp_target(p, B, dev)
    yield S.top_k_renorm_probs(probs, k), _single(probs, tk, None, False), False
    yield S.top_k_sampling_from_probs(probs, k, seed=11, offset=3), _single(probs, tk, None, True), True
    yield S.top_p_sampling_from_probs(probs, p, seed=11, offset=3), _single(probs, None, tp, True), True
    yield S.top_k_top_p_sampling_from_probs(probs, k, p, seed=11, offset=3), _single(probs, tk, tp, True), True


def _check(probs):
    for got, ref, draw in _cases(probs):
        if draw:
            assert torch.equal(got, ref)
        else:
            torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-7)


@pytest.fixture(autouse=True)
def _fresh_disabled(monkeypatch):
    monkeypatch.setattr(S, "_COOPERATIVE_DISABLED", set())


@pytest.mark.parametrize("B,V", [(1, 151936), (3, 248320), (8, 32000)])
def test_cooperative_launch_matches_one_cta_per_row(B, V):
    torch.manual_seed(B)
    probs = torch.softmax(torch.randn(B, V, device="cuda") * 3, -1)
    G, _ = S._fused_plan(B, V, probs.device)
    assert G > 1, "the shape must exercise the cooperative grid barrier"
    _check(probs)
    assert not S._COOPERATIVE_DISABLED


def test_rejected_cooperative_launch_retries_with_one_cta_per_row(monkeypatch):
    # a budget far past any device's co-residency makes the driver reject the cooperative launch
    monkeypatch.setattr(S, "_coop_ctas_per_sm", lambda device: 64)
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(1, 248320, device="cuda") * 3, -1)
    _check(probs)
    assert S._COOPERATIVE_DISABLED
