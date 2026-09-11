"""Qwen4-Exp components against the HF reference (`modeling_qwen4_exp`).

The architecture's residual stream is `hc_count` copies of hidden_size wide, and
every block mixes them in and injects its output back out. Getting that mixing
subtly wrong produces plausible shapes and wrong numbers, so it is pinned to the
reference implementation rather than eyeballed.
"""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.qwen4_exp")

HIDDEN, HC, LOWRANK, EPS = 64, 4, 40, 1e-6


class _Cfg:
    """Duck-typed stand-in for Qwen4ExpTextConfig's GatedResidual fields."""

    hidden_size = HIDDEN
    hc_count = HC
    hc_lowrank = LOWRANK
    rms_norm_eps = EPS


def _reference(use_combine: bool = True):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as ref

    torch.manual_seed(0)
    mod = ref.Qwen4ExpTextGatedResidual(_Cfg(), use_combine=use_combine).eval()
    for p in mod.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return mod


def _ours_from(mod, use_combine: bool = True):
    from freetoken.models.qwen4_exp.hyper import GatedResidual

    ours = GatedResidual(HIDDEN, HC, LOWRANK, EPS, use_combine=use_combine)
    ours.hc_norm.weight = mod.hc_norm.weight.detach().clone()
    ours.input_mix_weight_down.weight = mod.input_mix_weight_down.weight.detach().clone()
    ours.input_mix_weight_up.weight = mod.input_mix_weight_up.weight.detach().clone()
    if use_combine:
        ours.block_inject_weight.weight = mod.block_inject_weight.weight.detach().clone()
    return ours


@pytest.mark.parametrize("tokens", [1, 7])
def test_gated_residual_matches_reference(tokens):
    mod = _reference()
    ours = _ours_from(mod)
    x = torch.randn(tokens, HC * HIDDEN)

    with torch.no_grad():
        want_mixed, want_hyper, want_inject = mod(x)
    got_mixed, got_hyper, got_inject = ours.forward(x)

    torch.testing.assert_close(got_mixed, want_mixed, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_inject, want_inject, rtol=1e-5, atol=1e-5)
    assert torch.equal(got_hyper, want_hyper)          # passed through untouched


def test_gated_residual_without_combine_returns_only_the_mix():
    mod = _reference(use_combine=False)
    ours = _ours_from(mod, use_combine=False)
    x = torch.randn(5, HC * HIDDEN)
    with torch.no_grad():
        want = mod(x)
    torch.testing.assert_close(ours.forward(x), want, rtol=1e-5, atol=1e-5)


def test_recombine_matches_the_reference_injection():
    """The decoder layer's own line, which lives outside GatedResidual."""
    from freetoken.models.qwen4_exp.hyper import recombine

    mod = _reference()
    ours = _ours_from(mod)
    x = torch.randn(3, HC * HIDDEN)
    block_out = torch.randn(3, HIDDEN)

    with torch.no_grad():
        _, hyper, inject = mod(x)
        want = hyper + (block_out.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)
    _, ghyper, ginject = ours.forward(x)
    torch.testing.assert_close(recombine(ghyper, block_out, ginject), want,
                               rtol=1e-5, atol=1e-5)


def test_grouped_rmsnorm_normalizes_each_stream_independently():
    """A single norm over the flat tensor would couple the streams -- scaling one
    stream must not change any other."""
    from freetoken.models.qwen4_exp.hyper import GroupedRMSNorm

    norm = GroupedRMSNorm(HC * HIDDEN, EPS, group_size=HIDDEN)
    norm.weight = torch.zeros(HC * HIDDEN)
    x = torch.randn(2, HC * HIDDEN)
    base = norm.forward(x)

    scaled = x.clone()
    scaled[:, :HIDDEN] *= 7.5                       # perturb stream 0 only
    out = norm.forward(scaled)
    torch.testing.assert_close(out[:, HIDDEN:], base[:, HIDDEN:], rtol=1e-5, atol=1e-5)
    # and stream 0 is scale-invariant under RMS norm
    torch.testing.assert_close(out[:, :HIDDEN], base[:, :HIDDEN], rtol=1e-4, atol=1e-4)


def test_grouped_rmsnorm_matches_reference():
    from transformers.models.qwen4_exp import modeling_qwen4_exp as ref
    from freetoken.models.qwen4_exp.hyper import GroupedRMSNorm

    torch.manual_seed(1)
    want_mod = ref.Qwen4ExpTextRMSNorm(HC * HIDDEN, group_size=HIDDEN, eps=EPS)
    torch.nn.init.normal_(want_mod.weight, std=0.1)
    ours = GroupedRMSNorm(HC * HIDDEN, EPS, group_size=HIDDEN)
    ours.weight = want_mod.weight.detach().clone()

    x = torch.randn(4, HC * HIDDEN)
    with torch.no_grad():
        want = want_mod(x)
    torch.testing.assert_close(ours.forward(x), want, rtol=1e-5, atol=1e-5)
