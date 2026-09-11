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


# ---------------------------------------------------------------------------
# PLE n-gram addressing. The table is ~320M rows and a token reads 16 of them, so
# an off-by-one here reads valid rows of a valid table and looks like a quality
# problem rather than a bug. Every piece is pinned to the reference.
# ---------------------------------------------------------------------------

NGRAM_SIZE, HEADS_PER_NGRAM, VOCAB, BASE, EOS, SEED = 3, 8, 4096, 1009, 2, 7
NGRAM_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM


class _NCfg:
    ngram_size = NGRAM_SIZE
    heads_per_ngram = HEADS_PER_NGRAM
    vocab_size = VOCAB
    ngram_vocab_size_base = BASE
    make_ngram_vocab_size_divisible_by = 128
    seed = SEED
    eos_token_id = EOS


@pytest.mark.parametrize("ple_layer_index", [0, 1, 3])
def test_layer_multipliers_match_reference(ple_layer_index):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as ref
    from freetoken.models.qwen4_exp.ngram import build_layer_multipliers

    want = ref._build_layer_multipliers(VOCAB, NGRAM_SIZE, ple_layer_index, SEED)
    got = build_layer_multipliers(VOCAB, NGRAM_SIZE, ple_layer_index, SEED)
    assert torch.equal(got, want)
    assert bool((got % 2 == 1).all())          # odd by construction


@pytest.mark.parametrize("ple_layer_index", [0, 2])
def test_head_vocabularies_match_reference(ple_layer_index):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as ref
    from freetoken.models.qwen4_exp.ngram import head_vocabularies

    sizes, offsets, total = head_vocabularies(BASE, NGRAM_HEADS, ple_layer_index)
    want_sizes, want_offsets, want_total = [], [], 0
    for head_idx in range(NGRAM_HEADS):
        g = ple_layer_index * NGRAM_HEADS + head_idx
        s = ref._find_nth_prime_after(BASE - 1, g + 1)
        want_sizes.append(s)
        want_offsets.append(want_total)
        want_total += s
    assert sizes.tolist() == want_sizes
    assert offsets.tolist() == want_offsets
    assert total == want_total
    # distinct vocabularies, so two heads cannot address identically
    assert len(set(want_sizes)) == len(want_sizes)


def test_shift_right_ignore_eos_matches_reference():
    from transformers.models.qwen4_exp import modeling_qwen4_exp as ref
    from freetoken.models.qwen4_exp.ngram import shift_right_ignore_eos

    mod = ref.Qwen4ExpTextNGramEmbedding(_NCfg(), embedding_dim=NGRAM_HEADS * 4,
                                         layer_idx=0, ple_layer_index=0)
    torch.manual_seed(3)
    ids = torch.randint(0, VOCAB, (2, 11))
    ids[0, 4] = EOS
    ids[1, 0] = EOS
    ids[1, 7] = EOS
    for shift in range(NGRAM_SIZE):
        want = mod._shift_right_ignore_eos(ids, shift)
        assert torch.equal(shift_right_ignore_eos(ids, shift, EOS), want), shift


def test_ngram_row_ids_match_reference():
    """The full addressing path, against the reference's own id computation."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as ref
    from freetoken.models.qwen4_exp.ngram import (
        build_layer_multipliers, head_vocabularies, ngram_row_ids,
    )

    for ple_layer_index in (0, 1):
        mod = ref.Qwen4ExpTextNGramEmbedding(_NCfg(), embedding_dim=NGRAM_HEADS * 4,
                                             layer_idx=0, ple_layer_index=ple_layer_index)
        torch.manual_seed(4 + ple_layer_index)
        seq = torch.randint(0, VOCAB, (2, 9))
        seq[0, 3] = EOS
        context = torch.full((2, NGRAM_SIZE - 1), EOS, dtype=torch.long)
        history = torch.cat([context, seq], dim=-1)

        # Replicate the reference's block, which is inlined in its forward().
        shifted = [mod._shift_right_ignore_eos(history, s) for s in range(NGRAM_SIZE)]
        blocks = []
        for ngram in range(2, NGRAM_SIZE + 1):
            start = (ngram - 2) * HEADS_PER_NGRAM
            end = start + HEADS_PER_NGRAM
            mixed = shifted[0] * mod.layer_multipliers[0]
            for pos in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[pos] * mod.layer_multipliers[pos])
            ids = torch.remainder(mixed.unsqueeze(-1),
                                  mod.ngram_heads_vocab_sizes[start:end].view(1, 1, -1))
            blocks.append(ids + mod.ngram_heads_offsets[start:end].view(1, 1, -1))
        want = torch.cat(blocks, dim=-1)[:, -seq.shape[1]:]

        sizes, offsets, _ = head_vocabularies(BASE, NGRAM_HEADS, ple_layer_index)
        got = ngram_row_ids(
            history,
            multipliers=build_layer_multipliers(VOCAB, NGRAM_SIZE, ple_layer_index, SEED),
            head_sizes=sizes, head_offsets=offsets,
            ngram_size=NGRAM_SIZE, heads_per_ngram=HEADS_PER_NGRAM,
            eos_token_id=EOS, keep_last=seq.shape[1],
        )
        assert torch.equal(got, want), ple_layer_index
        assert got.shape == (2, seq.shape[1], NGRAM_HEADS)


def test_row_ids_stay_inside_their_head_vocabulary():
    """Every id must land in its own head's [offset, offset+size) band -- a bug here
    would read another head's rows, which is undetectable downstream."""
    from freetoken.models.qwen4_exp.ngram import (
        build_layer_multipliers, head_vocabularies, ngram_row_ids,
    )

    sizes, offsets, total = head_vocabularies(BASE, NGRAM_HEADS, 0)
    torch.manual_seed(5)
    history = torch.randint(0, VOCAB, (3, 16))
    ids = ngram_row_ids(
        history,
        multipliers=build_layer_multipliers(VOCAB, NGRAM_SIZE, 0, SEED),
        head_sizes=sizes, head_offsets=offsets,
        ngram_size=NGRAM_SIZE, heads_per_ngram=HEADS_PER_NGRAM, eos_token_id=EOS,
    )
    lo = offsets.view(1, 1, -1)
    hi = (offsets + sizes).view(1, 1, -1)
    assert bool(((ids >= lo) & (ids < hi)).all())
    assert int(ids.max()) < total
