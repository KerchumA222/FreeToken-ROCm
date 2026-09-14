"""Acceptance: the leading-run bookkeeping, and the property that makes strict
rejection sampling worth its cost -- the committed tokens are distributed exactly as
plain decoding from the target would have produced them."""

from __future__ import annotations

import torch

from freetoken.speculative.accept import accept_match, accept_strict


def _lens(*values: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64)


def test_match_accepts_every_draft_and_returns_the_bonus_token():
    draft = torch.tensor([[5, 6, 7]])
    target = torch.tensor([[5, 6, 7, 9]])
    out = accept_match(draft, target, _lens(3))
    assert out.num_accepted.tolist() == [3]
    assert out.correction.tolist() == [9]


def test_match_stops_at_the_first_disagreement():
    draft = torch.tensor([[5, 6, 7]])
    target = torch.tensor([[5, 42, 7, 9]])
    out = accept_match(draft, target, _lens(3))
    assert out.num_accepted.tolist() == [1]
    # The correction is the target's own draw at the rejected slot, not the draft's.
    assert out.correction.tolist() == [42]


def test_match_does_not_resume_after_a_disagreement():
    """A later agreement is worthless: the drafts after a rejection were conditioned on a
    token that is not being committed."""
    draft = torch.tensor([[1, 2, 3]])
    target = torch.tensor([[9, 2, 3, 4]])
    out = accept_match(draft, target, _lens(3))
    assert out.num_accepted.tolist() == [0]
    assert out.correction.tolist() == [9]


def test_match_respects_per_request_draft_depth():
    """Requests near their output budget stage fewer drafts; the padding columns must not
    be accepted even when they happen to agree."""
    draft = torch.tensor([[1, 2, 3], [1, 2, 3]])
    target = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    out = accept_match(draft, target, _lens(3, 1))
    assert out.num_accepted.tolist() == [3, 1]
    assert out.correction.tolist() == [4, 2]


def _one_hot(rows: list[int], vocab: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(rows), vocab).float()


def test_strict_always_accepts_when_the_proposer_matches_the_target():
    """p/q == 1, so the acceptance test passes for every uniform draw in [0, 1)."""
    vocab = 8
    p = torch.full((4, 2, vocab), 1.0 / vocab)
    q = torch.full((4, 1, vocab), 1.0 / vocab)
    draft = torch.zeros(4, 1, dtype=torch.int64)
    out = accept_strict(draft, q, p, _lens(1, 1, 1, 1))
    assert out.num_accepted.tolist() == [1, 1, 1, 1]


def test_strict_never_accepts_a_draft_the_target_rules_out():
    """p(d) == 0 makes the ratio 0, and the residual relu(p - q) is then just p."""
    vocab = 4
    p = _one_hot([2, 2], vocab).unsqueeze(0).expand(1, 2, vocab).contiguous()
    q = _one_hot([0], vocab).unsqueeze(0)
    draft = torch.zeros(1, 1, dtype=torch.int64)
    out = accept_strict(draft, q, p, _lens(1))
    assert out.num_accepted.tolist() == [0]
    assert out.correction.tolist() == [2]


def test_strict_reproduces_the_target_distribution():
    """The whole point of rejection sampling over cheaper token matching: whatever the
    proposer's distribution is, the committed token is drawn from the target's.

    Deliberately mismatched q: it favours token 0 where the target favours token 3."""
    torch.manual_seed(0)
    vocab, trials = 4, 40000
    p_row = torch.tensor([0.1, 0.2, 0.3, 0.4])
    q_row = torch.tensor([0.7, 0.1, 0.1, 0.1])

    p = p_row.view(1, 1, vocab).expand(trials, 2, vocab).contiguous()
    q = q_row.view(1, 1, vocab).expand(trials, 1, vocab).contiguous()
    draft = torch.multinomial(q_row, trials, replacement=True).view(trials, 1)

    out = accept_strict(draft, q, p, _lens(*([1] * trials)))
    # With k=1 the first committed token is the draft when accepted, else the correction.
    emitted = torch.where(out.num_accepted.bool(), draft.squeeze(1), out.correction)
    freq = torch.bincount(emitted, minlength=vocab).float() / trials
    assert torch.allclose(freq, p_row, atol=0.01), f"{freq.tolist()} != {p_row.tolist()}"


def test_strict_acceptance_rate_matches_the_theoretical_value():
    """E[accept] = sum_x min(p(x), q(x)) -- the overlap of the two distributions."""
    torch.manual_seed(1)
    vocab, trials = 4, 40000
    p_row = torch.tensor([0.1, 0.2, 0.3, 0.4])
    q_row = torch.tensor([0.7, 0.1, 0.1, 0.1])
    expected = torch.minimum(p_row, q_row).sum().item()

    p = p_row.view(1, 1, vocab).expand(trials, 2, vocab).contiguous()
    q = q_row.view(1, 1, vocab).expand(trials, 1, vocab).contiguous()
    draft = torch.multinomial(q_row, trials, replacement=True).view(trials, 1)
    out = accept_strict(draft, q, p, _lens(*([1] * trials)))
    rate = out.num_accepted.float().mean().item()
    assert abs(rate - expected) < 0.01, f"{rate} != {expected}"


def test_match_also_reproduces_the_target_distribution():
    """Token matching is not a temperature-0-only shortcut. Whatever it commits is the
    target's own draw at that position, so the committed token is distributed as p even
    when the proposer is drawing from something quite different."""
    torch.manual_seed(2)
    vocab, trials = 4, 40000
    p_row = torch.tensor([0.1, 0.2, 0.3, 0.4])
    q_row = torch.tensor([0.7, 0.1, 0.1, 0.1])

    draft = torch.multinomial(q_row, trials, replacement=True).view(trials, 1)
    target = torch.multinomial(p_row, 2 * trials, replacement=True).view(trials, 2)

    out = accept_match(draft, target, _lens(*([1] * trials)))
    emitted = torch.where(out.num_accepted.bool(), draft.squeeze(1), out.correction)
    freq = torch.bincount(emitted, minlength=vocab).float() / trials
    assert torch.allclose(freq, p_row, atol=0.01), f"{freq.tolist()} != {p_row.tolist()}"


def test_match_accepts_less_often_than_strict_against_a_sampling_proposer():
    """The real difference between the two: sum_x p(x)q(x) against sum_x min(p(x), q(x)).
    This is what a stochastic proposer would buy by paying for the probability rows."""
    torch.manual_seed(3)
    vocab, trials = 4, 40000
    p_row = torch.tensor([0.1, 0.2, 0.3, 0.4])
    q_row = torch.tensor([0.7, 0.1, 0.1, 0.1])

    draft = torch.multinomial(q_row, trials, replacement=True).view(trials, 1)
    target = torch.multinomial(p_row, 2 * trials, replacement=True).view(trials, 2)
    lens = _lens(*([1] * trials))
    match_rate = accept_match(draft, target, lens).num_accepted.float().mean().item()

    p = p_row.view(1, 1, vocab).expand(trials, 2, vocab).contiguous()
    q = q_row.view(1, 1, vocab).expand(trials, 1, vocab).contiguous()
    strict_rate = accept_strict(draft, q, p, lens).num_accepted.float().mean().item()

    assert abs(match_rate - (p_row * q_row).sum().item()) < 0.01
    assert abs(strict_rate - torch.minimum(p_row, q_row).sum().item()) < 0.01
    assert match_rate < strict_rate


def test_the_two_policies_agree_for_a_deterministic_proposer():
    """The MTP head drafts greedily, so q is a point mass and min(1, p/q) collapses to
    p(d) -- the same rate token matching gets. Strict buys nothing here."""
    torch.manual_seed(4)
    vocab, trials = 4, 40000
    p_row = torch.tensor([0.1, 0.2, 0.3, 0.4])
    draft = torch.full((trials, 1), 2, dtype=torch.int64)

    target = torch.multinomial(p_row, 2 * trials, replacement=True).view(trials, 2)
    lens = _lens(*([1] * trials))
    match_rate = accept_match(draft, target, lens).num_accepted.float().mean().item()

    p = p_row.view(1, 1, vocab).expand(trials, 2, vocab).contiguous()
    q = torch.zeros(trials, 1, vocab).scatter_(-1, draft.unsqueeze(-1), 1.0)
    strict_rate = accept_strict(draft, q, p, lens).num_accepted.float().mean().item()

    assert abs(match_rate - p_row[2].item()) < 0.01
    assert abs(strict_rate - p_row[2].item()) < 0.01
