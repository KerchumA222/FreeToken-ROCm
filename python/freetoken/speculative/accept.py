"""Turning one verify forward into a run of committed tokens.

A verify forward scores ``1 + k`` positions per request: the ``k`` drafts, plus one past
them. Acceptance is prefix-bound -- a draft only counts if every draft before it was also
accepted -- so both policies here reduce to "how long is the leading run of accepted
draws", and both end by emitting one token the target chose at the first position the
drafts stopped being usable. That last token is why a step always commits at least one.

Both policies are exact: what comes out is distributed as if the target had been decoded
on its own. They differ in how often a draft survives, and in what they need to decide
that -- token matching needs one sampled token per position, rejection sampling needs the
full probability rows from both models.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class AcceptResult(NamedTuple):
    num_accepted: torch.Tensor  # [bs] int64: drafts committed, per request
    correction: torch.Tensor    # [bs] int64: the target's own token at the first rejected slot


def _leading_run(ok: torch.Tensor) -> torch.Tensor:
    """Length of the leading run of True in each row. cumprod zeroes every column after
    the first False, so the row sum is exactly that length."""
    return ok.int().cumprod(dim=1).sum(dim=1)


def _valid_mask(draft_lens: torch.Tensor, k: int) -> torch.Tensor:
    """[bs, k] True where a request actually staged a draft (depth is per-request)."""
    return torch.arange(k, device=draft_lens.device).unsqueeze(0) < draft_lens.unsqueeze(1)


def accept_match(
    draft_tokens: torch.Tensor,   # [bs, k] int
    target_tokens: torch.Tensor,  # [bs, k+1] int, the target's own draw at each position
    draft_lens: torch.Tensor,     # [bs] int
) -> AcceptResult:
    """Accept a draft when it equals the token the target itself drew at that position.

    Exact at any temperature, and not only at zero. Whatever is committed here is
    ``target_tokens`` either way -- an accepted draft only because it equalled the
    target's draw, a correction because it did not -- so the committed run is literally a
    run of the target's own samples, drawn from the right conditionals: position ``j`` was
    forwarded over the drafts before it, and those equal the target's own draws whenever
    ``j`` is reached at all.

    This holds only because ``target_tokens`` is a fresh draw from the target at each
    position. Comparing drafts against the target's *argmax* while the request is sampling
    would be the biased shortcut this is often mistaken for.

    What it gives up against ``accept_strict`` is acceptance rate, not correctness: a
    draft survives with probability ``sum_x p(x)q(x)`` here against ``sum_x min(p, q)``
    there. The two are equal when the proposer is deterministic."""
    k = draft_tokens.shape[1]
    ok = (draft_tokens == target_tokens[:, :k]) & _valid_mask(draft_lens, k)
    num_accepted = _leading_run(ok)
    correction = target_tokens.gather(1, num_accepted.unsqueeze(1)).squeeze(1)
    return AcceptResult(num_accepted, correction)


def accept_strict(
    draft_tokens: torch.Tensor,   # [bs, k] int
    draft_probs: torch.Tensor,    # [bs, k, V] the proposer's distribution at each draft
    target_probs: torch.Tensor,   # [bs, k+1, V] the target's distribution at each position
    draft_lens: torch.Tensor,     # [bs] int
    generator: torch.Generator | None = None,
) -> AcceptResult:
    """Rejection sampling: keep draft ``d`` with probability ``min(1, p(d)/q(d))`` and on
    a rejection draw from the residual ``norm(relu(p - q))``.

    Also exact, and accepts at least as often as ``accept_match`` -- strictly more often
    when the proposer is stochastic, and exactly as often when it is not, since a point
    mass makes ``min(1, p(d)/q(d))`` equal to ``p(d)``. It costs the target's full
    probability rows, which token matching does not need at all, so it only pays for
    itself against a proposer that samples. The
    uniforms for every position are drawn up front; that is sound because the prefix
    product below discards the draws after the first rejection, and an unused draw does
    not bias the ones before it."""
    bs, k = draft_tokens.shape
    idx = draft_tokens.unsqueeze(-1).long()
    p_d = target_probs[:, :k].gather(-1, idx).squeeze(-1)
    q_d = draft_probs.gather(-1, idx).squeeze(-1)
    # q is what the proposer actually sampled from, so q(d) > 0 for any draft it produced;
    # the clamp only guards a proposer that returns an unnormalized or stale row.
    ratio = p_d / q_d.clamp_min(torch.finfo(q_d.dtype).tiny)
    uniform = torch.rand(bs, k, device=ratio.device, dtype=ratio.dtype, generator=generator)
    ok = (uniform < ratio) & _valid_mask(draft_lens, k)
    num_accepted = _leading_run(ok)

    # The correction is drawn at the first position the run stopped at. When every draft
    # was accepted that position is k, where no draft exists and the target's own
    # distribution is the right one -- this is the bonus token.
    rows = torch.arange(bs, device=num_accepted.device)
    p_at = target_probs[rows, num_accepted]
    all_accepted = num_accepted >= draft_lens
    q_at = draft_probs[rows, num_accepted.clamp(max=max(k - 1, 0))]
    residual = torch.clamp(p_at - q_at, min=0.0)
    total = residual.sum(dim=-1, keepdim=True)
    # p and q can coincide to within rounding, leaving nothing to draw from; the target's
    # own distribution is the correct fallback (the draft was rejected by an unlucky
    # uniform, not because the two disagree).
    residual = torch.where(total > 0, residual / total.clamp_min(torch.finfo(total.dtype).tiny), p_at)
    draw_from = torch.where(all_accepted.unsqueeze(1), p_at, residual)
    correction = torch.multinomial(draw_from, 1, generator=generator).squeeze(1)
    return AcceptResult(num_accepted, correction)


__all__ = ["AcceptResult", "accept_match", "accept_strict"]
