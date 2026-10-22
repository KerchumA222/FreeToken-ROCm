"""Greedy acceptance: which drafts a verify step keeps.

Row ``j`` of a verify scores the position draft ``j`` occupies, so the target's own token there
is ``targets[j]``. A draft survives only while every draft before it did, and the step always
commits the target's token just past the accepted run, so it advances at least one token. Only
greedy requests speculate. Acceptance keeps target argmaxes from the verify pass; batched verification may differ from a separate plain-decode pass because its matmul shapes differ, while sequential verification preserves plain-decode shapes.
"""

from __future__ import annotations

from typing import Sequence


def accept_length(drafts: Sequence[int], targets: Sequence[int]) -> int:
    """Length of the leading run where ``drafts[j] == targets[j]``; ``targets`` has one more entry than ``drafts``."""
    if len(targets) != len(drafts) + 1:
        raise ValueError(f"{len(drafts)} drafts need {len(drafts) + 1} target tokens, got {len(targets)}")
    n = 0
    while n < len(drafts) and drafts[n] == targets[n]:
        n += 1
    return n


def committed_tokens(drafts: Sequence[int], targets: Sequence[int], *, force_reject: bool = False) -> list[int]:
    """The tokens one verify step commits: the accepted drafts plus the target's token after them.

    ``force_reject`` keeps no draft (a debug mode that exercises the rollback every step)."""
    n = 0 if force_reject else accept_length(drafts, targets)
    if force_reject and len(targets) != len(drafts) + 1:
        raise ValueError(f"{len(drafts)} drafts need {len(drafts) + 1} target tokens, got {len(targets)}")
    return [int(t) for t in drafts[:n]] + [int(targets[n])]


def draft_depth(confidences: Sequence[float], min_p: float | None, k: int) -> int:
    """How many drafts (at most ``k``) one request verifies under FREETOKEN_SPEC_DRAFT_MIN_P.

    ``confidences[j]`` is the head's probability for draft ``j + 1``. Draft 1 is always verified;
    draft ``j`` joins only while drafts ``1..j-1`` were all at least ``min_p``, so the chain stops
    at (and still verifies) the first unsure draft. Unknown confidences count as confident."""
    if min_p is None:
        return k
    for depth, conf in enumerate(confidences[: k - 1], start=1):
        if conf < min_p:
            return depth
    return k


def batch_draft_depth(confidences: Sequence[Sequence[float]], min_p: float | None, k: int) -> int:
    """The verify depth of a batch: rows are step-major, so every request runs the deepest one's."""
    return max((draft_depth(c, min_p, k) for c in confidences), default=k)


__all__ = ["accept_length", "batch_draft_depth", "committed_tokens", "draft_depth"]
