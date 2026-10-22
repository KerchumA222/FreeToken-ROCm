"""Greedy MTP acceptance: the leading run of drafts the target agrees with, plus the target's own token."""
from __future__ import annotations

import pytest

from freetoken.speculative import accept_length, batch_draft_depth, committed_tokens, draft_depth


@pytest.mark.parametrize(
    "drafts, targets, n",
    [
        ([5], [5, 9], 1),
        ([5], [6, 9], 0),
        ([1, 2, 3], [1, 2, 3, 4], 3),
        ([1, 2, 3], [1, 7, 3, 4], 1),     # a later match after a miss does not count
        ([1, 2, 3], [9, 2, 3, 4], 0),
        ([1, 2, 3], [1, 2, 8, 4], 2),
    ],
)
def test_accept_length_is_the_leading_run(drafts, targets, n):
    assert accept_length(drafts, targets) == n


def test_committed_tokens_end_with_the_target_token_after_the_run():
    assert committed_tokens([1, 2, 3], [1, 2, 8, 4]) == [1, 2, 8]
    assert committed_tokens([1, 2, 3], [1, 2, 3, 4]) == [1, 2, 3, 4]
    assert committed_tokens([1, 2, 3], [9, 2, 3, 4]) == [9]


def test_committed_tokens_are_the_target_tokens():
    """Whatever is accepted, the committed run equals the target's own tokens at those rows: it is what plain greedy decoding emits."""
    drafts, targets = [4, 4, 1, 2], [4, 4, 1, 3, 0]
    out = committed_tokens(drafts, targets)
    assert out == targets[: len(out)]


def test_force_reject_commits_one_target_token():
    assert committed_tokens([1, 2, 3], [1, 2, 3, 4], force_reject=True) == [1]


def test_targets_must_cover_every_draft_plus_one():
    with pytest.raises(ValueError):
        accept_length([1, 2], [1, 2])
    with pytest.raises(ValueError):
        committed_tokens([1, 2], [1, 2], force_reject=True)


@pytest.mark.parametrize(
    "confidences, min_p, depth",
    [
        ([0.9, 0.9, 0.9], 0.5, 4),   # every draft before the last is confident: full depth
        ([0.1, 0.9, 0.9], 0.5, 1),   # the first draft is unsure: it alone is verified
        ([0.9, 0.2, 0.9], 0.5, 2),   # the chain stops at (and verifies) the first unsure draft
        ([0.9, 0.9, 0.2], 0.5, 3),
        ([0.5, 0.9, 0.9], 0.5, 4),   # exactly the threshold counts as confident
        ([0.9], 0.5, 4),             # confidences not yet known count as confident
        ([0.1, 0.1, 0.1], None, 4),  # flag unset: always the full depth
    ],
)
def test_draft_depth_stops_at_the_first_unsure_draft(confidences, min_p, depth):
    assert draft_depth(confidences, min_p, k=4) == depth


def test_batch_draft_depth_is_the_deepest_request():
    assert batch_draft_depth([[0.9, 0.2], [0.1, 0.9]], 0.5, k=3) == 2
    assert batch_draft_depth([[0.1, 0.9], [0.1, 0.9]], 0.5, k=3) == 1
    assert batch_draft_depth([[0.9, 0.9], [0.1, 0.9]], 0.5, k=3) == 3
    assert batch_draft_depth([[0.1], [0.1]], None, k=3) == 3
