"""Greedy MTP acceptance: the leading run of drafts the target agrees with, plus the target's own token."""
from __future__ import annotations

import pytest

from freetoken.speculative import accept_length, committed_tokens


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
