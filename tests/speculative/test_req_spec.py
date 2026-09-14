"""Req's speculative bookkeeping: staging drafts, committing a prefix of them, and the
invariant everything downstream depends on -- len(input_ids) == device_len, and
extend_len is exactly what the next forward must cover."""

from __future__ import annotations

import pytest
import torch

from freetoken.core import Req, SamplingParams


def _req(prompt_len: int = 4, output_len: int = 16) -> Req:
    req = Req(
        input_ids=torch.arange(prompt_len, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    # Bring it to the state a request is in at the top of a decode step: the last sampled
    # token is in input_ids but not yet in the KV cache.
    req.complete_one()
    req.append_host(torch.tensor([99], dtype=torch.int32))
    return req


def _drafts(*ids: int) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int32)


def test_a_plain_decode_step_extends_by_one():
    req = _req()
    assert req.extend_len == 1
    assert req.spec_draft_len == 0


def test_staging_drafts_widens_the_extend():
    req = _req()
    cached = req.cached_len
    req.propose(_drafts(7, 8))
    assert req.spec_draft_len == 2
    # The verify forward covers the last committed token plus both drafts.
    assert req.extend_len == 3
    assert req.cached_len == cached
    assert req.input_ids.numel() == req.device_len
    assert req.input_ids[-2:].tolist() == [7, 8]


def test_accepting_everything_commits_the_drafts_and_the_bonus_token():
    req = _req()
    cached = req.cached_len
    req.propose(_drafts(7, 8))
    rolled_back = req.accept(2, torch.tensor(55, dtype=torch.int32))
    assert rolled_back == 0
    assert req.spec_draft_len == 0
    # Committed: the token that was pending, plus both drafts.
    assert req.cached_len == cached + 3
    # And the bonus token is now the pending one, so the next step is an ordinary decode.
    assert req.extend_len == 1
    assert req.input_ids[-3:].tolist() == [7, 8, 55]
    assert req.input_ids.numel() == req.device_len


def test_a_partial_accept_drops_the_rest_and_reports_the_rollback():
    req = _req()
    cached = req.cached_len
    req.propose(_drafts(7, 8, 9))
    rolled_back = req.accept(1, torch.tensor(55, dtype=torch.int32))
    # Two staged positions were forwarded into the KV cache and must be freed.
    assert rolled_back == 2
    assert req.cached_len == cached + 2
    assert req.extend_len == 1
    assert req.input_ids[-2:].tolist() == [7, 55]
    assert req.input_ids.numel() == req.device_len


def test_rejecting_everything_still_commits_one_token():
    """The target's draw at the first rejected slot is always usable, so a verify step
    can never make less progress than a plain decode step."""
    req = _req()
    cached = req.cached_len
    req.propose(_drafts(7, 8))
    rolled_back = req.accept(0, torch.tensor(55, dtype=torch.int32))
    assert rolled_back == 2
    assert req.cached_len == cached + 1
    assert req.input_ids[-1].item() == 55
    assert req.extend_len == 1


def test_capacity_leaves_room_for_the_token_the_verify_step_emits():
    """A request one token from its budget must not draft: the verify forward would have
    nowhere to put the token the target itself produces."""
    req = _req(prompt_len=4, output_len=2)
    assert req.remain_len == 1
    assert req.spec_capacity == 0
    with pytest.raises(AssertionError):
        req.propose(_drafts(7))


def test_capacity_is_honoured_up_to_the_last_usable_slot():
    req = _req(prompt_len=4, output_len=4)
    assert req.spec_capacity == 2
    req.propose(_drafts(7, 8))
    req.accept(2, torch.tensor(55, dtype=torch.int32))
    assert req.device_len == req.max_device_len
    assert not req.can_decode


def test_dropping_drafts_restores_the_pre_proposal_state():
    """An abort or an abandoned batch has to unstage without a verify forward."""
    req = _req()
    before = (req.cached_len, req.device_len, req.input_ids.tolist())
    req.propose(_drafts(7, 8))
    assert req.drop_drafts() == 2
    assert (req.cached_len, req.device_len, req.input_ids.tolist()) == before
    assert req.drop_drafts() == 0


def test_double_proposal_is_rejected():
    req = _req()
    req.propose(_drafts(7))
    with pytest.raises(AssertionError):
        req.propose(_drafts(8))


@pytest.mark.parametrize("k,accepted,expected_freeable", [
    (1, 0, 0),   # depth 1, rejected: the correction takes the draft's own slot
    (1, 1, 0),   # depth 1, accepted: the bonus token takes the next slot
    (3, 0, 2),
    (3, 1, 1),
    (3, 3, 0),
])
def test_freeable_positions_after_acceptance(k, accepted, expected_freeable):
    """How many staged positions acceptance leaves unused.

    One less than the number of drafts rejected, because the correction token occupies the
    first rejected draft's position -- at depth 1 that means nothing is ever freeable.
    Freeing by the rejected-draft count instead returns a slot the request never owned,
    which corrupts whichever request is handed it next."""
    req = _req(output_len=32)
    req.propose(_drafts(*range(70, 70 + k)))
    staged_device_len = req.device_len
    req.accept(accepted, torch.tensor(55, dtype=torch.int32))
    # Negative when every draft was accepted: the bonus token needs a position past the
    # staged range, so there is nothing to free and the request has grown instead.
    assert max(0, staged_device_len - req.device_len) == expected_freeable
    # And the pending correction is always still inside device_len, never freed with them.
    assert req.device_len == req.cached_len + 1
