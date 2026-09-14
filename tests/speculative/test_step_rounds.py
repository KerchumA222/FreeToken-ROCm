"""Several speculative rounds end to end, on the real Req bookkeeping.

The property that matters is not that any particular draft is accepted -- it is that the
committed token sequence is exactly what plain decoding would have produced, whatever the
proposer guesses. These simulate a target with a known next-token rule and proposers that
range from perfect to always wrong, and compare against that rule.
"""

from __future__ import annotations

import torch

from freetoken.core import Req, SamplingParams
from freetoken.speculative.accept import accept_match
from freetoken.speculative.step import draftable, plan_redraft, plan_verify

VOCAB = 1000


def truth(token: int) -> int:
    """The target's rule: the next token is one past the last, wrapping."""
    return (token + 1) % VOCAB


def _req(prompt: list[int], output_len: int) -> Req:
    req = Req(
        input_ids=torch.tensor(prompt, dtype=torch.int32),
        table_idx=0, cached_len=0, output_len=output_len, uid=0,
        sampling_params=SamplingParams(), cache_handle=None,
    )
    req.complete_one()
    req.append_host(torch.tensor([truth(prompt[-1])], dtype=torch.int32))
    return req


def _run(proposer, k: int, rounds: int, output_len: int = 40):
    """Drive `rounds` verify steps and return (final ids, tokens committed per round)."""
    prompt = [5, 6, 7]
    req = _req(prompt, output_len)
    committed = []
    for _ in range(rounds):
        if not req.can_decode:
            break
        n = draftable(req, k)
        req.reserve_drafts(n)
        for j in range(n):
            req.write_draft(j, torch.tensor(proposer(req, j), dtype=torch.int32))

        plan = plan_verify([req])
        assert plan.total_rows == 1 + n
        # The target scores every staged row: row j predicts from the token at C + j.
        c = plan.first_index[0]
        target = torch.tensor(
            [[truth(int(req.input_ids[c + j])) for j in range(n + 1)]], dtype=torch.int64
        )
        drafts = req.input_ids[c + 1: c + 1 + n].to(torch.int64).view(1, -1)
        out = accept_match(drafts, target, torch.tensor([n]))
        a = int(out.num_accepted[0])

        req.accept(a, out.correction[0].to(torch.int32))
        committed.append(a + 1)
    return req, committed


def _expected(prompt: list[int], n: int) -> list[int]:
    seq = list(prompt)
    while len(seq) < len(prompt) + n:
        seq.append(truth(seq[-1]))
    return seq


def test_a_perfect_proposer_commits_k_plus_one_per_round():
    """The best case: every draft is right, so each verify forward retires k+1 tokens."""
    def perfect(req, j):
        return truth(int(req.input_ids[req.cached_len + j]))

    req, committed = _run(perfect, k=3, rounds=4)
    assert committed == [4, 4, 4, 4]
    assert req.input_ids.tolist() == _expected([5, 6, 7], req.input_ids.numel() - 3)


def test_a_useless_proposer_still_commits_one_per_round():
    """The worst case degrades to plain decoding, and must stay correct."""
    req, committed = _run(lambda req, j: 999, k=3, rounds=5)
    assert committed == [1, 1, 1, 1, 1]
    assert req.input_ids.tolist() == _expected([5, 6, 7], req.input_ids.numel() - 3)


def test_a_proposer_that_is_right_only_first_commits_two_per_round():
    def first_only(req, j):
        return truth(int(req.input_ids[req.cached_len + j])) if j == 0 else 999

    req, committed = _run(first_only, k=3, rounds=4)
    assert committed == [2, 2, 2, 2]
    assert req.input_ids.tolist() == _expected([5, 6, 7], req.input_ids.numel() - 3)


def test_output_is_identical_across_every_proposer_and_depth():
    """Speculation is a speedup, not a behaviour change: for the same number of committed
    tokens the sequence must not depend on k or on how good the drafts were."""
    def perfect(req, j):
        return truth(int(req.input_ids[req.cached_len + j]))

    def alternating(req, j):
        return truth(int(req.input_ids[req.cached_len + j])) if j % 2 == 0 else 12

    runs = []
    for proposer in (perfect, alternating, lambda r, j: 999):
        for k in (1, 2, 4):
            req, _ = _run(proposer, k=k, rounds=30, output_len=60)
            runs.append(req.input_ids[:30].tolist())
    assert all(r == runs[0] for r in runs), runs[:3]
    assert runs[0] == _expected([5, 6, 7], 27)


def test_the_request_never_runs_past_its_output_budget():
    def perfect(req, j):
        return truth(int(req.input_ids[req.cached_len + j]))

    req, _ = _run(perfect, k=4, rounds=100, output_len=11)
    assert req.device_len <= req.max_device_len
    assert not req.can_decode


def test_redraft_covers_exactly_the_indices_just_committed():
    """The head re-processes the accepted run so its KV describes the committed sequence
    and not the drafts that were rolled back."""
    req = _req([5, 6, 7], 40)
    req.reserve_drafts(3)
    for j in range(3):
        req.write_draft(j, torch.tensor(100 + j, dtype=torch.int32))
    plan = plan_verify([req])
    c = plan.first_index[0]

    req.accept(1, torch.tensor(77, dtype=torch.int32))
    redraft = plan_redraft(req, plan.row_offset[0], num_accepted=1)

    # One accepted draft plus the correction: two indices became KV-backed this round.
    assert redraft.num_rows == 2
    assert redraft.first_index == c
    assert req.cached_len == c + 2
    # Verify rows 0 and 1 held the target's hidden for indices c and c+1; row 2 was
    # computed over a draft that is being discarded.
    assert redraft.hidden_rows == (0, 1)
    # The head at index t reads the token at t+1, and the last of those is the correction.
    assert redraft.token_index == (c + 1, c + 2)
    assert int(req.input_ids[c + 2]) == 77


def test_redraft_after_a_full_accept_covers_every_draft():
    req = _req([5, 6, 7], 40)
    req.reserve_drafts(2)
    for j in range(2):
        req.write_draft(j, torch.tensor(100 + j, dtype=torch.int32))
    plan = plan_verify([req])
    req.accept(2, torch.tensor(77, dtype=torch.int32))
    redraft = plan_redraft(req, plan.row_offset[0], num_accepted=2)
    assert redraft.num_rows == 3
    assert redraft.hidden_rows == (0, 1, 2)
    assert redraft.token_index[-1] == req.cached_len


def test_verify_plan_lays_out_ragged_requests_contiguously():
    """Requests near their budget stage fewer drafts, so rows are ragged; the LM head has
    to score all of them and the offsets are what locate each request's rows."""
    reqs = []
    for i, budget in enumerate((40, 40, 3)):
        r = _req([5, 6, 7], budget)
        r.table_idx = i
        n = draftable(r, 2)
        r.reserve_drafts(n)
        reqs.append(r)
    plan = plan_verify(reqs)
    assert plan.num_rows == (3, 3, 2)
    assert plan.row_offset == (0, 3, 6)
    assert plan.total_rows == 8
    assert plan.logits_rows() == [0, 1, 2, 3, 4, 5, 6, 7]
