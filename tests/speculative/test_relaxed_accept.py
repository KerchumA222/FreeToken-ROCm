"""FREETOKEN_SPEC_ACCEPT_MIN_P: a verify row takes its draft when the target gives the draft enough probability."""
from __future__ import annotations

import math

import torch

from freetoken.engine.sample import BatchSamplingArgs, keep_likely_drafts


def _logits(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor([[math.log(p) for p in r] for r in rows])


def test_rows_take_drafts_above_the_threshold_and_keep_their_sample_otherwise():
    # k = 2 drafts for one request: rows 0..1 score drafts 0..1 (rows 1..2's inputs); row 2 is the bonus row
    logits = _logits([[0.6, 0.3, 0.1], [0.2, 0.25, 0.55], [0.1, 0.1, 0.8]])
    input_ids = torch.tensor([9, 1, 1])  # row 0's input is the last committed token; drafts are 1 and 1
    sampled = torch.tensor([0, 2, 2], dtype=torch.int32)
    greedy = BatchSamplingArgs(temperatures=None)
    out = keep_likely_drafts(sampled, logits, input_ids, greedy, num_reqs=1, k=2, min_p=0.25)
    assert out.tolist() == [1, 1, 2]  # p(draft) 0.3 and 0.25 clear 0.25; the bonus row is never touched
    out = keep_likely_drafts(sampled, logits, input_ids, greedy, num_reqs=1, k=2, min_p=0.5)
    assert out.tolist() == [0, 2, 2] and out.dtype == torch.int32


def test_the_threshold_reads_probabilities_at_the_row_temperature():
    logits = _logits([[0.6, 0.3, 0.1], [0.1, 0.1, 0.8]])
    input_ids = torch.tensor([9, 1])
    sampled = torch.tensor([0, 2], dtype=torch.int32)
    # at temperature 2 the draft's 0.3 flattens to sqrt(0.3)/sum(sqrt(p)) ~= 0.33
    hot = BatchSamplingArgs(temperatures=torch.tensor([2.0, 2.0]))
    assert keep_likely_drafts(sampled, logits, input_ids, hot, num_reqs=1, k=1, min_p=0.32).tolist() == [1, 2]
    cold = BatchSamplingArgs(temperatures=torch.tensor([0.5, 0.5]))
    assert keep_likely_drafts(sampled, logits, input_ids, cold, num_reqs=1, k=1, min_p=0.32).tolist() == [0, 2]


def test_step_major_rows_pair_each_request_with_its_own_draft():
    # two requests, k = 1: rows [r0 s0, r1 s0, r0 s1, r1 s1]; drafts are rows 2 and 3's inputs
    logits = _logits([[0.5, 0.5], [0.9, 0.1], [0.5, 0.5], [0.5, 0.5]])
    input_ids = torch.tensor([7, 7, 1, 1])
    sampled = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    out = keep_likely_drafts(sampled, logits, input_ids, BatchSamplingArgs(temperatures=None), num_reqs=2, k=1, min_p=0.4)
    assert out.tolist() == [1, 0, 0, 0]


def test_engine_sample_verify_reads_the_flag_value(monkeypatch):
    from types import SimpleNamespace

    from freetoken.engine.engine import Engine
    from freetoken.engine.sample import Sampler
    from freetoken.env import ENV

    logits = _logits([[0.6, 0.3, 0.1], [0.1, 0.1, 0.8]])
    batch = SimpleNamespace(spec=SimpleNamespace(num_reqs=1, k=1), input_ids=torch.tensor([9, 1]))
    engine = SimpleNamespace(sampler=Sampler(device=torch.device("cpu"), vocab_size=3))
    args = BatchSamplingArgs(temperatures=None)
    monkeypatch.setattr(ENV.SPEC_ACCEPT_MIN_P, "value", None)  # unset: plain argmax
    assert Engine._sample_verify(engine, batch, logits, args).tolist() == [0, 2]
    monkeypatch.setattr(ENV.SPEC_ACCEPT_MIN_P, "value", 0.25)
    assert Engine._sample_verify(engine, batch, logits, args).tolist() == [1, 2]
    plain = SimpleNamespace(spec=None, input_ids=batch.input_ids)  # not a verify: untouched
    assert Engine._sample_verify(engine, plain, logits, args).tolist() == [0, 2]
