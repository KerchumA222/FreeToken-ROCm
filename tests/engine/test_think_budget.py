"""FREETOKEN_THINKING_BUDGET: the </think> pressure schedule, thinking-span tracking, and its effect on sampling."""

from __future__ import annotations

import pytest
import torch

from freetoken.core import Req, SamplingParams, VerifyRow
from freetoken.engine.sample import BatchSamplingArgs, Sampler
from freetoken.engine.think_budget import FORCE_BIAS, ThinkBudget, apply_biases

START, END, NL, V = 7, 8, 9, 32


def _req(prompt: list[int], uid: int = 1, budget: int | None = None) -> Req:
    return Req(
        input_ids=torch.tensor(prompt, dtype=torch.int32), table_idx=0, cached_len=0, output_len=64, uid=uid,
        sampling_params=SamplingParams(thinking_budget=budget), cache_handle=None,
    )


def _emit(req: Req, tokens: list[int]) -> None:
    for t in tokens:
        req.append_host(torch.tensor([t], dtype=torch.int32))


def test_bias_ramps_quadratically_then_forces():
    tb = ThinkBudget(START, END, budget=100, ramp=0.5, max_bias=8.0)
    assert tb.bias(0, 100) == 0.0 and tb.bias(50, 100) == 0.0
    assert tb.bias(75, 100) == pytest.approx(2.0)  # half way up the ramp: 8 * 0.5^2
    assert 0 < tb.bias(60, 100) < tb.bias(90, 100) < 8.0
    assert tb.bias(100, 100) == FORCE_BIAS and tb.bias(150, 100) == FORCE_BIAS


def test_span_tracking_follows_the_prompt_and_the_output():
    tb = ThinkBudget(START, END, budget=10, ramp=0.5, max_bias=4.0)
    opened = _req([1, 2, START, 3])  # the template opened the block: 1 thinking token so far
    closed = _req([1, START, 3, END, 2], uid=2)  # thinking off (an empty block)
    assert tb.row_biases([opened, closed]) == []
    _emit(opened, [5] * 5)  # 6 thinking tokens: past the ramp (5)
    (row, b), = tb.row_biases([opened, closed])
    assert row == 0 and b == pytest.approx(tb.bias(6, 10))
    _emit(opened, [5] * 3 + [END])  # closed under pressure, one token before the budget
    assert tb.row_biases([opened]) == []
    assert tb.ended == {"natural": 0, "pressured": 1, "boundary": 0, "forced": 0}
    _emit(closed, [START] + [5] * 10)  # a block the model opened itself, run to the budget
    assert tb.row_biases([closed]) == [(0, FORCE_BIAS)]
    _emit(closed, [END])
    tb.row_biases([closed])
    assert tb.ended["forced"] == 1


def test_a_request_budget_below_the_server_one_wins():
    tb = ThinkBudget(START, END, budget=100, ramp=0.0, max_bias=4.0)
    small = _req([START], budget=4)
    big = _req([START], uid=2, budget=1000)
    _emit(small, [5] * 4)
    _emit(big, [5] * 4)
    biases = dict(tb.row_biases([small, big]))
    assert biases[0] == FORCE_BIAS and biases[1] == pytest.approx(tb.bias(4, 100))


def test_verify_rows_carry_their_request_bias():
    tb = ThinkBudget(START, END, budget=4, ramp=0.0, max_bias=4.0)
    a, b = _req([START, 5, 5, 5, 5], uid=1), _req([1], uid=2)
    rows = [VerifyRow(r, s) for s in range(3) for r in (a, b)]  # step-major
    assert tb.row_biases(rows) == [(0, FORCE_BIAS), (2, FORCE_BIAS), (4, FORCE_BIAS)]


def test_apply_biases_adds_to_the_listed_rows_only():
    logits = torch.zeros(3, V)
    out = apply_biases(logits, [(1, 2.5), (5, 1.0)], END)
    assert out[1, END] == 2.5 and out.sum() == 2.5 and logits.sum() == 0


@pytest.mark.parametrize("greedy", [True, False])
def test_sampler_closes_the_thought_at_the_budget(greedy):
    s = Sampler(torch.device("cpu"), V)
    s.think_budget = ThinkBudget(START, END, budget=3, ramp=0.0, max_bias=1.0)
    req = _req([START, 5, 5, 5])
    if not greedy:
        req.sampling_params.temperature = 1.0

    class _Batch:
        reqs = [req]

    args = s.prepare(_Batch())
    assert args.think_biases == [(0, FORCE_BIAS)] and args.think_end_id == END
    logits = torch.zeros(1, V)
    logits[0, 3] = 20.0  # the model wants to keep thinking
    if greedy:
        assert int(s.sample(logits, args)[0]) == END
    else:
        from freetoken.engine.think_budget import apply_biases as ab

        assert int(torch.argmax(ab(logits, args.think_biases, END))) == END


def test_anthropic_budget_tokens_reach_the_sampling_params():
    from freetoken.server import anthropic_api as A
    from freetoken.server.anthropic_models import AnthropicMessagesRequest

    def spec(thinking):
        req = AnthropicMessagesRequest.model_validate(
            {"model": "m", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], **({"thinking": thinking} if thinking else {})}
        )
        return A.convert_anthropic_to_genspec(req, {}).sampling_params.thinking_budget

    assert spec({"type": "enabled", "budget_tokens": 2048}) == 2048
    assert spec({"type": "disabled", "budget_tokens": 2048}) is None
    assert spec(None) is None


def test_a_line_break_inside_the_window_closes_the_thought():
    tb = ThinkBudget(START, END, budget=20, ramp=0.5, max_bias=4.0, boundary_ids=frozenset({NL}))
    req = _req([START])
    _emit(req, [5, NL] + [5] * 3)  # a line break before the window: no cut
    assert tb.row_biases([req]) == []
    _emit(req, [5] * 6)  # in the window (11 thinking tokens), mid-line: the ramp bias only
    (_, b), = tb.row_biases([req])
    assert 0 < b < 4.0
    _emit(req, [NL])  # a line ends inside the window: the next row closes the thought
    assert tb.row_biases([req]) == [(0, FORCE_BIAS)]
    # a verify's later rows follow unseen drafts and keep the ramp bias
    rows = [VerifyRow(req, s) for s in range(3)]
    biases = dict(tb.row_biases(rows))
    assert biases[0] == FORCE_BIAS and 0 < biases[1] < 4.0 and biases[1] == biases[2]
    _emit(req, [END])
    tb.row_biases([req])
    assert tb.ended["boundary"] == 1


def test_boundary_tokens_come_from_the_vocabulary():
    class _Tok:
        unk_token_id = 0
        vocab = ["<unk>", "a", "\u010a", "x\u010a\u010a", "<think>", "</think>", "b"]

        def convert_tokens_to_ids(self, t):
            return self.vocab.index(t) if t in self.vocab else 0

        def convert_ids_to_tokens(self, ids):
            return [self.vocab[i] for i in ids]

        def __len__(self):
            return len(self.vocab)

    tb = ThinkBudget.from_tokenizer(_Tok(), 100, 0.5, 4.0)
    assert (tb.start_id, tb.end_id) == (4, 5) and tb.boundary_ids == {2, 3}
