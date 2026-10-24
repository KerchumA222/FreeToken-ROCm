"""Thinking budget by gradual pressure (FREETOKEN_THINKING_BUDGET).

A request's thinking is the span after its last ``<think>`` with no ``</think>`` yet, in the prompt (a template
that opens the block) or in what it generated. Past ``ramp * budget`` thinking tokens, the sampler adds a bias to
the ``</think>`` logit that grows quadratically to ``max_bias`` at the budget, so the model can close its thought
where it was already close to. Mid-reasoning a model's ``</think>`` logit sits far below that (Ornith never took it
in a 300-token window at +10), so inside the window the first line break also closes the thought: the cut lands
between steps rather than mid-sentence. At the budget the bias forces ``</think>`` wherever it is (the backstop).

The budget is the server's ``FREETOKEN_THINKING_BUDGET``, or a request's smaller one (Anthropic
``thinking.budget_tokens``); with the server budget unset nothing is biased, as upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

# the backstop: far above any natural logit gap
FORCE_BIAS = 1.0e4


@dataclass
class _State:
    scanned: int  # tokens of input_ids already classified
    in_think: bool
    count: int  # thinking tokens since the last <think>
    peak_bias: float = 0.0
    boundary_cut: bool = False  # closed by the line-break rule


class ThinkBudget:
    def __init__(self, start_id: int, end_id: int, budget: int, ramp: float = 0.75, max_bias: float = 10.0,
                 boundary_ids: frozenset[int] = frozenset()):
        if budget < 1:
            raise ValueError(f"thinking budget must be positive, got {budget}")
        self.start_id, self.end_id, self.budget = int(start_id), int(end_id), int(budget)
        self.ramp = min(max(float(ramp), 0.0), 1.0)
        self.max_bias = float(max_bias)
        # tokens that end a line: inside the pressure window, thinking closes after one
        self.boundary_ids = frozenset(boundary_ids)
        self._states: dict[int, _State] = {}
        self.ended = {"natural": 0, "pressured": 0, "boundary": 0, "forced": 0}

    @classmethod
    def from_tokenizer(cls, tokenizer, budget: int, ramp: float, max_bias: float) -> "ThinkBudget | None":
        """The budget for a checkpoint whose vocabulary has single ``<think>`` / ``</think>`` tokens, else None."""
        ids = []
        for tag in ("<think>", "</think>"):
            tid = tokenizer.convert_tokens_to_ids(tag)
            if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
                return None
            ids.append(int(tid))
        boundary = frozenset(
            i for i, tok in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer)))))
            if tok and ("\n" in tok or "\u010a" in tok)  # byte-level BPE spells a newline as U+010A
        )
        return cls(ids[0], ids[1], budget, ramp, max_bias, boundary)

    def budget_for(self, req) -> int:
        own = getattr(req.sampling_params, "thinking_budget", None)
        return min(self.budget, int(own)) if own else self.budget

    def bias(self, count: int, budget: int) -> float:
        """The ``</think>`` bias after ``count`` thinking tokens."""
        if count >= budget:
            return FORCE_BIAS
        start = self.ramp * budget
        if count < start or budget <= start:
            return 0.0
        return self.max_bias * ((count - start) / (budget - start)) ** 2

    def _state(self, req) -> _State:
        ids = req.input_ids
        st = self._states.get(req.uid)
        if st is None:
            # classify the prompt once: is the last tag in it an open <think>?
            prompt = ids[: req.prompt_len]
            starts = (prompt == self.start_id).nonzero()
            ends = (prompt == self.end_id).nonzero()
            last_start = int(starts[-1]) if starts.numel() else -1
            last_end = int(ends[-1]) if ends.numel() else -1
            in_think = last_start > last_end
            st = _State(scanned=len(prompt), in_think=in_think, count=len(prompt) - 1 - last_start if in_think else 0)
            self._states[req.uid] = st
        n = len(ids)
        if n > st.scanned:
            budget = self.budget_for(req)
            for tok in ids[st.scanned : n].tolist():
                if tok == self.start_id:
                    st.in_think, st.count, st.peak_bias, st.boundary_cut = True, 0, 0.0, False
                elif tok == self.end_id and st.in_think:
                    st.in_think = False
                    kind = ("forced" if st.count >= budget else "boundary" if st.boundary_cut
                            else "pressured" if st.peak_bias > 0 else "natural")
                    self.ended[kind] += 1
                    logger.info(
                        "thinking ended after %d of %d budget tokens (%s); natural/pressured/boundary/forced so far %s",
                        st.count, budget, kind, "/".join(str(v) for v in self.ended.values()),
                    )
                elif st.in_think:
                    st.count += 1
            st.scanned = n
        return st

    def row_biases(self, rows) -> list[tuple[int, float]]:
        """``(row, bias)`` for every batch row whose request is thinking past the ramp; rows of a verify carry their request.

        The line-break cut applies to the row that follows the last committed token (a verify's step 0): later verify rows
        follow drafts the host has not seen, and keep the ramp bias."""
        out = []
        seen: dict[int, tuple[float, float]] = {}
        for row, r in enumerate(rows):
            req = getattr(r, "req", r)
            biases = seen.get(req.uid)
            if biases is None:
                st = self._state(req)
                budget = self.budget_for(req)
                ramp = self.bias(st.count, budget) if st.in_think else 0.0
                first = ramp
                if ramp and ramp < FORCE_BIAS and int(req.input_ids[-1]) in self.boundary_ids:
                    first = FORCE_BIAS
                    st.boundary_cut = True
                st.peak_bias = max(st.peak_bias, ramp)
                biases = seen[req.uid] = (first, ramp)
            b = biases[0] if getattr(r, "step", 0) == 0 else biases[1]
            if b:
                out.append((row, b))
        return out

    def forget(self, uid: int) -> None:
        self._states.pop(uid, None)


def apply_biases(logits: torch.Tensor, biases: list[tuple[int, float]], token_id: int) -> torch.Tensor:
    """``logits`` with ``bias`` added to ``token_id`` in each listed row (a copy; rows past the logits are skipped)."""
    rows = [(r, b) for r, b in biases if r < logits.shape[0]]
    if not rows:
        return logits
    out = logits.float().clone()
    idx = torch.tensor([r for r, _ in rows], device=logits.device)
    out[idx, token_id] += torch.tensor([b for _, b in rows], device=logits.device, dtype=out.dtype)
    return out


__all__ = ["FORCE_BIAS", "ThinkBudget", "apply_biases"]
