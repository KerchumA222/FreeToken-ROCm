"""The MTP (``nextn``) proposer: one decoder block, run k times.

The head predicts the token after next from the target's hidden state at position t and
the embedding of the token at t+1, so the first draft comes free from the verify forward
that just ran -- its hidden states are already in hand. Each further step has nothing new
from the target and feeds the head its own previous output, which is why acceptance falls
off quickly past a depth of 2 or 3: a one-layer head was trained to predict one token
ahead, not to be an autoregressive model.

The head is a decoder block with its own KV at ``layer_id``, so every step has to run
inside a batch context the caller has prepared -- ``step_ctx`` is that hook. The caller
owns cache slots and attention metadata; this class owns the head's arithmetic.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, ContextManager

import torch

from .proposer import Drafts, Proposer

# A step context is entered around one head forward, given the draft index.
StepCtx = Callable[[int], ContextManager]


class MTPProposer(Proposer):
    def __init__(
        self,
        head,
        embed_tokens,
        lm_head,
        num_draft_tokens: int,
        *,
        step_ctx: StepCtx | None = None,
        need_probs: bool = False,
    ) -> None:
        self.head = head
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self._k = num_draft_tokens
        self._step_ctx = step_ctx or (lambda _j: nullcontext())
        self.need_probs = need_probs

    @property
    def num_draft_tokens(self) -> int:
        return self._k

    def _logits(self, hidden: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One head step: returns (logits, the head's output hidden for the next step)."""
        out = self.head.forward(hidden, self.embed_tokens.forward(tokens))
        return self.lm_head.forward(out), out

    @torch.inference_mode()
    def propose(
        self,
        hidden: torch.Tensor,
        last_tokens: torch.Tensor,
        draft_lens: torch.Tensor,
    ) -> Drafts:
        bs = hidden.shape[0]
        k = self._k
        tokens = torch.empty(bs, k, dtype=torch.int64, device=hidden.device)
        probs = None

        h, prev = hidden, last_tokens.to(torch.int64)
        for j in range(k):
            with self._step_ctx(j):
                logits, h = self._logits(h, prev)
            # Greedy: the proposer's job is to be right often, not to be diverse. Sampling
            # here would only lower the acceptance rate without widening what the target
            # can emit -- the target's own distribution is what acceptance enforces.
            prev = logits.argmax(dim=-1)
            tokens[:, j] = prev
            if self.need_probs:
                if probs is None:
                    probs = torch.empty(bs, k, logits.shape[-1], dtype=torch.float32,
                                        device=logits.device)
                probs[:, j] = torch.softmax(logits.float(), dim=-1)

        if self.need_probs:
            # The proposal actually used is the argmax, not the softmax row it came from,
            # so q is a point mass. Handing strict acceptance the softmax row instead would
            # inflate p/q by 1/q and accept drafts the target dislikes. (With q a point
            # mass, strict accepts at exactly the rate token matching does, which is why
            # this is off by default.)
            probs = torch.zeros_like(probs).scatter_(-1, tokens.unsqueeze(-1), 1.0)
        return Drafts(tokens=tokens, probs=probs, lens=draft_lens)


__all__ = ["MTPProposer"]
