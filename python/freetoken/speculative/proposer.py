"""Draft proposers: what a speculative step asks for its guesses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch


class Drafts(NamedTuple):
    tokens: torch.Tensor       # [bs, k] int64
    probs: torch.Tensor | None  # [bs, k, V], the distribution each token was drawn from
    lens: torch.Tensor         # [bs] int64, per-request depth (a request near its budget drafts less)


class Proposer(ABC):
    """Produces ``k`` guesses per request from the target's last forward.

    ``probs`` is what separates a proposer usable with strict rejection sampling from one
    that can only be checked by token equality: the acceptance test needs the distribution
    the draft was actually drawn from, not just the draw."""

    @property
    @abstractmethod
    def num_draft_tokens(self) -> int: ...

    @abstractmethod
    def propose(
        self,
        hidden: torch.Tensor,      # [bs, H] target hidden at each request's last committed position
        last_tokens: torch.Tensor,  # [bs] the token that position predicted
        draft_lens: torch.Tensor,   # [bs] how many drafts each request has room for
    ) -> Drafts: ...


__all__ = ["Drafts", "Proposer"]
