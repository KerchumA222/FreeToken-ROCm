"""Configuration for speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Method = Literal["mtp"]
AcceptPolicy = Literal["strict", "match"]


@dataclass(frozen=True)
class SpeculativeConfig:
    """How many tokens to propose per step, with what, and how to judge them.

    ``num_draft_tokens`` is the depth a proposer is asked for. A one-layer MTP head can be
    run autoregressively past its training depth of 1; each extra step feeds the head its
    own previous output, so acceptance falls off quickly and 2-3 is the useful range.
    """

    method: Method = "mtp"
    num_draft_tokens: int = 2
    # Both policies are exact; they differ in acceptance rate and in cost. "match" keeps a
    # draft that equals the target's own draw at that position, and needs one sampled token
    # per position. "strict" is rejection sampling against the proposer's distribution, and
    # accepts strictly more often -- but only when the proposer samples, and it needs the
    # full probability rows from both models. The MTP head drafts greedily, where the two
    # accept at the same rate, so "match" is the default.
    accept: AcceptPolicy = "match"
    # Stop proposing for a request whose recent acceptance rate falls below this, and retry
    # after a while. Drafting is not free, so a request the head is bad at should pay for
    # one forward, not two. 0 disables the check.
    min_acceptance_rate: float = 0.0
    # Draft head path, when it does not ship inside the target checkpoint.
    draft_model_path: str | None = None

    def __post_init__(self) -> None:
        if self.num_draft_tokens < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {self.num_draft_tokens}")
        if not 0.0 <= self.min_acceptance_rate <= 1.0:
            raise ValueError(f"min_acceptance_rate out of range: {self.min_acceptance_rate}")


__all__ = ["SpeculativeConfig", "AcceptPolicy", "Method"]
