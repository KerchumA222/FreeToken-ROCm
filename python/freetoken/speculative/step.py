"""The shape of one speculative decode step.

A round is four phases over a request whose committed KV ends at ``cached_len = C``, with
the token at index ``C`` sampled last round and not yet forwarded:

  verify   forward indices [C, C+k]. 1+k rows, each scoring one staged draft except the
           last, which predicts past all of them.
  accept   keep the leading run of drafts the target agreed with, plus one token the
           target drew itself. Commits ``1+a`` indices, so ``cached_len`` becomes
           ``C+1+a`` and index ``C+1+a`` holds the new pending token.
  redraft  run the draft head over the indices just committed, then autoregressively.
  stage    reserve the next round's draft slots.

The redraft phase is the part that is easy to get wrong. The head is a decoder block with
its own KV, and on a partial accept the slots it wrote for rejected drafts are rolled back
and reused for different tokens -- so a head that only ever stepped forward would attend
to KV describing a sequence that was never committed. It therefore re-processes the
indices this round committed, as one extend of length ``1+a``, before stepping. That extend
is free of allocation concerns: those indices are exactly the ones ``accept`` just moved
below ``cached_len``.

Index bookkeeping only. Nothing here builds a batch or touches a device; the engine
supplies those through the phase descriptions this module produces, which is what lets the
arithmetic be tested without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from freetoken.core import Req


@dataclass(frozen=True)
class VerifyPlan:
    """The rows of a verify forward, per request, in batch order."""

    first_index: tuple[int, ...]   # the first sequence index each request forwards (its C)
    num_rows: tuple[int, ...]      # 1 + staged drafts
    row_offset: tuple[int, ...]    # where each request's rows start in the flattened batch

    @property
    def total_rows(self) -> int:
        return self.row_offset[-1] + self.num_rows[-1] if self.num_rows else 0

    def logits_rows(self) -> list[int]:
        """Flattened row indices the LM head must score: every row of every request."""
        return [off + j for off, n in zip(self.row_offset, self.num_rows) for j in range(n)]


@dataclass(frozen=True)
class RedraftPlan:
    """The head's catch-up extend for one request, after acceptance.

    ``hidden_rows`` are positions in the verify batch's *gathered* rows (the same rows the
    LM head scored), and ``token_index`` are sequence indices into ``input_ids``. The head
    at sequence index ``t`` consumes the target's hidden at ``t`` and the token at ``t+1``,
    so the two lists are offset by one by construction."""

    first_index: int          # first sequence index the head re-processes (C)
    hidden_rows: tuple[int, ...]
    token_index: tuple[int, ...]

    @property
    def num_rows(self) -> int:
        return len(self.hidden_rows)


def plan_verify(reqs: Sequence[Req]) -> VerifyPlan:
    first, rows, offsets, off = [], [], [], 0
    for req in reqs:
        first.append(req.cached_len)
        n = 1 + req.spec_draft_len
        assert req.extend_len == n, (
            f"verify batch expects extend_len == 1 + drafts, got {req.extend_len} != {n}"
        )
        rows.append(n)
        offsets.append(off)
        off += n
    return VerifyPlan(tuple(first), tuple(rows), tuple(offsets))


def plan_redraft(req: Req, verify_row_offset: int, num_accepted: int) -> RedraftPlan:
    """The head's catch-up over the indices ``accept`` just committed.

    Call after ``Req.accept``: it reads the post-commit ``cached_len``, so the indices it
    names are the ones now backed by KV."""
    n = num_accepted + 1
    first = req.cached_len - n
    # Verify row j held the target's hidden for index first + j; rows past the accepted run
    # were computed over drafts that are being discarded.
    hidden_rows = tuple(verify_row_offset + j for j in range(n))
    # The head at index t reads the token at t+1. For the last row that token is the one
    # the target just drew, which accept() has already written at cached_len.
    token_index = tuple(first + 1 + j for j in range(n))
    return RedraftPlan(first, hidden_rows, token_index)


def draftable(req: Req, k: int) -> int:
    """How many drafts to stage for ``req``: ``k``, less whatever its output budget cannot
    hold. 0 means run it as a plain decode step this round."""
    return max(0, min(k, req.spec_capacity))


__all__ = ["RedraftPlan", "VerifyPlan", "draftable", "plan_redraft", "plan_verify"]
