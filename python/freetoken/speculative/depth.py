"""Adaptive speculative depth.

A verify round at depth ``k`` commits ``accepted + 1`` tokens and costs one forward over
``k + 1`` rows, plus ``k - 1`` chained head steps to draft them. The best ``k`` depends on
how often the chain's later drafts land and on what an extra row costs. On the disk tier
that cost is the row's non-resident experts. Neither is known ahead of time and both move
with the content, so the scheduler measures them.

The rate of depth ``d`` is ``E[tokens per round] / E[seconds per round]``, each an
exponential moving average, estimated separately:

- **Tokens** come from every round at depth ``k >= d``, not just rounds at ``d``. The
  drafts at depth ``d`` are the first ``d`` of the same chain, so a round that accepted
  ``a`` drafts at depth ``k`` would have committed ``min(a, d) + 1`` at depth ``d``.
  Acceptance swings from prompt to prompt; estimating every depth from the same rounds
  keeps that swing out of the comparison.
- **Seconds** only come from rounds run at ``d``. The first ``settle`` rounds after a
  depth change are not measured: a new depth routes experts the cache evicted while the
  old one ran, so its first rounds pay reads the steady state does not. A probe measured
  cold always loses.

Depth 0 is a plain decode step: one token, no verify rows. On the disk tier a verify
round's extra experts can cost more than its drafts return, and then not speculating
is the best choice. A plain step still runs the draft head, so speculation can resume.

The selector runs the best depth. Every ``probe_every`` rounds it spends a short burst on
the depth whose timing is oldest, so it notices a depth that got better (the cache
warmed) and refreshes the token estimates of depths deeper than it has been running.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _ema(old: float, new: float, alpha: float, first: bool) -> float:
    return new if first else old + alpha * (new - old)


@dataclass
class _DepthStat:
    token_rounds: int = 0
    tokens: float = 0.0       # EMA tokens a round at this depth commits
    rounds: int = 0           # timed rounds (after settling)
    seconds: float = 0.0      # EMA wall seconds per round
    disk: float = 0.0         # EMA seconds blocked on disk reads per round
    last_round: int = -1      # selector round of the latest timing

    @property
    def rate(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 and self.token_rounds else 0.0


class DepthSelector:
    """Pick the draft depth in ``0..max_k`` with the best measured tokens per second.

    ``choose()`` names the depth of the NEXT round: the head's chain drafts that many
    tokens during the current verify."""

    def __init__(
        self,
        max_k: int,
        *,
        adapt: bool = True,
        alpha: float = 0.1,
        seed_rounds: int = 8,
        probe_every: int = 128,
        probe_rounds: int = 8,
        settle: int = 6,
        fixed_k: int | None = None,
    ) -> None:
        assert max_k >= 1
        self.max_k = max_k
        self.fixed_k = min(fixed_k or max_k, max_k)
        self.adapt = adapt and fixed_k is None
        self.alpha = alpha
        self.seed_rounds = seed_rounds
        self.probe_every = probe_every
        self.probe_rounds = probe_rounds
        self.settle = settle
        self.stats = {k: _DepthStat() for k in range(0, max_k + 1)}
        self.round = 0
        self._probe: int | None = None
        self._probe_left = 0
        self._last_k: int | None = None
        self._since_switch = 0

    def record(
        self, k: int, accepted: Sequence[int], seconds: float, disk: float = 0.0
    ) -> None:
        """One verify round at depth ``k`` over requests that accepted ``accepted`` drafts
        each, taking ``seconds`` of wall time, ``disk`` of it blocked on expert reads."""
        stat = self.stats.get(k)
        if stat is None or seconds <= 0 or not accepted:
            return
        for d in range(0, k + 1):
            s = self.stats[d]
            tokens = sum(min(a, d) + 1 for a in accepted) / len(accepted)
            s.tokens = _ema(s.tokens, tokens, self.alpha, s.token_rounds == 0)
            s.token_rounds += 1

        if k != self._last_k:
            self._last_k, self._since_switch = k, 0
        self._since_switch += 1
        if self._since_switch <= self.settle:
            return
        if self._probe == k and self._probe_left > 0:
            self._probe_left -= 1
        # A round that swallowed a scheduling gap (prefill of another request, a stalled
        # client) says nothing about the depth; drop it once the depth is seeded.
        if stat.rounds >= self.seed_rounds and seconds > 4 * stat.seconds:
            return
        self.round += 1
        first = stat.rounds == 0
        stat.seconds = _ema(stat.seconds, seconds, self.alpha, first)
        stat.disk = _ema(stat.disk, disk, self.alpha, first)
        stat.rounds += 1
        stat.last_round = self.round

    def best(self) -> int:
        return max(self.stats, key=lambda k: (self.stats[k].rate, k))

    def choose(self) -> int:
        """The depth to draft at for the next round."""
        if not self.adapt:
            return self.fixed_k
        # Time every depth first, deepest first: its rounds also seed the shallower
        # depths' token estimates.
        for k in sorted(self.stats, reverse=True):
            if self.stats[k].rounds < self.seed_rounds:
                return k
        if self._probe is not None and self._probe_left > 0:
            return self._probe
        self._probe = None
        best = self.best()
        if self.round and self.round % self.probe_every == 0:
            others = [k for k in self.stats if k != best]
            self._probe = min(others, key=lambda k: self.stats[k].last_round)
            self._probe_left = self.probe_rounds
            return self._probe
        return best

    def summary(self) -> str:
        parts = []
        for k, s in sorted(self.stats.items()):
            if s.rounds or s.token_rounds:
                parts.append(
                    f"k={k}: {s.rounds} rounds {s.tokens:.2f} tok {s.seconds * 1e3:.1f} ms "
                    f"(disk {s.disk * 1e3:.1f}) {s.rate:.1f} tok/s"
                )
        mode = f"best k={self.best()}" if self.adapt else f"fixed k={self.fixed_k}"
        return f"spec depth ({mode}): " + "; ".join(parts)


__all__ = ["DepthSelector"]
