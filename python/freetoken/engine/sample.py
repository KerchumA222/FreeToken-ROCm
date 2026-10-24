from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import torch
from freetoken.env import ENV
from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.engine.think_budget import ThinkBudget


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    greedy_mask: torch.Tensor | None = None
    penalties: list[tuple[int, torch.Tensor, float, float]] = field(default_factory=list)
    # the largest top_k over the sampled rows when every one of them sets it, else None (host int: FREETOKEN_TOPK_SAMPLER)
    max_top_k: int | None = None
    # FREETOKEN_THINKING_BUDGET: (row, bias) added to the </think> logit (token think_end_id)
    think_biases: list[tuple[int, float]] = field(default_factory=list)
    think_end_id: int = -1

    def first_rows(self, n: int) -> "BatchSamplingArgs":
        """The args of the first ``n`` rows of the batch."""
        cut = lambda t: None if t is None else t[:n]
        return BatchSamplingArgs(
            cut(self.temperatures), cut(self.top_k), cut(self.top_p), cut(self.greedy_mask),
            [p for p in self.penalties if p[0] < n], self.max_top_k,
            [b for b in self.think_biases if b[0] < n], self.think_end_id,
        )


# FREETOKEN_TOPK_SAMPLER takes the candidates with torch.topk up to this k; larger falls back to the fused kernels
_TOPK_SAMPLER_MAX_K = 256


def sample_top_k_first(
    logits: torch.Tensor, temperatures: torch.Tensor, top_k: torch.Tensor | None, top_p: torch.Tensor | None, k: int,
) -> torch.Tensor:
    """Sample from each row's top ``top_k`` (then top ``top_p``) tokens at its temperature, reading the full vocabulary
    once with ``torch.topk``. The same distribution as the fused top-k-first kernels, without their cooperative
    launch: on ROCm a cooperative kernel waits for every other stream's work, ~2 ms a decode step in a server."""
    vals, idx = torch.topk(logits, k, dim=-1)  # sorted, largest first
    scaled = vals.float() / temperatures[:, None]
    if top_k is not None:
        rank = torch.arange(k, device=logits.device)[None, :]
        scaled = scaled.masked_fill(rank >= top_k[:, None], float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    if top_p is not None:
        # keep the smallest prefix whose mass reaches top_p: drop a token once the mass before it already does
        before = probs.cumsum(dim=-1) - probs
        probs = probs.masked_fill(before >= top_p[:, None], 0.0)
    # exponential race: argmax(p / Exp(1)) draws from p (unnormalized is fine), with no host sync
    choice = torch.argmax(probs / torch.empty_like(probs).exponential_(), dim=-1, keepdim=True)
    return idx.gather(1, choice).squeeze(1)


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


def keep_likely_drafts(
    tokens: torch.Tensor, logits: torch.Tensor, input_ids: torch.Tensor, args: BatchSamplingArgs,
    num_reqs: int, k: int, min_p: float,
) -> torch.Tensor:
    """FREETOKEN_SPEC_ACCEPT_MIN_P: in a step-major verify, row ``s`` scores draft ``s``, which is row ``s + 1``'s
    input; where the target gives that draft at least ``min_p`` (at the row's temperature) the row takes the draft."""
    n = k * num_reqs
    scaled = logits[:n].float()
    if args.temperatures is not None:
        scaled = scaled / args.temperatures[:n, None]
    drafts = input_ids[num_reqs : n + num_reqs].long()
    p_draft = torch.softmax(scaled, dim=-1).gather(1, drafts[:, None]).squeeze(1)
    out = tokens.clone()
    out[:n] = torch.where(p_draft >= min_p, drafts.to(tokens.dtype), tokens[:n])
    return out


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int
    # FREETOKEN_THINKING_BUDGET (set by the scheduler, which has the tokenizer)
    think_budget: "ThinkBudget | None" = None

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        args = self._prepare(batch)
        if self.think_budget is not None:
            args.think_biases = self.think_budget.row_biases(batch.reqs)
            args.think_end_id = self.think_budget.end_id
        return args

    def _prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        is_greedy = [p.is_greedy for p in params]
        penalties = []
        for row, req in enumerate(batch.reqs):
            p = req.sampling_params
            if not (p.presence_penalty or p.frequency_penalty) or not req.can_decode:
                continue
            if req.output_token_counts is None:
                req.output_token_counts = torch.zeros(
                    self.vocab_size, dtype=torch.int32, device=self.device
                )
            penalties.append(
                (row, req.output_token_counts, p.presence_penalty, p.frequency_penalty)
            )
        if all(is_greedy):
            return BatchSamplingArgs(temperatures=None, penalties=penalties)

        MIN_P = MIN_T = 1e-6
        # Greedy outputs are selected explicitly in sample(); use neutral sampling
        # parameters for those rows instead of approximating argmax at low temperature.
        ts = [1.0 if g else max(p.temperature, MIN_T) for p, g in zip(params, is_greedy)]
        top_ks = [
            p.top_k if not g and p.top_k >= 1 else self.vocab_size
            for p, g in zip(params, is_greedy)
        ]
        top_ps = [
            1.0 if g else min(max(p.top_p, MIN_P), 1.0)
            for p, g in zip(params, is_greedy)
        ]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        greedy_mask = (
            make_device_tensor(is_greedy, torch.bool, self.device) if any(is_greedy) else None
        )
        sampled_ks = [k for k, g in zip(top_ks, is_greedy) if not g]
        max_top_k = max(sampled_ks) if all(k < self.vocab_size for k in sampled_ks) else None
        return BatchSamplingArgs(
            temperatures, top_k=top_k, top_p=top_p, greedy_mask=greedy_mask, penalties=penalties,
            max_top_k=max_top_k,
        )

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.think_biases:
                from freetoken.engine.think_budget import apply_biases

                logits = apply_biases(logits, args.think_biases, args.think_end_id)
            if args.penalties:
                logits = logits.float().clone()
                for row, counts, presence, frequency in args.penalties:
                    logits[row] -= frequency * counts + presence * (counts > 0)
            if args.temperatures is None:  # greedy sampling
                tokens = torch.argmax(logits, dim=-1)
            else:
                if ENV.TOPK_SAMPLER and args.max_top_k is not None and args.max_top_k <= _TOPK_SAMPLER_MAX_K:
                    tokens = sample_top_k_first(logits.float(), args.temperatures, args.top_k, args.top_p, args.max_top_k)
                else:
                    tokens = sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
                if args.greedy_mask is not None:
                    # Mixed batches still run probability sampling for all rows, but
                    # greedy rows must follow argmax's deterministic tie-breaking.
                    greedy_tokens = torch.argmax(logits, dim=-1).to(tokens.dtype)
                    tokens = torch.where(args.greedy_mask, greedy_tokens, tokens)
            # Update on the sampling stream: overlapped scheduling can prepare the next
            # batch before the previous token reaches Req.input_ids on the CPU.
            for row, counts, _, _ in args.penalties:
                counts.scatter_add_(
                    0, tokens[row : row + 1].long(), counts.new_ones(1)
                )
            return tokens
