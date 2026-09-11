"""Hyper-connections: Qwen4-Exp's multi-stream residual.

Where a transformer carries one residual stream of ``hidden_size``, this
architecture carries ``hc_count`` of them side by side -- the residual tensor is
``hc_count * hidden_size`` wide the whole way down the stack. Each block mixes the
streams down to a single ``hidden_size`` input, runs attention or the MoE on that,
and then injects the block's output back across all the streams with learned
per-stream weights.

Both the mixing weights and the injection weights are computed from the *normalized*
streams, so the norm output is reused rather than recomputed -- that sharing is why
this is one module and not a norm plus two projections.

Ported from ``transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextGatedResidual``;
``tests/models/test_qwen4_exp.py`` holds it to that reference numerically.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, LinearReplicated


class GroupedRMSNorm(BaseOP):
    """RMSNorm over independent groups of ``group_size`` within the last dim.

    The hyper-connection stream is one flat ``hc_count * hidden_size`` tensor, but
    each stream must be normalized on its own statistics -- normalizing the whole
    thing would couple them. Scale is Gemma-style ``(1 + weight)``, matching the
    reference; ``group_size=None`` is ordinary RMSNorm.
    """

    def __init__(self, size: int, eps: float, group_size: int | None = None) -> None:
        if group_size is not None and size % group_size:
            raise ValueError(f"size {size} is not divisible by group_size {group_size}")
        self.size = size
        self.eps = eps
        self.group_size = group_size
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        if self.group_size is not None:
            out = out.reshape(*out.shape[:-1], -1, self.group_size)
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.group_size is not None:
            out = out.flatten(-2)
        return (out * (1.0 + self.weight.float())).type_as(x)


class GatedResidual(BaseOP):
    """One block's hyper-connection: mix the streams in, inject the result back.

    ``forward`` returns ``(mixed, hyper_input, injection_weights)``. The caller runs
    its block on ``mixed`` and then recombines with
    ``hyper_input + (block_out.unsqueeze(-2) * injection_weights.unsqueeze(-1)).flatten(-2)``.
    Returning the untouched ``hyper_input`` rather than adding internally keeps the
    block's output out of this module, which is what lets attention and the MoE share it.
    """

    def __init__(self, hidden_size: int, hc_count: int, hc_lowrank: int, eps: float,
                 use_combine: bool = True) -> None:
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        hc_hidden = hc_count * hidden_size
        self.hc_norm = GroupedRMSNorm(hc_hidden, eps, group_size=hidden_size)
        self.input_mix_weight_down = LinearReplicated(hc_hidden, hc_lowrank, has_bias=False)
        self.input_mix_weight_up = LinearReplicated(hc_lowrank, hc_hidden, has_bias=False)
        self.block_inject_weight = (
            LinearReplicated(hc_hidden, hc_count, has_bias=False) if use_combine else None
        )

    def forward(self, hyper_input: torch.Tensor):
        hc, h = self.hc_count, self.hidden_size
        if hyper_input.shape[-1] != hc * h:
            raise ValueError(
                f"expected {hc * h} hyper-connection features, got {hyper_input.shape[-1]}"
            )
        normed = self.hc_norm.forward(hyper_input)
        # The /hc_count before each nonlinearity is the reference's scaling, not a
        # mean: it keeps the pre-activations stream-count independent.
        mix = F.silu(self.input_mix_weight_down.forward(normed) / hc)
        mix = torch.sigmoid(self.input_mix_weight_up.forward(mix))
        mixed = (mix.unflatten(-1, (hc, h)) * normed.unflatten(-1, (hc, h))).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed
        inject = 2 * torch.sigmoid(self.block_inject_weight.forward(normed) / hc)
        return mixed, hyper_input, inject


def recombine(hyper_input: torch.Tensor, block_out: torch.Tensor,
              injection_weights: torch.Tensor) -> torch.Tensor:
    """Scatter a block's output back across the streams and add to the residual."""
    return hyper_input + (block_out.unsqueeze(-2) * injection_weights.unsqueeze(-1)).flatten(-2)


__all__ = ["GroupedRMSNorm", "GatedResidual", "recombine"]
