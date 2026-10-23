from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.env import ENV
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    make_moe_layer,
    silu_and_mul,
)

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


_SMALL_ROUTER_ROWS = 4


def _router_linear(gate: LinearReplicated, x: torch.Tensor) -> torch.Tensor:
    """Optional broadcast-reduction path for small Qwen3.5 router GEMVs on HIP."""
    if (
        ENV.QWEN35_ROUTER_GEMV
        and torch.version.hip is not None
        and x.shape[0] <= _SMALL_ROUTER_ROWS
    ):
        return (x.unsqueeze(1) * gate.weight).sum(-1)
    return gate.forward(x)


def _fused_router(
    moe: Qwen3_5MoE, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Router logits and ``sigmoid(shared_expert_gate(x))`` (fp32 ``[rows]``) in one small-M
    GEMV launch, or None to use the separate linears. Graph-capturable: no host sync."""
    if not ENV.FUSED_ROUTER or torch.version.hip is None or not x.is_cuda or x.dim() != 2:
        return None
    from freetoken.kernel.triton.small_gemv import MAX_ROWS, flash_next_gemv_config, small_gemv

    w, gate_w = moe.gate.weight, moe.shared_expert_gate.weight.view(-1)
    rows = x.shape[0]
    if (rows == 0 or rows > MAX_ROWS or w.dtype != x.dtype or gate_w.dtype != x.dtype
            or w.dim() != 2 or w.stride(1) != 1 or gate_w.stride(0) != 1):
        return None
    block_n, block_k, num_warps = flash_next_gemv_config(rows, w.shape[0], w.shape[1], x.device.index)
    gate = torch.empty(rows, dtype=torch.float32, device=x.device)
    logits = small_gemv(x.contiguous(), w, block_n=block_n, block_k=block_k, num_warps=num_warps,
                        side_weight=gate_w, side_output=gate)
    return logits, gate


class _SharedExpert(BaseOP):
    """Always-present shared SwiGLU expert of width ``shared_expert_intermediate_size``."""

    def __init__(
        self, config: ModelConfig, hidden_size: int, intermediate_size: int, *, prefix: str = ""
    ):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = LinearRowParallel(
            intermediate_size, hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen3_5DenseMLP(_SharedExpert):
    """Dense (non-MoE) SwiGLU MLP for dense Qwen3.x checkpoints (e.g. 27B): ``gate_up_proj``
    (fused gate|up) + ``down_proj`` at full ``intermediate_size``. Same structure as the shared
    expert, so it reuses ``_SharedExpert`` directly and keeps the state-dict keys flat
    (``...layers.N.mlp.{gate_up_proj,down_proj}``)."""

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        super().__init__(config, config.hidden_size, config.intermediate_size, prefix=prefix)


class Qwen3_5MoE(BaseOP):
    """Routed MoE (256 experts, top-8) plus a gated shared expert:

        out = routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)

    Router softmaxes over all experts, takes top-k, and renormalizes (HF semantics).
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None, *, prefix: str = ""):
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        # routers stay bf16 whatever the checkpoint quantizes
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size,
            prefix=f"{prefix}.shared_expert",
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Compute the router + shared expert BEFORE the routed experts: the fused MoE
        # kernel may write into ``hidden_states`` in place, which would corrupt the
        # shared expert's input (HF also evaluates the shared expert first).
        fused = _fused_router(self, hidden_states)
        if fused is not None:
            router_logits, gate = fused
            shared = self.shared_expert.forward(hidden_states)
            shared = shared * gate.unsqueeze(1).to(shared.dtype)
        else:
            router_logits = _router_linear(self.gate, hidden_states)
            shared = self.shared_expert.forward(hidden_states)
            shared = shared * torch.sigmoid(_router_linear(self.shared_expert_gate, hidden_states))
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return (routed + shared).view(num_tokens, hidden_dim)


__all__ = ["Qwen3_5MoE", "Qwen3_5DenseMLP"]
