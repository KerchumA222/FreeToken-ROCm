from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.layers.q8_act import attach, wants_q8
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


def _fused_router(moe: "Qwen3_5MoE", x: torch.Tensor):
    """Router and shared-expert gate logits in one small-M GEMV, or None to use the linears.

    At decode batch sizes rocBLAS runs these [256, H] / [1, H] GEMVs as large Tensile tiles
    on RDNA (~160 us a layer on an RX 6800). One Triton GEMV over the concatenated
    [257, H] weight reads it once for every row in ~5 us."""
    from freetoken.kernel.triton.small_gemv import MAX_ROWS, small_gemv

    if torch.version.hip is None or x.shape[0] > MAX_ROWS:
        return None
    w = moe._router_weight
    if w is None:
        w = moe._router_weight = torch.cat(
            [moe.gate.weight, moe.shared_expert_gate.weight], dim=0).contiguous()
    out = small_gemv(x.contiguous(), w, block_n=1, block_k=2048, num_warps=2)
    e = moe.gate.weight.shape[0]
    return out[:, :e].contiguous(), out[:, e:]


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
        h = self.gate_up_proj.forward(x)
        if h.dim() == 2 and wants_q8(self.down_proj, h.shape[0]):
            # The activation writes the down projection's q8_1 input itself.
            from freetoken.kernel.triton.activation import silu_and_mul as triton_silu_and_mul

            return self.down_proj.forward(attach(*triton_silu_and_mul(h, q8=True)))
        return self.down_proj.forward(silu_and_mul(h))


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
        self._router_weight = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Compute the router + shared expert BEFORE the routed experts: the fused MoE
        # kernel may write into ``hidden_states`` in place, which would corrupt the
        # shared expert's input (HF also evaluates the shared expert first).
        fused = _fused_router(self, hidden_states)
        if fused is None:
            router_logits = self.gate.forward(hidden_states)
            shared_gate = self.shared_expert_gate.forward(hidden_states)
        else:
            router_logits, shared_gate = fused
        shared = self.shared_expert.forward(hidden_states)
        shared = shared * torch.sigmoid(shared_gate)
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return (routed + shared).view(num_tokens, hidden_dim)


__all__ = ["Qwen3_5MoE", "Qwen3_5DenseMLP"]
