from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
    make_moe_layer,
    silu_and_mul,
)
from freetoken.models.blocks import BaseLLMModel, embed_input_ids
from freetoken.utils import nvtx_annotate

from .attention import LagunaAttention

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def _laguna_route(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
    normalize: bool,
    scale: float,
    selection_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = logits.float().sigmoid()
    selection = scores + correction_bias.float()
    if selection_bias is not None:
        selection = selection + selection_bias
    selected = torch.topk(selection, top_k, dim=-1).indices
    weights = scores.gather(-1, selected)
    if normalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return (weights * scale).float().contiguous(), selected.to(torch.int32).contiguous()


class LagunaDenseMLP(BaseOP):
    def __init__(
        self, config: ModelConfig, intermediate_size: int, *, prefix: str = ""
    ):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            has_bias=False,
            quant_config=config.quant,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = LinearRowParallel(
            intermediate_size,
            config.hidden_size,
            has_bias=False,
            quant_config=config.quant,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(hidden)))


class LagunaMoE(BaseOP):
    """Sigmoid top-k routing with selection-only correction bias and shared expert."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.e_score_correction_bias = torch.empty(config.num_experts, dtype=torch.float32)
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        self.shared_expert = LagunaDenseMLP(
            config,
            config.shared_expert_intermediate_size,
            prefix=f"{prefix}.shared_expert",
        )

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from freetoken.moe.route_bias import selection_bias

        logits = F.linear(hidden.float(), self.gate.weight.float())
        return _laguna_route(
            logits,
            self.e_score_correction_bias,
            top_k=self.top_k,
            normalize=self.norm_topk_prob,
            scale=self.routed_scaling_factor,
            selection_bias=selection_bias(self.experts, hidden.shape[0], logits.device),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        n_tokens, hidden_size = hidden.shape
        hidden = hidden.view(-1, hidden_size)
        weights, experts = self._route(hidden)
        shared = self.shared_expert.forward(hidden)
        routed = self.experts.routed_forward(hidden, weights, experts)
        return (routed + shared).view(n_tokens, hidden_size)


class LagunaDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self._layer_id = layer_id
        self.self_attn = LagunaAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp = (
            LagunaDenseMLP(
                config, config.intermediate_size, prefix=f"{prefix}.mlp"
            )
            if layer_id < config.first_k_dense_replace
            else LagunaMoE(config, layer_id, prefix=f"{prefix}.mlp")
        )
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, residual = self.input_layernorm.forward(hidden, residual)
        hidden = self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class LagunaModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            quant_config=config.quant,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = OPList(
            [
                LagunaDecoderLayer(config, i, prefix=f"{prefix}.layers.{i}")
                for i in range(config.num_layers)
            ]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        hidden = embed_input_ids(self.embed_tokens, input_ids, batch)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            hidden, residual = layer.forward(hidden, residual)
        return self.norm.forward(hidden, residual)[0]


class LagunaForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = LagunaModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["LagunaForCausalLM", "LagunaModel", "LagunaAttention", "LagunaMoE"]
