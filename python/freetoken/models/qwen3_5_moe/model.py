from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                quant_config=config.quant,
                prefix=f"{prefix}.linear_attn",
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = (
            Qwen3_5MoE(config, layer_id, prefix=f"{prefix}.mlp")
            if config.moe_enabled
            else Qwen3_5DenseMLP(config, prefix=f"{prefix}.mlp")
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual



def _gguf_embed_type(config) -> int | None:
    """The token embedding's ggml type when the checkpoint packs it, else None.

    Every linear -- the LM head included -- resolves its layout through
    ``quant_method_for``, but ``VocabParallelEmbedding`` has no quant seam because it
    is a gather rather than a matmul. So the packed table is selected here, from the
    same QuantConfig the linears read, rather than through a parallel mechanism.
    """
    from freetoken.layers.quantization.scheme import QuantKind

    quant = getattr(config, "quant", None)
    if quant is None or config.tie_word_embeddings:
        return None
    scheme = quant.scheme_for("model.embed_tokens")
    if scheme is None or scheme.kind is not QuantKind.GGUF:
        return None
    from freetoken.models.gguf.dequant import GGML_NAME

    return {name: t for t, name in GGML_NAME.items()}[scheme.weight.elem]


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        # A tied lm_head reads embed_tokens.weight through ParallelLMHead, and a
        # packed embedding has no .weight -- so tied checkpoints keep the bf16 table.
        embed_type = _gguf_embed_type(config)
        if embed_type is not None:
            from freetoken.layers.gguf import GGUFEmbedding

            self.embed_tokens = GGUFEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                quant_type=embed_type,
            )
        else:
            self.embed_tokens = VocabParallelEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
            )
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        x, _ = self.norm.forward_add_residual(x, residual)
        return x


class Qwen3_5MoEForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
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


__all__ = ["Qwen3_5MoEForCausalLM"]
