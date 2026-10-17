"""Laguna-S-2.1 GGUF adapter for the verified llama.cpp conversion layout.

The converter stores per-layer query-head counts but not layer types. This adapter therefore
accepts only the verified Laguna-S-2.1 geometry, whose full/SWA placement is fixed by its
four-layer head-count pattern. Routed expert matrices remain packed in GGUF expert banks.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index
from freetoken.models.gguf.tensors import float_tensor, fused, matrix

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "laguna"
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_HEAD_COUNTS = (48, 72, 72, 72) * 12

_LAYER_LEAVES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "self_attn.q_norm": "attn_q_norm",
    "self_attn.k_norm": "attn_k_norm",
    "self_attn.g_proj": "attn_gate",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
    "mlp.gate": "ffn_gate_inp",
    "mlp.e_score_correction_bias": "exp_probs_b",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "mlp.shared_expert.gate_proj": "ffn_gate_shexp",
    "mlp.shared_expert.up_proj": "ffn_up_shexp",
    "mlp.shared_expert.down_proj": "ffn_down_shexp",
    "mlp.experts.0.gate_proj": "ffn_gate_exps",
    "mlp.experts.0.up_proj": "ffn_up_exps",
    "mlp.experts.0.down_proj": "ffn_down_exps",
}


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Checkpoint module name -> canonical Laguna GGUF tensor name(s), without `.weight`."""
    if name == "model.embed_tokens":
        return ("token_embd",)
    if name == "model.norm":
        return ("output_norm",)
    if name == "lm_head":
        return ("output",) if "output.weight" in tensors else ()
    match = _LAYER_RE.match(name)
    if match is None:
        return ()
    layer, leaf = match.groups()
    tensor = _LAYER_LEAVES.get(leaf)
    if tensor is None:
        return ()
    return (f"blk.{layer}.{tensor}",)


def _required(md: dict, suffix: str):
    key = f"{_ARCH}.{suffix}"
    try:
        return md[key]
    except KeyError as exc:
        raise ValueError(f"Laguna GGUF is missing required metadata {key!r}") from exc


def _head_counts(value) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Laguna GGUF attention.head_count must contain per-layer query-head counts")
    return tuple(int(heads) for heads in value)


def _one_kv_head_count(value, layers: int) -> int:
    if isinstance(value, (list, tuple)):
        values = tuple(int(heads) for heads in value)
        if len(values) != layers or len(set(values)) != 1:
            raise NotImplementedError("Laguna GGUF requires one constant KV-head count across layers")
        return values[0]
    return int(value)


def _validate_tensor_geometry(shim: "GgufConfigShim", config: ModelConfig) -> None:
    """Check GGUF dimensions that would otherwise fail later in projection or bank loading."""
    index = gguf_tensor_index(shim.model_path)
    if not index:  # metadata-only FTW source files intentionally have no tensor table
        return

    def check(name: str, expected: tuple[int, ...]) -> None:
        info = index.get(name if name.endswith((".weight", ".bias")) else f"{name}.weight")
        if info is None:
            return
        if tuple(info.ne) != expected:
            raise ValueError(f"{name}: GGUF dimensions {tuple(info.ne)} do not match expected {expected}")

    check("token_embd", (config.hidden_size, config.vocab_size))
    check("output_norm", (config.hidden_size,))
    if "output.weight" in index:
        check("output", (config.hidden_size, config.vocab_size))
    for layer, q_heads in enumerate(config.num_qo_heads_per_layer):
        blk = f"blk.{layer}"
        check(f"{blk}.attn_q", (config.hidden_size, q_heads * config.head_dim))
        check(f"{blk}.attn_k", (config.hidden_size, config.num_kv_heads * config.head_dim))
        check(f"{blk}.attn_v", (config.hidden_size, config.num_kv_heads * config.head_dim))
        check(f"{blk}.attn_output", (q_heads * config.head_dim, config.hidden_size))
        check(f"{blk}.attn_gate", (config.hidden_size, q_heads))
        check(f"{blk}.attn_q_norm", (config.head_dim,))
        check(f"{blk}.attn_k_norm", (config.head_dim,))
        check(f"{blk}.attn_norm", (config.hidden_size,))
        check(f"{blk}.ffn_norm", (config.hidden_size,))
        if layer < config.first_k_dense_replace:
            for proj, shape in (
                ("gate", (config.hidden_size, config.intermediate_size)),
                ("up", (config.hidden_size, config.intermediate_size)),
                ("down", (config.intermediate_size, config.hidden_size)),
            ):
                check(f"{blk}.ffn_{proj}", shape)
        else:
            check(f"{blk}.ffn_gate_inp", (config.hidden_size, config.num_experts))
            check(f"{blk}.exp_probs_b.bias", (config.num_experts,))
            for proj, shape in (
                ("gate_shexp", (config.hidden_size, config.shared_expert_intermediate_size)),
                ("up_shexp", (config.hidden_size, config.shared_expert_intermediate_size)),
                ("down_shexp", (config.shared_expert_intermediate_size, config.hidden_size)),
                ("gate_exps", (config.hidden_size, config.moe_intermediate_size, config.num_experts)),
                ("up_exps", (config.hidden_size, config.moe_intermediate_size, config.num_experts)),
                ("down_exps", (config.moe_intermediate_size, config.hidden_size, config.num_experts)),
            ):
                check(f"{blk}.ffn_{proj}", shape)


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    md = shim.metadata
    num_layers = int(_required(md, "block_count"))
    q_heads = _head_counts(_required(md, "attention.head_count"))
    if num_layers != 48 or q_heads != _HEAD_COUNTS:
        raise NotImplementedError(
            "Laguna GGUF currently supports only the verified Laguna-S-2.1 48-layer query-head profile"
        )
    hidden = int(_required(md, "embedding_length"))
    head_dim = int(_required(md, "attention.key_length"))
    kv_heads = _one_kv_head_count(_required(md, "attention.head_count_kv"), num_layers)
    context = int(_required(md, "context_length"))
    sliding_window = int(_required(md, "attention.sliding_window"))
    full_rotary_dim = int(_required(md, "rope.dimension_count"))
    swa_rotary_dim = int(_required(md, "rope.dimension_count_swa"))
    full_base = float(_required(md, "rope.freq_base"))
    swa_base = float(_required(md, "rope.freq_base_swa"))

    leading_dense = int(_required(md, "leading_dense_block_count"))
    experts = int(_required(md, "expert_count"))
    top_k = int(_required(md, "expert_used_count"))
    moe_intermediate = int(_required(md, "expert_feed_forward_length"))
    shared_intermediate = int(_required(md, "expert_shared_feed_forward_length"))
    norm_topk = bool(_required(md, "expert_weights_norm"))
    routed_scale = float(_required(md, "expert_weights_scale"))
    gating_func = str(_required(md, "expert_gating_func")).lower()
    if (hidden, head_dim, kv_heads, sliding_window) != (3072, 128, 8, 512):
        raise NotImplementedError("Laguna GGUF geometry does not match the verified Laguna-S-2.1 attention profile")
    if (full_rotary_dim, swa_rotary_dim, full_base, swa_base) != (64, 128, 500000.0, 10000.0):
        raise NotImplementedError("Laguna GGUF RoPE geometry does not match the verified Laguna-S-2.1 profile")
    if (leading_dense, experts, top_k, moe_intermediate, shared_intermediate) != (1, 256, 10, 1024, 1024):
        raise NotImplementedError("Laguna GGUF routing metadata does not match the verified Laguna-S-2.1 MoE profile")
    if not norm_topk or routed_scale != 2.5 or gating_func != "sigmoid":
        raise NotImplementedError("Laguna GGUF requires normalized sigmoid routing with routed scale 2.5")
    if not (0 < top_k <= experts) or full_rotary_dim % 2 or swa_rotary_dim != head_dim:
        raise ValueError("Laguna GGUF contains invalid expert or rotary dimensions")

    yarn_type = str(_required(md, "rope.scaling.type")).lower()
    if yarn_type != "yarn":
        raise NotImplementedError(f"Laguna GGUF full-attention RoPE type {yarn_type!r} is not supported")
    yarn_attn_factor = float(_required(md, "rope.scaling.yarn_attn_factor"))
    yarn = {
        "rope_type": "yarn",
        "factor": float(_required(md, "rope.scaling.factor")),
        "original_max_position_embeddings": int(_required(md, "rope.scaling.original_context_length")),
        "beta_fast": float(_required(md, "rope.scaling.yarn_beta_fast")),
        "beta_slow": float(_required(md, "rope.scaling.yarn_beta_slow")),
        # llama.cpp writes 1.0 here as a sentinel for its internally derived YaRN
        # mscale; FreeToken's rotary implementation derives the same scale when absent.
        "attention_factor": None if yarn_attn_factor == 1.0 else yarn_attn_factor,
        "truncate": bool(md.get(f"{_ARCH}.rope.scaling.yarn_truncate", True)),
    }
    if yarn["factor"] != 128.0 or yarn["original_max_position_embeddings"] != 8192:
        raise NotImplementedError("Laguna GGUF YaRN scale does not match the verified Laguna-S-2.1 profile")
    attention_factor = yarn["attention_factor"]
    if (yarn["beta_fast"], yarn["beta_slow"]) != (32.0, 1.0) or (
        attention_factor is not None
        and (not math.isfinite(attention_factor) or attention_factor <= 0)
    ):
        raise ValueError("Laguna GGUF contains invalid YaRN scaling metadata")

    full_rope = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=full_rotary_dim,
        max_position=context,
        base=full_base,
        scaling=yarn,
    )
    swa_rope = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=swa_rotary_dim,
        max_position=context,
        base=swa_base,
        scaling=None,
    )
    config = ModelConfig(
        num_layers=num_layers,
        num_qo_heads=max(q_heads),
        num_qo_heads_per_layer=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        intermediate_size=int(_required(md, "feed_forward_length")),
        hidden_act="silu",
        rms_norm_eps=float(_required(md, "attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=swa_rope,
        num_experts=experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=moe_intermediate,
        norm_topk_prob=norm_topk,
        model_type="laguna",
        architectures=list(shim.architectures),
        moe_enabled=True,
        expert_quant="gguf",
        use_qk_norm=True,
        shared_expert_intermediate_size=shared_intermediate,
        first_k_dense_replace=leading_dense,
        n_shared_experts=1,
        routed_scaling_factor=routed_scale,
        has_router_bias=True,
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=tuple(range(0, num_layers, 4)),
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                rotary_config=full_rope,
            ),
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=tuple(i for i in range(num_layers) if i % 4),
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                rotary_config=swa_rope,
                sliding_window=sliding_window,
            ),
        ),
    )
    _validate_tensor_geometry(shim, config)
    return config


def _require_tp1(what: str) -> None:
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(f"Laguna GGUF {what} currently supports TP=1 only")


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Load dense/shared/router state; routed expert matrices are loaded as GGUF banks."""
    if not include_non_moe:
        return
    from freetoken.utils import cached_load_hf_config

    _require_tp1("weight loading")
    config = parse_gguf_config(cached_load_hf_config(model_path))
    index = gguf_tensor_index(model_path)

    def t(name: str) -> GgufTensorInfo:
        return index[name if name.endswith((".weight", ".bias")) else f"{name}.weight"]

    embed = t("token_embd")
    yield from fused("model.embed_tokens", [embed], [matrix(embed)], config.hidden_size)
    if "output.weight" in index:
        output = t("output")
        yield from fused("lm_head", [output], [matrix(output)], config.hidden_size)
    yield "model.norm.weight", float_tensor(t("output_norm"))

    for layer in range(config.num_layers):
        blk, base = f"blk.{layer}", f"model.layers.{layer}"
        yield f"{base}.input_layernorm.weight", float_tensor(t(f"{blk}.attn_norm"))
        yield f"{base}.post_attention_layernorm.weight", float_tensor(t(f"{blk}.ffn_norm"))
        projections = [t(f"{blk}.attn_{part}") for part in ("q", "k", "v")]
        yield from fused(
            f"{base}.self_attn.qkv_proj",
            projections,
            [matrix(info) for info in projections],
            config.hidden_size,
        )
        gate = t(f"{blk}.attn_gate")
        yield from fused(f"{base}.self_attn.g_proj", [gate], [matrix(gate)], config.hidden_size)
        output = t(f"{blk}.attn_output")
        yield from fused(
            f"{base}.self_attn.o_proj", [output], [matrix(output)], int(output.ne[0])
        )
        yield f"{base}.self_attn.q_norm.weight", float_tensor(t(f"{blk}.attn_q_norm"))
        yield f"{base}.self_attn.k_norm.weight", float_tensor(t(f"{blk}.attn_k_norm"))

        mlp = f"{base}.mlp"
        if layer < config.first_k_dense_replace:
            gate, up, down = (t(f"{blk}.ffn_{p}") for p in ("gate", "up", "down"))
            yield from fused(f"{mlp}.gate_up_proj", [gate, up], [matrix(gate), matrix(up)], config.hidden_size)
            yield from fused(f"{mlp}.down_proj", [down], [matrix(down)], int(down.ne[0]))
            continue

        yield f"{mlp}.gate.weight", float_tensor(t(f"{blk}.ffn_gate_inp"))
        yield f"{mlp}.e_score_correction_bias", float_tensor(t(f"{blk}.exp_probs_b.bias"))
        gate, up, down = (t(f"{blk}.ffn_{p}_shexp") for p in ("gate", "up", "down"))
        yield from fused(
            f"{mlp}.shared_expert.gate_up_proj",
            [gate, up],
            [matrix(gate), matrix(up)],
            config.hidden_size,
        )
        yield from fused(
            f"{mlp}.shared_expert.down_proj",
            [down],
            [matrix(down)],
            config.shared_expert_intermediate_size,
        )


__all__ = ["iter_gguf_weights", "parse_gguf_config", "to_gguf"]
