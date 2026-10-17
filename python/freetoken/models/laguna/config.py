from __future__ import annotations

import math
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)

_FULL_TYPE = "full_attention"
_SWA_TYPE = "sliding_attention"


def _read(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _rope_config(
    params: dict[str, Any], *, head_dim: int, max_position: int
) -> RotaryConfig:
    partial = float(params.get("partial_rotary_factor", 1.0))
    rotary_dim = int(head_dim * partial)
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(f"Laguna partial rotary dimension must be positive, even, and <= {head_dim}; got {rotary_dim}")
    base = float(params.get("rope_theta", 10000.0))
    if not math.isfinite(base) or base <= 0:
        raise ValueError(f"Laguna RoPE base must be positive and finite; got {base}")

    scaling = {key: value for key, value in params.items() if key not in ("rope_theta", "partial_rotary_factor")}
    if scaling:
        if "rope_type" not in scaling:
            raise ValueError("Laguna RoPE scaling metadata must specify rope_type")
        rope_type = str(scaling["rope_type"]).lower()
        if rope_type not in {"default", "proportional", "llama3", "yarn"}:
            raise ValueError(f"Unsupported Laguna RoPE type {rope_type!r}")
        scaling["rope_type"] = rope_type
        if rope_type in {"llama3", "yarn"}:
            factor = float(scaling.get("factor", 0.0))
            original_max = int(scaling.get("original_max_position_embeddings", 0))
            if not math.isfinite(factor) or factor <= 0 or original_max <= 0:
                raise ValueError(f"Laguna {rope_type} scaling metadata must have a positive finite factor and original position limit")
            scaling["factor"] = factor
            scaling["original_max_position_embeddings"] = original_max
        if rope_type == "llama3":
            for key in ("low_freq_factor", "high_freq_factor"):
                value = float(scaling.get(key, 0.0))
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Laguna llama3 scaling metadata requires a positive finite {key}")
                scaling[key] = value
        if rope_type == "yarn":
            if base <= 1:
                raise ValueError(f"Laguna YaRN RoPE base must be greater than 1; got {base}")
            for key, default in (("beta_fast", 32.0), ("beta_slow", 1.0)):
                value = float(scaling.get(key, default))
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Laguna YaRN scaling metadata requires a positive finite {key}")
                scaling[key] = value
            attention_factor = scaling.get("attention_factor")
            if attention_factor is not None:
                attention_factor = float(attention_factor)
                if not math.isfinite(attention_factor) or attention_factor <= 0:
                    raise ValueError("Laguna YaRN scaling metadata requires a positive finite attention_factor")
                scaling["attention_factor"] = attention_factor

    return RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=base,
        scaling=scaling or None,
    )


def parse_config(hf_config: Any) -> ModelConfig:
    num_layers = int(_read(hf_config, "num_hidden_layers"))
    hidden_size = int(_read(hf_config, "hidden_size"))
    head_dim = int(_read(hf_config, "head_dim", hidden_size // int(_read(hf_config, "num_attention_heads"))))
    num_kv_heads = int(_read(hf_config, "num_key_value_heads"))
    max_position = int(_read(hf_config, "max_position_embeddings"))
    layer_types = tuple(_read(hf_config, "layer_types", ()))
    if len(layer_types) != num_layers or set(layer_types) - {_FULL_TYPE, _SWA_TYPE}:
        raise ValueError("Laguna layer_types must name every layer as full_attention or sliding_attention")

    q_heads = tuple(int(value) for value in _read(hf_config, "num_attention_heads_per_layer", ()))
    if len(q_heads) != num_layers or any(value <= 0 for value in q_heads):
        raise ValueError("Laguna num_attention_heads_per_layer must provide one positive count per layer")
    if any(value % num_kv_heads for value in q_heads):
        raise ValueError("Laguna query-head counts must be divisible by num_key_value_heads")

    gating_types = tuple(_read(hf_config, "gating_types", ()))
    if gating_types and (len(gating_types) != num_layers or set(gating_types) != {"per_head"}):
        raise NotImplementedError("Laguna currently supports per-head attention gating on every layer")
    if _read(hf_config, "gating", "per-head") not in ("per-head", "per_head"):
        raise NotImplementedError("Laguna currently supports per-head attention gating")

    mlp_only_layers = tuple(int(i) for i in _read(hf_config, "mlp_only_layers", (0,)))
    mlp_types = tuple(_read(hf_config, "mlp_layer_types", ()))
    if mlp_only_layers != (0,) or (mlp_types and (len(mlp_types) != num_layers or mlp_types[0] != "dense" or any(t != "sparse" for t in mlp_types[1:]))):
        raise NotImplementedError("Laguna currently supports a dense layer 0 followed by sparse MoE layers")
    if bool(_read(hf_config, "moe_apply_router_weight_on_input", False)):
        raise NotImplementedError("Laguna input-side routed expert weighting is not implemented")

    rope_params = _read(hf_config, "rope_parameters", {}) or {}
    full_rope = _rope_config(rope_params.get(_FULL_TYPE, {}), head_dim=head_dim, max_position=max_position)
    swa_rope = _rope_config(rope_params.get(_SWA_TYPE, {}), head_dim=head_dim, max_position=max_position)
    full_ids = tuple(i for i, kind in enumerate(layer_types) if kind == _FULL_TYPE)
    swa_ids = tuple(i for i, kind in enumerate(layer_types) if kind == _SWA_TYPE)
    num_experts = int(_read(hf_config, "num_experts"))
    experts_per_token = int(_read(hf_config, "num_experts_per_tok"))
    if num_experts <= 0 or not 0 < experts_per_token <= num_experts:
        raise ValueError("Laguna requires a positive expert count and valid top-k")

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=max(q_heads),
        num_qo_heads_per_layer=q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=int(_read(hf_config, "vocab_size")),
        intermediate_size=int(_read(hf_config, "intermediate_size")),
        rms_norm_eps=float(_read(hf_config, "rms_norm_eps")),
        rotary_config=swa_rope,
        hidden_act="silu",
        tie_word_embeddings=bool(_read(hf_config, "tie_word_embeddings", False)),
        num_experts=num_experts,
        num_experts_per_tok=experts_per_token,
        moe_intermediate_size=int(_read(hf_config, "moe_intermediate_size")),
        norm_topk_prob=bool(_read(hf_config, "norm_topk_prob", True)),
        model_type=str(_read(hf_config, "model_type", "laguna")),
        architectures=list(_read(hf_config, "architectures", ["LagunaForCausalLM"])),
        moe_enabled=True,
        shared_expert_intermediate_size=int(_read(hf_config, "shared_expert_intermediate_size")),
        first_k_dense_replace=1,
        n_shared_experts=1,
        routed_scaling_factor=float(_read(hf_config, "moe_routed_scaling_factor", 1.0)),
        has_router_bias=True,
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_ids,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=full_rope,
            ),
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=swa_ids,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=swa_rope,
                sliding_window=int(_read(hf_config, "sliding_window")),
            ),
        ),
    )


__all__ = ["parse_config"]
