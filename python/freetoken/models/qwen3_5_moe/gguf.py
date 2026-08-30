"""GGUF adapter for Qwen3.5/3.6 hybrid MoE (llama.cpp arch ``qwen35moe``).

Layer kinds follow ``qwen35moe.full_attention_interval`` (layer ``i`` is full
attention iff ``(i+1) % interval == 0``; the rest are GDN linear-attention).
Non-expert tensors dequantize to bf16 under FreeToken's HF names; the routed
experts stay in their native GGUF quant blocks and stream through the ``q4_0``
expert-bank path (generalized: any MMVQ-covered ggml type; a bank whose type is
not uniform across layers is promoted to Q8_0 at load, the only type gguf-py can
requantize to).

Facts verified against a real Qwen3.6-35B-A3B GGUF (unsloth UD-Q4_K_M header +
sampled tensors):

- Full layers carry ``attn_q`` with the q|gate interleaving already fused
  (out = 2 * heads * head_dim), matching FreeToken's ``qkv_proj`` q-slice.
- Linear layers store the GDN in_proj split as ``attn_qkv`` (conv_dim),
  ``attn_gate`` (value_dim), ``ssm_beta``/``ssm_alpha`` (num_v_heads each);
  FreeToken's fused ``in_proj`` concat order is qkv | z | b | a.
- Gemma-style ``+1`` is already baked into attn/post/output norms and q/k norms
  at conversion time (values center on 1.0) -> load as-is; ``ssm_norm`` is a
  plain gated RMS norm weight (no +1).
- ``ssm_a`` stores ``-exp(A_log)`` (all values negative) -> recover
  ``A_log = log(-a)``; ``ssm_dt`` (spelled ``ssm_dt.bias`` in some conversions) is
  the fp32 ``dt_bias`` verbatim.
- GDN geometry from the ssm KVs: num_key_heads = ``ssm.group_count``,
  num_value_heads = ``ssm.time_step_rank``, key_head_dim = ``ssm.state_size``,
  value_head_dim = ``ssm.inner_size // ssm.time_step_rank``.

TP=1 only. Experts require an offload-family ``--moe-backend`` (the resident
fused layer has no GGUF format); the CPU executor's inline dot only understands
literal Q4_0, so non-Q4_0 banks need ``--moe-backend offload``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

import torch

import os

from freetoken.models.gguf.dequant import (
    GGML_NAME,
    GGML_Q4_0,
    GGML_Q4_1,
    GGML_Q5_0,
    GGML_Q5_1,
    GGML_Q8_0,
    dequant_any,
    row_bytes,
)
from freetoken.models.gguf.reader import (
    GgufTensor,
    iter_gguf_tensors,
    load_gguf_metadata,
)

from .config import parse_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

logger = logging.getLogger(__name__)

_EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")


def _require_tp1(what: str) -> None:
    from freetoken.distributed import get_tp_info

    try:
        size = get_tp_info().size
    except RuntimeError:  # TP not initialized: offline tooling context
        return
    if size > 1:
        raise NotImplementedError(f"qwen35moe GGUF {what} supports TP=1 only")


def _is_full_attention(layer: int, interval: int) -> bool:
    return (layer + 1) % interval == 0


def _head_count(value, what: str) -> int:
    """Collapse a possibly per-layer head count to the model-wide one.

    llama.cpp writes ``attention.head_count[_kv]`` as an array with one entry per
    block for hybrid models, and stores 0 on the GDN linear-attention layers
    (which have no attention heads at all). Real Qwen3.5/3.6 files therefore hand
    us e.g. ``[0, 0, 0, 2, 0, 0, 0, 2, ...]`` where the synthetic fixtures had a
    scalar. Take the value the full-attention layers agree on; a scalar passes
    through unchanged.
    """
    if not isinstance(value, (list, tuple)):
        return int(value)
    counts = {int(v) for v in value if int(v)}
    if len(counts) != 1:
        raise ValueError(
            f"qwen35moe GGUF: {what} = {list(value)} - FreeToken needs a single "
            f"head count shared by every full-attention layer, got {sorted(counts)}"
        )
    return counts.pop()


def _expert_types(model_path: str) -> tuple[dict[str, dict[int, int]], dict[str, int]]:
    """Per-layer ggml types of the routed expert tensors, and the resolved bank
    type per bank (uniform type, else Q8_0 promotion)."""
    per: dict[str, dict[int, int]] = {s: {} for s in _EXPERT_SUFFIXES}
    for t in iter_gguf_tensors(model_path):
        if t.name.startswith("blk.") and t.name.split(".", 2)[2] in per:
            per[t.name.split(".", 2)[2]][int(t.name.split(".")[1])] = t.ggml_type

    # Mixed-type banks are requantized to one type at load; gguf-py can only
    # quantize the classic types, so Q5_1 (6 bpw) is the default promotion target
    # (Q8_0 would add ~4 GB of pinned host RAM on a 35B-A3B). Override with
    # FT_GGUF_BANK_PROMOTE=Q8_0|Q5_1|Q5_0|Q4_1|Q4_0.
    _PROMOTE = {
        "Q4_0": GGML_Q4_0, "Q4_1": GGML_Q4_1,
        "Q5_0": GGML_Q5_0, "Q5_1": GGML_Q5_1, "Q8_0": GGML_Q8_0,
    }[os.environ.get("FT_GGUF_BANK_PROMOTE", "Q5_1").upper()]

    def resolve(types: set[int], what: str) -> int:
        if len(types) == 1:
            return next(iter(types))
        names = sorted(GGML_NAME.get(t, str(t)) for t in types)
        logger.warning(
            "qwen35moe GGUF: %s experts mix %s across layers -> requantizing the "
            "whole bank to %s at load (set FT_GGUF_BANK_PROMOTE to change)",
            what, names, GGML_NAME[_PROMOTE],
        )
        return int(_PROMOTE)

    gate_types = set(per["ffn_gate_exps.weight"].values())
    up_types = set(per["ffn_up_exps.weight"].values())
    bank = {
        "gate_up": resolve(gate_types | up_types, "gate/up"),
        "down": resolve(set(per["ffn_down_exps.weight"].values()), "down"),
    }
    return per, bank


def parse_gguf_config(shim: "GgufConfigShim") -> "ModelConfig":
    m = shim.metadata
    prefix = shim.model_type  # "qwen35moe"

    def g(key: str, default=None):
        return m.get(f"{prefix}.{key}", default)

    num_layers = int(g("block_count"))
    interval = int(g("full_attention_interval", 4))
    layer_types = [
        "full_attention" if _is_full_attention(i, interval) else "linear_attention"
        for i in range(num_layers)
    ]
    head_dim = int(g("attention.key_length"))
    rotary_dim = int(g("rope.dimension_count", head_dim))
    num_v_heads = int(g("ssm.time_step_rank"))

    # no text_config attr: parse_config's getattr then falls back to the namespace itself
    hf_like = SimpleNamespace(
        num_hidden_layers=num_layers,
        num_attention_heads=_head_count(g("attention.head_count"), "attention.head_count"),
        num_key_value_heads=_head_count(g("attention.head_count_kv"), "attention.head_count_kv"),
        head_dim=head_dim,
        hidden_size=int(g("embedding_length")),
        intermediate_size=int(g("feed_forward_length", 0)),
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon", 1e-6)),
        max_position_embeddings=int(g("context_length", 262144)),
        rope_theta=float(g("rope.freq_base", 10_000_000.0)),
        partial_rotary_factor=rotary_dim / head_dim,
        rope_parameters=None,
        rope_scaling=None,
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        layer_types=layer_types,
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(g("expert_shared_feed_forward_length", 0)),
        norm_topk_prob=True,
        linear_num_key_heads=int(g("ssm.group_count")),
        linear_num_value_heads=num_v_heads,
        linear_key_head_dim=int(g("ssm.state_size")),
        linear_value_head_dim=int(g("ssm.inner_size")) // num_v_heads,
        linear_conv_kernel_dim=int(g("ssm.conv_kernel")),
        quantization_config=None,
        model_type="qwen3_5_moe",
        architectures=["Qwen35MoeGGUFForCausalLM"],
        torch_dtype="bfloat16",
    )
    config = parse_config(hf_like)
    # Routed experts ride the (generalized) q4_0 GGUF bank path.
    object.__setattr__(config, "expert_quant", "q4_0")
    object.__setattr__(config, "moe_weight_format", "q4_0")
    per_layer, bank_types = _expert_types(shim.model_path)
    object.__setattr__(config, "gguf_expert_bank_types", bank_types)
    object.__setattr__(config, "gguf_expert_layer_types", per_layer)
    return config


def _to_bf16(t: GgufTensor) -> torch.Tensor:
    return dequant_any(t).to(torch.bfloat16)


# suffix -> module-relative name, tensors that map 1:1 (dequant to bf16)
_SUFFIX_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",  # +1 baked at conversion
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
}

_QKV_SLOTS = ("attn_q", "attn_k", "attn_v")  # full-attention fuse order
_IN_PROJ_SLOTS = ("attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha")  # qkv|z|b|a


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Dense (non-routed-expert) weights in FreeToken naming, dequantized to bf16
    (fp32 for ``dt_bias``/``A_log``). Routed experts are delivered by
    :func:`load_q4_0_expert_sources`, never here."""
    assert not include_moe_experts, (
        "qwen35moe GGUF routed experts load as native packed banks "
        "(load_q4_0_expert_sources), not through iter_weights"
    )
    assert include_non_moe
    _require_tp1("weights")

    interval = int(load_gguf_metadata(model_path).get(
        "qwen35moe.full_attention_interval", 4
    ))

    fuse: dict[tuple[int, str], dict[str, torch.Tensor]] = {}

    def feed_fused(
        layer: int, group: str, slots: tuple[str, ...], slot: str, val: torch.Tensor, out_name: str
    ) -> Iterator[tuple[str, torch.Tensor]]:
        buf = fuse.setdefault((layer, group), {})
        buf[slot] = val
        if len(buf) == len(slots):
            del fuse[(layer, group)]
            yield out_name, torch.cat([buf[s] for s in slots], dim=0).contiguous()

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.weight", _to_bf16(t)
        elif name == "output_norm.weight":
            yield "model.norm.weight", _to_bf16(t)
        elif name == "output.weight":
            yield "lm_head.weight", _to_bf16(t)
        elif name.startswith("blk."):
            layer = int(name.split(".")[1])
            suffix = name.split(".", 2)[2]
            stem = f"model.layers.{layer}."
            if suffix in _EXPERT_SUFFIXES:
                continue
            if suffix == "ssm_a":
                # stored as -exp(A_log); FreeToken keeps A_log (fp32)
                a = dequant_any(t).to(torch.float32)
                assert (a < 0).all(), f"{name}: expected -exp(A_log) (negative values)"
                yield stem + "linear_attn.A_log", torch.log(-a)
                continue
            # Real Qwen3.5/3.6 conversions name this bias bare `ssm_dt` (the way
            # `ssm_a` is bare); other files spell it `ssm_dt.bias`. Either way it is
            # the fp32 dt_bias vector, one entry per value head.
            if suffix in ("ssm_dt.bias", "ssm_dt"):
                yield stem + "linear_attn.dt_bias", dequant_any(t).to(torch.float32)
                continue
            if suffix == "ssm_conv1d.weight":
                # ggml [K, conv_dim] -> torch (conv_dim, K) -> module [conv_dim, 1, K]
                yield stem + "linear_attn.conv1d.weight", _to_bf16(t).unsqueeze(1).contiguous()
                continue
            if suffix == "ffn_gate_inp_shexp.weight":
                yield stem + "mlp.shared_expert_gate.weight", _to_bf16(t).reshape(1, -1)
                continue
            proj = suffix.rsplit(".weight", 1)[0]
            if proj in _QKV_SLOTS and _is_full_attention(layer, interval):
                yield from feed_fused(
                    layer, "qkv", _QKV_SLOTS, proj, _to_bf16(t),
                    stem + "self_attn.qkv_proj.weight",
                )
                continue
            if proj in _IN_PROJ_SLOTS and not _is_full_attention(layer, interval):
                yield from feed_fused(
                    layer, "in_proj", _IN_PROJ_SLOTS, proj, _to_bf16(t),
                    stem + "linear_attn.in_proj.weight",
                )
                continue
            if proj in ("ffn_gate_shexp", "ffn_up_shexp"):
                yield from feed_fused(
                    layer, "shexp", ("ffn_gate_shexp", "ffn_up_shexp"), proj, _to_bf16(t),
                    stem + "mlp.shared_expert.gate_up_proj.weight",
                )
                continue
            rel = _SUFFIX_MAP.get(suffix)
            if rel is not None:
                yield stem + rel, _to_bf16(t)

    leftovers = sorted(fuse)
    assert not leftovers, f"incomplete fused groups: {leftovers}"


# --------------------------------------------------------------------------------------
# Routed-expert host banks (native GGUF quants) for the offload cache.
# --------------------------------------------------------------------------------------


def _bank_types(config: "ModelConfig") -> dict[str, int]:
    types = getattr(config, "gguf_expert_bank_types", None)
    assert types is not None, "config was not built by qwen35moe parse_gguf_config"
    return types


def _expert_specs(config: "ModelConfig") -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    E = config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    types = _bank_types(config)
    return {
        "gate_up": ((E, 2 * I, row_bytes(H, types["gate_up"])), torch.uint8),
        "down": ((E, H, row_bytes(I, types["down"])), torch.uint8),
    }


def _packed_as(t: GgufTensor, target_type: int) -> torch.Tensor:
    """Packed ``[rows, row_bytes(target)]`` bytes, requantizing when the tensor's
    own type differs from the bank type (Q8_0 promotion path)."""
    if t.ggml_type == target_type:
        return t.packed()
    import gguf
    from gguf.quants import quantize

    n_in = t.shape[-1]
    data = dequant_any(t).to(torch.float32).reshape(t.rows, n_in).numpy()
    q = quantize(data, gguf.GGMLQuantizationType(target_type))
    return torch.from_numpy(q)


def load_q4_0_expert_sources(
    model_path: str, config: "ModelConfig", *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Per-layer host banks of the routed experts' native GGUF block bytes.

    ``gate_up[li]`` is ``[E, 2I, row_bytes(H, bank_type)]`` (gate rows then up rows,
    matching ``silu_and_mul``'s halves) and ``down[li]`` ``[E, H, row_bytes(I,
    bank_type)]``. llama.cpp stores gate and up as separate ``ffn_gate_exps`` /
    ``ffn_up_exps`` tensors, so each layer's halves are copied into the row ranges of
    one bank. Layers whose tensor type differs from the bank type are requantized
    (see :func:`_expert_types`)."""
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    _require_tp1("expert banks")
    L, E = config.num_layers, config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    types = _bank_types(config)
    gu_bytes = row_bytes(H, types["gate_up"])
    dn_bytes = row_bytes(I, types["down"])
    hb = alloc_layer_banks(_expert_specs(config), L)  # lazy anon mmaps (unpinned)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    seen: dict[str, set[int]] = {s: set() for s in _EXPERT_SUFFIXES}

    def _load(sink) -> None:
        # 3 writes/layer: gate half, up half, down
        tracker = LayerCompletionTracker(3, hb, sink) if sink is not None else None
        for t in iter_gguf_tensors(model_path):
            if not t.name.startswith("blk."):
                continue
            suffix = t.name.split(".", 2)[2]
            if suffix not in _EXPERT_SUFFIXES:
                continue
            layer = int(t.name.split(".")[1])
            if suffix == "ffn_gate_exps.weight":
                banks["gate_up"][layer][:, :I].copy_(
                    _packed_as(t, types["gate_up"]).reshape(E, I, gu_bytes)
                )
            elif suffix == "ffn_up_exps.weight":
                banks["gate_up"][layer][:, I:].copy_(
                    _packed_as(t, types["gate_up"]).reshape(E, I, gu_bytes)
                )
            else:
                banks["down"][layer].copy_(
                    _packed_as(t, types["down"]).reshape(E, H, dn_bytes)
                )
            seen[suffix].add(layer)
            if tracker is not None:
                tracker.note(layer)

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)  # CUDA-less: mmap banks stay pageable, never pinned

    want = set(range(L))
    missing = {s: sorted(want - got) for s, got in seen.items() if got != want}
    assert not missing, f"missing expert layers: {missing}"
    return banks


def dummy_q4_0_expert_sources(config: "ModelConfig") -> dict[str, list[torch.Tensor]]:
    """Random banks shaped like :func:`load_q4_0_expert_sources` output."""
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    hb = alloc_layer_banks(_expert_specs(config), config.num_layers)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    for t in banks["gate_up"] + banks["down"]:
        t.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(hb)
    return banks


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "load_q4_0_expert_sources",
    "dummy_q4_0_expert_sources",
]
