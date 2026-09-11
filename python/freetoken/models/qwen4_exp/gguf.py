"""Qwen3.8-Flash-Next (``qwen4exp``) read from a GGUF checkpoint.

Upstream's reader serves the NVFP4 and block-fp8 safetensors releases; this is the
GGUF side. Two things make it cheaper than the safetensors path rather than harder:

* llama.cpp **precomputes the PLE tables** into the metadata --
  ``ple.head_vocab_sizes``, ``ple.head_offsets`` and ``ple.layer_multipliers`` -- so
  the n-gram addressing is read rather than re-derived from primes and splitmix64.
  Deriving it independently would be a second implementation to keep in agreement.
* The n-gram table arrives as one flat ``per_layer_token_embd`` rather than the
  checkpoint's 128 shards, so the row index needs no shard arithmetic.

Naming follows the ggml convention (``blk.N.*``), which is a different namespace from
the HF module paths the model is built with; the translation lives here, as it does
for every other GGUF adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from freetoken.models.gguf.reader import load_gguf_metadata

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "qwen4exp"


class _Text:
    """The ``text_config`` view ``parse_config`` reads, filled from GGUF metadata."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def _hf_like(shim: "GgufConfigShim"):
    md = load_gguf_metadata(shim.model_path)

    def g(key: str, default=None):
        value = md.get(f"{_ARCH}.{key}", default)
        if value is None and default is None:
            raise KeyError(f"GGUF metadata has no {_ARCH}.{key}")
        return value

    head_dim = int(g("attention.key_length"))
    rotary_dim = int(g("rope.dimension_count"))
    layers = int(g("block_count"))
    interval = int(g("full_attention_interval", 4))
    num_v_heads = int(g("ssm.time_step_rank"))

    # ggml records the PLE layer 0-indexed (it names the tensors ``blk.1.ple_*``);
    # parse_config takes HF's one-indexed form and subtracts, so shift here.
    ple_layers = tuple(int(i) + 1 for i in g("ple.layers", []))
    head_vocab_sizes = [int(v) for v in g("ple.head_vocab_sizes")]
    # The per-head vocabularies are the first primes above a round base; recovering
    # the base keeps the config field meaningful, but the sizes themselves are what
    # the addressing uses.
    base = 10 ** (len(str(min(head_vocab_sizes))) - 1) * (min(head_vocab_sizes) // 10 ** (len(str(min(head_vocab_sizes))) - 1))

    text = _Text(
        hidden_size=int(g("embedding_length")),
        num_hidden_layers=layers,
        num_attention_heads=int(g("attention.head_count")),
        num_key_value_heads=int(g("attention.head_count_kv")),
        head_dim=head_dim,
        partial_rotary_factor=rotary_dim / head_dim,
        rope_parameters={"rope_theta": float(g("rope.freq_base")), "rope_type": "default"},
        max_position_embeddings=int(g("context_length")),
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        hidden_act="silu",
        vocab_size=shim.vocab_size,
        eos_token_id=int(g("ple.eos_token_id")),
        # layer types: every ``interval``-th layer is full attention, as llama.cpp
        # derives them from full_attention_interval rather than storing a list
        layer_types=[
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(layers)
        ],
        full_attention_interval=interval,
        # MoE
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(g("expert_shared_feed_forward_length")),
        norm_topk_prob=True,
        # hyper-connections
        hc_count=int(g("hyper_connection.count")),
        hc_lowrank=int(g("hyper_connection.low_rank")),
        # PLE / n-gram
        ple_layer_ids=ple_layers,
        ple_embed_dim=int(g("embedding_length_per_layer_input")),
        ple_conv_kernel_size=int(g("ple.conv_kernel")),
        ngram_size=int(g("ple.ngram_size")),
        heads_per_ngram=int(g("ple.heads_per_ngram")),
        ngram_vocab_size_base=int(base),
        make_ngram_vocab_size_divisible_by=128,
        # one flat table here, not the checkpoint's 128 shards
        split_ngram_parts=1,
        ple_head_vocab_sizes=tuple(head_vocab_sizes),
        ple_head_offsets=tuple(int(v) for v in g("ple.head_offsets")),
        ple_layer_multipliers=tuple(int(v) for v in g("ple.layer_multipliers")),
        # QSA indexer
        indexer_n_heads=int(g("attention.indexer.head_count")),
        indexer_kv_heads=1,
        indexer_head_dim=int(g("attention.indexer.key_length")),
        indexer_budget=int(g("attention.indexer.top_k")),
        indexer_compress_ratio=max(int(r) for r in g("attention.compress_ratios", [1])) or 1,
        # GatedDeltaNet
        linear_num_key_heads=int(g("ssm.group_count")),
        linear_num_value_heads=num_v_heads,
        linear_key_head_dim=int(g("ssm.state_size")),
        linear_value_head_dim=int(g("ssm.inner_size")) // num_v_heads,
        linear_conv_kernel_dim=int(g("ssm.conv_kernel")),
        output_gate_type="silu",
        tie_word_embeddings=shim.tie_word_embeddings,
    )

    class _HF:
        text_config = text
        architectures = ["Qwen4ExpGGUFForCausalLM"]
        model_type = "qwen4_exp"
        quantization_config = None

    return _HF()


def parse_gguf_config(shim: "GgufConfigShim") -> "ModelConfig":
    """A ModelConfig for a GGUF Flash-Next, through upstream's own ``parse_config``."""
    from .config import parse_config

    return parse_config(_hf_like(shim))


# ggml tensor suffix -> the FreeToken module it feeds, for modules fed by exactly one
# tensor. Fused modules are listed separately below because their slots have to agree.
_ONE_TO_ONE = {
    "attn_output.weight": "self_attn.o_proj",
    "ssm_out.weight": "linear_attn.out_proj",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj",
    "hc_attn_up.weight": "attn_hyper_connection.input_mix_weight_up",
    "hc_ffn_up.weight": "mlp_hyper_connection.input_mix_weight_up",
    "ple_key.weight": "ple.key_proj",
    "ple_value.weight": "ple.value_proj",
}

# module -> the ggml tensors concatenated into it, in output order. A module is served
# packed only when every slot is a type the kernels implement, so these resolve as a
# group.
_FUSED = {
    "self_attn.qkv_proj": ("attn_q.weight", "attn_k.weight", "attn_v.weight"),
    "mlp.shared_expert.gate_up_proj": ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"),
    "mlp.experts": ("ffn_gate_exps.weight", "ffn_down_exps.weight"),
}

# Fusions that mix quantized and unquantized tensors. They are listed like any other:
# the dialect carries a type per slot and the method serves the dense ones dense
# alongside their packed siblings. Refusing them whole would cost ~14.8 ms/token on
# an RX 6800 -- they are 1.85 B parameters and the dense slots are 0.69% of the bytes.
_MIXED = {
    "linear_attn.in_proj": (
        "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight",
    ),
    "attn_hyper_connection.input_mix_weight_down_block_inject": (
        "hc_attn_down.weight", "hc_attn_inject.weight",
    ),
    "mlp_hyper_connection.input_mix_weight_down_block_inject": (
        "hc_ffn_down.weight", "hc_ffn_inject.weight",
    ),
}


def gguf_module_types(model_path: str) -> dict[str, tuple[str, ...]]:
    """FreeToken module prefix -> ggml type per fused slot, for what we serve packed.

    The GGUF side of the ``gguf`` quant dialect. Only this module knows both the ggml
    names and the FreeToken module paths, so the translation lives here rather than
    as NameMap rules.
    """
    from freetoken.models.gguf.dequant import GGML_NAME
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.models.qwen3_5_moe.gguf import _is_packable

    globals_: dict[str, int] = {}
    by_layer: dict[int, dict[str, int]] = {}
    for t in iter_gguf_tensors(model_path):
        # Every type is recorded, packable or not: a fused module needs each slot's
        # type to decide per slot, and _packable_group() drops groups with nothing
        # worth packing.
        if t.name.startswith("blk."):
            layer = int(t.name.split(".")[1])
            by_layer.setdefault(layer, {})[t.name.split(".", 2)[2]] = t.ggml_type
        else:
            globals_[t.name] = t.ggml_type

    name = GGML_NAME.__getitem__
    out: dict[str, tuple[str, ...]] = {}
    for tensor, module in (("output.weight", "lm_head"),
                           ("token_embd.weight", "model.embed_tokens")):
        t = globals_.get(tensor)
        if t is not None and _is_packable(t):
            out[module] = (name(t),)

    def group(types: dict[str, int], slots, *, all_packable: bool) -> tuple[str, ...] | None:
        got = [types.get(s) for s in slots]
        if any(t is None for t in got):
            return None            # slot missing from this checkpoint
        packable = [_is_packable(t) for t in got]
        keep = all(packable) if all_packable else any(packable)
        return tuple(name(t) for t in got) if keep else None

    for layer, types in by_layer.items():
        stem = f"model.layers.{layer}."
        for suffix, module in _ONE_TO_ONE.items():
            t = types.get(suffix)
            if t is not None and _is_packable(t):
                out[stem + module] = (name(t),)
        for module, slots in _FUSED.items():
            g = group(types, slots, all_packable=True)
            if g:
                out[stem + module] = g
        for module, slots in _MIXED.items():
            g = group(types, slots, all_packable=False)
            if g:
                out[stem + module] = g
    return out


__all__ = ["parse_gguf_config", "gguf_module_types"]
