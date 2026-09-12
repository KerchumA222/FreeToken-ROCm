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

from typing import TYPE_CHECKING, Any, Iterator

import torch

from freetoken.models.gguf.reader import load_gguf_metadata

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "qwen4exp"


def _num_ngram_heads(g) -> int:
    """One head group per n-gram order 2..ngram_size (Qwen3.8: 8 x 2-gram + 8 x 3-gram).
    Mirrors ``Qwen4Args.num_ngram_heads``; ggml records the two factors separately."""
    return (int(g("ple.ngram_size")) - 1) * int(g("ple.heads_per_ngram"))


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
        # ggml stores the PER-HEAD row width (FreeToken's ngram_head_dim); the model's
        # ple_embed_dim is the width of all heads concatenated, which is what key_proj
        # and value_proj actually consume (Qwen3.8: 160 x 16 = 2560).
        ple_embed_dim=int(g("embedding_length_per_layer_input")) * _num_ngram_heads(g),
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
        # Sigmoid, not the silu Qwen3.5 hardcodes: llama.cpp's graph gates the GDN
        # output norm with SIGMOID(z) and gdn_reference.py says the same. ggml records
        # no KV for it -- it is an architectural constant of qwen4exp. Feeding silu
        # multiplied the gate by an extra z and put layer 0's gated norm at -321.7
        # against the reference's -36.1.
        output_gate_type="sigmoid",
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
    from freetoken.models.qwen3_5_moe.gguf import _dense_types, _expert_types

    from .config import parse_config

    config = parse_config(_hf_like(shim))
    # Routed experts ride the (generalized) q4_0 GGUF bank path: the tag means "native
    # GGUF block bytes", and the actual ggml type per bank travels separately.
    object.__setattr__(config, "expert_quant", "q4_0")
    object.__setattr__(config, "moe_weight_format", "q4_0")
    per_layer, bank_types = _expert_types(shim.model_path)
    object.__setattr__(config, "gguf_expert_bank_types", bank_types)
    object.__setattr__(config, "gguf_expert_layer_types", per_layer)
    object.__setattr__(config, "gguf_dense_types", _dense_types(shim.model_path))
    return config


# --------------------------------------------------------------------------------------
# ggml <-> FreeToken naming.
#
# Three shapes of correspondence, because the model's buffers do not all come from one
# ggml tensor and the packed path can only serve some of them:
#
# * :data:`_SUFFIX_MAP` -- one ggml tensor fills one module buffer.
# * :data:`_MERGED` -- one ggml tensor per *slot* of a merged linear. The module keeps
#   the slots apart (``weight_0``, ``weight_1``, ...), so their ggml types may differ.
# * :data:`_ROW_CONCAT` -- several ggml tensors are stacked into ONE buffer. The module
#   is a plain ``LinearReplicated`` with a single weight, so the parts have to agree on
#   a type to stay packed: a row-wise concatenation of block-quantized rows is itself a
#   valid tensor of that type, but only when every row uses the same block layout.
# --------------------------------------------------------------------------------------

# Routed experts never come through the weight iterator; the offload banks read them.
_EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")

# suffix -> module-relative key, for the tensors that fill one buffer unchanged.
_SUFFIX_MAP = {
    "attn_output.weight": "self_attn.o_proj.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "hc_attn_up.weight": "attn_hyper_connection.input_mix_weight_up.weight",
    "hc_ffn_up.weight": "mlp_hyper_connection.input_mix_weight_up.weight",
    "ple_key.weight": "ple.key_proj.weight",
    "ple_value.weight": "ple.value_proj.weight",
}

# The zero-centered norms. HF stores ``scale - 1`` and FreeToken keeps it that way --
# GemmaPlusOneRMSNorm / GroupedPlusOneRMSNorm add the 1 back in fp32 at runtime, which is
# the whole point of the format. llama.cpp's converter instead bakes the +1 into the
# stored weight, so the 1 comes back off here. Measured on this checkpoint: every one of
# these is centered on ~1 (e.g. blk.1.ple_norm_key 0.674..1.863, mean 0.893) while the
# plain-scaled ``ssm_norm`` sits at 0.875..1.023 -- the two are only distinguishable by
# which norm class consumes them, not by their values.
_PLUS_ONE_MAP = {
    "hc_attn_norm.weight": "attn_hyper_connection.hc_norm.weight",
    "hc_ffn_norm.weight": "mlp_hyper_connection.hc_norm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
    "ple_norm_key.weight": "ple.norm_key.weight",
    "ple_norm_query.weight": "ple.norm_query.weight",
    "ple_norm_conv.weight": "ple.norm_conv.weight",
}

# Merged linears: module -> the ggml tensors feeding its slots, in output order.
# ``attn_q`` carries q AND the output gate (``_qkv_split`` is [2*qo, kv, kv]), which is
# why the full-attention fuse takes it whole.
_MERGED = {
    "self_attn.qkv_proj": ("attn_q.weight", "attn_k.weight", "attn_v.weight"),
    "linear_attn.in_proj": (
        "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight",
    ),
    "mlp.shared_expert.gate_up_proj": ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"),
}

# Single-buffer stacks: module -> (parts in row order, row alignment). The HC mix reads
# its low-rank projection and its injection logits out of one GEMM, padded to a multiple
# of 16 (vLLM hyperconnection.py); the pad rows are zero and their output is dropped.
_ROW_CONCAT = {
    "attn_hyper_connection.input_mix_weight_down_block_inject": (
        ("hc_attn_down.weight", "hc_attn_inject.weight"), 16,
    ),
    "mlp_hyper_connection.input_mix_weight_down_block_inject": (
        ("hc_ffn_down.weight", "hc_ffn_inject.weight"), 16,
    ),
    "self_attn.indexer.index_qk_proj": (
        ("indexer.q_proj.weight", "indexer.k_proj.weight"), 1,
    ),
}

# Whole-model tensors (no ``blk.N.`` prefix).
_GLOBAL_SUFFIX_MAP = {
    "output_hc_down.weight": "model.hyper_connection_mixer.input_mix_weight_down.weight",
    "output_hc_up.weight": "model.hyper_connection_mixer.input_mix_weight_up.weight",
}
_GLOBAL_PLUS_ONE = {
    "output_hc_norm.weight": "model.hyper_connection_mixer.hc_norm.weight",
}
# The n-gram table: 54.4 GiB of q8_0 here, never part of the dense state dict.
_PLE_TABLE = "per_layer_token_embd.weight"


def _pad_rows(n: int, align: int) -> int:
    return (-n) % align


def gguf_module_types(model_path: str) -> dict[str, tuple[str, ...]]:
    """FreeToken module prefix -> ggml type per fused slot, for what we serve packed.

    The GGUF side of the ``gguf`` quant dialect. Only this module knows both the ggml
    names and the FreeToken module paths, so the translation lives here rather than
    as NameMap rules.
    """
    from freetoken.models.gguf.dequant import GGML_NAME
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.models.qwen3_5_moe.gguf import (
        _expert_types,
        _is_packable,
        _v_head_permutation,
    )

    _per_layer, bank_types = _expert_types(model_path)
    globals_: dict[str, int] = {}
    by_layer: dict[int, dict[str, int]] = {}
    for t in iter_gguf_tensors(model_path):
        # Every type is recorded, packable or not: a merged module needs each slot's
        # type to decide per slot.
        if t.name.startswith("blk."):
            layer = int(t.name.split(".")[1])
            by_layer.setdefault(layer, {})[t.name.split(".", 2)[2]] = t.ggml_type
        else:
            globals_[t.name] = t.ggml_type

    # linear_attn.out_proj is the one module whose value-head permutation moves
    # COLUMNS rather than rows, so it can only be served packed when a head lands on
    # quant-block boundaries -- true for the 32-wide block types, false for a K-quant
    # whose 256-element superblock is wider than this model's 128-wide head. Claiming
    # it regardless left the reader with a permutation it could not apply.
    md = load_gguf_metadata(model_path)
    n_v = int(md[f"{_ARCH}.ssm.time_step_rank"])
    n_k = int(md[f"{_ARCH}.ssm.group_count"])
    d_v = int(md[f"{_ARCH}.ssm.inner_size"]) // n_v
    perm_needed = _v_head_permutation(n_v, n_k) is not None

    name = GGML_NAME.__getitem__
    out: dict[str, tuple[str, ...]] = {}

    def one(types: dict[str, int], suffix: str, module: str) -> None:
        t = types.get(suffix)
        if t is None or not _is_packable(t):
            return
        if (suffix == "ssm_out.weight" and perm_needed
                and _head_block_bytes(n_v * d_v, d_v, t) is None):
            return      # served dense; the reader permutes the columns there
        out[module] = (name(t),)

    for tensor, module in (("output.weight", "lm_head"),
                           ("token_embd.weight", "model.embed_tokens")):
        one(globals_, tensor, module)
    for suffix, module in _GLOBAL_SUFFIX_MAP.items():
        one(globals_, suffix, module.rsplit(".weight", 1)[0])

    for layer, types in by_layer.items():
        stem = f"model.layers.{layer}."
        for suffix, module in _SUFFIX_MAP.items():
            one(types, suffix, stem + module.rsplit(".weight", 1)[0])
        for module, slots in _MERGED.items():
            got = [types.get(s) for s in slots]
            # A slot missing means this layer does not have the module at all
            # (full-attention vs GDN); a group with nothing packable is left dense.
            if any(t is None for t in got) or not any(_is_packable(t) for t in got):
                continue
            out[stem + module] = tuple(name(t) for t in got)
        # The routed experts are one module with two banks, and gate_up and down may
        # differ in type. Reported from the RESOLVED bank types rather than this
        # layer's stored types, so the dialect (which is what _bank_types reads, being
        # the only carrier that survives the engine's dataclasses.replace) agrees with
        # the loader on a checkpoint whose layers disagree and get promoted.
        if all(s in types for s in _EXPERT_SUFFIXES):
            out[stem + "mlp.experts"] = (name(bank_types["gate_up"]), name(bank_types["down"]))
        for module, (parts, _align) in _ROW_CONCAT.items():
            got = [types.get(p) for p in parts]
            # One buffer, so one block layout: the parts must agree on a packable type
            # or the whole stack is served dense.
            if any(t is None for t in got) or len(set(got)) != 1 or not _is_packable(got[0]):
                continue
            out[stem + module] = (name(got[0]),)
    return out


# --------------------------------------------------------------------------------------
# Dense weights.
# --------------------------------------------------------------------------------------


def _to_bf16(t) -> "torch.Tensor":
    from freetoken.models.gguf.dequant import dequant_any

    return dequant_any(t).to(torch.bfloat16)


def _head_block_bytes(in_features: int, head_dim: int, ggml_type: int) -> int | None:
    """Bytes per value head inside one packed row, or None when a head does not land on
    a block boundary. Reordering whole heads along the INPUT axis is a byte permutation
    only if each head spans a whole number of quant blocks -- Qwen3.8 has head_v_dim 128
    over q8_0's 32-wide blocks, so a head is exactly 4 blocks (136 bytes)."""
    from freetoken.models.gguf.dequant import BLOCK_SHAPE, row_bytes

    if ggml_type not in BLOCK_SHAPE:
        return None
    block, type_size = BLOCK_SHAPE[ggml_type]
    if head_dim % block:
        return None
    total = row_bytes(in_features, ggml_type)
    per_head = head_dim // block * type_size
    return per_head if per_head * (in_features // head_dim) == total else None


def _permute_packed_heads(packed: "torch.Tensor", perm, per_head: int) -> "torch.Tensor":
    """Reorder whole value heads along the packed input axis of ``[out, row_bytes]``."""
    out = packed.shape[0]
    return (
        packed.view(out, -1, per_head)
        .index_select(1, perm)
        .reshape(out, -1)
        .contiguous()
    )


def _zero_rows(n: int, row_bytes: int) -> "torch.Tensor":
    """``n`` all-zero packed rows. Every quant type this path serves encodes zero as
    all-zero bytes (a q8_0 block is an fp16 scale of 0 over 32 zero weights), so the
    HC pad rows need no per-type construction."""
    return torch.zeros(n, row_bytes, dtype=torch.uint8)


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> "Iterator[tuple[str, torch.Tensor]]":
    """Dense (non-routed-expert) weights in FreeToken naming.

    Quantized tensors are handed over packed wherever the dialect said we would serve
    them that way, and dequantized to bf16 otherwise (fp32 for ``dt_bias``/``A_log``).
    Routed experts are delivered by the expert-bank reader and the n-gram table by
    :func:`load_gguf_ple_table`; neither comes through here.
    """
    from freetoken.models.gguf.dequant import dequant_any
    from freetoken.models.gguf.reader import iter_gguf_tensors, load_gguf_metadata
    from freetoken.models.qwen3_5_moe.gguf import (
        _is_full_attention,
        _permute_head_rows,
        _require_tp1,
        _v_head_permutation,
    )

    assert not include_moe_experts, (
        "qwen4exp GGUF routed experts load as native packed banks, not through iter_weights"
    )
    assert include_non_moe
    _require_tp1("weights")

    md = load_gguf_metadata(model_path)
    interval = int(md.get(f"{_ARCH}.full_attention_interval", 4))
    packed = gguf_module_types(model_path)

    # GQA head-order fix for the GDN layers, exactly as in qwen35moe: llama.cpp pairs
    # value head j with key head j % HK, the vendored fla kernels with j // (HV // HK).
    # Qwen3.8 has the same 3:1 split (48 value heads over 16 key heads) that made this
    # necessary there.
    n_v = int(md[f"{_ARCH}.ssm.time_step_rank"])
    n_k = int(md[f"{_ARCH}.ssm.group_count"])
    d_v = int(md[f"{_ARCH}.ssm.inner_size"]) // n_v
    d_k = int(md[f"{_ARCH}.ssm.state_size"])
    # Confirmed against llama.cpp's converter rather than inferred from the ratio:
    # qwen4exp's Qwen4ExpTextModel derives from _LinearAttentionVReorderBase, the same
    # base qwen35moe uses, which rewrites the V heads from HF's grouped order
    # [G0_v0..v{r-1}, G1_v0..] into ggml's tiled order [G0_v0, G1_v0, .., G0_v1, ..] so
    # that ggml_repeat can pair them. It reorders exactly the tensors permuted below --
    # in_proj_qkv (V rows only), in_proj_z, in_proj_a/b, A_log, dt_bias, conv1d (V
    # channels only) and out_proj (columns) -- and this permutation is its inverse.
    v_perm = _v_head_permutation(n_v, n_k)
    key_dim = n_k * d_k

    def _fix_v_rows(t: torch.Tensor) -> torch.Tensor:
        if v_perm is None:
            return t
        head, tail = t[: 2 * key_dim], t[2 * key_dim :]
        return torch.cat([head, _permute_head_rows(tail, v_perm, d_v)], dim=0).contiguous()

    # The n-gram addressing tables. The HF checkpoint ships these as tensors; llama.cpp
    # precomputes them into the KV section instead, so they are read from the metadata
    # rather than re-derived from primes and splitmix64 (a second derivation would be a
    # second thing to keep in agreement with the reference).
    for ple_layer in (int(i) for i in md.get(f"{_ARCH}.ple.layers", [])):
        base = f"model.layers.{ple_layer}.ple.ple_embedding."
        for key, field in (
            ("layer_multipliers", "ple.layer_multipliers"),
            ("ngram_heads_vocab_sizes", "ple.head_vocab_sizes"),
            ("ngram_heads_offsets", "ple.head_offsets"),
        ):
            yield base + key, torch.tensor(
                [int(v) for v in md[f"{_ARCH}.{field}"]], dtype=torch.int64
            )

    # A merged or stacked module only completes once every part has been seen, and the
    # tensors arrive in shard order rather than grouped.
    pending: dict[tuple[int, str], dict[str, Any]] = {}

    def feed(layer: int, module: str, parts: tuple[str, ...], part: str, val):
        buf = pending.setdefault((layer, module), {})
        buf[part] = val
        if len(buf) != len(parts):
            return None
        del pending[(layer, module)]
        return [buf[p] for p in parts]

    for t in iter_gguf_tensors(model_path):
        name = t.name

        if name == _PLE_TABLE:
            continue
        if name == "token_embd.weight":
            if "model.embed_tokens" in packed:
                yield "model.embed_tokens.qweight", t.packed()
            else:
                yield "model.embed_tokens.weight", _to_bf16(t)
            continue
        if name == "output.weight":
            yield "lm_head.weight", (
                t.packed() if "lm_head" in packed else _to_bf16(t)
            )
            continue
        if name in _GLOBAL_PLUS_ONE:
            yield _GLOBAL_PLUS_ONE[name], _to_bf16(t) - 1.0
            continue
        if name in _GLOBAL_SUFFIX_MAP:
            key = _GLOBAL_SUFFIX_MAP[name]
            module = key.rsplit(".weight", 1)[0]
            yield key, (t.packed() if module in packed else _to_bf16(t))
            continue
        if not name.startswith("blk."):
            raise ValueError(f"{name}: unrecognized qwen4exp GGUF tensor")

        layer = int(name.split(".")[1])
        suffix = name.split(".", 2)[2]
        stem = f"model.layers.{layer}."
        if suffix in _EXPERT_SUFFIXES:
            continue

        if suffix in _PLUS_ONE_MAP:
            yield stem + _PLUS_ONE_MAP[suffix], _to_bf16(t) - 1.0
            continue
        if suffix == "ssm_a":
            # stored as -exp(A_log); FreeToken keeps A_log (fp32)
            a = dequant_any(t).to(torch.float32)
            assert (a < 0).all(), f"{name}: expected -exp(A_log) (negative values)"
            if v_perm is not None:
                a = a[v_perm]
            yield stem + "linear_attn.A_log", torch.log(-a)
            continue
        if suffix == "ssm_dt.bias":
            dtb = dequant_any(t).to(torch.float32)
            yield stem + "linear_attn.dt_bias", dtb if v_perm is None else dtb[v_perm]
            continue
        if suffix == "ssm_conv1d.weight":
            # ggml [K, conv_dim] -> torch (conv_dim, K) -> module [conv_dim, 1, K]
            yield stem + "linear_attn.conv1d.weight", _fix_v_rows(
                _to_bf16(t)
            ).unsqueeze(1).contiguous()
            continue
        if suffix == "ple_conv1d.weight":
            yield stem + "ple.conv1d.weight", _to_bf16(t).unsqueeze(1).contiguous()
            continue
        if suffix == "ffn_gate_inp_shexp.weight":
            yield stem + "mlp.shared_expert_gate.weight", _to_bf16(t).reshape(1, -1)
            continue
        if suffix == "ssm_out.weight" and v_perm is not None:
            # [hidden, value_dim]: reorder the input columns to match the permuted heads.
            key = stem + "linear_attn.out_proj.weight"
            if stem + "linear_attn.out_proj" in packed:
                # gguf_module_types only marks this packed when the heads land on
                # block boundaries, so this holds by construction.
                per_head = _head_block_bytes(n_v * d_v, d_v, t.ggml_type)
                assert per_head is not None, (
                    f"{name}: value heads do not land on {t.ggml_type} block boundaries, "
                    "so the dialect should not have marked out_proj packed"
                )
                yield key, _permute_packed_heads(t.packed(), v_perm, per_head)
            else:
                cols = (v_perm[:, None] * d_v + torch.arange(d_v)).reshape(-1)
                yield key, _to_bf16(t)[:, cols].contiguous()
            continue

        done = False
        for module, slots in _MERGED.items():
            if suffix not in slots:
                continue
            if module == "self_attn.qkv_proj" and not _is_full_attention(layer, interval):
                continue
            if module == "linear_attn.in_proj" and _is_full_attention(layer, interval):
                continue
            i = slots.index(suffix)
            is_packed = stem + module in packed
            val = t.packed() if is_packed else _to_bf16(t)
            # Above the packed/dense split, not inside the dense arm: every one of
            # these is a permutation of whole ROWS, and a block-quantized row is
            # self-contained, so it applies to the packed bytes unchanged. Keeping it
            # on the dense side (which is where qwen3_5_moe can leave it, never packing
            # in_proj) silently skipped it for a checkpoint whose in_proj slots are all
            # packable -- leaving q/k/v/z/beta/alpha in ggml's tiled head order while
            # conv1d, A_log, dt_bias and out_proj were in the kernels'.
            if v_perm is not None and module == "linear_attn.in_proj":
                if suffix == "attn_qkv.weight":
                    val = _fix_v_rows(val)                       # [q | k | v]
                elif suffix == "attn_gate.weight":
                    val = _permute_head_rows(val, v_perm, d_v)   # z, per value head
                else:
                    val = val[v_perm]                            # beta / alpha
            if is_packed:
                yield f"{stem}{module}.weight_{i}", val.contiguous()
            else:
                group = feed(layer, module, slots, suffix, val)
                if group is not None:
                    yield f"{stem}{module}.weight", torch.cat(group, dim=0).contiguous()
            done = True
            break
        if done:
            continue

        for module, (parts, align) in _ROW_CONCAT.items():
            if suffix not in parts:
                continue
            is_packed = stem + module in packed
            group = feed(
                layer, module, parts, suffix,
                t.packed() if is_packed else _to_bf16(t),
            )
            if group is not None:
                rows = sum(g.shape[0] for g in group)
                pad = _pad_rows(rows, align)
                if pad:
                    group.append(
                        _zero_rows(pad, group[0].shape[1]) if is_packed
                        else group[0].new_zeros(pad, group[0].shape[1])
                    )
                yield f"{stem}{module}.weight", torch.cat(group, dim=0).contiguous()
            done = True
            break
        if done:
            continue

        rel = _SUFFIX_MAP.get(suffix)
        if rel is None:
            raise ValueError(f"{name}: unrecognized qwen4exp GGUF tensor")
        module = stem + rel.rsplit(".weight", 1)[0]
        yield stem + rel, (t.packed() if module in packed else _to_bf16(t))

    leftovers = sorted(pending)
    assert not leftovers, f"incomplete fused groups: {leftovers}"


__all__ = ["parse_gguf_config", "gguf_module_types", "iter_gguf_weights"]
