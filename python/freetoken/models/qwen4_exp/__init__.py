"""Qwen3.8-Flash-Next (model_type qwen4_exp), served text-only.

48 decoder layers on hc_count=4 hyper-connection residual streams R [T, 4*hidden]:
embed -> repeat(1, 4) -> [PLE at zero-based layer 1] -> per layer attn_hc.mix -> (GDN | QSA) -> attn_hc.combine -> mlp_hc.mix -> MoE -> mlp_hc.combine -> top-level mixer.mix -> lm_head.
Layer contract: forward(R [T, 4*hidden], batch) -> R' [T, 4*hidden].

Contracts shared across modules (do not rename):
- The PLE dilated-conv left context lives on the LinearStatePool slots as the declared slot state ``ple_conv`` (config.ple_slot_states -> ModelConfig.slot_states), read back with ``pool.slot_state("ple_conv", layer_id)``; same slot / COW / snapshot lifecycle as conv_states / recurrent_states.
- kvcache.qsa_pool.QSAKVCache(MHAKVCache): ``cmp_k_cache(slot) -> [rows, index_head_dim]`` (compressed index keys, row = kv slot // index_ratio), ``pending_ring(slot) -> [num_req_slots, ring_capacity, index_head_dim]`` (per-request pre-RoPE index-k tail indexed by table_idx, never cleared), ``cmp_scratch_base`` (int, first scratch row for non-closing decode writes). ``slot`` is the sparse layer's order in the attention backend.
"""

from .config import parse_config
from .gguf import (
    gguf_module_types,
    iter_gguf_weights,
    parse_gguf_config,
)
from .model import Qwen4ExpForCausalLM
from .weight import (
    ftw_side_files,
    nvfp4_expert_spec,
    iter_weights,
    load_ple_table,
)

# Official FP8 checkpoints share qwen3_5_moe's block-fp8 expert layout (same
# model.language_model.layers.* keys), so reuse its expert reader.
from freetoken.models.qwen3_5_moe.weight import iter_expert_pieces as _iter_expert_pieces_hf

# The GGUF routed experts are stored under the same ``blk.N.ffn_*_exps.weight`` names
# and read through the same per-layer banks as qwen3_5_moe's, and those readers key on
# nothing model-specific (the tensor names, the config's dims, and the bank types off
# the quant dialect), so they are reused rather than restated.
from freetoken.models.qwen3_5_moe.gguf import (
    dummy_q4_0_expert_sources,
    iter_gguf_expert_pieces,
    load_q4_0_expert_sources,
)


def iter_expert_pieces(model_path, config, kind, **kwargs):
    """Routed-expert pieces for every storage form this family reads.

    GGUF keeps its experts in llama.cpp's block layout, which upstream's readers do
    not cover; every other kind falls through to them unchanged.
    """
    from freetoken.layers.quantization import QuantKind

    if kind is QuantKind.GGUF:
        return iter_gguf_expert_pieces(model_path, config, **kwargs)
    return _iter_expert_pieces_hf(model_path, config, kind, **kwargs)

__all__ = [
    "ftw_side_files",
    "nvfp4_expert_spec",
    "Qwen4ExpForCausalLM",
    "iter_weights",
    "load_ple_table",
    "parse_config",
    "iter_expert_pieces",
    # GGUF adapter (fork-only: upstream has no qwen4_exp GGUF path)
    "parse_gguf_config",
    "gguf_module_types",
    "iter_gguf_weights",
    "load_q4_0_expert_sources",
    "dummy_q4_0_expert_sources",
]
