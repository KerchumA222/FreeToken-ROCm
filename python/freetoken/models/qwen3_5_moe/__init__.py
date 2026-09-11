from .config import parse_config
from .gguf import (
    gguf_module_types,
    iter_gguf_expert_pieces,
    dummy_q4_0_expert_sources,
    iter_gguf_weights,
    load_q4_0_expert_sources,
    parse_gguf_config,
)
from .model import Qwen3_5MoEForCausalLM
from .weight import (
    iter_expert_pieces as _iter_expert_pieces_hf,
    iter_weights,
    iter_weights_parallel,
    nvfp4_expert_spec,
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
    "Qwen3_5MoEForCausalLM",
    "parse_config",
    "iter_weights",
    "iter_weights_parallel",
    "iter_expert_pieces",
    "nvfp4_expert_spec",
    # GGUF adapter (fork-only: upstream has no qwen3_5_moe GGUF path)
    "parse_gguf_config",
    "gguf_module_types",
    "iter_gguf_weights",
    "load_q4_0_expert_sources",
    "dummy_q4_0_expert_sources",
]
