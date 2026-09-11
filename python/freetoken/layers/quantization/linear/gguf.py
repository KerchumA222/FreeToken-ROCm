"""GGUF-packed dense linear: the blocks stay packed and dequantize inside the kernel.

A GGUF checkpoint stores its dense projections quantized, and reading them back as
bf16 costs both the VRAM and the bandwidth: on an RX 6800 the fp16 rocBLAS GEMV of
this model's dense weights ran at 110-220 GB/s against ~500 GB/s for the packed MMVQ
kernel, and the lm_head alone went 9.23 -> 1.09 ms per token.

A fused module can mix ggml types across its slots (Q4_K_M puts Q6_K on attn_v over a
Q4_K body), which is why the scheme carries one type per slot and the weights are one
packed tensor per slot rather than a single concatenated buffer.
"""

from __future__ import annotations

from typing import Any

import torch

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import LinearConfig, LinearKernel, LinearMethod


def _types(cfg: LinearConfig) -> list[int]:
    """The scheme's per-slot ggml type ids, in output order."""
    from freetoken.models.gguf.dequant import GGML_NAME

    by_name = {name: t for t, name in GGML_NAME.items()}
    return [by_name[n] for n in cfg.scheme.weight.elem.split("+")]


class MmvqGgufLinearKernel(LinearKernel):
    """llama.cpp's MMVQ/MMQ pair: q8_1 activations dotted against the packed blocks."""

    name = "mmvq"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.layers.gguf import fused_mul_mat_gguf

        outs = [fused_mul_mat_gguf(x, q, t) for q, t in zip(layer.qweights, layer.gguf_types)]
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        return out + layer.bias.to(out.dtype) if layer.bias is not None else out


@register_method(QuantKind.GGUF, LayerKind.LINEAR)
class GgufLinearMethod(LinearMethod):
    candidates = (MmvqGgufLinearKernel,)

    def create_weights(self, layer: Any) -> None:
        from freetoken.models.gguf.dequant import row_bytes

        g = self.cfg
        types = _types(g)
        if len(types) != len(g.output_sizes):
            raise ValueError(
                f"gguf scheme has {len(types)} types for {len(g.output_sizes)} fused slots"
            )
        layer.gguf_types = types
        layer.qweights = [
            torch.empty(out, row_bytes(g.in_features, t), dtype=torch.uint8)
            for out, t in zip(g.output_sizes, types)
        ]
        # Single-slot layers keep the conventional attribute name so the weight
        # readers and state_dict see `qweight`, matching the other dialects' `weight`.
        if len(layer.qweights) == 1:
            layer.qweight = layer.qweights[0]
