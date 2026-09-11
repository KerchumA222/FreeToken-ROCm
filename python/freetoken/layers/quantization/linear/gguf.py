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


def _is_packed(ggml_type: int) -> bool:
    """Whether this slot's bytes stay in their blocks, or arrive dense."""
    from freetoken.layers.gguf import _UNQUANTIZED

    return ggml_type not in _UNQUANTIZED


class MmvqGgufLinearKernel(LinearKernel):
    """llama.cpp's MMVQ/MMQ pair: q8_1 activations dotted against the packed blocks."""

    name = "mmvq"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.layers.gguf import fused_mul_mat_gguf

        # A fused module may mix packed and dense slots; the dense ones are tiny
        # (0.69% of the bytes in Qwen4-Exp) but must still be multiplied as dense.
        outs = [
            fused_mul_mat_gguf(x, getattr(layer, n), t) if _is_packed(t)
            else torch.nn.functional.linear(x, getattr(layer, n))
            for n, t in zip(layer.gguf_slots, layer.gguf_types)
        ]
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        bias = getattr(layer, "bias", None)
        return out + bias.to(out.dtype) if bias is not None else out


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
        # One named tensor per slot rather than one concatenated buffer: slots of a
        # fused module can carry different ggml types, so their row_bytes differ and
        # there is no single [out, row_bytes] shape covering them. Named because
        # BaseOP's state_dict walks attributes -- a list would be invisible to the
        # weight readers. A plain linear keeps the conventional single `qweight`,
        # matching the other dialects' `weight`.
        # Named `weight` like every other dialect's tensor rather than `qweight`:
        # the element type is the scheme's business, and ParallelLMHead pre-creates a
        # bf16 `weight` that create_weights is expected to replace -- a differently
        # named tensor would leave that stale one behind for the weight reader to
        # demand.
        layer.gguf_slots = (
            ("weight",) if len(types) == 1
            else tuple(f"weight_{i}" for i in range(len(types)))
        )
        # Modules are built under torch_dtype(config.dtype), so the default dtype is
        # the model's -- which on a GPU without bf16 hardware is fp16, not bf16.
        dense_dtype = torch.get_default_dtype()
        for slot, out, t in zip(layer.gguf_slots, g.output_sizes, types):
            setattr(layer, slot, (
                torch.empty(out, row_bytes(g.in_features, t), dtype=torch.uint8)
                if _is_packed(t) else torch.empty(out, g.in_features, dtype=dense_dtype)
            ))
