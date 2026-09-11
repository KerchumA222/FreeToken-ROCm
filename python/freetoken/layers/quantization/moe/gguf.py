"""GGUF-packed experts: llama.cpp's block layout, dequantized inside the MoE kernel.

The banks hold the checkpoint's packed block bytes per output row and never
materialize a bf16 copy -- the same property the dense GGUF linear has, which is what
lets a model whose experts do not fit in VRAM stream them as their on-disk bytes.

A k-quant checkpoint gives gate_up and down different types (Q4_K_M mixes Q4_K and
Q6_K), so the scheme carries one per bank in that order.
"""

from __future__ import annotations

import torch

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import BankSpec, ExpertView, MoEConfig, MoEKernel, MoEMethod


def _bank_types(cfg: MoEConfig) -> tuple[int, int]:
    """(gate_up, down) ggml type ids from the scheme; one type means both banks share it."""
    from freetoken.models.gguf.dequant import GGML_NAME

    by_name = {name: t for t, name in GGML_NAME.items()}
    names = cfg.scheme.weight.elem.split("+")
    if len(names) == 1:
        names = names * 2
    if len(names) != 2:
        raise ValueError(f"gguf expert scheme needs 1 or 2 types, got {names}")
    return by_name[names[0]], by_name[names[1]]


class MmvqGgufMoEKernel(MoEKernel):
    """The borrowed ggml grouped GEMV/GEMM over packed expert rows."""

    name = "mmvq"
    # the banks are CPU-readable as-is, so the CPU executor serves them unchanged
    cpu_format = "q4_0"

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        from freetoken.models.gguf.dequant import row_bytes

        gu_type, dn_type = _bank_types(cfg)
        i = cfg.local_intermediate
        return {
            "gate_up": BankSpec((2 * i, row_bytes(cfg.hidden, gu_type)), torch.uint8),
            "down": BankSpec((cfg.hidden, row_bytes(i, dn_type)), torch.uint8),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        # The reader hands packed rows already in bank layout (see the models' GGUF
        # expert source loaders), so packing is a copy.
        out["gate_up"].copy_(pieces["gate_up"])
        out["down"].copy_(pieces["down"])
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.moe.fused_q4_0 import fused_experts_gguf

        gu_type, dn_type = _bank_types(layer.quant_method.cfg)
        return fused_experts_gguf(
            x, view.tensors["gate_up"], view.tensors["down"], topk_weights, topk_ids,
            layer.activation, gate_up_type=gu_type, down_type=dn_type,
        )


@register_method(QuantKind.GGUF, LayerKind.MOE)
class GgufMoEMethod(MoEMethod):
    candidates = (MmvqGgufMoEKernel,)

    def create_weights(self, layer) -> None:
        from freetoken.models.gguf.dequant import row_bytes

        g = self.cfg
        gu_type, dn_type = _bank_types(g)
        i = g.local_intermediate
        layer.gate_up_proj = torch.empty(
            g.num_experts, 2 * i, row_bytes(g.hidden, gu_type), dtype=torch.uint8
        )
        layer.down_proj = torch.empty(
            g.num_experts, g.hidden, row_bytes(i, dn_type), dtype=torch.uint8
        )

    def resident_view(self, layer) -> ExpertView:
        return ExpertView({"gate_up": layer.gate_up_proj, "down": layer.down_proj})
