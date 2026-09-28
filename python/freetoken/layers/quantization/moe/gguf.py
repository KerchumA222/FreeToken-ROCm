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


# Row bytes of the widest layer per bank, when a checkpoint's expert layers disagree on type
# (Unsloth's dynamic quants upgrade a few sensitive layers). The offload banks and slots
# are sized for this; a narrower layer's expert sits at the start of its slot and its
# kernels step between experts by the bank's stride. Empty when every layer agrees.
_ROW_BYTES_MAX: dict[str, int] = {}


def set_bank_row_bytes(gate_up: int, down: int) -> None:
    """Record the widest expert rows per bank (see ``_ROW_BYTES_MAX``)."""
    _ROW_BYTES_MAX.update(gate_up=int(gate_up), down=int(down))


def _typed_view(bank: torch.Tensor, rows: int, row_bytes: int) -> torch.Tensor:
    """``bank`` ([slots, rows, >= row_bytes] uint8) as this layer's [slots, rows, row_bytes]
    experts: each expert is a contiguous run at the start of its slot."""
    if bank.shape[-1] == row_bytes:
        return bank
    return torch.as_strided(bank, (bank.shape[0], rows, row_bytes), (bank.stride(0), row_bytes, 1))


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
        gu = max(row_bytes(cfg.hidden, gu_type), _ROW_BYTES_MAX.get("gate_up", 0))
        dn = max(row_bytes(i, dn_type), _ROW_BYTES_MAX.get("down", 0))
        return {
            "gate_up": BankSpec((2 * i, gu), torch.uint8),
            "down": BankSpec((cfg.hidden, dn), torch.uint8),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        # The reader hands packed rows already in bank layout (see the models' GGUF
        # expert source loaders), so packing is a copy.
        # A layer narrower than the bank (per-layer types) lands at the start of each
        # expert's slot; see _typed_view.
        for role in ("gate_up", "down"):
            src, dst = pieces[role], out[role]
            if src.shape[-1] == dst.shape[-1]:
                dst.copy_(src)
            else:
                n = src[0].numel()
                dst.view(dst.shape[0], -1)[:, :n].copy_(src.reshape(src.shape[0], -1))
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.moe.fused_q4_0 import fused_experts_gguf

        from freetoken.models.gguf.dequant import row_bytes

        cfg = layer.quant_method.cfg
        gu_type, dn_type = _bank_types(cfg)
        i = cfg.local_intermediate
        gate_up = _typed_view(view.tensors["gate_up"], 2 * i, row_bytes(cfg.hidden, gu_type))
        down = _typed_view(view.tensors["down"], cfg.hidden, row_bytes(i, dn_type))
        return fused_experts_gguf(
            x, gate_up, down, topk_weights, topk_ids,
            layer.activation, gate_up_type=gu_type, down_type=dn_type,
        )


@register_method(QuantKind.GGUF, LayerKind.MOE)
class GgufMoEMethod(MoEMethod):
    candidates = (MmvqGgufMoEKernel,)

    @property
    def ggml_types(self) -> tuple[int, int]:
        """Packed GGML type ids in (gate_up, down) bank order."""
        return _bank_types(self.cfg)

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
