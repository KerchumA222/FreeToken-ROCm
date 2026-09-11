"""GGUF: the block-quantized layout llama.cpp writes, read in place.

Unlike the other dialects this one is not announced by a ``quantization_config`` in
an HF config -- a GGUF file carries its types per tensor, in the tensor table. The
GGUF config shims synthesize ``{"quant_method": "gguf", "types": {...}}`` from that
table so the checkpoint reaches the same ``QuantConfig.from_hf`` seam as every other
family, and the dialect answers scheme questions straight out of the map.

Two things are specific to GGUF and shape the code below:

* **Scales live inside the block**, not in a sibling tensor. So a scheme has the one
  ``qweight`` role, no ``weight_scale``, and ``STORAGE`` is a single entry.
* **A fused module's slots can carry different types.** Q4_K_M puts Q6_K on attn_v
  and ffn_down over a Q4_K body, so ``self_attn.qkv_proj`` is genuinely three types.
  The base ``scheme_for`` rejects that mix; here it is the normal case, so this
  dialect overrides it to carry one type per slot in checkpoint order instead. A
  module whose slots are only *partly* quantized still resolves to None -- half a
  packed fusion has no kernel.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..names import NameMap
from ..registry import register_dialect
from ..scheme import QuantKind, QuantScheme, gguf_scheme
from .base import QuantConfig, Stored


@register_dialect
class GgufConfig(QuantConfig):
    """Per-tensor ggml types from the GGUF tensor table."""

    dialect = "gguf"

    # One role, one tensor: the packed [rows, row_bytes] blocks.
    STORAGE: ClassVar[dict[QuantKind, dict[str, str | Stored]]] = {
        QuantKind.GGUF: {"qweight": "qweight"},
    }

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map: NameMap | None = None,
                 unquantized: tuple[str, ...] = ()):
        super().__init__(name_map, unquantized)
        # FreeToken module prefix -> the ggml type name per fused slot, in output order.
        #
        # Keyed on the *module* rather than the checkpoint tensor, which is the one
        # place this dialect departs from the others. A NameMap translates attribute
        # paths into checkpoint names within one namespace; ggml's names are a
        # different namespace entirely (``blk.0.attn_q.weight`` for
        # ``model.layers.0.self_attn.qkv_proj``), and the per-family GGUF adapters
        # already own that translation. Duplicating it as NameMap rules would put the
        # same knowledge in two places and let them drift.
        self.module_types: dict[str, tuple[str, ...]] = {
            k: tuple(v) for k, v in (q.get("module_types") or {}).items()
        }

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        types = self.module_types.get(name)
        return gguf_scheme(types) if types else None

    def scheme_for(self, prefix: str) -> QuantScheme | None:
        """The module's scheme, carrying one ggml type per fused slot in output order.

        A k-quant checkpoint mixes types across a fusion's slots (Q4_K_M puts Q6_K on
        attn_v and ffn_down over a Q4_K body), which the base ``scheme_for`` treats as
        an error; here it is the normal case. A slot may also be *unquantized*, which
        the method serves dense alongside its packed siblings -- see ``gguf_scheme``
        for why that beats refusing the whole module.
        """
        if prefix not in self._schemes:
            self._schemes[prefix] = (
                None if self.unquantized(prefix) else self.scheme_for_name(prefix)
            )
        return self._schemes[prefix]


def gguf_quantization_config(module_types: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """The synthetic ``quantization_config`` a GGUF adapter hands to ``QuantConfig.from_hf``.

    ``module_types`` maps a FreeToken module prefix to its ggml type per fused slot.
    """
    return {"quant_method": GgufConfig.dialect, "module_types": module_types}


__all__ = ["GgufConfig", "gguf_quantization_config"]
