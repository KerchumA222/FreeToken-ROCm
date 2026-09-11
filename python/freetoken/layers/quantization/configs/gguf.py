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
        # checkpoint tensor name -> ggml type name, for the types we can serve packed.
        # A tensor absent from this map is served bf16 (F32/F16 tensors, and any type
        # the vendored kernels do not implement -- the shim filters those out).
        self.types: dict[str, str] = dict(q.get("types") or {})

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        ggml_type = self.types.get(name)
        return gguf_scheme((ggml_type,)) if ggml_type else None

    def scheme_for(self, prefix: str) -> QuantScheme | None:
        """One scheme per module, carrying a type per slot in checkpoint order.

        The base implementation requires every name of a fused module to agree; for a
        k-quant checkpoint they routinely do not, and that mix is exactly what
        ``GgufColSplits`` exists to serve. Partly-quantized fusions still fall back to
        bf16: there is no kernel for half a packed module.
        """
        if prefix in self._schemes:
            return self._schemes[prefix]
        names = self.name_map.to_checkpoint(prefix)
        types = [None if self.unquantized(n) else self.types.get(n) for n in names]
        scheme = gguf_scheme(types) if types and all(t is not None for t in types) else None
        self._schemes[prefix] = scheme
        return scheme


def gguf_quantization_config(types: dict[str, str]) -> dict[str, Any]:
    """The synthetic ``quantization_config`` a GGUF shim hands to ``QuantConfig.from_hf``."""
    return {"quant_method": GgufConfig.dialect, "types": types}


__all__ = ["GgufConfig", "gguf_quantization_config"]
