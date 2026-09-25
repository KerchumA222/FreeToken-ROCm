"""Activations that already carry their q8_1 blocks.

A packed GGUF GEMV (MMVQ) quantizes its fp16 input to q8_1 first: one extra launch per
matmul, 531 of a Qwen3.8-Flash-Next decode token's ~2,800 kernels. Two ways to skip it:

- a module whose fused projection has several packed runs quantizes its input once;
- a producer kernel whose output only feeds packed GEMVs writes the blocks itself
  (``kernel/triton/q8_1.py``) and attaches them to its fp16 output with :func:`attach`.

The blocks ride on the tensor object, so a view or any other tensor never picks them up;
outside inference mode they also remember its version, so an in-place change drops them.
"""

from __future__ import annotations

import os
from typing import Any

import torch

_ATTR = "_ft_q8"
# FT_Q8_FUSE=0: every packed GEMV quantizes its own input again (A/B switch).
ENABLED = os.environ.get("FT_Q8_FUSE", "1") != "0"


def _version(x: torch.Tensor) -> int | None:
    # Inference-mode tensors keep no version counter. Producers attach only to their own
    # fresh outputs, and every consumer reads before anything writes those in place.
    return None if x.is_inference() else x._version


def attach(x: torch.Tensor, q8: torch.Tensor | None) -> torch.Tensor:
    if q8 is not None:
        setattr(x, _ATTR, (_version(x), q8))
    return x


def attached(x: torch.Tensor) -> torch.Tensor | None:
    got = getattr(x, _ATTR, None)
    return got[1] if got is not None and got[0] == _version(x) else None


def wants_q8(layer: Any, rows: int) -> bool:
    """Whether ``layer`` will multiply a ``rows``-row input with MMVQ, the only consumer
    of attached blocks (larger batches take MMQ or the Triton GEMM)."""
    from freetoken.layers.gguf import mmvq_rows_ok

    types = getattr(layer, "gguf_types", None)
    return ENABLED and bool(types) and mmvq_rows_ok(rows) and all(_mmvq_type(t) for t in types)


def _mmvq_type(t) -> bool:
    from freetoken.layers.gguf import is_mmvq_type

    return is_mmvq_type(t)


__all__ = ["attach", "attached", "wants_q8"]
