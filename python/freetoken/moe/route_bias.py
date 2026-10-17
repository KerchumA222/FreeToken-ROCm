"""Selection-only bias toward experts resident in the GPU cache.

``FT_ROUTE_BIAS`` applies to decode-sized batches (up to eight rows);
``FT_ROUTE_BIAS_PREFILL`` applies to larger batches. Both are zero by default.
"""

from __future__ import annotations

import os

import torch

_ROUTE_BIAS = float(os.environ.get("FT_ROUTE_BIAS", "0") or 0)
_PREFILL_BIAS = float(os.environ.get("FT_ROUTE_BIAS_PREFILL", "0") or 0)
_DECODE_MAX_ROWS = 8


def selection_bias(experts, num_rows: int, device: torch.device) -> torch.Tensor | None:
    """Return per-expert score offsets for the current layer, or ``None`` when disabled."""
    margin = _ROUTE_BIAS if num_rows <= _DECODE_MAX_ROWS else _PREFILL_BIAS
    cache = getattr(experts, "offload_cache", None)
    if not margin or cache is None:
        return None

    slot_for_id = cache.slot_for_id
    layer_id = experts.layer_id
    if layer_id < 0 or layer_id >= slot_for_id.shape[0]:
        return None
    resident = slot_for_id[layer_id] >= 0
    return resident.to(device=device, dtype=torch.float32) * margin
