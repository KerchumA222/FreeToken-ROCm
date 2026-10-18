"""Selection-only bias toward experts that are cheap to run.

``FT_ROUTE_BIAS`` favours experts resident in the GPU cache for decode-sized batches (up to eight rows);
``FT_ROUTE_BIAS_PREFILL`` does the same for larger batches. ``FT_ROUTE_BIAS_HOST`` favours experts held in
the disk tier's pinned host pool (a miss there is a disk read) for decode-sized batches. All are zero by
default. The margins add to the scores the router selects on, so their scale is the router's: logits for
softmax routers, sigmoid probabilities for sigmoid routers. Routing weights are never biased.
"""

from __future__ import annotations

import os

import torch

_ROUTE_BIAS = float(os.environ.get("FT_ROUTE_BIAS", "0") or 0)
_PREFILL_BIAS = float(os.environ.get("FT_ROUTE_BIAS_PREFILL", "0") or 0)
_HOST_BIAS = float(os.environ.get("FT_ROUTE_BIAS_HOST", "0") or 0)
_DECODE_MAX_ROWS = 8


def selection_bias(experts, num_rows: int, device: torch.device) -> torch.Tensor | None:
    """Return per-expert score offsets for the current layer, or ``None`` when disabled."""
    decode = num_rows <= _DECODE_MAX_ROWS
    margin = _ROUTE_BIAS if decode else _PREFILL_BIAS
    host_margin = _HOST_BIAS if decode else 0.0
    cache = getattr(experts, "offload_cache", None)
    if (not margin and not host_margin) or cache is None:
        return None

    layer_id = experts.layer_id
    slot_for_id = cache.slot_for_id
    if layer_id < 0 or layer_id >= slot_for_id.shape[0]:
        return None
    bias = None
    if margin:
        bias = (slot_for_id[layer_id] >= 0).to(device=device, dtype=torch.float32) * margin
    host_tier = getattr(cache, "host_tier", None)
    if host_margin and host_tier is not None:
        host = host_tier.resident[layer_id].to(device=device, dtype=torch.float32) * host_margin
        bias = host if bias is None else bias + host
    return bias
