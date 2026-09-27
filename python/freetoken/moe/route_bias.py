"""Experimental routing bias toward experts already in the GPU cache (offload MoE).

FT_ROUTE_BIAS=<logits>: at decode sizes, a GPU-resident expert wins a top-k slot over a
non-resident one whose router logit is within this margin. Selection only: the chosen
experts keep their own router weights. It routes tokens to experts the router ranked
lower, so it trades quality for fewer expert loads.

FT_ROUTE_BIAS_PREFILL=<logits>: the same bias for larger (prefill / extend) batches. Off by
default: it changes how every prompt token is processed, not just the generated ones.

FT_ROUTE_BIAS_TRACE=<path.pt>: per decode step and layer, the router's own top-k and the
experts actually used, as int16 [rows, 2, k]. Eager only; it syncs.
"""

from __future__ import annotations

import os

import torch

_ROUTE_BIAS = float(os.environ.get("FT_ROUTE_BIAS", "0") or 0)
_PREFILL_BIAS = float(os.environ.get("FT_ROUTE_BIAS_PREFILL", "0") or 0)
_MAX_ROWS = 8  # decode and speculative verify batches
_TRACE = os.environ.get("FT_ROUTE_BIAS_TRACE", "")
_trace_rows: list = []


def bias_to_resident(
    logits: torch.Tensor, resident: torch.Tensor, top_k: int, bias: float | None = None
) -> torch.Tensor:
    """Router logits whose top-k is the top-k of ``logits + bias * resident``, with every
    other expert pushed far down, so the downstream softmax/top-k/renormalize weights the
    chosen experts by their original logits. Device-only (graph-safe)."""
    biased = logits + resident.to(logits.dtype) * (_ROUTE_BIAS if bias is None else bias)
    chosen = biased.topk(top_k, dim=-1).indices
    out = torch.full_like(logits, -1e4)
    return out.scatter_(-1, chosen, logits.gather(-1, chosen))


def _record(experts, raw: torch.Tensor, used: torch.Tensor) -> None:
    from freetoken.core import get_global_ctx

    if not get_global_ctx().batch.is_decode or torch.cuda.is_current_stream_capturing():
        return
    k = experts.top_k
    ids = torch.stack([raw.topk(k, dim=-1).indices, used.topk(k, dim=-1).indices], dim=1)
    _trace_rows.append((experts.layer_id, ids.to("cpu", torch.int16)))
    if len(_trace_rows) % 4800 == 0:
        torch.save(_trace_rows, _TRACE)


def apply(experts, router_logits: torch.Tensor) -> torch.Tensor:
    """The router logits ``experts`` should route with: biased toward its GPU-resident
    experts by FT_ROUTE_BIAS (decode-sized batches) or FT_ROUTE_BIAS_PREFILL (larger
    ones), else unchanged."""
    used = router_logits
    bias = _ROUTE_BIAS if router_logits.shape[0] <= _MAX_ROWS else _PREFILL_BIAS
    if bias:
        cache = getattr(experts, "offload_cache", None)
        if cache is not None:
            resident = cache.slot_for_id[experts.layer_id] >= 0
            used = bias_to_resident(router_logits, resident, experts.top_k, bias)
    if _TRACE:
        _record(experts, router_logits, used)
    return used
