"""Opt-in expert routing remap (``FT_EXPERT_REMAP=<path.pt>``).

A hard table is an integer ``[layers, experts]`` tensor mapping each source expert
onto one destination. A soft table is ``{"ids": [layers, experts, choices],
"w": [layers, experts, choices]}``; each source expert distributes its router
probability across destination experts according to a normalized weight row.

Mapped probability mass is aggregated by destination before top-k, so every token
still selects the configured number of distinct expert IDs.
"""

from __future__ import annotations

import os

import torch

_PATH = os.environ.get("FT_EXPERT_REMAP", "")
_table: tuple[torch.Tensor, torch.Tensor | None, tuple[int, ...]] | None = None


def enabled() -> bool:
    return bool(_PATH)


def _integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def _load(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None, tuple[int, ...]]:
    global _table
    if _table is None or _table[0].device != device:
        obj = torch.load(_PATH, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            if set(obj) != {"ids", "w"}:
                raise ValueError("soft expert remap must contain exactly 'ids' and 'w'")
            raw_ids = torch.as_tensor(obj["ids"])
            raw_weights = torch.as_tensor(obj["w"])
            if not _integer_dtype(raw_ids.dtype):
                raise ValueError("expert remap ids must use an integer dtype")
            if (
                raw_ids.ndim != 3
                or raw_weights.shape != raw_ids.shape
                or any(size == 0 for size in raw_ids.shape)
            ):
                raise ValueError("soft remap ids and weights must share nonempty shape [layers, experts, choices]")
            ids = raw_ids.to(dtype=torch.int64)
            weights = raw_weights.to(dtype=torch.float32)
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("soft remap weights must be finite and nonnegative")
            if not torch.allclose(
                weights.sum(dim=-1),
                torch.ones_like(weights[..., 0]),
                rtol=0,
                atol=1e-5,
            ):
                raise ValueError("soft remap weights must sum to 1 for each source expert")
        else:
            raw_ids = torch.as_tensor(obj)
            if not _integer_dtype(raw_ids.dtype) or raw_ids.ndim != 2:
                raise ValueError("hard expert remap must be an integer [layers, experts] tensor")
            ids, weights = raw_ids.to(dtype=torch.int64), None

        if ids.shape[0] == 0 or ids.shape[1] == 0:
            raise ValueError("expert remap table must contain layers and experts")
        if (ids < 0).any() or (ids >= ids.shape[1]).any():
            raise ValueError("expert remap destination IDs must be in [0, experts)")
        if weights is None:
            active_counts = tuple(torch.unique(ids[layer]).numel() for layer in range(ids.shape[0]))
        else:
            active_counts = tuple(
                torch.unique(ids[layer][weights[layer] > 0]).numel()
                for layer in range(ids.shape[0])
            )
        _table = (
            ids.to(device=device),
            None if weights is None else weights.to(device=device),
            active_counts,
        )
    return _table


def route(
    layer_id: int,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    selection_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Route with probabilities aggregated through the table, or return ``None`` for later layers."""
    ids_table, weight_table, active_counts = _load(gating_output.device)
    if layer_id < 0:
        raise ValueError("expert remap layer ID must be nonnegative")
    if layer_id >= ids_table.shape[0]:
        return None
    num_rows, num_experts = gating_output.shape
    if ids_table.shape[1] != num_experts:
        raise ValueError(
            f"expert remap has {ids_table.shape[1]} experts, router has {num_experts}"
        )
    if not 1 <= topk <= num_experts:
        raise ValueError(f"top-k {topk} must be in [1, {num_experts}]")

    layer_ids = ids_table[layer_id]
    if weight_table is None:
        destinations = layer_ids.unsqueeze(0).expand(num_rows, -1)
        mapped = torch.zeros_like(gating_output, dtype=torch.float32)
        mapped.scatter_add_(1, destinations, torch.softmax(gating_output.float(), dim=-1))
    else:
        layer_weights = weight_table[layer_id]
        destinations = layer_ids.unsqueeze(0).expand(num_rows, -1, -1).reshape(num_rows, -1)
        contributions = (
            torch.softmax(gating_output.float(), dim=-1).unsqueeze(-1) * layer_weights.unsqueeze(0)
        ).reshape(num_rows, -1)
        mapped = torch.zeros_like(gating_output, dtype=torch.float32)
        mapped.scatter_add_(1, destinations, contributions)
    available = active_counts[layer_id]
    if available < topk:
        raise ValueError(
            f"expert remap layer {layer_id} has only {available} active destinations for top-k {topk}"
        )

    if selection_bias is None:
        selection_scores = mapped
    else:
        if selection_bias.shape != (num_experts,):
            raise ValueError("selection bias must have one value per destination expert")
        selection_scores = mapped.clamp_min(torch.finfo(mapped.dtype).tiny).log()
        selection_scores = selection_scores + selection_bias.float()
        selection_scores = selection_scores.masked_fill(mapped == 0, -torch.inf)
    selected = torch.argsort(selection_scores, dim=-1, descending=True, stable=True)[:, :topk]
    weights = mapped.gather(1, selected)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    return weights.contiguous(), selected.to(torch.int32).contiguous()
