"""Experimental expert pruning by routing remap (FT_EXPERT_REMAP=<path.pt>).

The file holds an int tensor [num_layers, num_experts]: row ``l`` maps each expert id the
router picks in MoE layer ``l`` to the expert that serves it. A pruned expert maps to a
kept one; kept experts map to themselves. The router and its weights are unchanged, so a
token whose picks collide on one expert gets that expert's output at the summed weight.

This measures what dropping experts costs without writing a smaller checkpoint. Layers
past the table (an MTP draft head's) are left alone.
"""

from __future__ import annotations

import os

import torch

_PATH = os.environ.get("FT_EXPERT_REMAP", "")
_table: torch.Tensor | None = None


def enabled() -> bool:
    return bool(_PATH)


def _load(device: torch.device) -> torch.Tensor:
    global _table
    if _table is None or _table.device != device:
        table = torch.load(_PATH, map_location="cpu")
        _table = torch.as_tensor(table, dtype=torch.int64).to(device)
    return _table


def apply(layer_id: int, topk_ids: torch.Tensor) -> torch.Tensor:
    """``topk_ids`` with pruned experts replaced by the ones that absorbed them."""
    table = _load(topk_ids.device)
    if layer_id >= table.shape[0]:
        return topk_ids
    return table[layer_id].index_select(0, topk_ids.reshape(-1).long()).view_as(topk_ids).to(topk_ids.dtype)
