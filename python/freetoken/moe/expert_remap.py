"""Experimental expert pruning by routing remap (FT_EXPERT_REMAP=<path.pt>).

The file holds either
  * an int tensor [num_layers, num_experts] (hard remap): row ``l`` maps each expert id the
    router picks in MoE layer ``l`` to the expert that serves it -- a pruned expert maps to
    a kept one, kept experts to themselves; or
  * a dict {"ids": int [L, E, m], "w": float [L, E, m]} (soft remap): each pick of expert
    ``e`` is split across ``ids[l, e]`` in the proportions ``w[l, e]`` (rows sum to 1). Kept
    experts are ``[e, e, ...]`` with weights ``[1, 0, ...]``, so every token's routing widens
    from k to k*m columns and the zero-weight duplicates cost compute, not loads.

The router and the expert weights are unchanged, so this measures what dropping experts
costs without writing a smaller checkpoint. Layers past the table (an MTP draft head's)
are left alone.
"""

from __future__ import annotations

import os

import torch

_PATH = os.environ.get("FT_EXPERT_REMAP", "")
_table = None


def enabled() -> bool:
    return bool(_PATH)


def _load(device: torch.device):
    global _table
    if _table is None or _table[0].device != device:
        obj = torch.load(_PATH, map_location="cpu")
        if isinstance(obj, dict):
            _table = (torch.as_tensor(obj["ids"], dtype=torch.int64).to(device),
                      torch.as_tensor(obj["w"], dtype=torch.float32).to(device))
        else:
            _table = (torch.as_tensor(obj, dtype=torch.int64).to(device), None)
    return _table


def apply(layer_id: int, topk_weights: torch.Tensor, topk_ids: torch.Tensor):
    """(weights, ids) with pruned experts replaced by the one(s) that absorbed them."""
    ids_table, w_table = _load(topk_ids.device)
    if layer_id >= ids_table.shape[0]:
        return topk_weights, topk_ids
    flat = topk_ids.reshape(-1).long()
    if w_table is None:
        ids = ids_table[layer_id].index_select(0, flat).view_as(topk_ids).to(topk_ids.dtype)
        return topk_weights, ids
    rows, k = topk_ids.shape
    m = ids_table.shape[-1]
    ids = ids_table[layer_id].index_select(0, flat).view(rows, k * m).to(topk_ids.dtype)
    frac = w_table[layer_id].index_select(0, flat).view(rows, k, m)
    weights = (topk_weights.unsqueeze(-1).float() * frac).view(rows, k * m).to(topk_weights.dtype)
    return weights, ids.contiguous()
