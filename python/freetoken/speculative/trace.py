"""Diagnostic: log the target's top logits for every sampled row, keyed by the sequence
index each row predicts.

Enabled by ``FT_SPEC_TRACE=<path.jsonl>``. Comparing a speculative run against a plain
run of the same greedy request finds the first index the two commit differently, and the
margin between the top two logits there tells a near-tie flip (the verify forward's
extend kernels rounding differently from the decode kernels) from a bookkeeping bug
(a confident, wrong token). Forces a device sync per forward; never enable in serving.
"""

from __future__ import annotations

import json
import os

import torch

_PATH = os.environ.get("FT_SPEC_TRACE", "")
_TOPK = 5
_fh = None


def enabled() -> bool:
    return bool(_PATH)


def record(batch, logits: torch.Tensor, rows: int) -> None:
    global _fh
    if _fh is None:
        _fh = open(_PATH, "a", buffering=1)
    vals, ids = logits[:rows].float().topk(_TOPK, dim=-1)
    vals, ids = vals.cpu().tolist(), ids.cpu().tolist()
    kind = "verify" if batch.is_spec_verify else ("prefill" if batch.is_prefill else "decode")
    off = 0
    for i, req in enumerate(batch.reqs):
        c, d = req.cached_len, req.device_len
        fed = req.input_ids[c:d].tolist()
        if batch.is_spec_verify:
            preds = [(off + j, c + 1 + j) for j in range(d - c)]
            off += d - c
        else:
            preds = [(i, d)]
        for row, idx in preds:
            _fh.write(json.dumps({
                "uid": req.uid, "kind": kind, "fed_from": c, "fed": fed,
                "idx": idx, "ids": ids[row], "vals": vals[row],
            }) + "\n")


__all__ = ["enabled", "record"]
