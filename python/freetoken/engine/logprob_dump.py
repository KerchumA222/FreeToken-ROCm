"""FT_LOGPROB_DUMP=<path.jsonl>: append each forward's next-token distribution, for offline
KL studies (e.g. FT_ROUTE_BIAS against unbiased routing).

One line per request per forward: ``uid``, ``len`` (the request's device length, i.e. the
position the sampled token takes), ``tok`` (the sampled token) and the top
``FT_LOGPROB_TOPK`` (default 256) token ids with their log-probabilities. Eval only: it
syncs the stream every step. Speculative verify batches are skipped.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from freetoken.core import Batch

_PATH = os.environ.get("FT_LOGPROB_DUMP", "")
_TOPK = int(os.environ.get("FT_LOGPROB_TOPK", "256"))


def enabled() -> bool:
    return bool(_PATH)


def record(batch: Batch, logits: torch.Tensor, next_tokens: torch.Tensor) -> None:
    if batch.is_spec_verify:
        return
    rows = batch.size
    top = logits[:rows].float().log_softmax(-1).topk(_TOPK, dim=-1)
    ids, lps, toks = top.indices.tolist(), top.values.tolist(), next_tokens[:rows].tolist()
    with open(_PATH, "a") as f:
        for req, i, lp, tok in zip(batch.reqs, ids, lps, toks):
            f.write(json.dumps({"uid": req.uid, "len": req.device_len, "tok": tok,
                                "ids": i, "lp": [round(v, 5) for v in lp]}) + "\n")
