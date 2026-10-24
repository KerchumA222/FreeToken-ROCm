"""Per-shape choice among equivalent prefill GEMM formulations (FREETOKEN_PREFILL_GEMM_PICK).

rocBLAS on RDNA2 picks very different Tensile tiles for ``x @ W^T`` and ``(W @ x^T)^T``: a
``[4096, 10240] @ [10240, 336]`` hyper-connection projection runs 11.2 ms one way and 2.3 ms
the other, and the faster side flips with the shape. Each call site names its candidates; the
first call for a shape key times them once (outside graph capture) and later calls reuse the
winner. Candidates must compute the same product; only speed differs.
"""

from __future__ import annotations

import time
from typing import Callable, Hashable, Sequence

import torch

_choice: dict[Hashable, int] = {}


def row_bucket(rows: int) -> int:
    """Row counts share a choice per power of two (a prefill's last chunk is usually short)."""
    return 1 << max(0, rows - 1).bit_length()


def pick(key: Hashable, candidates: Sequence[Callable[[], torch.Tensor]]) -> torch.Tensor:
    """Run the fastest candidate for ``key``, timing all of them on the first call."""
    idx = _choice.get(key)
    if idx is not None:
        return candidates[idx]()
    if len(candidates) == 1 or torch.cuda.is_current_stream_capturing():
        return candidates[0]()
    best, out = None, None
    for i, fn in enumerate(candidates):
        r = fn()  # warm (kernel selection / JIT)
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        if best is None or dt < best[0]:
            best, out = (dt, i), r
    _choice[key] = best[1]
    return out


def flipped_linear(x2: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x2 @ w^T`` computed as ``(w @ x2^T)^T``, made contiguous."""
    return (w @ x2.t()).t().contiguous()


def choices() -> dict[Hashable, int]:
    return dict(_choice)


__all__ = ["choices", "flipped_linear", "pick", "row_bucket"]
