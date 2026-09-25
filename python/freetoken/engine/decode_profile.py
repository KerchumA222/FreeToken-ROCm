"""Diagnostic: profile a window of decode-side forwards with torch.profiler.

``FT_PROFILE_DECODE=<skip>,<count>,<path>`` skips ``skip`` decode/verify forwards, profiles
the next ``count`` and writes a per-kernel GPU-time table to ``path``. Run eager
(``--cuda-graph-max-bs 0``) so each kernel is recorded rather than one graph launch.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

_SPEC = os.environ.get("FT_PROFILE_DECODE", "")
_state = {"n": 0, "prof": None, "done": False}


def step(batch):
    """Context for one forward; a no-op outside the profiled window."""
    if not _SPEC or _state["done"] or batch.is_prefill and not batch.is_spec_verify:
        return nullcontext()
    skip, count, path = _SPEC.split(",", 2)
    skip, count = int(skip), int(count)
    n = _state["n"]
    _state["n"] = n + 1
    if n == skip:
        import torch

        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_stack=bool(os.environ.get("FT_PROFILE_STACK")),
            record_shapes=bool(os.environ.get("FT_PROFILE_SHAPES")),
        )
        prof.__enter__()
        _state["prof"] = prof
    if n == skip + count:
        import torch

        torch.cuda.synchronize()
        prof = _state["prof"]
        prof.__exit__(None, None, None)
        _state["done"] = True
        sort = os.environ.get("FT_PROFILE_SORT", "self_cuda_time_total")
        table = prof.key_averages().table(sort_by=sort, row_limit=int(os.environ.get("FT_PROFILE_ROWS", "60")),
                                          max_name_column_width=90)
        with open(path, "w") as f:
            f.write(f"forwards profiled: {count}\n{table}\n")
            if os.environ.get("FT_PROFILE_SHAPES"):
                f.write(prof.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cuda_time_total", row_limit=25, max_name_column_width=40,
                    max_shapes_column_width=120))
            if os.environ.get("FT_PROFILE_STACK"):
                f.write(prof.key_averages(group_by_stack_n=6).table(
                    sort_by="self_cuda_time_total", row_limit=40, max_name_column_width=60))
    return nullcontext()


__all__ = ["step"]
