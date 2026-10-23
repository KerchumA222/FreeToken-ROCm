"""Python face of the host-call extension (``kernel/csrc/host_call``): a captured graph calls back into Python and resumes when it returns.

``HostCall.submit`` / ``wait`` enqueue a stream-memop doorbell and a stream-memop wait, which record as graph nodes where the runtime supports it;
a host thread runs the callback in between. ``probe`` checks that on this GPU, end to end, once at startup.
"""

from __future__ import annotations

import time

import torch

from freetoken.kernel.pinned import alloc_pinned_tensor

_PROBE_TIMEOUT_S = 10.0


def load():
    """The extension module, or an ImportError naming how to build it."""
    import importlib

    try:
        return importlib.import_module("freetoken.kernel._host_call")
    except ImportError as exc:
        raise ImportError("freetoken.kernel._host_call is not built; rebuild FreeToken (python setup.py build_ext --inplace)") from exc


def probe(device: torch.device) -> str | None:
    """None when a graph captured on ``device`` can wait for a Python callback, else the reason it cannot.

    Replays a one-node graph that hands a value to the callback and takes the answer back, so a runtime that accepts the memops but records
    no node (the WAIT would run at capture and be gone from the replay) fails here, and never hangs: a graph that does not return is released."""
    try:
        ext = load()
    except ImportError as exc:
        return str(exc)
    stream = torch.cuda.current_stream(device)
    if not ext.probe(stream.cuda_stream):
        return "the runtime does not record stream memory operations in a captured graph"
    box = alloc_pinned_tensor(2, dtype=torch.int64)
    box.zero_()
    cells = box.numpy()

    def serve(task: int) -> None:
        cells[1] = cells[0] * 2 + task

    call = ext.HostCall(box.data_ptr(), serve)
    call.start()
    reason = None
    try:
        value = torch.zeros(1, dtype=torch.int64, device=device)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            side = torch.cuda.current_stream(device).cuda_stream
            box[:1].copy_(value, non_blocking=True)
            call.submit(side, 5)
            call.wait(side)
            value.copy_(box[1:], non_blocking=True)
        value.fill_(7)
        graph.replay()
        done = torch.cuda.Event()
        done.record()
        deadline = time.monotonic() + _PROBE_TIMEOUT_S
        while not done.query() and time.monotonic() < deadline:
            time.sleep(0.001)
        if not done.query():
            call.release()
            reason = f"a captured host call did not return within {_PROBE_TIMEOUT_S:.0f} s"
            done.synchronize()
        elif int(value.item()) != 7 * 2 + 5:
            reason = f"a captured host call returned {int(value.item())}, expected 19"
    except Exception as exc:
        reason = f"capturing a host call failed: {exc}"
    finally:
        call.stop()
    return reason


__all__ = ["load", "probe"]
