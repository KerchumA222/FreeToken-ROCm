"""The graph host call (``kernel/csrc/host_call``): a captured stream hands a value to a Python callback and resumes with its answer, on every
replay, and is released when the callback fails or the owner gives up."""

from __future__ import annotations

import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

host_call = pytest.importorskip("freetoken.kernel.host_call")


def _need_memops() -> None:
    reason = host_call.probe(torch.device("cuda"))
    if reason is not None:
        pytest.skip(reason)


def _wait(event: torch.cuda.Event, seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while not event.query():
        assert time.monotonic() < deadline, "the replay never returned from its host call"
        time.sleep(0.001)


def _graph(call, box, value, layers: int):
    """``layers`` host calls in a row, each doubling ``value`` through the host (task = layer + 1 is added)."""
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        side = torch.cuda.current_stream().cuda_stream
        for layer in range(layers):
            box[:1].copy_(value, non_blocking=True)
            call.submit(side, layer + 1)
            call.wait(side)
            value.copy_(box[1:], non_blocking=True)
    return graph


def test_probe_reports_a_reason_or_nothing():
    reason = host_call.probe(torch.device("cuda"))
    assert reason is None or isinstance(reason, str)


def test_replays_run_the_callback_between_graph_nodes():
    _need_memops()
    from freetoken.kernel.pinned import alloc_pinned_tensor

    box = alloc_pinned_tensor(2, dtype=torch.int64)
    box.zero_()
    cells = box.numpy()
    seen = []

    def serve(task: int) -> None:
        seen.append(task)
        cells[1] = cells[0] * 2 + task

    call = host_call.load().HostCall(box.data_ptr(), serve)
    call.start()
    try:
        value = torch.zeros(1, dtype=torch.int64, device="cuda")
        graph = _graph(call, box, value, layers=3)
        for start in (1, 5, 11):
            value.fill_(start)
            graph.replay()
            done = torch.cuda.Event()
            done.record()
            _wait(done)
            want = start
            for layer in range(3):
                want = want * 2 + layer + 1
            assert int(value.item()) == want
        assert seen == [1, 2, 3] * 3
        assert call.served() == 9
    finally:
        call.stop()


def test_a_zero_pending_count_skips_the_callback():
    _need_memops()
    from freetoken.kernel.pinned import alloc_pinned_tensor

    box = alloc_pinned_tensor(2, dtype=torch.int64)
    box.zero_()
    cells = box.numpy()
    seen = []
    call = host_call.load().HostCall(box.data_ptr(), lambda task: seen.append(task))
    call.start()
    try:
        value = torch.zeros(1, dtype=torch.int64, device="cuda")
        graph = _graph(call, box, value, layers=2)
        for _ in range(3):
            graph.replay()
            done = torch.cuda.Event()
            done.record()
            _wait(done)
        assert seen == [] and call.served() == 6 and int(cells[1]) == 0
    finally:
        call.stop()


def test_a_raising_callback_still_releases_the_stream():
    _need_memops()
    from freetoken.kernel.pinned import alloc_pinned_tensor

    box = alloc_pinned_tensor(2, dtype=torch.int64)
    box.zero_()

    def serve(task: int) -> None:
        raise ValueError("boom")

    call = host_call.load().HostCall(box.data_ptr(), serve)
    call.start()
    try:
        value = torch.ones(1, dtype=torch.int64, device="cuda")
        graph = _graph(call, box, value, layers=2)
        graph.replay()
        done = torch.cuda.Event()
        done.record()
        _wait(done)
    finally:
        call.stop()


def test_release_frees_a_stream_whose_thread_is_not_running():
    _need_memops()
    from freetoken.kernel.pinned import alloc_pinned_tensor

    box = alloc_pinned_tensor(2, dtype=torch.int64)
    box.zero_()
    call = host_call.load().HostCall(box.data_ptr(), lambda task: None)  # never started: nobody answers
    value = torch.ones(1, dtype=torch.int64, device="cuda")
    graph = _graph(call, box, value, layers=1)
    graph.replay()
    done = torch.cuda.Event()
    done.record()
    time.sleep(0.2)
    assert not done.query()
    call.release()
    _wait(done)
    call.stop()
