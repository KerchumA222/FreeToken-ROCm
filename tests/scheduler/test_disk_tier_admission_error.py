"""Disk-tier callback failures surface only after a forward's completion barrier."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from freetoken.scheduler.scheduler import Scheduler


def test_admission_error_after_barrier_prevents_final_batch_commit():
    completed = False
    parked_error = None

    class CopyDone:
        def synchronize(self):
            nonlocal completed, parked_error
            completed = True
            parked_error = RuntimeError("disk read failed in graph host node")

    class OffloadCache:
        def raise_admission_error(self):
            assert completed, "admission error checked before replay completion"
            if parked_error is not None:
                raise RuntimeError("disk-tier expert admission failed") from parked_error

    commits = []

    @contextmanager
    def lazy_free_region():
        commits.append("cache region entered")
        yield

    class Request:
        def __init__(self):
            self.input_ids = torch.tensor([1], dtype=torch.int32)
            self.aborted = False

        def append_host(self, tokens):
            commits.append(tokens.tolist())

    req = Request()
    batch = SimpleNamespace(reqs=[req], is_spec_verify=False, draft_last_rows=[])
    last_data = (
        SimpleNamespace(batch=batch),
        (None, torch.tensor([42], dtype=torch.int32), CopyDone()),
    )
    sent = []
    scheduler = SimpleNamespace(
        engine=SimpleNamespace(moe_offload_cache=OffloadCache()),
        cache_manager=SimpleNamespace(lazy_free_region=lazy_free_region),
        finished_reqs=set(),
        send_result=sent.extend,
    )

    with pytest.raises(RuntimeError, match="disk-tier expert admission failed"):
        Scheduler._process_last_data(scheduler, last_data)

    assert completed
    assert req.input_ids.tolist() == [1]
    assert commits == []
    assert sent == []
