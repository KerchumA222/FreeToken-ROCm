from array import array
from types import SimpleNamespace

import pytest
import torch

from freetoken.env import ENV
from freetoken.scheduler.scheduler import Scheduler


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [(False, "XYZ"), (True, "AB")],
)
def test_stop_match_order_preserves_upstream_unless_opted_in(monkeypatch, enabled, expected):
    class Tokenizer:
        def decode(self, token_ids):
            return "prefix AB later XYZ"

    monkeypatch.setattr(ENV.EARLIEST_STOP_MATCH, "value", enabled)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tokenizer = Tokenizer()
    req = SimpleNamespace(
        input_ids=torch.tensor([0, 1]),
        max_device_len=6,
        output_len=5,
        sampling_params=SimpleNamespace(stop_strs=["XYZ", "AB"]),
    )

    assert scheduler._match_stop_str(req) == expected
