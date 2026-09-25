"""Run coalescing behind the prefill tier fill's DMA copies (``moe/offload_cache.py``)."""

import torch

from freetoken.moe.offload_cache import _copy_runs, _row_runs


def test_row_runs_coalesce_only_when_both_sides_step():
    assert _row_runs([3, 4, 5, 9, 10], range(5)) == [(3, 0, 3), (9, 3, 2)]
    # destination consecutive but source not: separate runs
    assert _row_runs([1, 2, 3], [7, 8, 20]) == [(1, 7, 2), (3, 20, 1)]
    assert _row_runs([], []) == []


def test_copy_runs_matches_index_copy():
    src = torch.arange(10 * 4, dtype=torch.uint8).view(10, 4)
    dst_rows, src_rows = [0, 1, 2, 6, 8, 9], [4, 5, 6, 0, 1, 2]
    dst = torch.zeros(10, 4, dtype=torch.uint8)
    _copy_runs(dst, src, _row_runs(dst_rows, src_rows))
    ref = torch.zeros_like(dst)
    ref[dst_rows] = src[src_rows]
    assert torch.equal(dst, ref)
