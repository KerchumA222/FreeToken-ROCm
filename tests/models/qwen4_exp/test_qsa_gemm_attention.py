"""Prefill QSA through batched GEMMs: splitting a tile group under the byte budget must not
change the result, and both match the tiled Triton kernel."""

from __future__ import annotations

import pytest
import torch

from .common import requires_cuda

BLOCK = 64


def _case(seed: int, ctx: int, rows: int, width: int, hq: int = 24, kvh: int = 2, dim: int = 64):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n_pages = (ctx + rows + BLOCK - 1) // BLOCK
    k = torch.randn(n_pages, BLOCK, kvh, dim, device="cuda", generator=g).half()
    v = torch.randn(n_pages, BLOCK, kvh, dim, device="cuda", generator=g).half()
    q = torch.randn(rows, hq, dim, device="cuda", generator=g).half()
    block_table = torch.randperm(n_pages, device="cuda", generator=g).int()[None, :]
    indices = torch.full((rows, width), -1, dtype=torch.int32, device="cuda")
    for r in range(rows):
        visible = ctx + r + 1
        n = min(width, visible)
        indices[r, :n] = torch.randperm(visible, device="cuda", generator=g)[:n].sort().values.int()
    return q, k, v, indices, block_table, [0, rows]


@requires_cuda
@pytest.mark.parametrize("budget", [1, 1 << 20])
def test_gemm_budget_split_matches_unsplit_and_tiled(monkeypatch, budget: int):
    from freetoken.kernel.triton.qsa import attend_tiled as at

    q, k, v, indices, block_table, cu = _case(0, ctx=1500, rows=300, width=256)
    whole = at.qsa_gemm_attention(q, k, v, indices, block_table, cu)
    tiled = at.qsa_tiled_attention(q, k, v, indices, block_table, cu)
    monkeypatch.setattr(at, "_GEMM_BYTES", budget)
    split = at.qsa_gemm_attention(q, k, v, indices, block_table, cu)
    # rocBLAS may pick a different kernel for a smaller batch: fp16 rounding, not bits.
    torch.testing.assert_close(split.float(), whole.float(), atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(split.float(), tiled.float(), atol=2e-3, rtol=2e-2)
