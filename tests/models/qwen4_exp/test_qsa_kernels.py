"""The modified and original QSA Triton kernels against pure-torch references.

Only kernels FreeToken changed or wrote get unit tests: the compression kernel (re-addressed
pending ring, its own torch check) and the block top-k (original radix select, checked against
torch.topk and, through the expansion chain, against the vLLM reference semantics). score.py
and attend.py are vendored from vLLM and are covered by the backend and e2e tests.
``_qsa_mqa_paged_reference`` / ``_qsa_relative_topk_reference`` / ``_expand_qsa_indices_reference``
are transcribed from ``vllm/tests/test_qsa_reference.py`` (Apache-2.0).
"""

from __future__ import annotations

import math

import pytest
import torch

from .common import Fixture, requires_cuda, parsed_config

PAGE_SIZE = 64
RATIO = 4
BUDGET = 2048
INDEX_DIM = 128
CMP_PAGE = PAGE_SIZE // RATIO


# --------------------------------------------------------------------------------------
# vLLM pure-torch references (tests/test_qsa_reference.py:87-227)
# --------------------------------------------------------------------------------------


def _qsa_mqa_paged_reference(q, k_cache, page_table, token_to_req, visible_lengths):
    pages = page_table.index_select(0, token_to_req.long()).long()
    keys = k_cache[pages, :, 0, :].flatten(1, 2)
    scores = torch.einsum("rhd,rnd->rnh", q.float(), keys.float())
    logits = torch.relu(scores).sum(dim=-1) / math.sqrt(q.shape[-1])
    positions = torch.arange(keys.shape[1], device=q.device).unsqueeze(0)
    return logits.masked_fill(positions >= visible_lengths.unsqueeze(1), -torch.inf)


def _qsa_relative_topk_reference(logits, row_starts, row_ends, topk):
    output = torch.full((logits.shape[0], topk), -1, dtype=torch.int32, device=logits.device)
    for row in range(logits.shape[0]):
        start = int(row_starts[row].item())
        length = int((row_ends[row] - row_starts[row]).item())
        width = min(length, topk)
        if width:
            output[row, :width] = torch.topk(
                logits[row, start : start + length], width
            ).indices.to(torch.int32)
    return output


def _expand_qsa_indices_reference(
    block_indices, query_positions, sequence_lengths, compress_ratio, token_topk
):
    rows = block_indices.shape[0]
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    offsets = torch.arange(compress_ratio, device=block_indices.device)
    blocks = block_indices.long()
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)
    ).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=block_indices.device)
    visible_tokens = query_positions + 1
    tail_start = visible_tokens // compress_ratio * compress_ratio
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_count = (visible_tokens - tail_start).unsqueeze(1)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count) & (
        tail < sequence_lengths.unsqueeze(1)
    )
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat((expanded, tail), dim=1)
    order = torch.arange(output_width, device=result.device).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + output_width)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


class _Case:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def _paged_case(length: int, bs: int, rows_per_req: int, seed: int, index_heads: int = 4):
    """Synthetic paged geometry: shuffled pages, the last ``rows_per_req`` queries per request."""
    device = torch.device("cuda")
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    pages_per_req = -(-length // PAGE_SIZE)
    total_pages = bs * pages_per_req
    block_table = (
        torch.randperm(total_pages, device=device).reshape(bs, pages_per_req).to(torch.int32)
    )
    q = torch.randn(
        bs * rows_per_req, index_heads, INDEX_DIM, device=device, dtype=torch.bfloat16,
        generator=generator,
    )
    token_to_req = torch.repeat_interleave(
        torch.arange(bs, device=device, dtype=torch.int32), rows_per_req
    )
    query_positions = torch.cat(
        [
            torch.arange(length - rows_per_req, length, device=device, dtype=torch.int32)
            for _ in range(bs)
        ]
    )
    seq_lens = torch.full((bs,), length, device=device, dtype=torch.int32)
    return _Case(
        device=device,
        generator=generator,
        length=length,
        bs=bs,
        pages_per_req=pages_per_req,
        total_pages=total_pages,
        block_table=block_table,
        q=q,
        token_to_req=token_to_req,
        query_positions=query_positions,
        seq_lens=seq_lens,
    )


@requires_cuda
@pytest.mark.parametrize("length", [20000])
@pytest.mark.parametrize("bs", [1])
@pytest.mark.parametrize("torch_topk", [False, True])
def test_top_blocks_and_expansion_match_vllm_reference(length: int, bs: int, torch_topk: bool):
    from freetoken.kernel.triton.qsa import expand_qsa_block_indices, qsa_mqa_paged

    config = parsed_config()
    fixture = Fixture(config, num_pages=4, max_running_req=2)
    case = _paged_case(length, bs, rows_per_req=4, seed=7 * length + bs)
    cache = torch.randn(
        case.total_pages, CMP_PAGE, 1, INDEX_DIM, device=case.device,
        dtype=torch.bfloat16, generator=case.generator,
    )
    rows, columns = case.q.shape[0], case.pages_per_req * CMP_PAGE
    logits = torch.empty(rows, columns, dtype=torch.float32, device=case.device)
    visible = torch.empty(rows, dtype=torch.int32, device=case.device)
    qsa_mqa_paged(
        case.q, cache, case.block_table, case.token_to_req, case.query_positions,
        case.seq_lens, RATIO, logits, visible,
    )
    blocks = torch.empty(rows, BUDGET // RATIO, dtype=torch.int32, device=case.device)
    reference_logits = _qsa_mqa_paged_reference(
        case.q, cache, case.block_table, case.token_to_req, visible
    )
    if torch_topk:
        fixture.backend._block_topk_kernel = None
    fixture.backend._top_blocks(logits, visible, blocks)

    expected_blocks = _qsa_relative_topk_reference(
        reference_logits, torch.zeros_like(visible), visible, BUDGET // RATIO
    )
    # Ties between equal scores may land on either index; the SET is what selection means.
    torch.testing.assert_close(blocks.sort(-1).values, expected_blocks.sort(-1).values)

    row_seq_lens = case.seq_lens.index_select(0, case.token_to_req.long())
    indices = torch.empty(rows, BUDGET + RATIO - 1, dtype=torch.int32, device=case.device)
    expand_qsa_block_indices(
        expected_blocks, case.query_positions, case.seq_lens, case.token_to_req,
        RATIO, BUDGET, indices,
    )
    expected = _expand_qsa_indices_reference(
        expected_blocks, case.query_positions, row_seq_lens, RATIO, BUDGET
    )
    torch.testing.assert_close(indices, expected)


@requires_cuda
@pytest.mark.parametrize("ring_capacity", [4, 8])
def test_compression_reads_both_sources(ring_capacity: int):
    """Members already consumed come from the ring, the rest from this forward's raw rows."""
    from freetoken.kernel.triton.qsa import qsa_compress_groups, qsa_store_rows

    device = torch.device("cuda")
    dim, slots = 8, 3
    pairs = [(0, p) for p in range(2, 9)] + [(1, p) for p in range(5, 11)]

    def key(request: int, position: int) -> torch.Tensor:
        return (torch.arange(dim, dtype=torch.float32) + request * 1000 + position * 10).to(
            torch.bfloat16
        )

    raw = torch.stack([key(*pair) for pair in pairs]).to(device)
    token_to_req = torch.tensor([r for r, _ in pairs], dtype=torch.int32, device=device)
    positions = torch.tensor([p for _, p in pairs], dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor([0, 7, 13], dtype=torch.int32, device=device)
    ring_slots = torch.tensor([2, 0], dtype=torch.int32, device=device)
    ring = torch.zeros(slots, ring_capacity, dim, device=device, dtype=torch.bfloat16)
    for request, position, slot in ((0, 0, 2), (0, 1, 2), (1, 4, 0)):
        ring[slot, position % ring_capacity] = key(request, position).to(device)

    pooled = torch.empty(len(pairs), dim, device=device, dtype=torch.bfloat16)
    first = torch.empty(len(pairs), dtype=torch.int32, device=device)
    qsa_compress_groups(
        raw, ring, ring_slots, token_to_req, cu_seqlens, positions, RATIO, pooled, first
    )

    for row, (request, position) in enumerate(pairs):
        if (position + 1) % RATIO:
            continue
        group = torch.stack([key(request, position - RATIO + 1 + k).float() for k in range(RATIO)])
        expected = group.mean(0).to(torch.bfloat16).to(device)
        assert torch.equal(pooled[row], expected), (request, position)
        assert int(first[row]) == position - RATIO + 1

    # The ring keeps only each request's last ring_capacity rows.
    rows = torch.arange(len(pairs), device=device)
    ends = cu_seqlens.long().index_select(0, token_to_req.long() + 1)
    slot = torch.where(
        rows >= ends - ring_capacity,
        ring_slots.long().index_select(0, token_to_req.long()) * ring_capacity
        + positions.long() % ring_capacity,
        torch.full_like(rows, -1),
    )
    qsa_store_rows(ring, slot.to(torch.int32), raw)
    for request, ring_slot in ((0, 2), (1, 0)):
        for position in [p for r, p in pairs if r == request][-ring_capacity:]:
            assert torch.equal(
                ring[ring_slot, position % ring_capacity], key(request, position).to(device)
            )


# --------------------------------------------------------------------------------------
# Block top-k (kernel/triton/qsa/topk.py)
# --------------------------------------------------------------------------------------

TOPK_SHAPES = [(512, 512), (4096, 512)]


def _torch_topk_blocks(logits, visible, width):
    """The torch.topk fallback of ``qsa_sparse._top_blocks``, kept here as the reference."""
    columns = logits.shape[1]
    out = torch.full((logits.shape[0], width), -1, dtype=torch.int32, device=logits.device)
    column = torch.arange(columns, device=logits.device)
    masked = logits.masked_fill(column.unsqueeze(0) >= visible.unsqueeze(1), -float("inf"))
    take = min(width, columns)
    values, chosen = torch.topk(masked, take, dim=-1)
    out[:, :take] = torch.where(values > -float("inf"), chosen.to(torch.int32), -1)
    return out


def _topk_case(n_blocks: int, bs: int, mode: str, seed: int):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(bs, n_blocks, device=device, generator=generator)
    visible = torch.full((bs,), n_blocks, dtype=torch.int32, device=device)
    if mode == "ties":
        # Three distinct scores over every column: almost every selection sits on a tie.
        logits = torch.randint(0, 3, (bs, n_blocks), device=device, generator=generator).float()
    if mode == "ragged":
        visible = torch.randint(
            0, n_blocks + 1, (bs,), dtype=torch.int32, device=device, generator=generator
        )
    if mode == "dead":
        logits[:, ::5] = -float("inf")
    return logits, visible


@requires_cuda
@pytest.mark.parametrize("n_blocks,width", TOPK_SHAPES)
@pytest.mark.parametrize("bs", [4])
@pytest.mark.parametrize("mode", ["random", "ties", "ragged", "dead"])
def test_block_topk_matches_torch_topk(n_blocks: int, width: int, bs: int, mode: str):
    from freetoken.kernel.triton.qsa import qsa_block_topk

    logits, visible = _topk_case(n_blocks, bs, mode, seed=31 * n_blocks + 7 * width + bs)
    blocks = torch.empty(bs, width, dtype=torch.int32, device=logits.device)
    qsa_block_topk(logits, visible, blocks)
    # torch.topk does not define which members of a tie it keeps (on ROCm, at 4096 columns,
    # not the lowest columns), so on ties compare against the kernel's documented policy.
    reference = _policy_topk_blocks if mode == "ties" else _torch_topk_blocks
    expected = reference(logits, visible, width)

    # Selection is a set: torch.topk orders by descending score, the kernel by column id.
    torch.testing.assert_close(blocks.sort(-1).values, expected.sort(-1).values)
    live = (blocks >= 0).sum(-1)
    torch.testing.assert_close(live, (expected >= 0).sum(-1))
    # expand.py reads ranks [0, complete_blocks), so a -1 may only sit in the tail.
    ranks = torch.arange(width, device=blocks.device)
    assert torch.equal(blocks >= 0, ranks.unsqueeze(0) < live.unsqueeze(1))
    assert bool(((blocks[:, 1:] > blocks[:, :-1]) | (blocks[:, 1:] < 0)).all())


@requires_cuda
def test_block_topk_replays_in_a_cuda_graph():
    """Fixed grid, no host read: one capture serves every later sequence length."""
    from freetoken.kernel.triton.qsa import qsa_block_topk

    rows, columns, width = 4, 4096, 512
    device = torch.device("cuda")
    logits = torch.randn(rows, columns, device=device)
    visible = torch.full((rows,), columns, dtype=torch.int32, device=device)
    blocks = torch.empty(rows, width, dtype=torch.int32, device=device)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        qsa_block_topk(logits, visible, blocks)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qsa_block_topk(logits, visible, blocks)

    for lengths in ([columns] * rows, [columns // 2, 900, 37, 0], [4095, 512, 511, 4096]):
        logits.normal_()
        visible.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        blocks.fill_(0)
        graph.replay()
        expected = _torch_topk_blocks(logits, visible, width)
        torch.testing.assert_close(blocks.sort(-1).values, expected.sort(-1).values)


def test_torch_topk_env_picks_the_fallback(monkeypatch):
    from freetoken.attention.qsa_sparse import TORCH_TOPK_ENV, _resolve_block_topk

    assert _resolve_block_topk() is not None
    monkeypatch.setenv(TORCH_TOPK_ENV, "1")
    assert _resolve_block_topk() is None


# --------------------------------------------------------------------------------------
# Block top-k, split + merge path (wide buffers)
# --------------------------------------------------------------------------------------

# (columns, _split_plan's chunk). A 1M-token window scores 262,144 compressed rows, and its 32
# splits leave more candidates than one program holds, so a reduction level runs before the merge.
SPLIT_SHAPES = [(65536, 4096), (262144, 8192)]


def _split_chunk(columns: int, width: int) -> int:
    from freetoken.kernel.triton.qsa.topk import _split_plan

    plan = _split_plan(columns, width)
    assert plan is not None, f"{columns} columns must take the split path"
    return plan.chunk


def _policy_topk_blocks(logits, visible, width):
    """The kernel's documented order: highest score wins, lowest column breaks a tie."""
    columns = logits.shape[1]
    column = torch.arange(columns, device=logits.device)
    masked = logits.masked_fill(column.unsqueeze(0) >= visible.unsqueeze(1), -float("inf"))
    take = min(width, columns)
    order = masked.argsort(dim=-1, descending=True, stable=True)[:, :take]
    out = torch.full((logits.shape[0], width), -1, dtype=torch.int32, device=logits.device)
    out[:, :take] = torch.where(
        masked.gather(1, order) > -float("inf"), order.to(torch.int32), -1
    )
    return out


def _split_topk_case(n_blocks: int, chunk: int, width: int, bs: int, mode: str, seed: int):
    if mode != "boundary":
        return _topk_case(n_blocks, bs, mode, seed)
    # width - 212 columns beat the tie, so the 212 remaining winners start 100 columns below
    # a chunk boundary and run past it into a chunk that is all tie. The last row cuts at the
    # middle split boundary, which is where a reduction level divides the candidates.
    logits = torch.zeros(bs, n_blocks, device="cuda")
    splits = n_blocks // chunk
    for row in range(bs):
        cut = chunk * (splits // 2 if row == bs - 1 else 1 + row % (splits - 1))
        logits[row, : width - 212] = 2.0
        logits[row, cut - 100 :] = 1.0
    return logits, torch.full((bs,), n_blocks, dtype=torch.int32, device="cuda")


@requires_cuda
@pytest.mark.parametrize("n_blocks,chunk", SPLIT_SHAPES)
@pytest.mark.parametrize("bs", [4])
@pytest.mark.parametrize("mode", ["random", "boundary", "ragged", "dead"])
def test_block_topk_split_path_matches_torch_topk(n_blocks: int, chunk: int, bs: int, mode: str):
    from freetoken.kernel.triton.qsa import qsa_block_topk, qsa_block_topk_scratch_width

    width = 512
    assert qsa_block_topk_scratch_width(n_blocks, width) > 0, "case must take the split path"
    assert _split_chunk(n_blocks, width) == chunk
    logits, visible = _split_topk_case(
        n_blocks, chunk, width, bs, mode, seed=n_blocks + bs + len(mode)
    )
    blocks = torch.empty(bs, width, dtype=torch.int32, device=logits.device)
    qsa_block_topk(logits, visible, blocks)

    torch.testing.assert_close(
        blocks.sort(-1).values, _torch_topk_blocks(logits, visible, width).sort(-1).values
    )
    # Tie determinism: the winners are the exact set the lowest-column-first policy names,
    # including the ties that straddle a chunk boundary.
    torch.testing.assert_close(
        blocks.sort(-1).values, _policy_topk_blocks(logits, visible, width).sort(-1).values
    )
    live = (blocks >= 0).sum(-1)
    ranks = torch.arange(width, device=blocks.device)
    assert torch.equal(blocks >= 0, ranks.unsqueeze(0) < live.unsqueeze(1))
    assert bool(((blocks[:, 1:] > blocks[:, :-1]) | (blocks[:, 1:] < 0)).all())


@requires_cuda
@pytest.mark.parametrize("preallocated", [False, True])
def test_block_topk_split_path_replays_in_a_cuda_graph(preallocated: bool):
    """The split geometry comes from the buffer width, so one capture serves every length."""
    from freetoken.kernel.triton.qsa import qsa_block_topk, qsa_block_topk_scratch_width

    rows, columns, width = 4, 65536, 512
    device = torch.device("cuda")
    scratch_width = qsa_block_topk_scratch_width(columns, width)
    assert scratch_width > 0
    logits = torch.randn(rows, columns, device=device)
    visible = torch.full((rows,), columns, dtype=torch.int32, device=device)
    blocks = torch.empty(rows, width, dtype=torch.int32, device=device)
    # The scratch never needs clearing: every split rewrites its own slots on every replay.
    scratch = (
        torch.empty(rows, scratch_width, dtype=torch.int32, device=device)
        if preallocated
        else None
    )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        qsa_block_topk(logits, visible, blocks, scratch)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qsa_block_topk(logits, visible, blocks, scratch)

    for lengths in ([columns] * rows, [4096, 40000, 0, 65535], [1, 4097, 8192, columns]):
        logits.normal_()
        visible.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        blocks.fill_(0)
        graph.replay()
        expected = _torch_topk_blocks(logits, visible, width)
        torch.testing.assert_close(blocks.sort(-1).values, expected.sort(-1).values)


@requires_cuda
def test_block_topk_split_path_cost_tracks_live_blocks():
    """A wide buffer with a short row must not pay for the splits past its visible tail."""
    from freetoken.kernel.triton.qsa import qsa_block_topk, qsa_block_topk_scratch_width

    # enough rows that the live splits fill the card, so their work outweighs the launch floor
    rows, columns, width = 8, 262144, 512
    assert qsa_block_topk_scratch_width(columns, width) > 0, "case must take the split path"
    chunk = _split_chunk(columns, width)
    device = torch.device("cuda")
    logits = torch.randn(rows, columns, device=device)
    visible = torch.full((rows,), columns, dtype=torch.int32, device=device)
    blocks = torch.empty(rows, width, dtype=torch.int32, device=device)
    scratch = torch.empty(
        rows, qsa_block_topk_scratch_width(columns, width), dtype=torch.int32, device=device
    )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        qsa_block_topk(logits, visible, blocks, scratch)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            qsa_block_topk(logits, visible, blocks, scratch)

    def replay_us(live: int) -> float:
        visible.fill_(live)
        best = float("inf")
        for _ in range(5):
            start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            graph.replay()
            stop.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(stop) * 1000.0 / 20)
        return best

    full, short = replay_us(columns), replay_us(chunk)
    assert full > 2.0 * short, f"{columns} live {full:.1f}us vs {chunk} live {short:.1f}us"


@requires_cuda
def test_capture_graph_provisions_the_block_topk_scratch():
    from freetoken.kernel.triton.qsa import qsa_block_topk_scratch_width

    fixture = Fixture(parsed_config(), num_pages=320, max_running_req=2)
    backend = fixture.backend
    table_width = fixture.page_table.shape[1]
    columns = table_width // PAGE_SIZE * CMP_PAGE
    width = qsa_block_topk_scratch_width(columns, backend.block_topk)
    assert width > 0, "the fixture must be wide enough to reach the split path"

    backend.init_capture_graph(table_width, [2])
    static = backend._graph["topk_scratch"]
    assert static.shape[1] == width
    assert backend._scratch("topk_scratch", 2, width, dtype=torch.int32).data_ptr() == (
        static.data_ptr()
    )


# --------------------------------------------------------------------------------------
# TurboQuant K/V pages and the tiled prefill
# --------------------------------------------------------------------------------------

KV_HEADS, Q_HEADS, HEAD_DIM = 2, 4, 256


def _kv_pool(kv_cache_dtype: str, num_pages: int):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.mha_pool import MHAKVCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    return MHAKVCache(
        num_kv_heads=KV_HEADS, num_layers=1, head_dim=HEAD_DIM, num_pages=num_pages,
        page_size=PAGE_SIZE, dtype=torch.float16, device=torch.device("cuda"),
        kv_cache_dtype=kv_cache_dtype,
    )


def _filled_pool(kv_cache_dtype: str, num_pages: int, seed: int = 5):
    """A pool whose every slot holds random K/V; returns it with the unquantized K/V."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pool = _kv_pool(kv_cache_dtype, num_pages)
    n = num_pages * PAGE_SIZE
    k = torch.randn(n, KV_HEADS * HEAD_DIM, device="cuda", dtype=torch.float16, generator=gen)
    v = torch.randn(n, KV_HEADS * HEAD_DIM, device="cuda", dtype=torch.float16, generator=gen)
    pool.store_kv(k, v, torch.arange(n, dtype=torch.int32, device="cuda"), 0)
    return pool, k, v


def _quant_meta(pool):
    return dict(
        k_norms=pool.k_norm_cache(0), v_scales=pool.v_scale_cache(0), v_zeros=pool.v_zero_cache(0),
        centroids=pool.turboquant_centroids, rotation=pool.turboquant_rotation,
    )


def _selection(rows, width, seed=3):
    """Random causal selections: row ``(req, position)`` picks up to ``width`` distinct tokens of
    its request's logical prefix, -1 padded."""
    gen = torch.Generator().manual_seed(seed)
    indices = torch.full((len(rows), width), -1, dtype=torch.int32)
    for i, (_, position) in enumerate(rows):
        pick = torch.randperm(position + 1, generator=gen)[:width]
        indices[i, : pick.numel()] = pick.to(torch.int32)
    return indices.cuda()


def _paged_batch(rows, pages_per_req, num_pages, seed=9):
    """Physical pages are a shuffled assignment, so logical -> physical translation matters."""
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_pages, generator=gen)
    num_reqs = max(r for r, _ in rows) + 1
    table = order[: num_reqs * pages_per_req].view(num_reqs, pages_per_req).to(torch.int32).cuda()
    token_to_req = torch.tensor([r for r, _ in rows], dtype=torch.int32).cuda()
    return table, token_to_req


def _dense_reference(q, pool, kv_cache_dtype, indices, table, token_to_req):
    """fp32 attention over each row's selected tokens, K/V read back through the pool's own
    dequantization (K in the rotated basis, paired with the rotated query)."""
    from freetoken.kernel.triton.turboquant import dequant_turboquant_4bit_nc

    n = pool.k_cache(0).shape[0] * PAGE_SIZE
    if kv_cache_dtype == "turboquant_4bit_nc":
        meta = _quant_meta(pool)
        k, v = dequant_turboquant_4bit_nc(
            k_cache=pool.k_cache(0).view(-1, KV_HEADS, HEAD_DIM // 2),
            v_cache=pool.v_cache(0).view(-1, KV_HEADS, HEAD_DIM // 2),
            k_norms=meta["k_norms"], v_scales=meta["v_scales"], v_zeros=meta["v_zeros"],
            centroids=meta["centroids"], indices=torch.arange(n, dtype=torch.int32, device="cuda"),
            dtype=torch.float32,
        )
        q = q.float() @ meta["rotation"].T
    else:
        k = pool.k_cache(0).reshape(n, KV_HEADS, HEAD_DIM).float()
        v = pool.v_cache(0).reshape(n, KV_HEADS, HEAD_DIM).float()
        q = q.float()
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    group = q.shape[1] // KV_HEADS
    for row in range(q.shape[0]):
        logical = indices[row][indices[row] >= 0].long()
        slots = table[token_to_req[row].long(), logical // PAGE_SIZE].long() * PAGE_SIZE + logical % PAGE_SIZE
        for head in range(q.shape[1]):
            kv = head // group
            weights = torch.softmax(k[slots, kv] @ q[row, head] * HEAD_DIM**-0.5, dim=0)
            out[row, head] = weights @ v[slots, kv]
    return out


def _row_sets():
    decode = [(0, 700), (1, 130), (2, 1999), (3, 0)]  # one row per request
    verify = [(0, 500 + i) for i in range(4)]  # 1 + depth rows of one request
    prefill = [(0, 40 + i) for i in range(150)] + [(1, 10 + i) for i in range(70)]
    return {"decode": decode, "verify": verify, "prefill": prefill}


@requires_cuda
@pytest.mark.parametrize("kv_cache_dtype", ["model", "turboquant_4bit_nc"])
@pytest.mark.parametrize("case", ["decode", "verify", "prefill"])
@pytest.mark.parametrize("width", [100, 2051])
def test_sparse_attention_matches_dense_reference(kv_cache_dtype, case, width):
    """The per-row sparse kernel over model-dtype and TurboQuant pages equals fp32 attention over
    the same (dequantized) K/V, for decode rows, MTP verify rows and prefill rows."""
    from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

    rows = _row_sets()[case]
    pages_per_req, num_pages = 32, 160
    pool, _, _ = _filled_pool(kv_cache_dtype, num_pages)
    gen = torch.Generator(device="cuda").manual_seed(1)
    q = torch.randn(len(rows), Q_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16, generator=gen)
    indices = _selection(rows, width)
    table, token_to_req = _paged_batch(rows, pages_per_req, num_pages)
    meta = _quant_meta(pool) if kv_cache_dtype == "turboquant_4bit_nc" else {}
    got = qsa_sparse_paged_attention(
        q, pool.k_cache(0), pool.v_cache(0), indices, table, token_to_req, **meta
    )
    reference = _dense_reference(q, pool, kv_cache_dtype, indices, table, token_to_req)
    torch.testing.assert_close(got.float(), reference, rtol=2e-2, atol=2e-3)


@requires_cuda
@pytest.mark.parametrize("tuning", [(16, 1, 4), (16, 8, 2), (32, 64, 4)])
def test_turboquant_sparse_attention_is_independent_of_the_split_profile(tuning):
    from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

    rows = _row_sets()["verify"]
    pool, _, _ = _filled_pool("turboquant_4bit_nc", 64)
    q = torch.randn(len(rows), Q_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16)
    indices = _selection(rows, 400)
    table, token_to_req = _paged_batch(rows, 8, 64)
    run = lambda **kw: qsa_sparse_paged_attention(
        q, pool.k_cache(0), pool.v_cache(0), indices, table, token_to_req, **_quant_meta(pool), **kw
    )
    torch.testing.assert_close(run(tuning=tuning).float(), run().float(), rtol=1e-2, atol=2e-3)


@requires_cuda
@pytest.mark.parametrize("kv_cache_dtype", ["model", "turboquant_4bit_nc"])
@pytest.mark.parametrize("budget", ["default", "batched", "fallback"])
def test_tiled_prefill_matches_the_sparse_kernel(kv_cache_dtype, budget, monkeypatch):
    """GEMM prefill is the same attention as the per-row kernel, and its scratch obeys the byte
    cap: tiles batch to fit it, and a tile whose union does not fit runs through the sparse kernel."""
    import freetoken.kernel.triton.qsa.attend_tiled as tiled
    from freetoken.kernel.triton.qsa import qsa_gemm_attention, qsa_sparse_paged_attention

    rows = [(0, 2300 + i) for i in range(200)] + [(1, 900 + i) for i in range(45)]
    pages_per_req, num_pages = 48, 120
    pool, _, _ = _filled_pool(kv_cache_dtype, num_pages)
    gen = torch.Generator(device="cuda").manual_seed(2)
    q = torch.randn(len(rows), Q_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16, generator=gen)
    indices = _selection(rows, 300)
    table, token_to_req = _paged_batch(rows, pages_per_req, num_pages)
    meta = _quant_meta(pool) if kv_cache_dtype == "turboquant_4bit_nc" else {}
    turbo = bool(meta)
    group = Q_HEADS // KV_HEADS
    one_tile = tiled._tile_bytes(41 * PAGE_SIZE, group, HEAD_DIM, 2, turbo)
    limit = {"default": None, "batched": 3 * one_tile, "fallback": 1}[budget]

    calls = []
    original = tiled._gemm_tiles

    def spy(q_, k_cache, v_cache, out, pages, mask, row_ids, kvh, group_, length, dim, **kw):
        calls.append((row_ids.shape[0], length))
        return original(q_, k_cache, v_cache, out, pages, mask, row_ids, kvh, group_, length, dim, **kw)

    monkeypatch.setattr(tiled, "_gemm_tiles", spy)
    got = qsa_gemm_attention(
        q, pool.k_cache(0), pool.v_cache(0), indices, table, token_to_req, [0, 200, 245],
        gemm_bytes=limit, max_union_ratio=0, **meta,
    )
    expected = qsa_sparse_paged_attention(
        q, pool.k_cache(0), pool.v_cache(0), indices, table, token_to_req, **meta
    )
    torch.testing.assert_close(got.float(), expected.float(), rtol=3e-2, atol=3e-3)
    if budget == "fallback":
        assert not calls
    else:
        assert calls
    if limit is not None and budget != "fallback":
        assert len(calls) > 1
        for tiles, length in calls:
            assert tiles * tiled._tile_bytes(length, group, HEAD_DIM, 2, turbo) <= limit


@requires_cuda
@pytest.mark.parametrize("ratio", [0, 4.0, 20.0])
def test_tiled_prefill_union_ratio_routes_wide_tiles_to_the_sparse_kernel(ratio, monkeypatch):
    """A tile whose page union exceeds ``ratio`` selections' worth of tokens runs through the
    per-row kernel instead (0 disables the cap); either way the attention is the same."""
    import freetoken.kernel.triton.qsa.attend_tiled as tiled
    from freetoken.kernel.triton.qsa import qsa_gemm_attention, qsa_sparse_paged_attention

    rows = [(0, 2300 + i) for i in range(200)] + [(1, 900 + i) for i in range(45)]
    pool, _, _ = _filled_pool("turboquant_4bit_nc", 120)
    gen = torch.Generator(device="cuda").manual_seed(4)
    q = torch.randn(len(rows), Q_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16, generator=gen)
    width = 300
    indices = _selection(rows, width)
    table, token_to_req = _paged_batch(rows, 48, 120)
    meta = _quant_meta(pool)

    lengths = []
    original = tiled._gemm_tiles

    def spy(q_, k_cache, v_cache, out, pages, mask, row_ids, kvh, group_, length, dim, **kw):
        lengths.append(length)
        return original(q_, k_cache, v_cache, out, pages, mask, row_ids, kvh, group_, length, dim, **kw)

    monkeypatch.setattr(tiled, "_gemm_tiles", spy)
    got = qsa_gemm_attention(
        q, pool.k_cache(0), pool.v_cache(0), indices, table, token_to_req, [0, 200, 245],
        max_union_ratio=ratio, **meta,
    )
    expected = qsa_sparse_paged_attention(
        q, pool.k_cache(0), pool.v_cache(0), indices, table, token_to_req, **meta
    )
    torch.testing.assert_close(got.float(), expected.float(), rtol=3e-2, atol=3e-3)
    if ratio == 0:
        assert lengths
    else:
        assert all(length <= ratio * width for length in lengths)
        # the long request's tiles (unions past 4 selections) run as GEMMs only under the loose cap
        assert any(length > 4 * width for length in lengths) == (ratio == 20.0)


@requires_cuda
@pytest.mark.parametrize("kv_cache_dtype", ["model", "turboquant_4bit_nc"])
def test_amd_prefill_tuning_is_the_same_attention(kv_cache_dtype, monkeypatch):
    """FREETOKEN_QSA_AMD_PREFILL_TUNING picks narrow tiles, one split and four warps for a prefill-sized launch (more than 512
    row-head programs); the output matches the default launch."""
    from freetoken.env import ENV
    from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

    if torch.version.hip is None:
        pytest.skip("the tuning is HIP-only")
    rows = [(0, 1500 + i) for i in range(300)]  # 300 rows x 2 KV heads = 600 programs
    pool, _, _ = _filled_pool(kv_cache_dtype, 40)
    gen = torch.Generator(device="cuda").manual_seed(8)
    q = torch.randn(len(rows), Q_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16, generator=gen)
    indices = _selection(rows, 200)
    table, token_to_req = _paged_batch(rows, 30, 40)
    meta = _quant_meta(pool) if kv_cache_dtype == "turboquant_4bit_nc" else {}

    def run():
        return qsa_sparse_paged_attention(q, pool.k_cache(0), pool.v_cache(0), indices, table, token_to_req, **meta)

    monkeypatch.setattr(ENV.QSA_AMD_PREFILL_TUNING, "value", False)
    expected = run()
    monkeypatch.setattr(ENV.QSA_AMD_PREFILL_TUNING, "value", True)
    torch.testing.assert_close(run().float(), expected.float(), rtol=1e-2, atol=2e-3)
