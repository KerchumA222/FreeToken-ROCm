"""MTP verify bookkeeping on the scheduler, CPU only: staging a verify, committing the accepted run,
and rolling the request back to it -- lengths, token pool, KV pages and the GDN state, which is
restored from the snapshot taken after the last committed row."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams, VerifyRow
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager

EOS = 99
NUM_PAGES = 64


def _scheduler(page_size: int, k: int, max_running: int = 4):
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=group, num_slots=max_running + 1, dtype=torch.float32,
                           device=torch.device("cpu"), tp_size=1, spec_slots=max_running * k)
    page_table = torch.zeros(max_running + 1, 256, dtype=torch.int32)
    s = Scheduler.__new__(Scheduler)
    s.engine = SimpleNamespace(linear_state_pool=pool)
    s.cache_manager = CacheManager(NUM_PAGES, page_size, page_table, "naive", linear_state_pool=pool)
    s.table_manager = TableManager(max_running, page_table)
    s.token_pool = s.table_manager.token_pool
    s.decode_manager = DecodeManager(page_size)
    s.config = SimpleNamespace(speculative_num_draft_tokens=k, page_size=page_size)
    s.device = torch.device("cpu")
    s._mtp = SimpleNamespace()
    s.spec_stats = {"rounds": 0, "drafted": 0, "accepted": 0}
    s.eos_token_ids = {EOS}
    s.toolcall_anchor_id = None
    s.finished_reqs = set()
    s._match_stop_str = lambda req: None
    return s, pool


def _decoding_req(s, prompt_len: int, uid: int = 0, output_len: int = 16) -> Req:
    """A request right after its prefill: prompt KV allocated, first token sampled, one draft pending."""
    cm = s.cache_manager
    ids = torch.arange(10, 10 + prompt_len, dtype=torch.int32)
    handle = cm.match_req(SimpleNamespace(input_ids=ids, input_len=prompt_len)).cuda_handle
    req = Req(input_ids=ids, table_idx=s.table_manager.allocate(), cached_len=0, output_len=output_len,
              uid=uid, sampling_params=SamplingParams(max_tokens=output_len), cache_handle=handle)
    cm.lock(handle)
    s.token_pool[req.table_idx, :prompt_len] = ids
    cm.allocate_paged([req])
    req.complete_one()
    req.append_host(torch.tensor([7], dtype=torch.int32))
    s.token_pool[req.table_idx, req.cached_len] = 7
    s.decode_manager.filter_reqs([req])
    return req


def _stage(s, reqs, first_drafts):
    for req, d in zip(reqs, first_drafts):
        req.pending_draft = d
    verify = s._stage_speculation(Batch(reqs=list(reqs), phase="decode"))
    assert verify.spec is not None
    s.cache_manager.allocate_paged(verify.reqs)
    return verify


def _fill_states(pool, reqs, k):
    """Mark the live state and each step's snapshot as the GDN layer would leave them: snapshot s
    holds ``100 * (i + 1) + s``, the live slot the value after the last row (``... + k``)."""
    for i, req in enumerate(reqs):
        live = req.table_idx  # naive GDN keys the state by table_idx
        for s in range(k):
            pool.recurrent_states[:, pool.spec_slot(i * k + s)] = 100 * (i + 1) + s
            pool.conv_states[:, pool.spec_slot(i * k + s)] = 100 * (i + 1) + s
        pool.recurrent_states[:, live] = 100 * (i + 1) + k
        pool.conv_states[:, live] = 100 * (i + 1) + k


def _output(targets_by_row, chain=None, drafts_by_row=None):
    rows = len(targets_by_row)
    return SimpleNamespace(
        next_tokens_cpu=torch.tensor(targets_by_row, dtype=torch.int32),
        chain_tokens_cpu=None if chain is None else torch.tensor(chain, dtype=torch.int32),
        draft_tokens_cpu=torch.tensor(drafts_by_row or [1000 + r for r in range(rows)], dtype=torch.int32),
    )


def _kept_pages(s, req) -> int:
    return -(-req.cached_len // s.cache_manager.page_size)


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize(
    "drafts, targets, committed",
    [
        ([21, 22, 23], [21, 22, 50, 51], [21, 22, 50]),   # two accepted, the target's token at row 2
        ([21, 22, 23], [21, 22, 23, 51], [21, 22, 23, 51]),  # all accepted plus the bonus token
        ([21, 22, 23], [40, 22, 23, 51], [40]),           # none accepted
    ],
)
def test_verify_commits_the_accepted_run_and_rolls_back_the_rest(page_size, drafts, targets, committed):
    k = len(drafts)
    s, pool = _scheduler(page_size, k)
    req = _decoding_req(s, prompt_len=6)
    c = req.cached_len
    verify = _stage(s, [req], [drafts[0]])
    assert [r.cached_len for r in verify.reqs] == [c + j for j in range(k + 1)]
    assert req.device_len == c + 1 + k and int(s.token_pool[req.table_idx, c + 1]) == drafts[0]
    s.token_pool[req.table_idx, c + 2 : c + 1 + k] = torch.tensor(drafts[1:], dtype=torch.int32)
    _fill_states(pool, [req], k)

    reply, finished = [], set()
    s._commit_verified(verify, 0, req, _output(targets, [drafts[1:]]), reply, finished)

    m = len(committed)
    assert [msg.next_token for msg in reply] == committed and not finished
    assert req.cached_len == c + m and req.device_len == c + m + 1
    assert req.input_ids[-m:].tolist() == committed
    # the token pool holds the committed sequence up to the next step's input
    assert s.token_pool[req.table_idx, c + 1 : c + 1 + m].tolist() == committed
    # every page past the committed positions is free again
    assert len(s.cache_manager.free_slots) == NUM_PAGES - _kept_pages(s, req)
    # the GDN state is the one after the last committed row (row m - 1)
    assert pool.recurrent_states[0, req.table_idx, 0, 0, 0].item() == 100 + (m - 1)
    assert pool.conv_states[1, req.table_idx, 0, 0].item() == 100 + (m - 1)
    # the next draft is the head's prediction at that row
    assert req.pending_draft == 1000 + (m - 1) and req.pending_slot == m - 1
    assert s.spec_stats == {"rounds": 1, "drafted": k, "accepted": m - 1}

    # the next step allocates exactly the pages it needs, with no slot handed out twice
    req.device_len += 2
    s.cache_manager.allocate_paged([VerifyRow(req, j) for j in range(3)])
    row = s.cache_manager.page_table[req.table_idx, : req.device_len]
    assert len(set(row.tolist())) == req.device_len
    free = set(s.cache_manager.free_slots.tolist())
    assert not free & set((row - row % page_size).tolist())


def test_verify_rows_are_step_major_across_requests():
    k = 2
    s, pool = _scheduler(1, k)
    a, b = _decoding_req(s, 5, uid=0), _decoding_req(s, 9, uid=1)
    verify = _stage(s, [a, b], [31, 41])
    assert [(r.req.uid, r.step) for r in verify.reqs] == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    _fill_states(pool, [a, b], k)
    # rows: (a,0) (b,0) (a,1) (b,1) (a,2) (b,2); a accepts both drafts, b none
    targets = [31, 98, 32, 0, 33, 0]
    out = _output(targets, chain=[[32], [42]])
    reply, finished = [], set()
    s._commit_verified(verify, 0, a, out, reply, finished)
    s._commit_verified(verify, 1, b, out, reply, finished)
    assert [m.next_token for m in reply] == [31, 32, 33, 98]
    assert a.pending_draft == 1004 and b.pending_draft == 1001
    assert pool.recurrent_states[0, a.table_idx, 0, 0, 0].item() == 100 + k       # all accepted: live
    assert pool.recurrent_states[0, b.table_idx, 0, 0, 0].item() == 200           # row 0 snapshot


def test_a_finish_inside_the_run_rolls_back_to_the_finishing_token():
    """EOS accepted mid-run: the request finishes there, its state and pages describe exactly the
    committed prefix when it is freed, and nothing leaks."""
    k = 3
    s, pool = _scheduler(1, k)
    req = _decoding_req(s, prompt_len=6)
    c = req.cached_len
    verify = _stage(s, [req], [21])
    _fill_states(pool, [req], k)
    live = req.table_idx
    reply, finished = [], set()
    s._commit_verified(verify, 0, req, _output([21, EOS, 23, 24], [[EOS, 23]]), reply, finished)
    assert [m.next_token for m in reply] == [21, EOS]
    assert reply[-1].finished and reply[-1].finish_reason == "stop"
    assert finished == {req} and req.table_idx == -1
    assert req.cached_len == c + 2
    assert pool.recurrent_states[0, live, 0, 0, 0].item() == 101   # restored to row 1, then freed
    s.cache_manager.check_integrity()


def test_force_reject_commits_only_the_first_target(monkeypatch):
    from freetoken.env import ENV

    monkeypatch.setattr(ENV.SPEC_FORCE_REJECT, "value", True)
    k = 2
    s, pool = _scheduler(1, k)
    req = _decoding_req(s, prompt_len=4)
    c = req.cached_len
    verify = _stage(s, [req], [21])
    _fill_states(pool, [req], k)
    reply = []
    s._commit_verified(verify, 0, req, _output([21, 22, 23], [[22]]), reply, set())
    assert [m.next_token for m in reply] == [21]
    assert req.cached_len == c + 1
    assert pool.recurrent_states[0, req.table_idx, 0, 0, 0].item() == 100


def test_stop_inside_accepted_run_counts_only_committed_drafts():
    k = 3
    s, _pool = _scheduler(1, k)
    req = _decoding_req(s, prompt_len=4)
    req.sampling_params.stop_strs = ["STOP"]
    s._match_stop_str = lambda current: "STOP" if int(current.input_ids[-1]) == 22 else None
    verify = _stage(s, [req], [21])
    reply = []
    s._commit_verified(
        verify,
        0,
        req,
        _output([21, 22, 23, 24], [[22, 23]]),
        reply,
        set(),
    )

    assert [msg.next_token for msg in reply] == [21, 22]
    assert reply[-1].finished and reply[-1].finish_reason == "stop"
    assert s.spec_stats == {"rounds": 1, "drafted": 3, "accepted": 2}


def test_eos_inside_accepted_run_counts_only_committed_drafts():
    k = 3
    s, _pool = _scheduler(1, k)
    req = _decoding_req(s, prompt_len=4)
    verify = _stage(s, [req], [21])
    reply = []
    s._commit_verified(
        verify,
        0,
        req,
        _output([21, EOS, 23, 24], [[EOS, 23]]),
        reply,
        set(),
    )

    assert [msg.next_token for msg in reply] == [21, EOS]
    assert reply[-1].finished and reply[-1].finish_reason == "stop"
    assert s.spec_stats == {"rounds": 1, "drafted": 3, "accepted": 2}


def test_a_batch_that_cannot_speculate_stays_a_plain_step():
    s, _ = _scheduler(1, 2)
    greedy, sampled = _decoding_req(s, 4, uid=0), _decoding_req(s, 4, uid=1)
    sampled.sampling_params = SamplingParams(temperature=0.7, max_tokens=16)
    for r in (greedy, sampled):
        r.pending_draft = 5
    batch = Batch(reqs=[greedy, sampled], phase="decode")
    assert s._stage_speculation(batch) is batch
    assert greedy.device_len == greedy.cached_len + 1

    # one token of output budget left: no room for a draft and the token after it
    s2, _ = _scheduler(1, 2)
    short = _decoding_req(s2, 4, output_len=2)
    short.pending_draft = 5
    batch = Batch(reqs=[short], phase="decode")
    assert short.remain_len == 1 and s2._stage_speculation(batch) is batch


def test_depth_is_capped_by_the_remaining_budget():
    s, _ = _scheduler(1, 3)
    req = _decoding_req(s, 4, output_len=3)
    assert req.remain_len == 2
    verify = _stage(s, [req], [5])
    assert verify.spec.k == 1 and req.device_len == req.max_device_len - 1


def test_mtp_prefill_mixed_chunked_batch_keeps_prompt_boundary_token():
    from freetoken.speculative.mtp import MtpDrafter

    chunked = ChunkedReq(
        input_ids=torch.tensor([10, 11], dtype=torch.int32), table_idx=0, cached_len=0,
        output_len=8, uid=0, sampling_params=SamplingParams(), cache_handle=None,
    )
    complete = Req(
        input_ids=torch.tensor([20, 21, 22], dtype=torch.int32), table_idx=1, cached_len=0,
        output_len=8, uid=1, sampling_params=SamplingParams(), cache_handle=None,
    )
    batch = Batch(reqs=[chunked, complete], phase="prefill")
    batch.padded_reqs = [chunked, complete]

    s = Scheduler.__new__(Scheduler)
    s.device = torch.device("cpu")
    s._mtp = SimpleNamespace(store_index=lambda table, row: table * 2 + row)
    mapping = (torch.tensor([0, 1]), torch.tensor([0, 0]))
    plan = s._mtp_plan(batch, mapping)

    assert plan.draft_rows.tolist() == [1, 4]
    assert plan.sampled_rows.tolist() == [4]
    assert plan.sampled_src_rows.tolist() == [1]

    batch.mtp_plan = plan
    batch.hidden_states = torch.zeros((5, 1))
    # The first sampler result belongs to the chunked request and must not replace its
    # staged next-prompt token. The complete request's result comes from sampler row 1.
    plan.next_ids = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int32)
    seen = {}

    class Embed:
        def forward(self, tokens):
            seen["tokens"] = tokens.clone()
            return tokens.to(torch.float32).unsqueeze(-1)

    class Head:
        def forward_chain(self, hidden, embed):
            out = hidden + embed
            return out, out

    drafter = object.__new__(MtpDrafter)
    drafter.device = torch.device("cpu")
    drafter.depth = 1
    drafter.store = None
    drafter._page_table = lambda: torch.zeros((2, 1), dtype=torch.int32)
    drafter.embed = Embed()
    drafter.head = Head()
    drafter.logits = lambda out: out

    drafts = drafter.head_pass(batch, torch.tensor([77, 88], dtype=torch.int32))
    assert seen["tokens"].tolist() == [10, 11, 20, 21, 88]
    assert drafts.shape == (2,)


def test_mtp_verify_snapshots_hybrid_toolcall_anchor_before_forward():
    from freetoken.scheduler.scheduler import ForwardInput

    s, pool = _scheduler(page_size=1, k=1)
    s.cache_manager = CacheManager(
        NUM_PAGES, 1, s.cache_manager.page_table, "hybrid_radix", linear_state_pool=pool
    )
    s.toolcall_anchor_id = 123
    req = _decoding_req(s, prompt_len=6)
    anchor = req.cached_len
    req.toolcall_anchor_len = anchor
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 0
    live_state = torch.full_like(pool.recurrent_states[:, req.linear_slot_idx], 3.5)
    live_conv = torch.full_like(pool.conv_states[:, req.linear_slot_idx], 2.5)
    pool.recurrent_states[:, req.linear_slot_idx].copy_(live_state)
    pool.conv_states[:, req.linear_slot_idx].copy_(live_conv)

    verify = _stage(s, [req], [21])
    assert verify.reqs[0].req is req  # verify rows wrap the underlying request
    events = []
    result = object()

    class Engine:
        def draft_chain(self, _batch, _token_pool):
            events.append("draft")
            assert req.mamba_last_track_seqlen == anchor
            assert req.mamba_next_track_idx == 1
            assert torch.equal(pool.recurrent_states[:, req.mamba_ping_pong[0]], live_state)
            assert torch.equal(pool.conv_states[:, req.mamba_ping_pong[0]], live_conv)
            return object()

        def forward_batch(self, _batch, _sample_args, _chain):
            events.append("forward")
            return result

    s.engine = Engine()
    s.decode_manager = SimpleNamespace(filter_reqs=lambda _reqs: events.append("filter"))
    mapping = (torch.tensor([req.table_idx]), torch.tensor([req.cached_len]))
    output = s._forward(ForwardInput(verify, None, mapping, None))

    assert output is result
    assert events == ["draft", "forward", "filter"]
