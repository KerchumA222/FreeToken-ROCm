"""FREETOKEN_SPEC_DRAFT_MIN_P (confidence-adaptive draft depth) beyond the unit cases: consecutive verify steps
whose depth changes, batches whose requests want different depths, finishes inside a truncated verify,
sampled / relaxed-acceptance pairing of rows with drafts, and the flag-off path. CPU only."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from test_verify_rollback import EOS, NUM_PAGES, _decoding_req, _kept_pages, _prepare_verify, _scheduler

from freetoken.core import Batch, SamplingParams
from freetoken.engine.sample import BatchSamplingArgs, Sampler, keep_likely_drafts
from freetoken.scheduler.scheduler import ForwardInput
from freetoken.speculative import batch_draft_depth, draft_depth


def _ref_depth(confs, p, k):
    """Draft j joins while drafts 1..j-1 all had confidence >= p (written independently of draft_depth)."""
    depth = 1
    while depth < k and confs[depth - 1] >= p:
        depth += 1
    return depth


class _PlannedEngine:
    """Fake engine. draft_chain follows MtpDrafter.draft_chain under the flag: it extends to the deepest
    request's depth, writes those drafts into the token pool and returns [B, depth - 1] (None for depth 1)."""

    def __init__(self, s, reqs, chain, confs):
        self.linear_state_pool = s.engine.linear_state_pool
        self.attn_backend = SimpleNamespace(prepare_metadata=self.prepare_metadata)
        self.s, self.reqs, self.chain, self.confs = s, reqs, chain, confs
        self.metadata_rows = []
        self.forwarded = None
        self.input_ids = None
        self.last_chain = None
        self.first_conf_seen = "unset"

    def prepare_metadata(self, batch):
        self.metadata_rows.append([(r.req.uid, r.step, r.cached_len, r.device_len) for r in batch.padded_reqs])

    def draft_chain(self, batch, token_pool):
        from freetoken.env import ENV

        spec = batch.spec
        self.first_conf_seen = spec.first_conf
        if spec.k < 2:
            return None
        min_p = ENV.SPEC_DRAFT_MIN_P.value
        if min_p is None:
            depth = spec.k
        else:
            confs = [[fc] + self.confs[i] for i, fc in enumerate(spec.first_conf)]
            depth = batch_draft_depth(confs, min_p, spec.k)
        if depth < 2:
            return None
        for i, req in enumerate(self.reqs):
            c = req.cached_len
            token_pool[req.table_idx, c + 2 : c + 1 + depth] = torch.tensor(self.chain[i][: depth - 1], dtype=torch.int32)
        self.last_chain = torch.tensor([row[: depth - 1] for row in self.chain], dtype=torch.int32)
        return self.last_chain

    def forward_batch(self, batch, sample_args, chain):
        self.forwarded = (batch, sample_args, chain)
        self.input_ids = batch.input_ids.tolist()
        return "output"


def _launch(s, reqs, chain, confs, first=None, args=None):
    """Stage and forward one verify through the real Scheduler._stage_speculation / _forward."""
    if first is not None:
        for req, d in zip(reqs, first):
            req.pending_draft = d
    verify = s._stage_speculation(Batch(reqs=list(reqs), phase="decode"))
    assert verify.spec is not None
    s.cache_manager.allocate_paged(verify.reqs)
    sample_args, input_mapping, write_mapping = _prepare_verify(s, verify)
    if args is not None:
        sample_args = args(verify.size)
    engine = _PlannedEngine(s, reqs, chain, confs)
    s.engine = engine
    assert s._forward(ForwardInput(verify, sample_args, input_mapping, write_mapping)) == "output"
    return verify, engine


def _outputs(verify, engine, accepts, tag, out_conf=None):
    """The step-major result of a forward: request i accepts ``accepts[i]`` drafts, then the target
    disagrees (or gives the bonus token)."""
    spec = verify.spec
    B, k = spec.num_reqs, spec.k
    drafts = [[spec.first_drafts[i]] + (engine.last_chain[i].tolist() if k > 1 else []) for i in range(B)]
    rows = [0] * ((k + 1) * B)
    for i in range(B):
        a = min(accepts[i], k)
        for st in range(k + 1):
            rows[st * B + i] = drafts[i][st] if st < a else (60 + 10 * i + st if st == a else 0)
    out = SimpleNamespace(
        next_tokens_cpu=torch.tensor(rows, dtype=torch.int32),
        chain_tokens_cpu=engine.last_chain,
        draft_tokens_cpu=torch.tensor([1000 * (tag + 1) + 10 * st + i for st in range(k + 1) for i in range(B)], dtype=torch.int32),
    )
    if out_conf is not None:
        out.draft_conf_cpu = torch.tensor([out_conf[i][st] for st in range(k + 1) for i in range(B)], dtype=torch.float32)
    return out, drafts, rows


def _mark_states(pool, reqs, k_config, spec_k, tag):
    """Snapshot ``sl`` of request i holds tag*1000 + 100*(i+1) + sl; the live state is the one after the
    last staged row. Snapshot slots past the verify's depth are poisoned, so reading one shows."""
    for i, req in enumerate(reqs):
        base = tag * 1000 + 100 * (i + 1)
        for sl in range(k_config):
            v = base + sl if sl < spec_k else -1
            pool.recurrent_states[:, pool.spec_slot(i * k_config + sl)] = v
            pool.conv_states[:, pool.spec_slot(i * k_config + sl)] = v
        pool.recurrent_states[:, req.table_idx] = base + spec_k
        pool.conv_states[:, req.table_idx] = base + spec_k


def _assert_pages_exact(s, reqs):
    cm, ps = s.cache_manager, s.cache_manager.page_size
    kept = sum(_kept_pages(s, r) for r in reqs)
    free = cm.free_slots.tolist()
    assert len(free) == NUM_PAGES - kept, "page leak or double free"
    assert len(set(free)) == len(free)
    held = set()
    for r in reqs:
        row = cm.page_table[r.table_idx, : r.cached_len]
        held |= set((row - row % ps).tolist())
    assert len(held) == kept and not held & set(free)


def _commit_all(s, verify, reqs, out):
    reply, finished = [], set()
    for i, req in enumerate(reqs):
        s._commit_verified(verify, i, req, out, reply, finished)
    return reply, finished


def _run_steps(s, pool, reqs, k, steps, first, p=0.5):
    """Drive consecutive verify steps; ``steps`` is a list of dicts with per-request ``chain`` (drafts 2..k),
    ``confs`` (confidence of drafts 2..k), ``accept`` and ``out_conf`` (the head's confidence at each
    row of the next forward, k + 1 entries). Everything is checked against an independent model."""
    B = len(reqs)
    base = [r.input_ids.numel() for r in reqs]
    expected_tokens = [[7] for _ in reqs]
    exp_first = list(first)
    exp_conf = [1.0] * B  # no pending_conf yet: counts as confident
    stats = {"rounds": 0, "drafted": 0, "accepted": 0}
    verifies = []
    for n, st in enumerate(steps):
        c0 = [r.cached_len for r in reqs]
        verify, engine = _launch(s, reqs, st["chain"], st["confs"], first=first if n == 0 else None)
        assert [r.pending_draft for r in reqs] == [None] * B
        assert verify.spec.first_conf == pytest.approx(exp_conf) and verify.spec.first_drafts == exp_first
        depth = max(_ref_depth([exp_conf[i]] + st["confs"][i], p, k) for i in range(B))
        spec = verify.spec
        assert spec.k == depth and spec.staged_k == k and len(spec.snap_slots) == depth
        assert len(engine.metadata_rows) == (1 if depth < k else 0)
        assert verify.size == len(verify.padded_reqs) == (depth + 1) * B
        assert [(r.req.uid, r.step) for r in verify.reqs] == [(reqs[i].uid, sidx) for sidx in range(depth + 1) for i in range(B)]
        assert verify.positions.tolist() == [c0[i] + sidx for sidx in range(depth + 1) for i in range(B)]
        assert verify.out_loc.tolist() == [
            int(s.cache_manager.page_table[reqs[i].table_idx, c0[i] + sidx]) for sidx in range(depth + 1) for i in range(B)
        ]
        last = [int(s.token_pool[r.table_idx, r.cached_len]) for r in reqs]
        assert last == [t[-1] for t in expected_tokens]
        assert engine.input_ids == [
            last[i] if sidx == 0 else ([exp_first[i]] + st["chain"][i])[sidx - 1] for sidx in range(depth + 1) for i in range(B)
        ]
        assert [r.device_len for r in reqs] == [c + 1 + depth for c in c0]
        assert len(verify.mtp_plan.store_idx) == len(verify.fla_metadata.cache_indices) == (depth + 1) * B
        assert len(verify.fla_metadata.cu_seqlens) == (depth + 1) * B + 1

        out, drafts, rows = _outputs(verify, engine, st["accept"], n, st["out_conf"])
        _mark_states(pool, reqs, k, depth, n + 1)
        reply, finished = _commit_all(s, verify, reqs, out)
        assert not finished
        for i, req in enumerate(reqs):
            a = min(st["accept"][i], depth)
            m = a + 1
            tokens = drafts[i][:a] + [rows[a * B + i]]
            expected_tokens[i] += tokens
            assert [x.next_token for x in reply if x.uid == req.uid] == tokens
            assert req.cached_len == c0[i] + m and req.device_len == req.cached_len + 1
            assert req.input_ids[base[i] - 1 :].tolist() == expected_tokens[i]
            assert s.token_pool[req.table_idx, base[i] - 1 : req.cached_len + 1].tolist() == expected_tokens[i]
            assert req.pending_slot == m - 1
            assert req.pending_draft == 1000 * (n + 1) + 10 * (m - 1) + i
            assert req.pending_conf == pytest.approx(st["out_conf"][i][m - 1])
            assert pool.recurrent_states[0, req.table_idx, 0, 0, 0].item() == (n + 1) * 1000 + 100 * (i + 1) + (m - 1)
            assert pool.conv_states[1, req.table_idx, 0, 0].item() == (n + 1) * 1000 + 100 * (i + 1) + (m - 1)
            stats["accepted"] += a
            exp_first[i], exp_conf[i] = req.pending_draft, st["out_conf"][i][m - 1]
        stats["rounds"] += B
        stats["drafted"] += depth * B
        assert s.spec_stats == stats
        _assert_pages_exact(s, reqs)
        verifies.append(verify)
    return verifies


@pytest.fixture
def flag(monkeypatch):
    from freetoken.env import ENV

    def set_p(p):
        monkeypatch.setattr(ENV.SPEC_DRAFT_MIN_P, "value", p)

    return set_p


HI, LO = 0.9, 0.1


@pytest.mark.parametrize("page_size", [1, 4])
def test_depth_changes_between_consecutive_steps_on_one_request(flag, page_size):
    """deep (4, accept 2) -> shallow (1, all accepted) -> deep (3, none accepted) -> mid (2) -> full (4, all accepted)."""
    flag(0.5)
    k = 4
    s, pool = _scheduler(page_size, k)
    req = _decoding_req(s, 6)
    steps = [
        dict(chain=[[32, 33, 34]], confs=[[HI, HI, HI]], accept=[2], out_conf=[[0.9, 0.9, LO, 0.9, 0.9]]),
        dict(chain=[[42, 43, 44]], confs=[[HI, HI, HI]], accept=[1], out_conf=[[0.9, 0.95, 0.9, 0.9, 0.9]]),
        dict(chain=[[52, 53, 54]], confs=[[HI, 0.2, HI]], accept=[0], out_conf=[[HI, HI, HI, HI, HI]]),
        dict(chain=[[62, 63, 64]], confs=[[0.3, HI, HI]], accept=[1], out_conf=[[HI, HI, HI, HI, HI]]),
        dict(chain=[[72, 73, 74]], confs=[[HI, HI, HI]], accept=[4], out_conf=[[HI, HI, HI, HI, HI]]),
    ]
    verifies = _run_steps(s, pool, [req], k, steps, first=[31])
    assert [v.spec.k for v in verifies] == [4, 1, 3, 2, 4]
    assert [v.spec.staged_k for v in verifies] == [4] * 5


@pytest.mark.parametrize("page_size", [1, 4])
def test_requests_with_different_depths_in_one_batch_over_several_steps(flag, page_size):
    """Three requests; each step verifies the deepest one's depth, and every request commits with its own acceptance."""
    flag(0.5)
    k = 4
    s, pool = _scheduler(page_size, k)
    reqs = [_decoding_req(s, 5, uid=0), _decoding_req(s, 9, uid=1), _decoding_req(s, 6, uid=2)]
    chain = [[32, 33, 34], [42, 43, 44], [52, 53, 54]]
    steps = [
        # own depths 2, 3, 2 -> the batch runs 3 of 4
        dict(chain=chain, confs=[[LO, HI, HI], [HI, 0.2, HI], [0.4, HI, HI]], accept=[3, 1, 0],
             out_conf=[[HI] * 3 + [LO, HI], [HI, 0.8, HI, HI, HI], [0.1, HI, HI, HI, HI]]),
        # first confs are now (0.1, 0.8, 0.1): depths 1, 2, 1 -> the batch runs 2
        dict(chain=chain, confs=[[HI, HI, HI], [0.3, HI, HI], [HI, HI, HI]], accept=[1, 2, 0],
             out_conf=[[LO] * 5, [HI] * 5, [LO] * 5]),
        # first confs 0.1, 0.9, 0.1: only request 1 can extend, and its draft 2 is unsure: depth 2
        dict(chain=chain, confs=[[HI, HI, HI], [LO, HI, HI], [HI, HI, HI]], accept=[1, 0, 1],
             out_conf=[[HI] * 5, [HI] * 5, [HI] * 5]),
        # everything confident: the full, untruncated depth
        dict(chain=chain, confs=[[HI, HI, HI]] * 3, accept=[4, 0, 2], out_conf=[[HI] * 5] * 3),
    ]
    verifies = _run_steps(s, pool, reqs, k, steps, first=[31, 41, 51])
    assert [v.spec.k for v in verifies] == [3, 2, 2, 4]


def test_staged_pages_are_freed_per_request_from_the_staged_depth(flag):
    """Requests whose staged page ranges end on different pages: after a truncated verify each gives back exactly its own surplus."""
    flag(0.5)
    k, page_size = 4, 4
    s, pool = _scheduler(page_size, k)
    # cached_len 6 and 7: the full verify reaches positions 6..10 and 7..11, one page beyond what each holds; depth 1 needs less
    a, b = _decoding_req(s, 6, uid=0), _decoding_req(s, 7, uid=1)
    a.pending_conf = b.pending_conf = LO  # depth 1
    free_before = len(s.cache_manager.free_slots)
    verify, engine = _launch(s, [a, b], [[32, 33, 34], [42, 43, 44]], [[LO, HI, HI]] * 2, first=[31, 41])
    assert verify.spec.k == 1 and verify.spec.staged_k == 4
    assert free_before - len(s.cache_manager.free_slots) == 2  # one extra page each, staged for the full depth
    _mark_states(pool, [a, b], k, 1, 1)
    out, _drafts, _rows = _outputs(verify, engine, [1, 0], 0)
    _commit_all(s, verify, [a, b], out)
    assert a.cached_len == 8 and b.cached_len == 8
    _assert_pages_exact(s, [a, b])


def test_state_is_restored_from_the_snapshot_at_the_last_committed_row_of_a_truncated_verify(flag):
    flag(0.5)
    k = 4
    s, pool = _scheduler(1, k)
    reqs = [_decoding_req(s, 5, uid=0), _decoding_req(s, 8, uid=1)]
    verify, engine = _launch(s, reqs, [[32, 33, 34], [42, 43, 44]], [[HI, LO, HI]] * 2, first=[31, 41])
    assert verify.spec.k == 3  # both stop at draft 3
    _mark_states(pool, reqs, k, 3, 1)
    out, _d, _r = _outputs(verify, engine, [2, 3], 0)
    _commit_all(s, verify, reqs, out)
    # request 0: m = 3 -> snapshot 2; request 1: all 3 accepted + bonus, m = 4 -> the live state (after row 3), not slot 3
    assert pool.recurrent_states[0, reqs[0].table_idx, 0, 0, 0].item() == 1000 + 100 + 2
    assert pool.recurrent_states[0, reqs[1].table_idx, 0, 0, 0].item() == 1000 + 200 + 3
    assert pool.conv_states[1, reqs[1].table_idx, 0, 0].item() == 1000 + 200 + 3
    _assert_pages_exact(s, reqs)


# ---- EOS, stop strings and the output budget -----------------------------------------------------------------


@pytest.mark.parametrize("page_size", [1, 4])
def test_eos_inside_the_accepted_run_of_a_truncated_verify(flag, page_size):
    flag(0.5)
    k = 4
    s, pool = _scheduler(page_size, k)
    a, b = _decoding_req(s, 5, uid=0), _decoding_req(s, 9, uid=1)
    verify, engine = _launch(s, [a, b], [[EOS, 33, 34], [42, 43, 44]], [[HI, LO, HI], [HI, LO, HI]], first=[31, 41])
    assert verify.spec.k == 3 and verify.spec.staged_k == 4
    ca, cb = a.cached_len, b.cached_len
    tbl_a = a.table_idx
    _mark_states(pool, [a, b], k, 3, 1)
    out, _d, _r = _outputs(verify, engine, [3, 3], 0)  # a's second draft is EOS: accepted, then the run ends
    reply, finished = _commit_all(s, verify, [a, b], out)
    assert [m.next_token for m in reply if m.uid == 0] == [31, EOS]
    assert reply[1].finished and reply[1].finish_reason == "stop" and finished == {a}
    assert a.table_idx == -1 and a.cached_len == ca + 2 and tbl_a != b.table_idx
    assert [m.next_token for m in reply if m.uid == 1] == [41, 42, 43, 73]
    assert b.cached_len == cb + 4 and b.pending_draft is not None
    assert s.spec_stats == {"rounds": 2, "drafted": 6, "accepted": 2 + 3}
    # b's pages are intact and a's all went back
    _assert_pages_exact(s, [b])
    assert pool.recurrent_states[0, b.table_idx, 0, 0, 0].item() == 1000 + 200 + 3


def test_stop_string_inside_a_truncated_verify_finishes_at_the_matching_token(flag):
    flag(0.5)
    k = 4
    s, pool = _scheduler(1, k)
    req = _decoding_req(s, 5)
    req.sampling_params.stop_strs = ["STOP"]
    s._match_stop_str = lambda current: "STOP" if int(current.input_ids[-1]) == 32 else None
    verify, engine = _launch(s, [req], [[32, 33, 34]], [[HI, LO, HI]], first=[31])
    assert verify.spec.k == 3
    out, _d, _r = _outputs(verify, engine, [3], 0)
    reply, finished = _commit_all(s, verify, [req], out)
    assert [m.next_token for m in reply] == [31, 32] and reply[-1].finish_reason == "stop"
    assert reply[-1].matched_stop == "STOP" and finished == {req} and req.table_idx == -1
    assert s.spec_stats == {"rounds": 1, "drafted": 3, "accepted": 2}
    s.cache_manager.check_integrity()


def test_output_budget_caps_k_and_a_length_finish_in_a_truncated_verify(flag):
    flag(0.5)
    k = 4
    s, pool = _scheduler(1, k)
    # remain_len 4 -> k = 3 drafts; the chain stops at depth 2 (truncated from the staged 3)
    req = _decoding_req(s, 5, output_len=5)
    assert req.remain_len == 4
    verify, engine = _launch(s, [req], [[32, 33, 34]], [[LO, HI, HI]], first=[31])
    assert verify.spec.staged_k == 3 and verify.spec.k == 2 and req.device_len == req.cached_len + 3
    _mark_states(pool, [req], k, 2, 1)
    out, _d, _r = _outputs(verify, engine, [2], 0)
    reply, finished = _commit_all(s, verify, [req], out)
    assert [m.next_token for m in reply] == [31, 32, 62]
    assert not finished and req.remain_len == 1  # 3 of the 4 remaining tokens committed
    _assert_pages_exact(s, [req])
    # the last token's step has room for no draft: a plain step
    plain = s._stage_speculation(Batch(reqs=[req], phase="decode"))
    assert plain.spec is None


def test_all_accepted_to_the_budget_finishes_with_length(flag):
    flag(0.5)
    k = 4
    s, pool = _scheduler(1, k)
    req = _decoding_req(s, 5, output_len=4)  # remain_len 3 -> k = 2, which is also the depth: no truncation
    verify, engine = _launch(s, [req], [[32, 33, 34]], [[HI, HI, HI]], first=[31])
    assert verify.spec.k == verify.spec.staged_k == 2 and engine.metadata_rows == []
    out, _d, _r = _outputs(verify, engine, [2], 0)
    reply, finished = _commit_all(s, verify, [req], out)
    assert [m.next_token for m in reply] == [31, 32, 60 + 2] and finished == {req}
    assert reply[-1].finish_reason == "length" and reply[0].finish_reason is None


# ---- sampled requests and relaxed acceptance -----------------------------------------------------------------


def _sampled_params(i):
    return SamplingParams(temperature=[0.7, 1.3, 0.0][i], top_k=[50, -1, -1][i], top_p=[1.0, 0.8, 1.0][i], max_tokens=16)


def _step_major_args(B, params):
    """Sampler args for a staged verify, built the way Sampler.prepare does for its rows (step-major over B requests)."""
    def make(n):
        rows = [params[r % B] for r in range(n)]
        greedy = [p.is_greedy for p in rows]
        return BatchSamplingArgs(
            temperatures=torch.tensor([1.0 if g else p.temperature for p, g in zip(rows, greedy)]),
            top_k=torch.tensor([p.top_k if p.top_k >= 1 and not g else 1000 for p, g in zip(rows, greedy)], dtype=torch.int32),
            top_p=torch.tensor([1.0 if g else p.top_p for p, g in zip(rows, greedy)]),
            greedy_mask=torch.tensor(greedy),
            penalties=[],
        )

    return make


def test_sampled_requests_speculate_with_the_flag_and_get_their_own_sampling_rows_after_truncation(flag, monkeypatch):
    from freetoken.env import ENV

    flag(0.5)
    k, B = 4, 3
    s, pool = _scheduler(1, k)
    reqs = [_decoding_req(s, 5 + i, uid=i) for i in range(B)]
    params = [_sampled_params(i) for i in range(B)]
    for r, p in zip(reqs, params):
        r.sampling_params = p
        r.pending_draft = 5
    # a sampled request does not speculate without FREETOKEN_SPEC_SAMPLED
    probe = Batch(reqs=list(reqs), phase="decode")
    assert s._stage_speculation(probe) is probe
    monkeypatch.setattr(ENV.SPEC_SAMPLED, "value", True)
    make = _step_major_args(B, params)
    verify, engine = _launch(s, reqs, [[32, 33, 34]] * B, [[HI, LO, HI]] * B, first=[31, 41, 51], args=make)
    assert verify.spec.k == 3
    got = engine.forwarded[1]
    want = make(verify.size)
    assert verify.size == 12 and make(verify.spec.staged_k * B + B).temperatures.numel() == 15
    for name in ("temperatures", "top_k", "top_p", "greedy_mask"):
        assert getattr(got, name).tolist() == getattr(want, name).tolist(), name
    # row r belongs to request r % B
    assert got.temperatures.tolist() == pytest.approx([0.7, 1.3, 1.0] * 4)
    assert got.greedy_mask.tolist() == [False, False, True] * 4
    assert [r.step for r in verify.reqs] == [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3


@pytest.mark.parametrize("depth_conf", [[LO, HI, HI], [HI, LO, HI], [HI, HI, LO]])  # depths 2, 3, 4 (not cut)
def test_relaxed_acceptance_pairs_every_row_with_its_own_draft_after_truncation(flag, monkeypatch, depth_conf):
    """Engine._sample_verify reads drafts from batch.input_ids rows B..; after the verify was cut those are still the right ones."""
    from freetoken.engine.engine import Engine
    from freetoken.env import ENV

    flag(0.5)
    monkeypatch.setattr(ENV.SPEC_ACCEPT_MIN_P, "value", 0.25)
    k, B, V = 4, 3, 12
    s, pool = _scheduler(1, k)
    reqs = [_decoding_req(s, 5 + i, uid=i) for i in range(B)]
    params = [_sampled_params(i) for i in range(B)]
    for r, p in zip(reqs, params):
        r.sampling_params = p
    monkeypatch.setattr(ENV.SPEC_SAMPLED, "value", True)
    chain = [[2, 3, 4], [6, 7, 8], [9, 10, 1]]
    verify, engine = _launch(s, reqs, chain, [depth_conf] * B, first=[1, 5, 8], args=_step_major_args(B, params))
    spec = verify.spec
    depth = spec.k
    assert depth == _ref_depth([1.0] + depth_conf, 0.5, k)
    drafts = [[[1, 5, 8][i]] + chain[i] for i in range(B)]
    assert verify.input_ids.tolist() == [7 if st == 0 else drafts[i][st - 1] for st in range(depth + 1) for i in range(B)]
    # the probability the target gives the draft scored at (request, step); the rest goes to token 11, the remainder to 0
    probs = {(i, st): 0.05 + 0.07 * ((3 * st + 2 * i) % 6) for i in range(B) for st in range(depth)}
    temps = [params[i].temperature if not params[i].is_greedy else 1.0 for i in range(B)]
    logits = torch.zeros(verify.size, V)
    for st in range(depth + 1):
        for i in range(B):
            row = st * B + i
            dist = torch.full((V,), 1e-6)
            if st < depth:
                pd = probs[(i, st)]
                dist[drafts[i][st]] = pd
                dist[11] = (1 - pd) * 0.7
                dist[0] = (1 - pd) * 0.3
            else:
                dist[11] = 1.0
            logits[row] = temps[i] * torch.log(dist)
    stub = SimpleNamespace(sampler=SimpleNamespace(sample=lambda lg, args: lg.argmax(dim=-1)))
    out = Engine._sample_verify(stub, verify, logits, engine.forwarded[1])
    for row in range(verify.size):
        st, i = divmod(row, B)
        kept = st < depth and probs[(i, st)] >= 0.25
        assert out[row].item() == (drafts[i][st] if kept else 11), (row, st, i)
    assert out.dtype == torch.int32 and out.numel() == verify.size


# ---- BatchSamplingArgs.first_rows ----------------------------------------------------------------------------


def test_first_rows_cuts_every_per_row_tensor_and_keeps_none():
    args = BatchSamplingArgs(temperatures=None)
    cut = args.first_rows(3)
    assert cut.temperatures is cut.top_k is cut.top_p is cut.greedy_mask is None and cut.penalties == []

    full = BatchSamplingArgs(
        temperatures=torch.arange(1.0, 7.0), top_k=torch.arange(10, 16, dtype=torch.int32),
        top_p=torch.linspace(0.1, 0.6, 6), greedy_mask=torch.tensor([True, False] * 3),
    )
    cut = full.first_rows(4)
    assert cut.temperatures.tolist() == [1.0, 2.0, 3.0, 4.0] and cut.top_k.tolist() == [10, 11, 12, 13]
    assert cut.top_p.shape == (4,) and cut.greedy_mask.tolist() == [True, False, True, False]
    assert cut.greedy_mask.dtype == torch.bool and cut.top_k.dtype == torch.int32
    assert full.temperatures.numel() == 6  # the original is untouched
    assert full.first_rows(0).temperatures.numel() == 0
    assert full.first_rows(99).temperatures.tolist() == full.temperatures.tolist()


def test_first_rows_keeps_the_penalty_rows_that_remain_and_shares_their_counts():
    counts = [torch.zeros(5, dtype=torch.int32) for _ in range(3)]
    args = BatchSamplingArgs(temperatures=None, penalties=[(0, counts[0], 0.5, 0.1), (3, counts[1], 0.0, 0.2), (5, counts[2], 1.0, 0.0)])
    cut = args.first_rows(4)
    assert [p[0] for p in cut.penalties] == [0, 3] and cut.penalties[1][1] is counts[1]
    assert [p[0] for p in args.first_rows(3).penalties] == [0] and args.first_rows(0).penalties == []
    assert len(args.penalties) == 3


def test_sampling_a_cut_batch_with_penalties_matches_sampling_its_prefix():
    """Greedy rows with penalties: cutting the logits and the args together gives the prefix of the full result,
    and the dropped rows' counts are not touched."""
    V = 6
    sampler = Sampler(device=torch.device("cpu"), vocab_size=V)
    torch.manual_seed(0)
    logits = torch.randn(6, V)
    counts_a, counts_b = torch.tensor([0, 2, 0, 0, 0, 0], dtype=torch.int32), torch.zeros(V, dtype=torch.int32)
    ref_counts_a = counts_a.clone()
    args = BatchSamplingArgs(temperatures=None, penalties=[(1, counts_a, 0.5, 1.0), (4, counts_b, 1.0, 0.0)])
    ref = sampler.sample(logits.clone(), BatchSamplingArgs(temperatures=None, penalties=[(1, ref_counts_a, 0.5, 1.0)]))
    out = sampler.sample(logits[:3].clone(), args.first_rows(3))
    assert out.tolist() == ref[:3].tolist()
    assert counts_a.tolist() == ref_counts_a.tolist() and counts_b.tolist() == [0] * V


def test_sampling_a_cut_batch_passes_the_cut_tensors_and_honours_the_greedy_mask(monkeypatch):
    import freetoken.engine.sample as sample_mod

    seen = {}

    def fake_impl(logits, temperatures, top_k, top_p):
        seen.update(logits=logits.shape[0], t=temperatures.tolist(), k=top_k.tolist(), p=top_p.tolist())
        return torch.full((logits.shape[0],), 5)

    monkeypatch.setattr(sample_mod, "sample_impl", fake_impl)
    sampler = Sampler(device=torch.device("cpu"), vocab_size=8)
    logits = torch.zeros(6, 8)
    logits[torch.arange(6), torch.tensor([1, 2, 3, 4, 6, 7])] = 3.0
    args = BatchSamplingArgs(
        temperatures=torch.tensor([0.5, 1.0, 2.0, 1.0, 1.0, 1.0]), top_k=torch.tensor([4, 8, 8, 8, 8, 8], dtype=torch.int32),
        top_p=torch.tensor([1.0, 0.9, 1.0, 1.0, 1.0, 1.0]), greedy_mask=torch.tensor([False, True, False, True, True, True]),
    )
    out = sampler.sample(logits[:4], args.first_rows(4))
    assert out.tolist() == [5, 2, 5, 4]  # rows 1 and 3 are greedy: argmax, not the fake sample
    assert seen == dict(logits=4, t=[0.5, 1.0, 2.0, 1.0], k=[4, 8, 8, 8], p=pytest.approx([1.0, 0.9, 1.0, 1.0]))


# ---- flag unset ----------------------------------------------------------------------------------------------


def test_without_the_flag_nothing_is_tracked_staged_or_truncated(monkeypatch):
    k = 3
    s, pool = _scheduler(1, k)

    def boom(*_a, **_kw):
        raise AssertionError("_truncate_verify called without the flag")

    monkeypatch.setattr(s, "_truncate_verify", boom)
    reqs = [_decoding_req(s, 5, uid=0), _decoding_req(s, 9, uid=1)]
    reqs[0].pending_conf = 0.01  # a stale low confidence must not matter
    verify, engine = _launch(s, reqs, [[32, 33], [42, 43]], [[LO, LO], [LO, LO]], first=[31, 41])
    spec = verify.spec
    assert spec.first_conf is None and engine.first_conf_seen is None
    assert spec.k == spec.staged_k == k and verify.size == (k + 1) * 2 and engine.metadata_rows == []
    assert [r.device_len for r in reqs] == [r.cached_len + 1 + k for r in reqs]
    assert verify.draft_conf is None and engine.forwarded[2].shape == (2, k - 1)
    _mark_states(pool, reqs, k, k, 1)
    out, _d, _r = _outputs(verify, engine, [3, 1], 0)  # the engine reports no confidences
    assert not hasattr(out, "draft_conf_cpu")
    reply, _fin = _commit_all(s, verify, reqs, out)
    assert [reqs[0].pending_conf, reqs[1].pending_conf] == [0.01, None]  # untouched
    assert s.spec_stats == {"rounds": 2, "drafted": 6, "accepted": 4}
    _assert_pages_exact(s, reqs)


def test_the_stats_line_has_no_mean_depth_without_the_flag(monkeypatch):
    logged = []
    monkeypatch.setattr("freetoken.scheduler.scheduler.logger", SimpleNamespace(info_rank0=logged.append))
    s, pool = _scheduler(1, 2)
    req = _decoding_req(s, 5)
    s.spec_stats = {"rounds": 255, "drafted": 510, "accepted": 255}
    verify, engine = _launch(s, [req], [[32]], [[HI]], first=[31])
    _mark_states(pool, [req], 2, 2, 1)
    out, _d, _r = _outputs(verify, engine, [0], 0)
    _commit_all(s, verify, [req], out)
    assert logged and "mean depth" not in logged[-1]


# ---- boundary values -----------------------------------------------------------------------------------------


def test_p_zero_never_stops_the_chain_even_at_zero_confidence(flag):
    flag(0.0)
    k = 4
    s, pool = _scheduler(1, k)
    req = _decoding_req(s, 5)
    req.pending_conf = 0.0
    verify, engine = _launch(s, [req], [[32, 33, 34]], [[0.0, 0.0, 0.0]], first=[31])
    assert verify.spec.first_conf == [0.0] and verify.spec.k == k and engine.metadata_rows == []
    assert draft_depth([0.0, 0.0, 0.0], 0.0, 4) == 4


def test_p_one_stops_after_draft_one_unless_a_confidence_is_exactly_one(flag):
    flag(1.0)
    k = 4
    s, pool = _scheduler(1, k)
    req = _decoding_req(s, 5)
    verify, engine = _launch(s, [req], [[32, 33, 34]], [[0.999999, 1.0, 1.0]], first=[31])
    assert verify.spec.k == 2  # first_conf defaults to 1.0 (passes), draft 2's 0.999999 does not
    s2, _ = _scheduler(1, k)
    req = _decoding_req(s2, 5)
    verify, engine = _launch(s2, [req], [[32, 33, 34]], [[1.0, 1.0, 1.0]], first=[31])
    assert verify.spec.k == 4
    s3, _ = _scheduler(1, k)
    req = _decoding_req(s3, 5)
    req.pending_conf = 0.9999999
    verify, engine = _launch(s3, [req], [[32, 33, 34]], [[1.0, 1.0, 1.0]], first=[31])
    assert verify.spec.k == 1 and verify.size == 2 and engine.last_chain is None


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_k_one_never_chains_or_truncates(flag, p):
    flag(p)
    s, pool = _scheduler(1, 1)
    req = _decoding_req(s, 5)
    req.pending_conf = 0.0
    verify, engine = _launch(s, [req], [[]], [[]], first=[31])
    assert verify.spec.k == verify.spec.staged_k == 1 and verify.size == 2 and engine.metadata_rows == []
    assert engine.forwarded[2] is None
    _mark_states(pool, [req], 1, 1, 1)
    out, _d, _r = _outputs(verify, engine, [1], 0, out_conf=[[0.3, 0.6]])
    _commit_all(s, verify, [req], out)
    assert req.cached_len == 5 + 2 and req.pending_conf == pytest.approx(0.6)
    _assert_pages_exact(s, [req])


def test_the_threshold_is_read_from_the_environment(monkeypatch):
    from freetoken.env import ENV

    monkeypatch.setattr(ENV.SPEC_DRAFT_MIN_P, "value", None)
    for text, want in (("0", 0.0), ("1", 1.0), ("0.35", 0.35)):
        monkeypatch.setenv("FREETOKEN_SPEC_DRAFT_MIN_P", text)
        ENV.SPEC_DRAFT_MIN_P._init("FREETOKEN_SPEC_DRAFT_MIN_P")
        assert ENV.SPEC_DRAFT_MIN_P.value == want and ENV.SPEC_DRAFT_MIN_P.value is not None


def test_depth_with_nan_confidence_counts_as_confident():
    """NaN never compares below the threshold; the drafter's confidences are NaN-free, so this only pins the fallback."""
    assert draft_depth([float("nan"), 0.9], 0.5, 3) == 3
    assert batch_draft_depth([], 0.5, 3) == 3


# ---- the real drafter chain, several requests ----------------------------------------------------------------


def _chain_drafter(monkeypatch, logits_by_step, reqs_n):
    from freetoken import core
    from freetoken.speculative.mtp import MtpDrafter

    steps = []

    class FakeBatch:
        def __init__(self, reqs, phase):
            self.reqs = reqs
            steps.append(self)

    class Head:
        def forward_chain(self, hidden, token_embed):
            return hidden, hidden

    drafter = object.__new__(MtpDrafter)
    drafter.depth = 4
    drafter.device = torch.device("cpu")
    drafter.store = torch.zeros(8 * 5, 1)
    drafter.embed = SimpleNamespace(forward=lambda tokens: tokens.to(torch.float32).unsqueeze(-1))
    drafter.head = Head()
    calls = iter(logits_by_step)
    drafter.logits = lambda out: next(calls)
    drafter._page_table = lambda: torch.arange(reqs_n * 20, dtype=torch.int32).reshape(reqs_n, 20)
    original_tensor = torch.tensor
    monkeypatch.setattr(core, "Batch", FakeBatch)
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: original_tensor(*a, **{k: v for k, v in kw.items() if k != "pin_memory"}))
    context = SimpleNamespace(forward_batch=lambda _batch: nullcontext())
    backend = SimpleNamespace(prepare_metadata=lambda batch: None)
    return drafter, steps, context, backend


def _peaked(token, sure, V=8):
    row = torch.full((V,), 1.0)
    row[token] = 9.0 if sure else 1.1
    return row


def test_real_draft_chain_follows_each_requests_own_confidences(monkeypatch, flag):
    flag(0.5)
    # request 0: sure at every step; request 1: unsure first draft; request 2: unsure at draft 2
    logits = [
        torch.stack([_peaked(2, True), _peaked(3, True), _peaked(4, False)]),   # draft 2
        torch.stack([_peaked(5, True), _peaked(6, True), _peaked(7, True)]),    # draft 3
        torch.stack([_peaked(1, True), _peaked(1, True), _peaked(1, True)]),    # draft 4
    ]
    drafter, steps, context, backend = _chain_drafter(monkeypatch, logits, 3)
    reqs = [SimpleNamespace(table_idx=i, pending_slot=0, cached_len=5 + i) for i in range(3)]
    spec = SimpleNamespace(k=4, reqs=reqs, first_drafts=[1, 1, 1], first_conf=[0.9, 0.1, 0.9])
    token_pool = torch.zeros((3, 16), dtype=torch.int32)
    drafts = drafter.draft_chain(SimpleNamespace(spec=spec), token_pool, context, backend, mrope=False)
    assert len(steps) == 3 and drafts.shape == (3, 3)  # request 0 is confident to the end: the full chain
    assert drafts.tolist() == [[2, 5, 1], [3, 6, 1], [4, 7, 1]]
    for i, r in enumerate(reqs):  # draft j lands at cached_len + j
        assert token_pool[i, r.cached_len + 2 : r.cached_len + 5].tolist() == drafts[i].tolist()


def test_real_draft_chain_stops_when_only_confident_requests_have_run_out(monkeypatch, flag):
    flag(0.5)
    logits = [torch.stack([_peaked(2, False), _peaked(3, True)]), torch.stack([_peaked(5, True), _peaked(6, False)])]
    drafter, steps, context, backend = _chain_drafter(monkeypatch, logits, 2)
    reqs = [SimpleNamespace(table_idx=i, pending_slot=0, cached_len=5) for i in range(2)]
    # request 0 is sure of draft 1 but not of draft 2 (depth 2); request 1 is sure of drafts 1 and 2 (depth >= 3)
    spec = SimpleNamespace(k=4, reqs=reqs, first_drafts=[1, 1], first_conf=[0.9, 0.9])
    drafts = drafter.draft_chain(SimpleNamespace(spec=spec), torch.zeros((2, 16), dtype=torch.int32), context, backend, mrope=False)
    # draft 3 still runs for request 1; draft 4 does not, because request 1's draft 3 is unsure
    assert len(steps) == 2 and drafts.shape == (2, 2)


@pytest.mark.parametrize("p, sure_gap, steps_run", [(0.0, 1.0001, 3), (1.0, 100.0, 3), (1.0, 8.0, 0), (0.999, 12.0, 3)])
def test_real_draft_chain_boundary_thresholds(monkeypatch, flag, p, sure_gap, steps_run):
    """p = 0 never stops; p = 1.0 continues only on a confidence of exactly 1.0 (a 100-logit gap rounds to it in fp32)."""
    flag(p)
    row = torch.zeros(8)
    row[3] = sure_gap
    drafter, steps, context, backend = _chain_drafter(monkeypatch, [row[None]] * 3, 1)
    first_conf = [float(torch.exp(row.max() - torch.logsumexp(row, 0)))]
    reqs = [SimpleNamespace(table_idx=0, pending_slot=0, cached_len=5)]
    spec = SimpleNamespace(k=4, reqs=reqs, first_drafts=[1], first_conf=first_conf)
    drafts = drafter.draft_chain(SimpleNamespace(spec=spec), torch.zeros((1, 16), dtype=torch.int32), context, backend, mrope=False)
    assert len(steps) == steps_run
    assert (drafts is None) == (steps_run == 0)


def test_drafter_confidences_are_finite_probabilities():
    from freetoken.speculative.mtp import MtpDrafter

    drafter = object.__new__(MtpDrafter)
    inf = float("inf")
    rows = torch.tensor([[1e4, -1e4, 0.0], [0.0, 0.0, 0.0], [-inf, 5.0, -inf], [3.0, 2.9, -inf]])
    drafter.logits = lambda out: out
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        token, conf = drafter._draft(rows.to(dtype), True)
        assert token.dtype == torch.int32 and token.tolist() == [0, 0, 1, 0]
        assert torch.isfinite(conf).all() and ((conf > 0) & (conf <= 1)).all()
        assert conf[0].item() == 1.0 and conf[2].item() == 1.0 and conf[1].item() == pytest.approx(1 / 3, rel=1e-6)
    token, conf = drafter._draft(rows, False)
    assert conf is None
