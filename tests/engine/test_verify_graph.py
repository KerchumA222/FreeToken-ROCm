"""MTP verify under CUDA graphs (FREETOKEN_SPEC_VERIFY_GRAPH): GraphRunner capture and replay of a one-request verify
at every draft depth, the chained draft step, and the engine's choice between replaying a graph and running eagerly.

The model, attention backend and draft head are small fakes with the real contracts: a verify is a decode batch of
``1 + k`` VerifyRows, the forward advances a per-slot state row by row and snapshots it after each of the first ``k``
rows (as GDN does), and everything it reads is a static buffer or a device tensor, so a replay must equal the eager run."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Context, MtpPlan, SpecVerify, VerifyRow, set_global_ctx

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

VOCAB, WIDTH, K = 64, 8, 3
NUM_SLOTS = 6
DUMMY_SLOT = 0


@pytest.fixture(autouse=True)
def _runtime():
    import freetoken.core as core
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = None


class FakeBackend:
    """Decode-shaped staging like the real backends: capture buffers, metadata restaged per replay."""

    supports_verify_graph = True

    def __init__(self):
        self.capture_bs = []
        self.replays = []

    def init_capture_graph(self, max_seq_len, bs_list):
        self.capture_bs = sorted(bs_list)
        self.kvlen = torch.zeros(max(bs_list), dtype=torch.int32, device="cuda")

    def prepare_metadata(self, batch):
        kv = torch.tensor([r.device_len for r in batch.padded_reqs], dtype=torch.int32, device="cuda")
        batch.attn_metadata = SimpleNamespace(kvlen=kv)

    def prepare_for_capture(self, batch):
        assert batch.size in self.capture_bs
        batch.attn_metadata = SimpleNamespace(kvlen=self.kvlen[: batch.size])

    def prepare_for_replay(self, batch):
        assert batch.padded_size in self.capture_bs
        md = batch.attn_metadata
        self.kvlen[: batch.padded_size].copy_(md.kvlen)
        md.kvlen = self.kvlen[: batch.padded_size]
        self.replays.append(batch.padded_size)

    def reset_capture(self):
        self.capture_bs = []


class FakeModel:
    """Advances ``state[slot]`` one row at a time from the row's token, position and KV length, snapshots it after
    each of the first k rows, and returns logits whose argmax is a function of the new state."""

    def forward(self):
        from freetoken.core import get_global_ctx

        ctx = get_global_ctx()
        batch, pool = ctx.batch, ctx.linear_state_pool
        spec = batch.spec
        state = pool.recurrent_states[0, :, 0, 0, 0]
        slots = batch.fla_metadata.cache_indices
        kvlen = batch.attn_metadata.kvlen
        outs = []
        for s in range(spec.k + 1):
            idx = slots[s : s + 1].long()
            new = state.index_select(0, idx) * 2 + batch.input_ids[s : s + 1].float() + batch.positions[s : s + 1].float()
            state.index_copy_(0, idx, new)
            if s < spec.k:
                state.index_copy_(0, spec.snap_slots[s], new)
            outs.append(new + kvlen[s : s + 1].float() + 1.0)
        out = torch.cat(outs)
        batch.hidden_states = out[:, None].expand(-1, WIDTH).contiguous()
        target = out.long() % VOCAB
        return torch.nn.functional.one_hot(target, VOCAB).to(torch.float16)


class FakeMtp:
    """The draft head's two entry points the runner calls: a tail over a verify and one chained step."""

    def __init__(self):
        self.store = torch.zeros(NUM_SLOTS * (K + 1), WIDTH, device="cuda")

    def store_index(self, table_idx, row):
        return table_idx * (K + 1) + row

    def head_pass(self, batch, sampled):
        self.store.index_copy_(0, batch.mtp_plan.store_idx, batch.hidden_states)
        batch.draft_conf = None
        return (sampled + 1) % VOCAB

    def chain_step(self, state, token, with_conf):
        return state + 1, ((token.long() + state[:, 0].long()) % VOCAB).to(torch.int32), None


def _tail(mtp):
    def tail(batch, logits):
        tokens = torch.argmax(logits, dim=-1).to(torch.int32)
        drafts = mtp.head_pass(batch, tokens)
        return tokens, drafts, batch.draft_conf

    return tail


class Rig:
    def __init__(self, depths=range(1, K + 1)):
        from freetoken.engine.graph import GraphRunner
        from freetoken.kvcache.linear_state_pool import LinearStatePool
        from freetoken.models.config import LinearGatedDeltaGroupConfig

        group = LinearGatedDeltaGroupConfig(
            name="linear", layer_ids=(0,), num_key_heads=1, num_value_heads=1,
            key_head_dim=2, value_head_dim=2, conv_kernel_dim=2, output_gate="silu",
        )
        self.pool = LinearStatePool(group, NUM_SLOTS, dtype=torch.float32, device=torch.device("cuda"), tp_size=1, spec_slots=K)
        self.ctx = Context(page_size=1)
        self.ctx.linear_state_pool = self.pool
        set_global_ctx(self.ctx)
        self.backend = FakeBackend()
        self.mtp = FakeMtp()
        self.stream = torch.cuda.Stream()
        self.dummy = SimpleNamespace(table_idx=NUM_SLOTS - 1, linear_slot_idx=DUMMY_SLOT, cached_len=0, device_len=1, extend_len=1)
        with torch.cuda.stream(self.stream):
            self.runner = GraphRunner(
                stream=self.stream, device=torch.device("cuda"), model=FakeModel(), attn_backend=self.backend,
                cuda_graph_bs=[], cuda_graph_max_bs=1, free_memory=1 << 30, max_seq_len=64, vocab_size=VOCAB,
                dummy_req=self.dummy, verify_depths=depths, verify_tail=_tail(self.mtp), mtp=self.mtp,
            )
        torch.cuda.synchronize()

    def verify_batch(self, k, cached_len, slot, tokens, num_reqs=1):
        """A live one-request verify the way the scheduler stages it (own tensors, own snapshot slot tensors)."""
        req = SimpleNamespace(table_idx=slot, linear_slot_idx=slot, cached_len=cached_len, device_len=cached_len + 1 + k, extend_len=1 + k)
        rows = [VerifyRow(req, s) for s in range(k + 1)]
        batch = Batch(reqs=rows, phase="decode")
        batch.padded_reqs = rows
        dev = "cuda"
        batch.spec = SpecVerify(
            reqs=[req] * num_reqs, k=k, first_drafts=[0],
            snap_slots=[torch.tensor([self.pool.spec_slot(s)], dtype=torch.int64, device=dev) for s in range(k)],
        )
        batch.capture_hidden = True
        batch.input_ids = torch.tensor(tokens[: k + 1], dtype=torch.int32, device=dev)
        batch.positions = torch.arange(cached_len, cached_len + k + 1, dtype=torch.int32, device=dev)
        batch.out_loc = batch.positions.clone()
        batch.linear_table_idx = torch.full((k + 1,), slot, dtype=torch.int32, device=dev)
        from freetoken.attention.linear import FLAMetadata

        batch.fla_metadata = FLAMetadata(cu_seqlens=torch.arange(k + 2, dtype=torch.int32, device=dev), cache_indices=batch.linear_table_idx)
        batch.mtp_plan = MtpPlan(store_idx=torch.tensor([self.mtp.store_index(slot, s) for s in range(k + 1)], dtype=torch.int64, device=dev))
        self.backend.prepare_metadata(batch)
        return batch

    def state(self):
        return self.pool.recurrent_states.clone()

    def eager(self, batch):
        """Forward + tail on the live batch, as the engine's eager verify does."""
        with self.ctx.forward_batch(batch):
            logits = FakeModel().forward()
            tokens = torch.argmax(logits, dim=-1).to(torch.int32)
            drafts = self.mtp.head_pass(batch, tokens)
        return logits.float(), tokens, drafts, batch.hidden_states

    def replay(self, batch, with_tail=True):
        with torch.cuda.stream(self.stream), self.ctx.forward_batch(batch):
            logits, tail = self.runner.replay_verify(batch, with_tail)
        torch.cuda.synchronize()
        return logits, tail


@pytest.mark.parametrize("k", [1, 2, 3])
def test_replay_of_each_depth_equals_the_eager_verify(k):
    rig = Rig()
    assert sorted(rig.runner.verify_graphs) == [1, 2, 3]
    tokens = [5, 9, 13, 2]
    start = rig.state()
    start[0, 3, 0, 0, 0] = 7.0  # the request's live state, away from the dummy slot the graphs were captured on

    rig.pool.recurrent_states.copy_(start)
    eager_logits, eager_tokens, eager_drafts, eager_hidden = rig.eager(rig.verify_batch(k, 10, 3, tokens))
    eager_state, eager_store = rig.state(), rig.mtp.store.clone()

    rig.pool.recurrent_states.copy_(start)
    rig.mtp.store.zero_()
    batch = rig.verify_batch(k, 10, 3, tokens)
    logits, tail = rig.replay(batch)

    assert torch.equal(logits.float(), eager_logits)
    assert torch.equal(batch.hidden_states, eager_hidden)
    assert torch.equal(tail[0], eager_tokens) and torch.equal(tail[1], eager_drafts)
    assert torch.equal(rig.state()[0, 3:4], eager_state[0, 3:4]), "live state"
    for s in range(k):
        slot = rig.pool.spec_slot(s)
        assert torch.equal(rig.state()[0, slot], eager_state[0, slot]), f"snapshot {s}"
    assert torch.equal(rig.mtp.store[3 * (K + 1) : 3 * (K + 1) + k + 1], eager_store[3 * (K + 1) : 3 * (K + 1) + k + 1])
    assert rig.backend.replays[-1] == k + 1
    # the dummy slot the graphs were captured on and the other requests' slots are untouched
    assert torch.equal(rig.state()[0, [1, 2, 4, 5]], start[0, [1, 2, 4, 5]])


def test_consecutive_replays_restage_their_inputs():
    """Two different requests, depths and contexts through the same graphs: nothing of the first leaks into the second."""
    rig = Rig()
    rig.replay(rig.verify_batch(3, 20, 1, [4, 6, 8, 10]))
    mid, mid_store = rig.state(), rig.mtp.store.clone()
    second = rig.verify_batch(2, 33, 4, [1, 2, 3])
    logits, tail = rig.replay(second)
    graph_state, graph_store = rig.state(), rig.mtp.store.clone()

    rig.pool.recurrent_states.copy_(mid)
    rig.mtp.store.copy_(mid_store)
    eager = rig.eager(rig.verify_batch(2, 33, 4, [1, 2, 3]))
    assert torch.equal(logits.float(), eager[0]) and torch.equal(tail[0], eager[1]) and torch.equal(tail[1], eager[2])
    assert torch.equal(graph_state, rig.state()) and torch.equal(graph_store, rig.mtp.store)
    assert rig.backend.replays[-2:] == [4, 3]


def test_a_replay_without_the_tail_leaves_the_sample_and_head_to_the_caller():
    rig = Rig()
    batch = rig.verify_batch(2, 5, 2, [3, 1, 4])
    store_before = rig.mtp.store.clone()
    logits, tail = rig.replay(batch, with_tail=False)
    assert tail is None
    assert batch.hidden_states.shape == (3, WIDTH)
    assert torch.equal(rig.mtp.store, store_before), "the draft head ran in the graph"
    assert logits.shape == (3, VOCAB)


def test_only_a_single_request_verify_of_a_captured_depth_replays():
    rig = Rig(depths=[1, 2])
    runner = rig.runner
    assert sorted(runner.verify_graphs) == [1, 2]
    assert runner.can_replay_verify(rig.verify_batch(2, 5, 1, [1, 2, 3]))
    assert not runner.can_replay_verify(rig.verify_batch(3, 5, 1, [1, 2, 3, 4])), "depth 3 was not captured"
    assert not runner.can_replay_verify(rig.verify_batch(2, 5, 1, [1, 2, 3], num_reqs=2)), "two requests"
    sequential = rig.verify_batch(2, 5, 1, [1, 2, 3])
    sequential.spec.step = 0
    assert not runner.can_replay_verify(sequential), "a sequential verify runs one step per forward"
    plain = Batch(reqs=[], phase="decode")
    assert not runner.can_replay_verify(plain)


def test_capture_writes_head_state_only_to_the_dummy_requests_rows():
    rig = Rig()
    first_dummy_row = rig.mtp.store_index(rig.dummy.table_idx, 0)
    assert rig.mtp.store[:first_dummy_row].abs().sum().item() == 0
    assert rig.mtp.store[first_dummy_row:].abs().sum().item() > 0


def test_the_attention_capture_covers_every_row_count():
    rig = Rig()
    assert rig.backend.capture_bs == [1, 2, 3, 4]  # chain step, then 1 + depth rows for each depth


def test_no_verify_depths_captures_nothing():
    rig = Rig(depths=[])
    assert rig.runner.verify_graphs == {} and rig.runner.chain_graph is None
    assert not rig.runner.can_replay_verify(rig.verify_batch(1, 5, 1, [1, 2]))


def test_the_chained_step_replays_like_the_eager_chain():
    rig = Rig()
    runner, mtp = rig.runner, rig.mtp
    assert runner.chain_graph is not None
    spec = SimpleNamespace(num_reqs=1)
    assert runner.can_replay_chain(spec) and not runner.can_replay_chain(SimpleNamespace(num_reqs=2))
    token_pool = torch.zeros(NUM_SLOTS, 32, dtype=torch.int32, device="cuda")
    req = SimpleNamespace(table_idx=2, cached_len=9)

    def eager(state0, token0, steps):
        state, token, out = state0.clone(), token0.clone(), []
        for _ in range(steps):
            state, token, _ = mtp.chain_step(state, token, False)
            out.append(int(token))
        return out

    def replayed(state0, token0, steps):
        runner.begin_chain(state0, token0)
        with torch.cuda.stream(rig.stream):
            for j in range(steps):
                step = Batch(reqs=[VerifyRow(req, j)], phase="decode")
                step.padded_reqs = step.reqs
                step.positions = torch.tensor([9 + j], dtype=torch.int32, device="cuda")
                step.out_loc = step.positions.clone()
                rig.backend.prepare_metadata(step)
                assert runner.replay_chain_step(step) is None
            drafts = runner.finish_chain(token_pool, req, steps)
        torch.cuda.synchronize()
        return drafts

    # two chains in a row, the second shorter: each starts from its own state and fills the drafts from the first slot
    for steps, base, first in [(2, 3.0, 17), (1, 11.0, 5)]:
        state0 = torch.arange(WIDTH, dtype=torch.float32, device="cuda")[None] + base
        token0 = torch.tensor([first], dtype=torch.int32, device="cuda")
        expected = eager(state0, token0, steps)
        token_pool.zero_()
        drafts = replayed(state0, token0, steps)
        assert drafts.tolist() == [expected]
        assert token_pool[2, 11 : 11 + steps].tolist() == expected and token_pool.sum().item() == sum(expected)
    assert rig.backend.replays[-3:] == [1, 1, 1]


class FakeGraphs:
    """What the engine asks of the GraphRunner for a verify."""

    def __init__(self, replayable=True, tail=True):
        self.replayable, self.tail = replayable, tail
        self.calls = []

    def can_use_cuda_graph(self, batch):
        return False

    def can_replay_verify(self, batch):
        return self.replayable

    def replay_verify(self, batch, with_tail):
        self.calls.append(with_tail)
        batch.hidden_states = torch.zeros(batch.size, WIDTH, device="cuda")
        logits = torch.zeros(batch.size, VOCAB, device="cuda")
        tail = None
        if with_tail and self.tail:
            tail = (torch.full((batch.size,), 7, dtype=torch.int32, device="cuda"), torch.full((batch.size,), 8, dtype=torch.int32, device="cuda"), None)
        return logits, tail


def _engine(graphs):
    from freetoken.engine.engine import Engine
    from freetoken.engine.sample import Sampler

    e = Engine.__new__(Engine)
    e.stream = torch.cuda.current_stream()
    e.ctx = Context(page_size=1)
    e.graph_runner = graphs
    e.model = SimpleNamespace(forward_host_ctx=lambda batch, use_graph: nullcontext(), forward=lambda: pytest.fail("eager forward ran"))
    e.cpu_moe_executor = None
    e.sampler = Sampler(torch.device("cuda"), VOCAB)
    e.eager_calls = []
    e.mtp = SimpleNamespace(
        head_pass=lambda batch, sampled: e.eager_calls.append("head") or torch.full_like(sampled, 9),
    )
    return e


def _spec_batch(rig_free=True, num_reqs=1, k=2):
    req = SimpleNamespace(table_idx=1, cached_len=4, device_len=4 + 1 + k, extend_len=1 + k)
    rows = [VerifyRow(req, s) for s in range(k + 1)]
    batch = Batch(reqs=rows, phase="decode")
    batch.padded_reqs = rows
    batch.spec = SpecVerify(reqs=[req] * num_reqs, k=k, first_drafts=[0])
    batch.capture_hidden = True
    batch.input_ids = torch.zeros(k + 1, dtype=torch.int32, device="cuda")
    return batch


def test_a_greedy_verify_takes_the_graphs_sample_and_drafts():
    from freetoken.engine.sample import BatchSamplingArgs

    graphs = FakeGraphs()
    out = _engine(graphs).forward_batch(_spec_batch(), BatchSamplingArgs(temperatures=None))
    out.copy_done_event.synchronize()
    assert graphs.calls == [True]
    assert out.next_tokens_cpu.tolist() == [7, 7, 7] and out.draft_tokens_cpu.tolist() == [8, 8, 8]


def test_a_sampled_verify_replays_the_forward_and_samples_eagerly(monkeypatch):
    from freetoken.engine.sample import BatchSamplingArgs

    graphs = FakeGraphs()
    engine = _engine(graphs)
    seen = []
    monkeypatch.setattr(engine, "_sample_verify", lambda batch, logits, args: seen.append(tuple(logits.shape)) or torch.full((batch.size,), 5, dtype=torch.int32, device="cuda"))
    args = BatchSamplingArgs(temperatures=torch.full((3,), 0.7, device="cuda"))
    out = engine.forward_batch(_spec_batch(), args)
    out.copy_done_event.synchronize()
    assert graphs.calls == [False] and seen == [(3, VOCAB)] and engine.eager_calls == ["head"]
    assert out.next_tokens_cpu.tolist() == [5, 5, 5] and out.draft_tokens_cpu.tolist() == [9, 9, 9]


def test_a_verify_the_graphs_cannot_replay_runs_the_eager_forward():
    from freetoken.engine.sample import BatchSamplingArgs

    engine = _engine(FakeGraphs(replayable=False))
    engine.model.forward = lambda: torch.zeros(3, VOCAB, device="cuda")
    engine.sampler = SimpleNamespace(sample=lambda logits, args: torch.full((logits.shape[0],), 6, device="cuda"))
    out = engine.forward_batch(_spec_batch(), BatchSamplingArgs(temperatures=None))
    out.copy_done_event.synchronize()
    assert engine.graph_runner.calls == [] and engine.eager_calls == ["head"]
    assert out.next_tokens_cpu.tolist() == [6, 6, 6]


@pytest.mark.parametrize(
    "flag, mtp, max_bs, supports, sequential, expected",
    [
        (True, True, 1, True, False, "graphs"),
        (False, True, 1, True, False, None),
        (True, False, 1, True, False, None),
        (True, True, 0, True, False, "CUDA graphs are off"),
        (True, True, 1, False, False, "does not capture a verify"),
        (True, True, 1, True, True, "one forward at a time"),
    ],
)
def test_the_engine_captures_verify_graphs_only_where_they_are_sound(monkeypatch, flag, mtp, max_bs, supports, sequential, expected):
    from freetoken.engine import engine as engine_module
    from freetoken.engine.engine import Engine

    monkeypatch.setattr(engine_module.ENV.SPEC_VERIFY_GRAPH, "value", flag)
    monkeypatch.setattr(engine_module.ENV.SPEC_SEQUENTIAL_VERIFY, "value", sequential)
    messages = []
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda msg, *a, **k: messages.append(msg))
    e = Engine.__new__(Engine)
    e.mtp = object() if mtp else None
    e.attn_backend = SimpleNamespace(supports_verify_graph=supports)
    config = SimpleNamespace(cuda_graph_max_bs=max_bs, speculative_num_draft_tokens=4)
    args = e._verify_graph_args(config)
    if expected == "graphs":
        assert list(args["verify_depths"]) == [1, 2, 3, 4] and args["mtp"] is e.mtp and callable(args["verify_tail"])
    else:
        assert args == {}
        assert (expected is None and not messages) or (expected in messages[0])


def test_the_speculative_config_keeps_the_graph_budget_only_for_verify_graphs(monkeypatch):
    from freetoken.engine import engine as engine_module

    def adjust(flag, max_bs):
        monkeypatch.setattr(engine_module.ENV.SPEC_VERIFY_GRAPH, "value", flag)
        config = SimpleNamespace(
            tp_info=SimpleNamespace(size=1), model_config=SimpleNamespace(is_moe=False), cuda_graph_max_bs=max_bs,
            cuda_graph_bs=None, moe_strategy="offload", mtp_source=SimpleNamespace(path="x", block=0),
        )
        applied = {}
        engine_module._adjust_speculative_config(config, applied.__setitem__)
        return applied

    assert adjust(False, 1) == {"cuda_graph_bs": [], "cuda_graph_max_bs": 0}
    assert adjust(True, 1) == {"cuda_graph_bs": []}, "decode graphs are dropped, the budget stays for the verify"
    assert adjust(True, 0) == {}
