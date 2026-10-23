"""Qwen3.5/3.6 MTP draft head: the parts that are decided, not computed.

The head's arithmetic is the target's own decoder block, which is already covered.
What is specific to the head -- and what a checkpoint can disagree with -- is where
its block sits in the layer pattern and how its metadata is read.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    RotaryConfig,
)
from freetoken.models.qwen3_5_moe.mtp import _with_full_attention


@dataclass(frozen=True)
class _HeadCfg:
    attention_groups: tuple
    hidden_size: int = 2
    rms_norm_eps: float = 1e-6
    quant: object = None


@dataclass(frozen=True)
class _Cfg:
    """Minimal stand-in: _with_full_attention only reads attention_groups, and needs
    a dataclass because it rebuilds the config with dataclasses.replace."""

    attention_groups: tuple


def _hybrid_groups():
    """A pattern like Qwen3.5's: every 4th layer full attention, the rest linear."""
    full_ids = tuple(i for i in range(40) if (i + 1) % 4 == 0)
    linear_ids = tuple(i for i in range(40) if (i + 1) % 4 != 0)
    rotary = RotaryConfig(head_dim=256, rotary_dim=64, max_position=4096,
                          base=1e7, scaling=None)
    full = FullAttentionGroupConfig(
        name="full", layer_ids=full_ids, num_kv_heads=2, head_dim=256,
        rotary_config=rotary,
    )
    linear = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=linear_ids, num_key_heads=16, num_value_heads=32,
        key_head_dim=128, value_head_dim=128, conv_kernel_dim=4, output_gate="silu",
    )
    return full, linear


def test_external_draft_gguf_resolves_target_ftw_metadata(monkeypatch):
    from freetoken.models.gguf import reader
    from freetoken.models.qwen3_5_moe import mtp

    sources = {
        "target.ftw": "target.ftw/source_metadata.gguf",
        "draft.gguf": "draft.gguf",
    }
    monkeypatch.setattr(reader, "gguf_config_source", lambda path: sources.get(path))
    architecture_paths = []

    def architecture(path):
        architecture_paths.append(path)
        return "qwen35moe"

    monkeypatch.setattr(reader, "gguf_architecture", architecture)
    draft_paths = []

    def draft_layer(path):
        draft_paths.append(path)
        return 40 if path == "draft.gguf" else None

    monkeypatch.setattr(mtp, "mtp_draft_layer", draft_layer)

    source = mtp.QWEN3_5_MTP.locate("target.ftw", "draft.gguf")

    assert architecture_paths == ["target.ftw/source_metadata.gguf", "draft.gguf"]
    assert draft_paths == ["draft.gguf"]
    assert source.path == "draft.gguf"
    assert source.block == 40


def test_ftw_metadata_without_external_draft_reports_missing_nextn(monkeypatch):
    from freetoken.models.gguf import reader
    from freetoken.models.qwen3_5_moe import mtp

    monkeypatch.setattr(
        reader,
        "gguf_config_source",
        lambda path: "target.ftw/source_metadata.gguf" if path == "target.ftw" else None,
    )
    scanned = []
    monkeypatch.setattr(mtp, "mtp_draft_layer", lambda path: scanned.append(path))

    with pytest.raises(ValueError, match="carries no nextn.*pass the draft head's GGUF"):
        mtp.QWEN3_5_MTP.locate("target.ftw", None)

    assert scanned == ["target.ftw/source_metadata.gguf"]


def test_draft_block_is_moved_into_the_full_attention_group():
    """The draft block sits one past the last layer, where the pattern says linear;
    the checkpoint carries attn_q/k/v for it, so the config has to agree."""
    full, linear = _hybrid_groups()
    cfg = _Cfg((full, linear))
    mtp_layer = 40
    assert mtp_layer not in full.layer_ids

    out = _with_full_attention(cfg, mtp_layer)
    groups = {type(g).__name__: g for g in out.attention_groups}
    assert mtp_layer in groups["FullAttentionGroupConfig"].layer_ids
    assert mtp_layer not in groups["LinearGatedDeltaGroupConfig"].layer_ids
    # every other layer keeps its group
    assert set(groups["FullAttentionGroupConfig"].layer_ids) - {mtp_layer} == set(full.layer_ids)
    assert set(groups["LinearGatedDeltaGroupConfig"].layer_ids) == set(linear.layer_ids)


def test_a_layer_already_full_is_left_alone():
    full, linear = _hybrid_groups()
    cfg = _Cfg((full, linear))
    out = _with_full_attention(cfg, 3)          # (3+1) % 4 == 0 -> already full
    groups = {type(g).__name__: g for g in out.attention_groups}
    assert set(groups["FullAttentionGroupConfig"].layer_ids) == set(full.layer_ids)
    assert set(groups["LinearGatedDeltaGroupConfig"].layer_ids) == set(linear.layer_ids)


def test_attributes_set_outside_the_dataclass_fields_survive():
    """The GGUF readers stash the expert bank types with object.__setattr__, and
    dataclasses.replace drops those -- the block would build its MoE against nothing."""
    full, linear = _hybrid_groups()
    cfg = _Cfg((full, linear))
    object.__setattr__(cfg, "gguf_expert_bank_types", {"gate_up": 12, "down": 7})
    out = _with_full_attention(cfg, 40)
    assert getattr(out, "gguf_expert_bank_types", None) == {"gate_up": 12, "down": 7}


def test_nextn_token_module_overrides_use_gguf_matrix_layout(monkeypatch):
    from freetoken.models.qwen3_5_moe import gguf

    values = torch.arange(128, dtype=torch.float32)
    raw = values.numpy().view("uint8")
    info = SimpleNamespace(
        ggml_type=0,
        ne=(8, 16),
        nbytes=raw.nbytes,
        data=lambda: raw,
    )
    index = {
        "blk.40.nextn.eh_proj.weight": info,
        "blk.40.nextn.embed_tokens.weight": info,
        "blk.40.nextn.shared_head_head.weight": info,
    }
    monkeypatch.setattr(gguf, "gguf_tensor_index", lambda _path: index)

    weights = dict(
        gguf.iter_gguf_mtp_token_weights(
            "synthetic.gguf", config=SimpleNamespace(hidden_size=8)
        )
    )

    assert set(weights) == {"mtp.embed_tokens.weight", "mtp.shared_head_head.weight"}
    expected = values.reshape(16, 8).to(torch.bfloat16)
    for weight in weights.values():
        assert weight.shape == (16, 8)
        torch.testing.assert_close(weight, expected)
    tensors = {name: 0 for name in index}
    assert gguf.to_gguf("mtp.embed_tokens", tensors) == (
        "blk.40.nextn.embed_tokens",
    )
    assert gguf.to_gguf("mtp.shared_head_head", tensors) == (
        "blk.40.nextn.shared_head_head",
    )


def test_mtp_adapter_builds_separate_token_modules_for_checkpoint_overrides(monkeypatch):
    from freetoken.models.qwen3_5_moe import mtp
    from freetoken.speculative.mtp import MtpSource

    from freetoken.models.gguf import reader

    index = {
        "blk.40.nextn.embed_tokens.weight": object(),
        "blk.40.nextn.shared_head_head.weight": object(),
    }
    monkeypatch.setattr(reader, "gguf_tensor_index", lambda _path: index)
    built = []

    class TokenModule:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            built.append(self)

    monkeypatch.setattr("freetoken.layers.VocabParallelEmbedding", TokenModule)
    monkeypatch.setattr("freetoken.layers.ParallelLMHead", TokenModule)
    target_embed, target_head = object(), object()
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=target_embed), lm_head=target_head)
    config = SimpleNamespace(vocab_size=32, hidden_size=8, quant=object())

    embed, head = mtp.QWEN3_5_MTP.build_token_modules(
        target, config, MtpSource("synthetic.gguf", 40), dtype=torch.bfloat16
    )

    assert len(built) == 2
    assert embed is not target_embed and head is not target_head
    assert embed.kwargs["prefix"] == "mtp.embed_tokens"
    assert head.kwargs["prefix"] == "mtp.shared_head_head"
    assert head.kwargs["tie_word_embeddings"] is False


@pytest.mark.parametrize("override_kind", ["embed", "head"])
def test_mtp_drafter_loads_one_sided_token_override_and_borrows_other(
    monkeypatch, override_kind
):
    from contextlib import nullcontext

    from freetoken.speculative import mtp
    from freetoken.speculative.mtp import MtpAdapter, MtpSource

    class FakeModule:
        def __init__(self, weights):
            self.weights = weights
            self.loaded = None

        def state_dict(self):
            return self.weights

        def load_state_dict(self, state):
            self.loaded = state

    @dataclass(frozen=True)
    class ModelConfig:
        num_layers: int = 40
        quant: object = None

    source = MtpSource("synthetic.gguf", 40)
    draft_weight = torch.tensor([2.0, 3.0])
    override_weight = torch.tensor([5.0, 7.0])
    head = FakeModule({"block.weight": torch.empty(2)})
    override = FakeModule({"weight": torch.empty(2)})
    borrowed_embedding, borrowed_head = object(), object()
    token_prefix = (
        "mtp.embed_tokens" if override_kind == "embed" else "mtp.shared_head_head"
    )
    token_weights = ((f"{token_prefix}.weight", override_weight),)
    finalized = []

    class Adapter(MtpAdapter):
        def build_head(self, _config, _layer_id):
            return head

        def iter_weights(self, _source, _config):
            return iter((("block.weight", draft_weight),))

        def build_token_modules(self, _model, _config, _source, *, dtype):
            assert dtype is torch.float32
            if override_kind == "embed":
                return override, borrowed_head
            return borrowed_embedding, override

        def iter_token_weights(self, _source, _config):
            return iter(token_weights)

        def embed_tokens(self, model):
            return model.model.embed_tokens

        def lm_head(self, model):
            return model.lm_head

    model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=borrowed_embedding), lm_head=borrowed_head
    )
    config = SimpleNamespace(
        model_spec=SimpleNamespace(parse_config="parse_gguf_config"),
        mtp_source=source,
        speculative_num_draft_tokens=1,
        model_config=ModelConfig(),
        dtype=torch.float32,
    )

    monkeypatch.setattr(mtp, "mtp_adapter", lambda _spec: Adapter())
    monkeypatch.setattr(mtp, "head_quant_config", lambda _source, _spec: object())
    monkeypatch.setattr("freetoken.distributed.get_tp_info", lambda: SimpleNamespace(size=1))
    monkeypatch.setattr("freetoken.layers.quantization.finalize_quant", finalized.append)
    monkeypatch.setattr("freetoken.utils.torch_dtype", lambda _dtype: nullcontext())
    monkeypatch.setattr(mtp, "logger", SimpleNamespace(info_rank0=lambda _message: None))

    drafter = mtp.MtpDrafter(config, model, device=torch.device("cpu"), page_table_ref=lambda: None)

    assert head.loaded["block.weight"].tolist() == [2.0, 3.0]
    assert override.loaded["weight"].tolist() == [5.0, 7.0]
    assert drafter.embed is (override if override_kind == "embed" else borrowed_embedding)
    assert drafter.lm_head is (override if override_kind == "head" else borrowed_head)
    assert finalized == [head, override]


def test_mtp_head_pass_applies_sampled_tokens_and_stores_selected_rows():
    from freetoken.core import MtpPlan
    from freetoken.speculative.mtp import MtpDrafter

    class Embedding:
        def forward(self, tokens):
            return tokens.to(torch.float32).unsqueeze(-1)

    class Head:
        def forward_chain(self, hidden, token_embed):
            out = hidden + token_embed
            return out, out + 10

    drafter = object.__new__(MtpDrafter)
    drafter.embed = Embedding()
    drafter.head = Head()
    drafter.depth = 1
    drafter.device = torch.device("cpu")
    drafter.store = None
    drafter._page_table = lambda: torch.empty((2, 1), dtype=torch.int32)
    drafter.logits = lambda out: torch.cat((out, -out), dim=-1)
    plan = MtpPlan(
        store_idx=torch.tensor([0, 1]),
        draft_rows=torch.tensor([1, 2]),
        sampled_rows=torch.tensor([1]),
        next_ids=torch.tensor([101, 102, 103]),
    )
    batch = SimpleNamespace(
        mtp_plan=plan,
        hidden_states=torch.tensor([[0.0], [20.0], [-200.0]]),
    )

    draft_tokens = drafter.head_pass(batch, torch.tensor([99], dtype=torch.int32))

    assert draft_tokens.tolist() == [0, 1]
    assert plan.next_ids.tolist() == [101, 102, 103]
    assert drafter.store.shape == (4, 1)
    # row 1 takes the sampled token: 20 + 99 + 10; row 2 keeps its next id: -200 + 103 + 10
    torch.testing.assert_close(drafter.store[:2, 0], torch.tensor([129.0, -87.0]))


def test_mtp_draft_chain_feeds_back_state_and_writes_following_tokens(monkeypatch):
    from contextlib import nullcontext

    from freetoken import core
    from freetoken.speculative.mtp import MtpDrafter

    steps = []

    class FakeBatch:
        def __init__(self, reqs, phase):
            self.reqs = reqs
            self.phase = phase
            steps.append(self)

    class Embedding:
        def forward(self, tokens):
            return tokens.to(torch.float32).unsqueeze(-1)

    class Head:
        def __init__(self):
            self.inputs = []

        def forward_chain(self, hidden, token_embed):
            self.inputs.append((hidden.clone(), token_embed.clone()))
            out = hidden + token_embed
            return out, out + 100

    class Context:
        def forward_batch(self, _batch):
            return nullcontext()

    class AttentionBackend:
        def prepare_metadata(self, batch):
            batch.metadata_prepared = True

    head = Head()
    drafter = object.__new__(MtpDrafter)
    drafter.depth = 3
    drafter.device = torch.device("cpu")
    drafter.store = torch.tensor([[10.0], [0.0], [0.0], [0.0]])
    drafter.embed = Embedding()
    drafter.head = head
    drafter.logits = lambda out: torch.cat((torch.zeros_like(out), out), dim=-1)
    req = SimpleNamespace(table_idx=0, pending_slot=0, cached_len=5)
    batch = SimpleNamespace(
        spec=SimpleNamespace(k=3, reqs=[req], first_drafts=[2])
    )
    page_table = torch.arange(20, dtype=torch.int32).reshape(1, 20)
    drafter._page_table = lambda: page_table
    token_pool = torch.zeros((1, 12), dtype=torch.int32)

    original_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(core, "Batch", FakeBatch)
    monkeypatch.setattr(torch, "tensor", cpu_tensor)

    drafts = drafter.draft_chain(batch, token_pool, Context(), AttentionBackend(), mrope=False)

    assert drafts.tolist() == [[1, 1]]
    assert token_pool[0, 7:9].tolist() == [1, 1]
    assert [step.positions.tolist() for step in steps] == [[5], [6]]
    assert [step.out_loc.tolist() for step in steps] == [[5], [6]]
    assert all(step.metadata_prepared for step in steps)
    torch.testing.assert_close(head.inputs[0][0], torch.tensor([[10.0]]))
    torch.testing.assert_close(head.inputs[0][1], torch.tensor([[2.0]]))
    torch.testing.assert_close(head.inputs[1][0], torch.tensor([[112.0]]))
    torch.testing.assert_close(head.inputs[1][1], torch.tensor([[1.0]]))


def test_mtp_head_combines_embedding_then_hidden_and_returns_chained_state(monkeypatch):
    from freetoken.models.qwen3_5_moe import model, mtp

    class FakeNorm:
        count = 0

        def __init__(self, _hidden, eps):
            self.offset = (1.0, 2.0, 3.0)[FakeNorm.count]
            FakeNorm.count += 1

        def forward(self, value):
            return value + self.offset

        def forward_add_residual(self, value, residual):
            return value + residual + self.offset, None

    class FakeProjection:
        def __init__(self, in_features, out_features, **_kwargs):
            assert (in_features, out_features) == (4, 2)
            self.weight = torch.tensor([[1.0, 10.0, 100.0, 1000.0], [2.0, 20.0, 200.0, 2000.0]])

        def forward(self, value):
            return value @ self.weight.T

    class FakeDecoderLayer:
        def __init__(self, _config, _layer_id, *, prefix):
            assert prefix == "mtp.layer"
            self._is_linear = False

        def forward(self, value, residual):
            assert residual is None
            return 2 * value, 3 * value

    monkeypatch.setattr(mtp, "GemmaRMSNorm", FakeNorm)
    monkeypatch.setattr(mtp, "LinearReplicated", FakeProjection)
    monkeypatch.setattr(model, "Qwen3_5DecoderLayer", FakeDecoderLayer)
    FakeNorm.count = 0
    full, linear = _hybrid_groups()
    head = mtp.Qwen3_5MTPHead(_HeadCfg((full, linear)), 40)

    hidden = torch.tensor([[3.0, 4.0]])
    token_embed = torch.tensor([[5.0, 6.0]])
    out, next_state = head.forward_chain(hidden, token_embed)
    expected_z = torch.cat((token_embed + 1, hidden + 2), dim=-1) @ head.eh_proj.weight.T
    expected = 5 * expected_z + 3

    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(next_state, expected)


def test_mtp_head_pass_reports_draft_confidence_only_with_the_flag(monkeypatch):
    from freetoken.core import MtpPlan
    from freetoken.env import ENV
    from freetoken.speculative.mtp import MtpDrafter

    class Embedding:
        def forward(self, tokens):
            return tokens.to(torch.float32).unsqueeze(-1)

    class Head:
        def forward_chain(self, hidden, token_embed):
            return hidden, hidden

    drafter = object.__new__(MtpDrafter)
    drafter.embed = Embedding()
    drafter.head = Head()
    drafter.depth = 1
    drafter.device = torch.device("cpu")
    drafter.store = None
    drafter._page_table = lambda: torch.empty((2, 1), dtype=torch.int32)
    logits = torch.tensor([[2.0, 0.0, 1.0], [0.0, 0.0, 3.0]])
    drafter.logits = lambda out: logits

    def run():
        plan = MtpPlan(store_idx=torch.tensor([0, 1]))
        batch = SimpleNamespace(mtp_plan=plan, hidden_states=torch.zeros(2, 1), draft_conf=None)
        tokens = drafter.head_pass(batch, torch.tensor([5, 6], dtype=torch.int32))
        return tokens, batch.draft_conf

    monkeypatch.setattr(ENV.SPEC_DRAFT_MIN_P, "value", None)
    tokens, conf = run()
    assert tokens.tolist() == [0, 2] and conf is None

    monkeypatch.setattr(ENV.SPEC_DRAFT_MIN_P, "value", 0.5)
    tokens, conf = run()
    assert tokens.tolist() == [0, 2]
    torch.testing.assert_close(conf, torch.softmax(logits, dim=-1).max(dim=-1).values)


def _chain_drafter(monkeypatch, logits_by_step):
    """A drafter whose head step ``j - 1`` returns ``logits_by_step[j - 1]`` ([B, vocab]); returns it with the visited steps."""
    from contextlib import nullcontext

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
    drafter.store = torch.zeros(4, 1)
    drafter.embed = SimpleNamespace(forward=lambda tokens: tokens.to(torch.float32).unsqueeze(-1))
    drafter.head = Head()
    calls = iter(logits_by_step)
    drafter.logits = lambda out: next(calls)
    drafter._page_table = lambda: torch.arange(20, dtype=torch.int32).reshape(1, 20)
    original_tensor = torch.tensor
    monkeypatch.setattr(core, "Batch", FakeBatch)
    monkeypatch.setattr(
        torch, "tensor", lambda *a, **kw: original_tensor(*a, **{k: v for k, v in kw.items() if k != "pin_memory"})
    )
    context = SimpleNamespace(forward_batch=lambda _batch: nullcontext())
    backend = SimpleNamespace(prepare_metadata=lambda batch: None)
    return drafter, steps, context, backend


@pytest.mark.parametrize(
    "first_conf, step_conf, steps_run",
    [
        (0.2, 0.9, 0),   # the first draft is unsure: no further draft is verified, so none is computed
        (0.9, 0.2, 1),   # draft 2 is computed (it is verified) but draft 3 is not
        (0.9, 0.9, 2),   # confident all the way: the full chain
    ],
)
def test_mtp_draft_chain_stops_when_no_request_can_extend(monkeypatch, first_conf, step_conf, steps_run):
    from freetoken.env import ENV

    unsure, sure = [1.0, 1.0, 1.0], [0.0, 8.0, 0.0]  # softmax max 1/3 and ~0.997
    logits = [torch.tensor([sure if step_conf > 0.5 else unsure])] * 2
    drafter, steps, context, backend = _chain_drafter(monkeypatch, logits)
    monkeypatch.setattr(ENV.SPEC_DRAFT_MIN_P, "value", 0.5)
    req = SimpleNamespace(table_idx=0, pending_slot=0, cached_len=5)
    spec = SimpleNamespace(k=3, reqs=[req], first_drafts=[2], first_conf=[first_conf])
    token_pool = torch.zeros((1, 12), dtype=torch.int32)

    drafts = drafter.draft_chain(SimpleNamespace(spec=spec), token_pool, context, backend, mrope=False)

    assert len(steps) == steps_run
    assert (drafts is None) if steps_run == 0 else drafts.shape == (1, steps_run)


class _ChainGraphs:
    """The GraphRunner side of a captured chained step: one replay per draft, the drafts kept in its own buffer."""

    def __init__(self, replayable=True, confs=None):
        self.replayable, self.confs = replayable, confs
        self.started, self.replayed, self.finished = None, [], None

    def can_replay_chain(self, spec):
        return self.replayable and spec.num_reqs == 1

    def begin_chain(self, state, token):
        self.started = (state.clone(), token.clone())

    def replay_chain_step(self, step):
        self.replayed.append((step.positions.tolist(), step.out_loc.tolist(), step.active_table_idx.tolist(), step.metadata_prepared))
        return None if self.confs is None else torch.tensor([self.confs[len(self.replayed) - 1]])

    def finish_chain(self, token_pool, req, steps):
        self.finished = (req.table_idx, steps)
        token_pool[req.table_idx, req.cached_len + 2 : req.cached_len + 2 + steps] = 40 + torch.arange(steps, dtype=torch.int32)
        return torch.arange(steps, dtype=torch.int32).view(1, steps)


def test_mtp_draft_chain_replays_one_captured_step_per_draft(monkeypatch):
    drafter, steps, context, backend = _chain_drafter(monkeypatch, [])
    drafter.logits = lambda out: pytest.fail("the eager head ran")
    drafter.store = torch.tensor([[10.0], [0.0], [0.0], [0.0]])
    backend.prepare_metadata = lambda batch: setattr(batch, "metadata_prepared", True)
    req = SimpleNamespace(table_idx=0, pending_slot=0, cached_len=5)
    spec = SimpleNamespace(k=3, reqs=[req], first_drafts=[2], num_reqs=1)
    graphs = _ChainGraphs()
    token_pool = torch.zeros((1, 12), dtype=torch.int32)

    drafts = drafter.draft_chain(SimpleNamespace(spec=spec), token_pool, context, backend, mrope=False, graphs=graphs)

    assert drafts.tolist() == [[0, 1]] and graphs.finished == (0, 2)
    assert token_pool[0, 7:9].tolist() == [40, 41]
    torch.testing.assert_close(graphs.started[0], torch.tensor([[10.0]]))
    assert graphs.started[1].tolist() == [2]
    assert graphs.replayed == [([5], [5], [0], True), ([6], [6], [0], True)]


def test_mtp_draft_chain_keeps_the_adaptive_depth_when_it_replays(monkeypatch):
    from freetoken.env import ENV

    drafter, steps, context, backend = _chain_drafter(monkeypatch, [])
    drafter.logits = lambda out: pytest.fail("the eager head ran")
    monkeypatch.setattr(ENV.SPEC_DRAFT_MIN_P, "value", 0.5)
    backend.prepare_metadata = lambda batch: setattr(batch, "metadata_prepared", True)
    req = SimpleNamespace(table_idx=0, pending_slot=0, cached_len=5)

    def run(first_conf, confs):
        spec = SimpleNamespace(k=4, reqs=[req], first_drafts=[2], first_conf=[first_conf], num_reqs=1)
        graphs = _ChainGraphs(confs=confs)
        out = drafter.draft_chain(SimpleNamespace(spec=spec), torch.zeros((1, 12), dtype=torch.int32), context, backend, mrope=False, graphs=graphs)
        return out, graphs

    out, graphs = run(0.2, [0.9, 0.9, 0.9])
    assert out is None and graphs.replayed == [] and graphs.finished is None, "an unsure first draft runs no chain step"
    out, graphs = run(0.9, [0.9, 0.2, 0.9])
    assert out.shape == (1, 2) and len(graphs.replayed) == 2, "the step that gives an unsure draft still runs, the next does not"
    out, graphs = run(0.9, [0.9, 0.9, 0.9])
    assert out.shape == (1, 3)


def test_mtp_draft_chain_runs_eagerly_where_no_chain_graph_applies(monkeypatch):
    drafter, steps, context, backend = _chain_drafter(monkeypatch, [torch.tensor([[0.0, 0.0, 9.0]])])
    req = SimpleNamespace(table_idx=0, pending_slot=0, cached_len=5)
    spec = SimpleNamespace(k=2, reqs=[req], first_drafts=[2], num_reqs=1)
    graphs = _ChainGraphs(replayable=False)
    token_pool = torch.zeros((1, 12), dtype=torch.int32)

    drafts = drafter.draft_chain(SimpleNamespace(spec=spec), token_pool, context, backend, mrope=False, graphs=graphs)

    assert drafts.tolist() == [[2]] and token_pool[0, 7].item() == 2 and len(steps) == 1
    assert graphs.started is None and graphs.replayed == []


def test_mtp_draft_chain_confidence_is_the_softmax_max_and_a_mixed_batch_runs_on(monkeypatch):
    from freetoken.env import ENV

    # request 0 is unsure of draft 2 (flat logits), request 1 is sure: step 3 still runs for request 1
    flat, sharp = [0.0, 0.0, 0.0, 0.0], [0.0, 9.0, 0.0, 0.0]
    drafter, steps, context, backend = _chain_drafter(monkeypatch, [torch.tensor([flat, sharp])] * 2)
    drafter.store = torch.zeros(10, 1)
    monkeypatch.setattr(ENV.SPEC_DRAFT_MIN_P, "value", 0.5)
    reqs = [SimpleNamespace(table_idx=i, pending_slot=0, cached_len=5) for i in range(2)]
    spec = SimpleNamespace(k=3, reqs=reqs, first_drafts=[2, 2], first_conf=[0.9, 0.9])
    drafter._page_table = lambda: torch.arange(40, dtype=torch.int32).reshape(2, 20)
    drafts = drafter.draft_chain(SimpleNamespace(spec=spec), torch.zeros((2, 12), dtype=torch.int32), context, backend, mrope=False)
    assert len(steps) == 2 and drafts.shape == (2, 2)


def test_the_qwen3_5_adapter_is_an_mtp_adapter_whose_head_keeps_the_targets_config():
    """MtpDrafter asks every adapter for head_config; the in-checkpoint Qwen3.5/3.6 head shares the target's."""
    from freetoken.models.qwen3_5_moe.mtp import QWEN3_5_MTP
    from freetoken.speculative.mtp import MtpAdapter

    assert isinstance(QWEN3_5_MTP, MtpAdapter)
    config = SimpleNamespace(num_experts=8)
    assert QWEN3_5_MTP.head_config(config, source=None) is config
