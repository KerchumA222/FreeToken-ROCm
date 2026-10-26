"""Qwen3.8-Flash-Next MTP draft head: sidecar resolution and loading, the head's math, and the
verify-time state it leans on (PLE contexts and conv history, the QSA index ring).

The head's arithmetic is checked against a transcription of llama.cpp's ``graph_mtp`` written here from
the maths (``W_e e + W_h h_s`` per stream, not the fused concat), with the decoder block stubbed: the block
itself is the trunk's own layer and is covered by the other files.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from freetoken.layers.base import StateLessOP
from freetoken.models.register import get_model_spec
from freetoken.speculative.mtp import MtpSource, mtp_adapter, with_draft_layer

from .common import Fixture, fresh_ctx, parsed_config, requires_cuda, toy_hf_config
from .test_skeleton import _group_norm, _hf_gated_residual

GGUF_ARCH = "Qwen4ExpGGUFForCausalLM"
SIDECAR = "mtp-sidecar.gguf"


def _toy_config():
    from freetoken.models.qwen4_exp.config import parse_config

    return parse_config(toy_hf_config(4))


# --------------------------------------------------------------------------------------
# registration and config
# --------------------------------------------------------------------------------------


def test_only_the_gguf_spec_registers_an_mtp_adapter():
    from freetoken.models.qwen4_exp import QWEN4_EXP_MTP

    assert get_model_spec(GGUF_ARCH).mtp == "QWEN4_EXP_MTP"
    assert mtp_adapter(get_model_spec(GGUF_ARCH)) is QWEN4_EXP_MTP
    with pytest.raises(ValueError, match="no MTP draft head support"):
        mtp_adapter(get_model_spec("Qwen4ExpForConditionalGeneration"))


def _qsa_group(config):
    from freetoken.models.config import FullAttentionGroupConfig

    return next(g for g in config.attention_groups if isinstance(g, FullAttentionGroupConfig))


def test_draft_block_joins_the_qsa_group_with_its_own_index_layer():
    config = _toy_config()
    full = _qsa_group(config)
    assert full.layer_ids == (3,) and full.num_index_layers == 1

    out = with_draft_layer(config)
    group = _qsa_group(out)
    assert group.layer_ids == (3, 4) and group.num_index_layers == 2
    assert 4 not in next(g for g in out.attention_groups if g is not group).layer_ids
    (spec,) = [s for s in out.kv_cache_group_specs() if s.layer_ids]
    assert spec.num_index_layers == 2 and spec.index_ratio == config.qwen4_args.index_ratio
    # the indexer count is the group's, not the model's: the plain target config is untouched
    assert _qsa_group(config).num_index_layers == 1


def _metadata(**overrides):
    md = {"qwen4exp.nextn_predict_layers": 1, "qwen4exp.nextn_shared_target_tensors": True}
    md.update(overrides)
    return md


def test_sidecar_metadata_must_be_one_block_borrowing_the_target_tensors():
    from freetoken.models.qwen4_exp.mtp import check_head_metadata

    check_head_metadata(SIDECAR, _metadata())
    with pytest.raises(NotImplementedError, match="one draft block"):
        check_head_metadata(SIDECAR, _metadata(**{"qwen4exp.nextn_predict_layers": 2}))
    with pytest.raises(NotImplementedError, match="own embedding"):
        check_head_metadata(SIDECAR, _metadata(**{"qwen4exp.nextn_shared_target_tensors": False}))


def test_locate_reads_the_sidecar_block_and_checks_its_metadata(monkeypatch):
    from freetoken.models.gguf import reader
    from freetoken.models.qwen4_exp import mtp

    monkeypatch.setattr(reader, "gguf_config_source", lambda path: path)
    monkeypatch.setattr(reader, "gguf_architecture", lambda path: "qwen4exp")
    monkeypatch.setattr(mtp, "mtp_draft_layer", lambda path: 48 if path == SIDECAR else None)
    monkeypatch.setattr(reader, "load_gguf_metadata", lambda path: _metadata())

    assert mtp.QWEN4_EXP_MTP.locate("target.gguf", SIDECAR) == MtpSource(path=SIDECAR, block=48)
    with pytest.raises(ValueError, match="carries no nextn.*pass the draft head's GGUF"):
        mtp.QWEN4_EXP_MTP.locate("target.gguf", None)

    monkeypatch.setattr(reader, "load_gguf_metadata", lambda path: _metadata(**{"qwen4exp.nextn_predict_layers": 0}))
    with pytest.raises(NotImplementedError):
        mtp.QWEN4_EXP_MTP.locate("target.gguf", SIDECAR)

    monkeypatch.setattr(reader, "gguf_architecture", lambda path: "qwen35moe" if path == SIDECAR else "qwen4exp")
    with pytest.raises(ValueError, match="draft head .* is 'qwen35moe'"):
        mtp.QWEN4_EXP_MTP.locate("target.gguf", SIDECAR)


def test_head_keeps_the_sidecars_expert_geometry(monkeypatch):
    """A sidecar cut from the 512-expert base model beside a 256-expert fine-tune: the head's MoE is the sidecar's."""
    from freetoken.models.gguf import reader
    from freetoken.models.qwen4_exp.mtp import QWEN4_EXP_MTP

    config = _toy_config()
    md = {
        "qwen4exp.embedding_length": config.hidden_size, "qwen4exp.expert_count": 2 * config.num_experts,
        "qwen4exp.expert_used_count": 10, "qwen4exp.expert_feed_forward_length": 96,
        "qwen4exp.expert_shared_feed_forward_length": 80,
    }
    monkeypatch.setattr(reader, "load_gguf_metadata", lambda path: md)
    object.__setattr__(config, "gguf_expert_bank_types", {"down": 7})
    head = QWEN4_EXP_MTP.head_config(config, MtpSource(path=SIDECAR, block=4))
    assert (head.num_experts, head.num_experts_per_tok) == (2 * config.num_experts, 10)
    assert (head.moe_intermediate_size, head.shared_expert_intermediate_size) == (96, 80)
    assert head.gguf_expert_bank_types == {"down": 7} and config.num_experts_per_tok != 10

    md["qwen4exp.embedding_length"] = config.hidden_size + 1
    with pytest.raises(ValueError, match="wide"):
        QWEN4_EXP_MTP.head_config(config, MtpSource(path=SIDECAR, block=4))


def test_draft_block_is_found_from_the_nextn_tensors(monkeypatch):
    from freetoken.models.gguf import reader
    from freetoken.models.qwen4_exp.mtp import mtp_draft_layer

    names = {"blk.47.attn_q.weight": 0, "blk.48.nextn.eh_proj.weight": 0, "blk.48.attn_q.weight": 0}
    monkeypatch.setattr(reader, "gguf_tensor_index", lambda path: names)
    assert mtp_draft_layer(SIDECAR) == 48
    monkeypatch.setattr(reader, "gguf_tensor_index", lambda path: {"blk.47.attn_q.weight": 0})
    assert mtp_draft_layer(SIDECAR) is None


def test_head_modules_map_onto_the_sidecar_tensors():
    from freetoken.models.qwen4_exp.gguf import to_gguf

    tensors = {"blk.48.nextn.eh_proj.weight": 0, "blk.48.attn_q.weight": 0}
    assert to_gguf("mtp.eh_proj", tensors) == ("blk.48.nextn.eh_proj",)
    assert to_gguf("mtp.hyper_connection_mixer.input_mix_weight_down", tensors) == ("blk.48.nextn.hc_head_down",)
    assert to_gguf("mtp.hyper_connection_mixer.input_mix_weight_up", tensors) == ("blk.48.nextn.hc_head_up",)
    assert to_gguf("mtp.layer.self_attn.q_proj", tensors) == ("blk.48.attn_q",)
    # upstream maps the indexer's halves separately; the packed mapping fuses them into index_qk_proj
    assert to_gguf("mtp.layer.self_attn.indexer.index_q_proj", tensors) == ("blk.48.indexer.q_proj",)
    assert to_gguf("mtp.layer.self_attn.indexer.index_k_proj", tensors) == ("blk.48.indexer.k_proj",)
    assert to_gguf("mtp.layer.attn_hyper_connection.block_inject_weight", tensors) == ("blk.48.hc_attn_inject",)
    assert to_gguf("mtp.layer.mlp.experts.0.down_proj", tensors) == ("blk.48.ffn_down_exps",)
    assert to_gguf("mtp.hyper_connection_mixer.hc_norm", tensors) == ()
    # a GGUF without a draft block maps nothing, and the trunk's names are unchanged
    assert to_gguf("mtp.eh_proj", {"blk.47.attn_q.weight": 0}) == ()
    assert to_gguf("model.language_model.layers.47.self_attn.q_proj", tensors) == ("blk.47.attn_q",)


# --------------------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------------------


@pytest.fixture
def cpu_rope():
    """The rope tables of a model built on meta live on the rope device; keep both, and the cache of tables, out of other tests."""
    from freetoken.layers import rotary

    previous = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))
    rotary.get_rope.cache_clear()
    yield
    rotary.set_rope_device(previous)
    rotary.get_rope.cache_clear()


def _fake_info(t: torch.Tensor) -> SimpleNamespace:
    raw = t.float().contiguous().numpy().view(np.uint8)
    return SimpleNamespace(ggml_type=0, ne=tuple(reversed(t.shape)), nbytes=raw.nbytes, data=lambda raw=raw: raw)


def _sidecar_tensors(config, block: int = 4) -> dict[str, torch.Tensor]:
    """The sidecar's tensors at toy dims, in torch order, filled with distinct random values."""
    a, g = config.qwen4_args, torch.Generator().manual_seed(0)
    h, hc, lr = config.hidden_size, a.hc_count, a.hc_lowrank
    q_out = config.num_qo_heads * config.head_dim
    kv_out = config.num_kv_heads * config.head_dim
    idx_q = a.index_n_heads * a.index_head_dim
    shapes = {
        "attn_q": (2 * q_out, h), "attn_k": (kv_out, h), "attn_v": (kv_out, h), "attn_output": (h, q_out),
        "attn_q_norm": (config.head_dim,), "attn_k_norm": (config.head_dim,),
        "indexer.q_proj": (idx_q, h), "indexer.k_proj": (a.index_head_dim, h),
        "indexer.q_norm": (a.index_head_dim,), "indexer.k_norm": (a.index_head_dim,),
        "ffn_gate_inp": (config.num_experts, h), "ffn_gate_inp_shexp": (h,),
        "ffn_gate_shexp": (config.shared_expert_intermediate_size, h),
        "ffn_up_shexp": (config.shared_expert_intermediate_size, h),
        "ffn_down_shexp": (h, config.shared_expert_intermediate_size),
        "ffn_gate_exps": (config.num_experts, config.moe_intermediate_size, h),
        "nextn.eh_proj": (h, 2 * h), "nextn.enorm": (h,), "nextn.hnorm": (hc * h,),
        "nextn.hc_head_down": (lr, hc * h), "nextn.hc_head_up": (hc * h, lr), "nextn.hc_head_norm": (hc * h,),
    }
    for site in ("attn", "ffn"):
        shapes.update({
            f"hc_{site}_down": (lr, hc * h), f"hc_{site}_up": (hc * h, lr),
            f"hc_{site}_inject": (hc, hc * h), f"hc_{site}_norm": (hc * h,),
        })
    return {f"blk.{block}.{k}.weight": torch.randn(*s, generator=g) for k, s in shapes.items()}


def test_sidecar_weights_fill_exactly_the_head_state_dict(monkeypatch, cpu_rope):
    from freetoken.models.gguf import reader
    from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPHead, iter_gguf_mtp_weights
    from freetoken.utils import torch_dtype

    config = with_draft_layer(_toy_config())
    tensors = _sidecar_tensors(config)
    monkeypatch.setattr(reader, "gguf_tensor_index", lambda path: {k: _fake_info(v) for k, v in tensors.items()})
    weights = dict(iter_gguf_mtp_weights(SIDECAR, config))

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        head = Qwen4ExpMTPHead(config, 4)
    reference = head.state_dict()
    assert set(weights) == {f"mtp.{k}" for k in reference}
    for key, tensor in reference.items():
        assert weights[f"mtp.{key}"].shape == tensor.shape, key

    b = lambda name: tensors[f"blk.4.{name}.weight"]
    # a norm is stored as 1 + w and the module applies 1 + w itself
    torch.testing.assert_close(weights["mtp.enorm.weight"], b("nextn.enorm") - 1.0)
    torch.testing.assert_close(weights["mtp.hnorm.weight"], b("nextn.hnorm") - 1.0)
    torch.testing.assert_close(weights["mtp.hyper_connection_mixer.hc_norm.weight"], b("nextn.hc_head_norm") - 1.0)
    torch.testing.assert_close(weights["mtp.layer.attn_hyper_connection.hc_norm.weight"], b("hc_attn_norm") - 1.0)
    bf = lambda t: t.to(torch.bfloat16)
    # eh_proj keeps the [W_e | W_h] column order, the mixer its unmerged down/up
    assert torch.equal(weights["mtp.eh_proj.weight"], bf(b("nextn.eh_proj")))
    assert torch.equal(weights["mtp.hyper_connection_mixer.input_mix_weight_down.weight"], bf(b("nextn.hc_head_down")))
    assert torch.equal(weights["mtp.hyper_connection_mixer.input_mix_weight_up.weight"], bf(b("nextn.hc_head_up")))
    lowrank = config.qwen4_args.hc_lowrank
    merged = weights["mtp.layer.mlp_hyper_connection.input_mix_weight_down_block_inject.weight"]
    assert torch.equal(merged[:lowrank], bf(b("hc_ffn_down")))
    assert torch.equal(merged[lowrank : lowrank + config.qwen4_args.hc_count], bf(b("hc_ffn_inject")))
    assert not merged[lowrank + config.qwen4_args.hc_count :].any()
    assert torch.equal(
        weights["mtp.layer.self_attn.qkv_proj.weight"], bf(torch.cat([b("attn_q"), b("attn_k"), b("attn_v")]))
    )
    assert torch.equal(
        weights["mtp.layer.self_attn.indexer.index_qk_proj.weight"], bf(torch.cat([b("indexer.q_proj"), b("indexer.k_proj")]))
    )
    assert weights["mtp.layer.mlp.shared_expert_gate.weight"].shape == (1, config.hidden_size)


@pytest.mark.parametrize("leaf", ["embed_tokens", "shared_head_head"])
def test_a_sidecar_with_its_own_embedding_or_head_is_refused(monkeypatch, leaf):
    from freetoken.models.gguf import reader
    from freetoken.models.qwen4_exp.mtp import iter_gguf_mtp_weights

    config = with_draft_layer(_toy_config())
    tensors = _sidecar_tensors(config)
    tensors[f"blk.4.nextn.{leaf}.weight"] = torch.zeros(2, 2)
    monkeypatch.setattr(reader, "gguf_tensor_index", lambda path: {k: _fake_info(v) for k, v in tensors.items()})
    with pytest.raises(NotImplementedError, match=leaf):
        list(iter_gguf_mtp_weights(SIDECAR, config))


def test_trunk_reader_is_unchanged_by_the_block_split(monkeypatch, cpu_rope):
    """The trunk's tensors now come from the shared per-layer reader; its names must stay what the model builds."""
    from freetoken.models.gguf import reader
    from freetoken.models.qwen4_exp.gguf import _iter_layer
    from freetoken.utils import torch_dtype

    config = _toy_config()
    tensors = _sidecar_tensors(config, block=3)
    monkeypatch.setattr(reader, "gguf_tensor_index", lambda path: {})
    index = {k: _fake_info(v) for k, v in tensors.items() if "nextn" not in k}
    names = {name for name, _ in _iter_layer(index, config, None, 3, "model.layers.3", linear=False)}

    from freetoken.models.qwen4_exp.model import Qwen4ExpDecoderLayer

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        layer = Qwen4ExpDecoderLayer(config, 3, prefix="model.layers.3")
    assert names == {f"model.layers.3.{k}" for k in layer.state_dict()}


# --------------------------------------------------------------------------------------
# the head's math
# --------------------------------------------------------------------------------------


class _StubBlock(StateLessOP):
    """Stands in for the trunk block: records what it was given and applies a fixed affine map."""

    def __init__(self, *args, **kwargs):
        self._is_linear = False
        self.ple = None
        self.seen = []

    def forward(self, streams, batch):
        self.seen.append((streams.clone(), batch))
        return streams * 0.75 + 0.125


def _reference_head(R, token_embed, w, hc, hidden, eps, block):
    """llama.cpp ``graph_mtp``: per-stream ``W_e e_norm + W_h h_norm_s``, the block, then the hc_head mixer."""
    tokens = R.shape[0]
    h_norm = _group_norm(R, w["hnorm"], eps, hc).view(tokens, hc, hidden)
    e_norm = _group_norm(token_embed, w["enorm"], eps, 1)
    w_e, w_h = w["eh"][:, :hidden], w["eh"][:, hidden:]
    streams = torch.stack([e_norm @ w_e.T + h_norm[:, s] @ w_h.T for s in range(hc)], dim=1)
    streams = block(streams.flatten(-2))
    mixed, _ = _hf_gated_residual(hc, streams, w["mix_norm"], w["down"], w["up"], None, hidden, eps)
    return mixed, streams


def test_head_forward_matches_a_reference_transcription(monkeypatch):
    from freetoken.models.qwen4_exp import model as model_module
    from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPHead
    from freetoken.utils import torch_dtype

    monkeypatch.setattr(model_module, "Qwen4ExpDecoderLayer", _StubBlock)
    torch.manual_seed(3)
    config = with_draft_layer(_toy_config())
    a, h, hc = config.qwen4_args, config.hidden_size, config.qwen4_args.hc_count
    with torch_dtype(torch.float32):
        head = Qwen4ExpMTPHead(config, 4)
    gen = torch.Generator().manual_seed(4)
    for tensor in head.state_dict().values():
        tensor.normal_(0.0, 0.1, generator=gen)
    batch = object()
    ctx = fresh_ctx()

    tokens = 5
    R = torch.randn(tokens, a.ple_state_width, generator=gen)
    embed = torch.randn(tokens, h, generator=gen)
    weights = dict(
        hnorm=head.hnorm.weight, enorm=head.enorm.weight, eh=head.eh_proj.weight,
        mix_norm=head.hyper_connection_mixer.hc_norm.weight,
        down=head.hyper_connection_mixer.input_mix_weight_down.weight,
        up=head.hyper_connection_mixer.input_mix_weight_up.weight,
    )
    expected, expected_streams = _reference_head(
        R, embed, weights, hc, h, config.rms_norm_eps, lambda s: s * 0.75 + 0.125
    )

    with ctx.forward_batch(batch):
        out, streams = head.forward_chain(R, embed)
        plain = head.forward(R, embed)
        chained, _ = head.forward_chain(streams, embed.flip(0))
    assert out.shape == (tokens, h) and streams.shape == (tokens, a.ple_state_width)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(streams, expected_streams, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(plain, expected, rtol=1e-5, atol=1e-6)
    # the block is entered with the streams that eh_proj produced, inside the active batch
    entered, seen_batch = head.layer.seen[0]
    assert seen_batch is batch and entered.shape == (tokens, a.ple_state_width)

    # a chained step takes the block output, not the mixed LM-head input
    expected_chain, _ = _reference_head(
        expected_streams, embed.flip(0), weights, hc, h, config.rms_norm_eps, lambda s: s * 0.75 + 0.125
    )
    torch.testing.assert_close(chained, expected_chain, rtol=1e-5, atol=1e-6)


def test_every_stream_sees_the_same_embedding_but_its_own_hidden(monkeypatch):
    from freetoken.models.qwen4_exp import model as model_module
    from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPHead
    from freetoken.utils import torch_dtype

    monkeypatch.setattr(model_module, "Qwen4ExpDecoderLayer", _StubBlock)
    config = with_draft_layer(_toy_config())
    h, hc = config.hidden_size, config.qwen4_args.hc_count
    with torch_dtype(torch.float32):
        head = Qwen4ExpMTPHead(config, 4)
    head.hnorm.weight.zero_()
    head.enorm.weight.zero_()
    head.eh_proj.weight.copy_(torch.eye(h, 2 * h))  # picks the embedding half only
    ctx = fresh_ctx()
    R = torch.randn(2, hc * h)
    embed = torch.randn(2, h)
    with ctx.forward_batch(object()):
        head.forward_chain(R, embed)
    entered = head.layer.seen[0][0].view(2, hc, h)
    expected = _group_norm(embed, head.enorm.weight, config.rms_norm_eps, 1)
    for s in range(hc):
        torch.testing.assert_close(entered[:, s], expected)

    head.eh_proj.weight.copy_(torch.eye(h, 2 * h).roll(h, dims=1))  # the hidden half only
    with ctx.forward_batch(object()):
        head.forward_chain(R, embed)
    entered = head.layer.seen[1][0].view(2, hc, h)
    streams = _group_norm(R, head.hnorm.weight, config.rms_norm_eps, hc).view(2, hc, h)
    torch.testing.assert_close(entered, streams)


# --------------------------------------------------------------------------------------
# verify-time state: PLE contexts and conv history
# --------------------------------------------------------------------------------------


def test_verify_contexts_roll_through_the_rows_before_them():
    from freetoken.models.qwen4_exp.ple import _verify_contexts

    context = torch.tensor([[1, 2], [3, 4]])  # two requests, ctx_len 2
    ids = torch.tensor([10, 20, 11, 21, 12, 22])  # step-major: three steps of two requests
    got = _verify_contexts(context.repeat(3, 1), ids, 2)
    assert got.tolist() == [[1, 2], [3, 4], [2, 10], [4, 20], [10, 11], [20, 21]]


def _verify_meta(config, k: int, num_reqs: int, slots, snap, step=None):
    from freetoken.models.qwen4_exp.ple import PLEMetadata

    spec = SimpleNamespace(num_reqs=num_reqs, k=k, snap_slots=snap, step=step)
    rows = num_reqs * (k + 1 if step is None else 1)
    return PLEMetadata(
        input_ids=torch.arange(rows) + 100,
        cu_seqlens=torch.arange(rows + 1, dtype=torch.int32),
        seq_lens=(1,) * rows,
        ngram_context=torch.zeros(rows, config.qwen4_args.ngram_size - 1, dtype=torch.int64),
        state_slots=slots.repeat(rows // num_reqs),
        fresh_slots=None,
        is_decode=True,
        spec=spec,
    )


@pytest.mark.parametrize("num_reqs, k", [(1, 2), (2, 3)])
def test_ngram_context_commit_keeps_the_last_row_live_and_snapshots_the_rest(num_reqs, k):
    from freetoken.models.qwen4_exp.ple import _verify_contexts, commit_ngram_context

    config = _toy_config()
    ctx_len = config.qwen4_args.ngram_size - 1
    live = torch.arange(1, num_reqs + 1)
    snap = [torch.arange(8 + s * num_reqs, 8 + (s + 1) * num_reqs) for s in range(k)]
    pool = torch.zeros(8 + k * num_reqs, ctx_len, dtype=torch.int32)
    start = torch.arange(num_reqs * ctx_len).view(num_reqs, ctx_len) + 50
    pool[live] = start.to(torch.int32)
    meta = _verify_meta(config, k, num_reqs, live, snap)
    meta.ngram_context = _verify_contexts(start.repeat(k + 1, 1), meta.input_ids, num_reqs)

    commit_ngram_context(meta, None, pool)

    tokens = meta.input_ids.view(k + 1, num_reqs)
    for i in range(num_reqs):
        history = start[i].tolist() + tokens[:, i].tolist()
        assert pool[live[i]].tolist() == history[-ctx_len:]
        for s in range(k):
            assert pool[snap[s][i]].tolist() == history[s + 1 : s + 1 + ctx_len]


def test_one_step_of_a_sequential_verify_commits_its_own_snapshot():
    from freetoken.models.qwen4_exp.ple import commit_ngram_context

    config = _toy_config()
    k, ctx_len = 2, config.qwen4_args.ngram_size - 1
    live, snap = torch.tensor([1]), [torch.tensor([5]), torch.tensor([6])]
    pool = torch.zeros(8, ctx_len, dtype=torch.int32)
    meta = _verify_meta(config, k, 1, live, snap, step=1)
    meta.ngram_context = torch.tensor([[7, 8]])
    commit_ngram_context(meta, None, pool)
    assert pool[1].tolist() == [8, 100] and pool[6].tolist() == [8, 100] and not pool[5].any()
    # the last step of a verify has no snapshot: the live slot is the state after it
    meta = _verify_meta(config, k, 1, live, snap, step=k)
    meta.ngram_context = torch.tensor([[7, 8]])
    pool.zero_()
    commit_ngram_context(meta, None, pool)
    assert pool[1].tolist() == [8, 100] and not pool[5:7].any()


@pytest.mark.parametrize("step", [None, 1])
def test_verify_conv_equals_decoding_each_step_and_snapshots_between(step):
    from freetoken.models.qwen4_exp.ple import PLELayer

    config = _toy_config()
    args = config.qwen4_args
    num_reqs, k = 2, 3
    torch.manual_seed(6)
    layer = PLELayer(config, args.ple_layer_ids[0])
    layer.conv1d.weight.normal_(0.0, 0.5)
    steps = k + 1 if step is None else 1
    live = torch.tensor([1, 2])
    snap = [torch.tensor([6 + s * num_reqs, 7 + s * num_reqs]) for s in range(k)]
    meta = _verify_meta(config, k, num_reqs, live, snap, step=step)
    x = torch.randn(steps * num_reqs, args.ple_state_width)
    states = torch.randn(6 + k * num_reqs, args.ple_state_width, args.ple_conv_state_len)
    reference = states.clone()

    got = layer._short_conv(x, meta, states)

    outs = []
    for j in range(steps):
        rows = slice(j * num_reqs, (j + 1) * num_reqs)
        plain = _verify_meta(config, k, num_reqs, live, snap, step=0)
        outs.append(layer._decode_conv(x[rows], plain, reference))
        s = j if step is None else step
        if s < k:
            assert torch.equal(states[snap[s]], reference[live]), s
    assert torch.equal(got, torch.cat(outs))
    assert torch.equal(states[live], reference[live])


def test_disk_ple_fill_reads_a_verifys_context_off_the_ngram_state_slot():
    from freetoken.models.qwen4_exp.config import PLE_NGRAM_STATE
    from freetoken.models.qwen4_exp.ple_disk import DiskRowTable

    committed = torch.tensor([[0, 0], [0, 0], [31, 32], [41, 42]], dtype=torch.int32)
    pool = SimpleNamespace(slot_state=lambda name: committed if name == PLE_NGRAM_STATE else None)
    fresh_ctx(linear_state_pool=pool)
    reqs = [SimpleNamespace(linear_slot_idx=2, table_idx=0), SimpleNamespace(linear_slot_idx=None, table_idx=3)]
    batch = SimpleNamespace(
        input_ids=torch.tensor([10, 20, 11, 21, 12, 22], dtype=torch.int32),
        spec=SimpleNamespace(num_reqs=2, reqs=reqs),
    )
    runs = DiskRowTable._verify_runs(None, batch)
    assert [r.tolist() for r in runs] == [
        [31, 32, 10], [41, 42, 20], [32, 10, 11], [42, 20, 21], [10, 11, 12], [20, 21, 22],
    ]


# --------------------------------------------------------------------------------------
# the QSA index ring under a verify
# --------------------------------------------------------------------------------------


def test_the_qsa_ring_widens_with_the_draft_depth():
    from freetoken.kvcache.qsa_pool import QSAKVCache

    config = parsed_config()
    ratio = config.qwen4_args.index_ratio
    assert QSAKVCache.ring_capacity_for(ratio) == ratio
    for k in (1, 3, 4, 7):
        capacity = QSAKVCache.ring_capacity_for(ratio, k)
        # a verify's k + 1 rows and the ratio - 1 members before the first of them all stay addressable
        assert capacity % ratio == 0 and capacity >= ratio + k


def test_kv_cost_and_pool_size_the_widened_ring():
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.qsa_pool import QSAKVCache

    config = with_draft_layer(parsed_config())
    k, slots = 3, 4

    def engine(algorithm):
        return SimpleNamespace(
            model_config=config, max_running_req=slots - 1, page_size=64, tp_info=SimpleNamespace(size=1),
            dtype=torch.bfloat16, speculative_algorithm=algorithm, speculative_num_draft_tokens=k,
        )

    plain, spec = QSAKVCache.kv_cost(engine(None)), QSAKVCache.kv_cost(engine("mtp"))
    a = config.qwen4_args
    wider = QSAKVCache.ring_capacity_for(a.index_ratio, k) - QSAKVCache.ring_capacity_for(a.index_ratio)
    assert spec[0] == plain[0]
    assert spec[1] - plain[1] == slots * a.index_head_dim * 2 * 2 * wider  # two index layers, 2 bytes

    pool = create_kvcache_pool(
        model_config=config, num_pages=2, page_size=64, dtype=torch.bfloat16, device=torch.device("cpu"),
        num_req_slots=slots, num_speculative_tokens=k,
    )
    assert pool.ring_capacity == QSAKVCache.ring_capacity_for(a.index_ratio, k)
    plain_pool = create_kvcache_pool(
        model_config=config, num_pages=2, page_size=64, dtype=torch.bfloat16, device=torch.device("cpu"),
        num_req_slots=slots,
    )
    assert plain_pool.ring_capacity == a.index_ratio


def _group_rows(fixture: Fixture, table_idx: int, groups: int) -> torch.Tensor:
    ratio = fixture.config.qwen4_args.index_ratio
    return fixture.page_table[table_idx, : groups * ratio : ratio].long() // ratio


def _verify_batch(fixture: Fixture, rows, num_reqs: int):
    batch = fixture.batch(rows, "decode")
    batch.spec = SimpleNamespace(num_reqs=num_reqs)
    return batch


def _qsa_verify_scenario(num_speculative_tokens: int):
    """Prefill two requests, then run a 4-row verify of which only the first two rows are accepted.

    The reference decodes the same accepted rows and then the rest one step at a time. Returns the
    two fixtures and the compressed index rows of each, per logical group."""
    layer_id, k, accepted = 3, 3, 2
    config = parsed_config()
    lengths = [301, 210]
    gen = torch.Generator(device="cuda").manual_seed(5)
    hidden = config.hidden_size
    prompt = [torch.randn(n, hidden, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.5 for n in lengths]
    rows = [torch.randn(k + 1, hidden, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.5 for _ in lengths]
    junk = [torch.randn(k + 1, hidden, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.5 for _ in lengths]

    def start(num_spec):
        fixture = Fixture(config, num_pages=64, num_speculative_tokens=num_spec)
        attn = fixture.layer(layer_id)
        reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
        attn.forward(torch.cat(prompt), fixture.batch(reqs, "prefill"))
        for r in reqs:
            fixture.allocate(r.table_idx, r.device_len, r.device_len + k + 1)
        return fixture, attn

    def row(r, s):
        return SimpleNamespace(
            table_idx=r.table_idx, cached_len=lengths[r.table_idx] + s,
            device_len=lengths[r.table_idx] + s + 1, extend_len=1,
        )

    reference, attn = start(0)
    for s in range(k + 1):
        attn.forward(torch.stack([rows[i][s] for i in range(2)]), reference.batch([row(SimpleNamespace(table_idx=i), s) for i in range(2)], "decode"))

    fixture, attn = start(num_speculative_tokens)
    sent = [[rows[i][s] if s < accepted else junk[i][s] for i in range(2)] for s in range(k + 1)]
    batch = _verify_batch(fixture, [row(SimpleNamespace(table_idx=i), s) for s in range(k + 1) for i in range(2)], 2)
    attn.forward(torch.stack([x for step in sent for x in step]), batch)
    for s in range(accepted, k + 1):
        step_rows = [row(SimpleNamespace(table_idx=i), s) for i in range(2)]
        attn.forward(torch.stack([rows[i][s] for i in range(2)]), fixture.batch(step_rows, "decode"))
    return reference, fixture, lengths, k


@requires_cuda
def test_rejected_verify_rows_leave_the_index_cache_as_plain_decoding_would():
    reference, fixture, lengths, k = _qsa_verify_scenario(num_speculative_tokens=3)
    ratio = reference.config.qwen4_args.index_ratio
    for table_idx, n in enumerate(lengths):
        groups = (n + k + 1) // ratio
        got = fixture.pool.cmp_k_cache(0)[_group_rows(fixture, table_idx, groups)]
        want = reference.pool.cmp_k_cache(0)[_group_rows(reference, table_idx, groups)]
        torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)
    # the ring holds the last positions' raw keys, wherever each ring puts them
    for table_idx, n in enumerate(lengths):
        for position in range(n + k + 1 - ratio, n + k + 1):
            got = fixture.pool.pending_ring(0)[table_idx, position % fixture.pool.ring_capacity]
            want = reference.pool.pending_ring(0)[table_idx, position % reference.pool.ring_capacity]
            torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
def test_a_ring_sized_without_the_drafts_loses_a_group_member():
    """The control for the test above: with the plain ring the rejected rows overwrite a member of the open group."""
    reference, fixture, lengths, k = _qsa_verify_scenario(num_speculative_tokens=0)
    ratio = reference.config.qwen4_args.index_ratio
    first = lengths[0] // ratio  # the group that was open at the start of the verify and closes inside it
    got = fixture.pool.cmp_k_cache(0)[_group_rows(fixture, 0, first + 1)[first]].float()
    want = reference.pool.cmp_k_cache(0)[_group_rows(reference, 0, first + 1)[first]].float()
    assert not torch.allclose(got, want, rtol=2e-2, atol=2e-2)
