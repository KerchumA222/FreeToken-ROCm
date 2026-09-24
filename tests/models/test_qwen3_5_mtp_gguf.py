"""The MTP draft head as a real checkpoint ships it.

These run against unsloth/Qwen3.6-35B-A3B-MTP-GGUF, which carries the nextn block as
``blk.40`` on top of a 40-layer target. What is worth pinning here is the disagreement
between the checkpoint's own counts and the target's: ``block_count`` includes the draft
block, so reading it as the layer count builds a 41st target layer that the hybrid
pattern then misreads as linear attention -- against a block that carries attn_q/k/v.
"""

from __future__ import annotations

import os

import pytest

MODEL = os.environ.get(
    "FT_TEST_QWEN36_MTP_GGUF",
    os.path.expanduser("~/models/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL), reason=f"checkpoint not present: {MODEL}"
)


@pytest.fixture(scope="module")
def config():
    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.models.qwen3_5_moe.gguf import parse_gguf_config

    return parse_gguf_config(build_gguf_shim(MODEL))


def test_the_draft_block_is_not_counted_as_a_target_layer(config):
    from freetoken.models.gguf.reader import load_gguf_metadata

    md = load_gguf_metadata(MODEL)
    assert md["qwen35moe.block_count"] == 41
    assert md["qwen35moe.nextn_predict_layers"] == 1
    assert config.num_layers == 40
    assert config.num_nextn_layers == 1
    # And no attention group claims it, so nothing builds a block for it by accident.
    for group in config.attention_groups:
        assert 40 not in group.layer_ids


def test_the_draft_count_survives_the_engine_config_rebuild(config):
    """EngineConfig.model_config rebuilds the config with dataclasses.replace, which keeps
    only declared fields. The draft count has to be one, or the engine never learns the
    checkpoint has a head -- this is the trap that eats object.__setattr__ stashes."""
    from dataclasses import replace

    rebuilt = replace(config, quant=None)
    assert rebuilt.num_nextn_layers == 1
    assert not hasattr(rebuilt, "gguf_expert_bank_types")  # the stash, for contrast


def test_the_draft_layer_is_found_from_the_checkpoint(config):
    from freetoken.models.qwen3_5_moe.mtp import mtp_draft_layer

    assert mtp_draft_layer(MODEL) == config.num_layers


def test_the_draft_block_joins_the_full_attention_group(config):
    """It carries attn_q/k/v and no ssm_*, but sits at an index the hybrid pattern would
    call linear."""
    from freetoken.models.config import FullAttentionGroupConfig
    from freetoken.models.qwen3_5_moe.mtp import _with_full_attention

    out = _with_full_attention(config, 40)
    full = [g for g in out.attention_groups if isinstance(g, FullAttentionGroupConfig)]
    assert len(full) == 1 and 40 in full[0].layer_ids
    # The KV pool sizes itself from len(layer_ids), so the head's block gets its own layer.
    before = [g for g in config.attention_groups if isinstance(g, FullAttentionGroupConfig)][0]
    assert len(full[0].layer_ids) == len(before.layer_ids) + 1
    # The expert bank types are stashed outside the dataclass fields; losing them here
    # would leave the head's MoE with nothing to load.
    assert out.gguf_expert_bank_types == config.gguf_expert_bank_types


def test_the_head_translates_to_the_modules_that_build_it(config):
    from freetoken.models.qwen3_5_moe.mtp import iter_gguf_mtp_weights

    got = {name: tuple(t.shape) for name, t in iter_gguf_mtp_weights(MODEL)}
    hidden = config.hidden_size
    assert got["mtp.eh_proj.weight"] == (hidden, 2 * hidden)
    for norm in ("mtp.enorm.weight", "mtp.hnorm.weight", "mtp.shared_head_norm.weight"):
        assert got[norm] == (hidden,)
    # q is doubled by the attention output gate, then k and v at one head_dim per kv head.
    g = [gr for gr in config.attention_groups if hasattr(gr, "num_kv_heads")][0]
    q_rows = 2 * config.num_qo_heads * g.head_dim
    assert got["mtp.layer.self_attn.qkv_proj.weight"] == (q_rows + 2 * g.num_kv_heads * g.head_dim, hidden)
    assert got["mtp.layer.mlp.shared_expert_gate.weight"] == (1, hidden)
    # The routed experts are the bank reader's job, not this iterator's.
    assert not [k for k in got if "experts" in k]


def test_the_head_borrows_the_target_embedding_and_output(config):
    """``mtp_use_dedicated_embeddings`` is false upstream, so the checkpoint ships neither
    -- the head has to read the target's, and a loader that expects its own would fail."""
    from freetoken.models.gguf.reader import iter_gguf_tensors

    names = {t.name for t in iter_gguf_tensors(MODEL) if t.name.startswith("blk.40.")}
    assert "blk.40.nextn.embed_tokens.weight" not in names
    assert "blk.40.nextn.shared_head_head.weight" not in names
    assert "blk.40.nextn.eh_proj.weight" in names


@pytest.mark.slow
def test_the_target_load_stops_at_the_last_target_layer():
    """The head's block must not reach the target's weight loader. Its index is past the
    target stack, so the layer pattern would call it linear attention and translate its
    attn_q/k/v onto module names the target does not have; its routed bank would index one
    past the end of the per-layer banks."""
    import re

    import torch

    from freetoken.models.qwen3_5_moe.gguf import iter_gguf_weights

    names = [
        n for n, _ in iter_gguf_weights(
            MODEL, torch.device("cpu"), include_moe_experts=False, include_non_moe=True
        )
    ]
    layers = {int(m.group(1)) for n in names if (m := re.match(r"model\.layers\.(\d+)\.", n))}
    assert layers == set(range(40))
    assert not [n for n in names if "nextn" in n or "mtp" in n]
    assert {n for n in names if not n.startswith("model.layers.")} == {
        "lm_head.weight", "model.embed_tokens.qweight", "model.norm.weight",
    }


@pytest.fixture(scope="module")
def head(config):
    """The draft head built against the real checkpoint's geometry."""
    import torch

    from freetoken.distributed import set_tp_info
    from freetoken.models.qwen3_5_moe.mtp import Qwen3_5MTPHead, _with_full_attention
    from freetoken.utils.torch_utils import torch_dtype

    try:
        set_tp_info(rank=0, size=1)
    except Exception:  # already set by another test in this session
        pass
    spec_config = _with_full_attention(config, config.num_layers)
    # The engine builds every model under the compute dtype; without it the norms would
    # come out fp32 and reject the checkpoint's bf16.
    with torch_dtype(torch.bfloat16):
        return Qwen3_5MTPHead(spec_config, config.num_layers)


def test_the_loader_and_the_module_name_the_same_tensors(head):
    """Loader/module disagreement is the failure mode that does not raise: a name that
    lands nowhere leaves a module at its initialization and the model emits plausible
    garbage. Every tensor the reader produces must have a home, and every parameter the
    head has must be fed -- apart from the routed experts, which arrive as banks."""
    from freetoken.models.qwen3_5_moe.mtp import iter_gguf_mtp_weights

    module = {k: tuple(v.shape) for k, v in head.state_dict().items()}
    loaded = {
        name.removeprefix("mtp."): tuple(t.shape)
        for name, t in iter_gguf_mtp_weights(MODEL)
    }
    experts = {k for k in module if ".mlp.experts." in k}
    assert experts, "the draft block has its own routed bank"

    assert set(loaded) == set(module) - experts
    mismatched = {k: (loaded[k], module[k]) for k in loaded if loaded[k] != module[k]}
    assert not mismatched, mismatched


def test_the_head_accepts_the_checkpoint_weights(head):
    """The load itself, so a dtype or transpose disagreement surfaces here rather than as
    wrong numbers at serving time."""
    import torch

    from freetoken.models.qwen3_5_moe.mtp import iter_gguf_mtp_weights

    state = {name.removeprefix("mtp."): t for name, t in iter_gguf_mtp_weights(MODEL)}
    for key, ref in head.state_dict().items():
        if ".mlp.experts." in key:  # delivered as packed banks, not through this path
            state[key] = torch.zeros_like(ref)
    head.load_state_dict(state)


def test_the_draft_block_gets_a_bank_slot_only_when_speculation_is_on(config):
    """The draft block owns a routed bank of its own, past the target's last layer. It
    needs a slot in the expert banks and the offload cache when it is going to run, and
    must not consume one otherwise -- a checkpoint with a head still serves normally."""
    from dataclasses import replace

    assert config.num_speculative_tokens == 0
    assert config.num_moe_layers == config.num_layers

    speculating = replace(config, num_speculative_tokens=2)
    assert speculating.num_moe_layers == config.num_layers + 1


def test_the_expert_piece_reader_skips_the_draft_block_when_speculation_is_off(monkeypatch):
    """The streaming reader must bound blocks the same way the complete-bank loader does:
    with speculation off, ``blk.40``'s routed bank has no slot, and yielding it fails bank
    construction with "expert piece out of range: layer 40". Synthetic tensors, no
    checkpoint needed."""
    from types import SimpleNamespace

    import torch

    import freetoken.models.qwen3_5_moe.gguf as g

    E, H, I, L = 2, 32, 32, 3
    types = {"gate_up": 2, "down": 2}  # Q4_0: 18 bytes per 32 values
    rb = g.row_bytes(32, 2)
    tensors = [
        SimpleNamespace(name=f"blk.{li}.{s}")
        for li in range(L + 1)
        for s in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
    ]
    monkeypatch.setattr(g, "_iter_expert_tensors", lambda path, cfg: iter(tensors))
    monkeypatch.setattr(g, "_bank_types", lambda cfg: types)
    monkeypatch.setattr(g, "_packed_as", lambda t, ty: torch.zeros(E * 32 * rb, dtype=torch.uint8))

    def layers(addressable):
        cfg = SimpleNamespace(num_addressable_layers=addressable, num_layers=L, num_experts=E,
                              hidden_size=H, moe_intermediate_size=I)
        return [li for li, *_ in g.iter_gguf_expert_pieces("unused", cfg)]

    assert layers(L) == list(range(L))
    assert layers(L + 1) == list(range(L + 1))
