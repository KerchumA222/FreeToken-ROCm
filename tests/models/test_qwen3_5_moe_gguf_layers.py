"""qwen35moe GGUF: which blocks are model layers, and which kind each one is.

Two ways the adapter used to get this wrong, both silent:

* ``block_count`` includes ``nextn_predict_layers`` MTP draft blocks. Ornith-1.5-35B-A3B
  reports 41 blocks for a 40-layer model and names the draft head ``blk.40.*``, not
  ``mtp.*`` the way Qwen3.5's own GGUF does.
* Layer kinds came from the ``full_attention_interval`` pattern alone, so any stack that
  is not purely that pattern was typed wrong.
"""

from __future__ import annotations

import pytest

from freetoken.models.gguf import reader
from freetoken.models.qwen3_5_moe.gguf import _layer_types, _num_model_layers

# Qwen3.5/3.6 geometry: 40 blocks, full attention on every 4th (3, 7, ... 39).
INTERVAL = 4
FULL = [i for i in range(40) if (i + 1) % INTERVAL == 0]
SSM_SUFFIXES = ("ssm_a", "ssm_conv1d.weight", "ssm_out.weight")


def _names(num_blocks: int, ssm_layers: set[int]) -> set[str]:
    out = {"token_embd.weight", "output_norm.weight", "output.weight"}
    for i in range(num_blocks):
        out |= {f"blk.{i}.attn_norm.weight", f"blk.{i}.ffn_gate_exps.weight"}
        if i in ssm_layers:
            out |= {f"blk.{i}.{s}" for s in SSM_SUFFIXES}
        else:
            out |= {f"blk.{i}.attn_q.weight", f"blk.{i}.attn_k.weight"}
    return out


@pytest.fixture
def tensor_names(monkeypatch):
    """Install a synthetic tensor table for ``_layer_types`` to read."""

    def install(names: set[str]) -> str:
        monkeypatch.setattr(reader, "gguf_tensor_names", lambda _path: names)
        return "synthetic.gguf"

    return install


def test_nextn_blocks_are_not_model_layers():
    # Real Ornith-1.5-35B-A3B header values.
    md = {"qwen35moe.block_count": 41, "qwen35moe.nextn_predict_layers": 1}
    assert _num_model_layers(md, "qwen35moe") == 40


def test_block_count_stands_without_nextn():
    assert _num_model_layers({"qwen35moe.block_count": 40}, "qwen35moe") == 40
    # llama.cpp writes an explicit 0 on models with no MTP head.
    assert _num_model_layers(
        {"qwen35moe.block_count": 40, "qwen35moe.nextn_predict_layers": 0}, "qwen35moe"
    ) == 40


def test_layer_kinds_come_from_tensor_names(tensor_names):
    path = tensor_names(_names(40, set(range(40)) - set(FULL)))
    types = _layer_types(path, 40, INTERVAL)
    assert [i for i, t in enumerate(types) if t == "full_attention"] == FULL


def test_ornith_extra_block_is_ignored_not_mistyped(tensor_names):
    """Block 40 has no ssm tensors, but it is past num_layers and must not appear."""
    path = tensor_names(_names(41, set(range(40)) - set(FULL)))
    types = _layer_types(path, _num_model_layers(
        {"qwen35moe.block_count": 41, "qwen35moe.nextn_predict_layers": 1}, "qwen35moe"
    ), INTERVAL)
    assert len(types) == 40
    assert [i for i, t in enumerate(types) if t == "full_attention"] == FULL


def test_names_win_when_they_disagree_with_the_interval(tensor_names):
    """A stack the interval pattern cannot describe: full attention at 0, 4, 8 (i % 4)."""
    off_pattern = {i for i in range(40) if i % INTERVAL != 0}
    path = tensor_names(_names(40, off_pattern))
    types = _layer_types(path, 40, INTERVAL)
    got = [i for i, t in enumerate(types) if t == "full_attention"]
    assert got == [i for i in range(40) if i % INTERVAL == 0]
    assert got != FULL, "the two conventions must actually differ for this to prove anything"


def test_falls_back_to_the_interval_without_ssm_tensors(tensor_names):
    """Metadata-only shims and synthetic fixtures carry no ssm tensors."""
    path = tensor_names({"token_embd.weight"})
    types = _layer_types(path, 40, INTERVAL)
    assert [i for i, t in enumerate(types) if t == "full_attention"] == FULL
