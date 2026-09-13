"""Qwen3.5/3.6 MTP draft head: the parts that are decided, not computed.

The head's arithmetic is the target's own decoder block, which is already covered.
What is specific to the head -- and what a checkpoint can disagree with -- is where
its block sits in the layer pattern and how its metadata is read.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    RotaryConfig,
)
from freetoken.models.qwen3_5_moe.mtp import _with_full_attention


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
