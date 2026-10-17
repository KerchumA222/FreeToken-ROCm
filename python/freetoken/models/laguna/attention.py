from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.attention import AttentionSpec
from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearOProj, LinearQKVMerged, RMSNorm
from freetoken.layers.rotary import get_rope
from freetoken.models.config import SWAAttentionGroupConfig
from freetoken.utils import div_even, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class LagunaAttention(BaseOP):
    """Laguna GQA with per-head softplus gating and layer-specific query geometry."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        group = config.attention_group_for_layer(layer_id)
        q_heads = config.num_qo_heads_for_layer(layer_id)
        tp_size = get_tp_info().size
        self.layer_id = layer_id
        self.head_dim = group.head_dim
        self.num_qo_heads = div_even(q_heads, tp_size)
        self.num_kv_heads = div_even(group.num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * self.head_dim
        self.kv_attn_dim = self.num_kv_heads * self.head_dim
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=self.head_dim,
            num_qo_heads=q_heads,
            num_kv_heads=group.num_kv_heads,
            has_bias=False,
            quant_config=config.quant,
            prefix=f"{prefix}.qkv_proj",
        )
        self.g_proj = LinearColParallelMerged(
            config.hidden_size,
            [q_heads],
            has_bias=False,
            quant_config=config.quant,
            prefix=f"{prefix}.g_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary = group.rotary_config
        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=rotary.rotary_dim,
            max_position=rotary.max_position,
            base=rotary.base,
            rope_scaling=tuple(rotary.scaling.items()) if rotary.scaling else None,
        )
        self.o_proj = LinearOProj(
            q_heads * self.head_dim,
            config.hidden_size,
            has_bias=False,
            quant_config=config.quant,
            prefix=f"{prefix}.o_proj",
        )
        self.attn_spec = AttentionSpec(
            sliding_window=(group.sliding_window if isinstance(group, SWAAttentionGroupConfig) else None)
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        qkv = self.qkv_proj.forward(x)
        q, k, v = qkv.split(
            [self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1
        )
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        self.q_norm.forward_inplace(q)
        self.k_norm.forward_inplace(k)
        q = q.reshape(-1, self.qo_attn_dim)
        k = k.reshape(-1, self.kv_attn_dim)
        q, k = self.rotary.forward(ctx.batch.get_attn_positions(), q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        v = v.contiguous()
        attended = ctx.attn_backend.forward(
            q, k, v, self.layer_id, ctx.batch, attn_spec=self.attn_spec
        )
        gate = F.softplus(self.g_proj.forward(x).float()).to(attended.dtype)
        gate = gate.view(-1, self.num_qo_heads, 1)
        return self.o_proj.forward((attended * gate).reshape(-1, self.qo_attn_dim))


__all__ = ["LagunaAttention"]
