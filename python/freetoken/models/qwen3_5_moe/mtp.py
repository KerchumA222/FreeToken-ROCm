"""Qwen3.5/3.6 MTP (``nextn``) draft head.

One decoder block that predicts the token after next, from the target model's hidden
state at position t and the embedding of the token at t+1::

    e    = enorm(embed_tokens[x])          # target's embedding, borrowed
    h    = hnorm(hidden)                   # target's hidden state at t
    z    = eh_proj(concat(e, h))           # [2*hidden] -> [hidden]
    z    = decoder_block(z)                # a normal Qwen3.5 block, full attention
    out  = shared_head_norm(z)             # -> target's lm_head

Follows llama.cpp's ``llama_model_qwen35moe::graph_mtp`` rather than reinterpreting it:
the concat is ``[e | h]`` in that order, the block is the standard pre-norm hybrid one
(its residual stream starts at the eh_proj output), and the head borrows the target's
token embedding and output projection -- a checkpoint may override either with
``nextn.embed_tokens`` / ``nextn.shared_head_head``, which the draft-head-only GGUFs
do not ship.

The block is ALWAYS full attention, whatever the target's layer pattern would say for
an index one past the end: the checkpoints carry ``attn_q/k/v`` for it and no ``ssm_*``
(Qwen3.8-Flash-Next says the same thing in config as ``mtp.layer_types``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.layers import BaseOP, GemmaRMSNorm
from freetoken.layers.linear import LinearReplicated

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def _with_full_attention(config: "ModelConfig", layer_id: int) -> "ModelConfig":
    """``config`` with ``layer_id`` moved into the full-attention group.

    The draft block sits one past the target's last layer, where the hybrid pattern
    would place a linear-attention layer; the checkpoints carry ``attn_q/k/v`` for it.
    llama.cpp sidesteps this by building the MTP graph explicitly rather than from the
    layer pattern -- this is the same statement, made to the config.
    """
    from dataclasses import replace

    from freetoken.models.config import FullAttentionGroupConfig

    groups = []
    for group in config.attention_groups or ():
        ids = set(group.layer_ids)
        if isinstance(group, FullAttentionGroupConfig):
            ids.add(layer_id)
        else:
            ids.discard(layer_id)
        groups.append(replace(group, layer_ids=tuple(sorted(ids))))
    out = replace(config, attention_groups=tuple(groups))
    # replace() keeps only declared fields, and the GGUF readers stash the expert
    # bank types with object.__setattr__ -- carry those across or the block builds
    # its MoE against nothing.
    for key, value in config.__dict__.items():
        if key not in out.__dict__:
            object.__setattr__(out, key, value)
    return out


class Qwen3_5MTPHead(BaseOP):
    """The ``nextn`` draft block. ``layer_id`` is its slot in the attention backend and
    KV pool -- one past the target's last layer, which is also how the checkpoint names
    it (``blk.<n_layer>.*``)."""

    def __init__(self, config: "ModelConfig", layer_id: int, *, prefix: str = "mtp") -> None:
        from .model import Qwen3_5DecoderLayer

        hidden = config.hidden_size
        self.layer_id = layer_id
        self.enorm = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self.hnorm = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self.eh_proj = LinearReplicated(
            2 * hidden, hidden, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.eh_proj",
        )
        self.layer = Qwen3_5DecoderLayer(
            _with_full_attention(config, layer_id), layer_id, prefix=f"{prefix}.layer"
        )
        # The block index the target would assign is one past the end, where the
        # hybrid pattern may say "linear"; the checkpoint says full attention.
        assert not self.layer._is_linear, (
            f"MTP block {layer_id} was built as linear attention; the nextn checkpoints "
            "carry attn_q/k/v for it, so the config's layer pattern disagrees with the "
            "weights"
        )
        self.shared_head_norm = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)

    def forward(self, hidden: torch.Tensor, token_embed: torch.Tensor) -> torch.Tensor:
        """``hidden`` [T, H] from the target at t, ``token_embed`` [T, H] for t+1;
        returns [T, H] for the target's lm_head."""
        e = self.enorm.forward(token_embed)
        h = self.hnorm.forward(hidden)
        z = self.eh_proj.forward(torch.cat([e, h], dim=-1))
        # The block's residual stream starts at the projection output (llama.cpp's
        # inpSA), so it is entered with residual=None.
        out, residual = self.layer.forward(z, None)
        out, _ = self.shared_head_norm.forward_add_residual(out, residual)
        return out


# --------------------------------------------------------------------------------------
# Weights.
# --------------------------------------------------------------------------------------

MTP_DRAFT_LAYER_KV = "mtp.draft_layer"


def mtp_draft_layer(model_path: str) -> int | None:
    """The block index a draft-head GGUF carries, or ``None`` if it is not one."""
    from freetoken.models.gguf.reader import gguf_architecture, load_gguf_metadata

    md = load_gguf_metadata(model_path)
    arch = gguf_architecture(model_path)
    value = md.get(f"{arch}.{MTP_DRAFT_LAYER_KV}")
    if value is not None:
        return int(value)
    # A full checkpoint that happens to carry the head: nextn_predict_layers > 0 puts
    # it at the end of the block range.
    n_nextn = md.get(f"{arch}.nextn_predict_layers")
    n_layer = md.get(f"{arch}.block_count")
    if n_nextn and n_layer:
        return int(n_layer) - int(n_nextn)
    return None


def iter_gguf_mtp_weights(model_path: str, prefix: str = "mtp"):
    """Yield ``(name, tensor)`` for the draft head, in this module's naming (see ``gguf.iter_gguf_mtp_weights``)."""
    from .gguf import iter_gguf_mtp_weights as _iter

    yield from _iter(model_path, prefix)


__all__ = ["Qwen3_5MTPHead", "iter_gguf_mtp_weights", "mtp_draft_layer"]
