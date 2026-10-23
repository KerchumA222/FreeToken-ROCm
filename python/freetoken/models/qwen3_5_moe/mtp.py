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
    """``config`` with ``layer_id`` moved into the full-attention group (the draft block's KV layer).

    The draft block sits one past the target's last layer, where the hybrid pattern
    would place a linear-attention layer; the checkpoints carry ``attn_q/k/v`` for it.
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
    # replace() keeps only declared fields; carry the rest across
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

    def forward_chain(self, hidden: torch.Tensor, token_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The head output twice: a chained draft step takes the LM-head input as its next hidden state."""
        out = self.forward(hidden, token_embed)
        return out, out

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
    """The block index a GGUF's draft head occupies (its ``nextn`` tensors), or ``None`` if it has none."""
    from freetoken.models.gguf.reader import gguf_tensor_index

    from .gguf import nextn_block

    return nextn_block(gguf_tensor_index(model_path))


def iter_gguf_mtp_weights(model_path: str, prefix: str = "mtp", config: "ModelConfig | None" = None):
    """Yield ``(name, tensor)`` for the draft head, in this module's naming (see ``gguf.iter_gguf_mtp_weights``)."""
    from .gguf import iter_gguf_mtp_weights as _iter

    yield from _iter(model_path, prefix, config)


class Qwen3_5MTP:
    """``--speculative-algorithm mtp`` for qwen35 / qwen35moe GGUFs: the nextn block of the checkpoint
    itself, or of a draft-head-only GGUF (``mtp-*.gguf``, ``<arch>.mtp.draft_only``) beside it."""

    def draft_layer(self, path: str) -> int | None:
        return mtp_draft_layer(path)

    def locate(self, model_path: str, draft_path: str | None):
        from freetoken.models.gguf.reader import gguf_architecture, gguf_config_source
        from freetoken.speculative.mtp import MtpSource

        model_source = gguf_config_source(model_path) or model_path
        path = (
            (gguf_config_source(draft_path) or draft_path)
            if draft_path
            else model_source
        )
        if draft_path is not None:
            model_arch = gguf_architecture(model_source)
            draft_arch = gguf_architecture(path)
            if draft_arch != model_arch:
                raise ValueError(
                    f"draft head {draft_path} is {draft_arch!r}, the checkpoint {model_arch!r}"
                )
        block = self.draft_layer(path)
        if block is None:
            hint = "" if draft_path else "; pass the draft head's GGUF with --speculative-draft-model-path"
            raise ValueError(f"{path} carries no nextn (MTP) block{hint}")
        return MtpSource(path=path, block=block)

    def build_head(self, config: "ModelConfig", layer_id: int):
        return Qwen3_5MTPHead(config, layer_id)

    def iter_weights(self, source, config: "ModelConfig"):
        for name, tensor in iter_gguf_mtp_weights(source.path, "mtp", config):
            yield name.removeprefix("mtp."), tensor

    def embed_tokens(self, model):
        return model.model.embed_tokens

    def lm_head(self, model):
        return model.lm_head

    def build_token_modules(self, model, config: "ModelConfig", source, *, dtype):
        from freetoken.layers import ParallelLMHead, VocabParallelEmbedding
        from freetoken.models.gguf.reader import gguf_tensor_index
        from freetoken.utils import torch_dtype

        index = gguf_tensor_index(source.path)
        prefix = f"blk.{source.block}.nextn."
        build_embed = f"{prefix}embed_tokens.weight" in index
        build_head = f"{prefix}shared_head_head.weight" in index
        embed, lm_head = self.embed_tokens(model), self.lm_head(model)
        if not (build_embed or build_head):
            return embed, lm_head

        with torch.device("meta"), torch_dtype(dtype):
            if build_embed:
                embed = VocabParallelEmbedding(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=config.quant,
                    prefix="mtp.embed_tokens",
                )
            if build_head:
                lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    tie_word_embeddings=False,
                    quant_config=config.quant,
                    prefix="mtp.shared_head_head",
                )
        return embed, lm_head

    def iter_token_weights(self, source, config: "ModelConfig"):
        from .gguf import iter_gguf_mtp_token_weights

        yield from iter_gguf_mtp_token_weights(source.path, "mtp", config)


QWEN3_5_MTP = Qwen3_5MTP()


__all__ = ["QWEN3_5_MTP", "Qwen3_5MTPHead", "iter_gguf_mtp_weights", "mtp_draft_layer"]
