"""Qwen3.8-Flash-Next MTP (``nextn``) draft head.

One trunk-shaped block that predicts the token after next. Unlike the Qwen3.5/3.6 head,
which is conditioned on a plain hidden state, this one folds the next token's embedding
into the trunk's *wide hyper-connection residual* and stays there::

    h  = hnorm(R)                          # R [T, hc*hidden], the trunk's wide residual
    e  = enorm(embed_tokens[x])            # [T, hidden], broadcast to every stream
    R' = eh_proj(concat(e, h))             # per stream: fc_embedding @ e + fc_hidden @ h
    R' = decoder_block(R')                 # a trunk block: dense attention + MoE, HC-wrapped
    out = hyper_connection_mixer(R')       # the head's OWN mixer, which is also its output norm
                                           # -> the trunk's lm_head

Follows llama.cpp PR #27836 (``qwen4exp : add NextN/MTP draft head``) rather than
reinterpreting it. Three things that graph settles and shapes alone do not:

- ``eh_proj`` is applied to each hyper-connection stream separately, with ``e`` repeated
  across them. Pooling the streams first would discard what the residual is for.
- The mixer is on the OUTPUT. qwen4exp has no final norm anywhere, and this stands in for
  one, mirroring the trunk's own ``output_hc_*``.
- ``e`` comes first in the concat: the converter fuses ``fc_embedding`` and ``fc_hidden``
  in that order, since ``W_e @ e + W_h @ h == [W_e|W_h] @ concat(e, h)``.

The block runs QSA like the trunk's full-attention layers, using the indexer weights the
checkpoint ships for it. llama.cpp's v1 attends densely instead -- the converter writes a
compress ratio of 0 for the trailing blocks -- on the grounds that dense is a numerical
superset of a 2048-token prune and the target verifies the drafts either way. The HF head
carries ``self_attn.indexer.*`` of its own, so using it is the closer match; dense stays
available by giving the block its own attention group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.layers import BaseOP
from freetoken.layers.linear import LinearReplicated

from .hc import GatedResidual, GroupedPlusOneRMSNorm

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig

MTP_DRAFT_LAYER_KV = "mtp.draft_layer"


class Qwen4ExpMTPHead(BaseOP):
    """The ``nextn`` draft block. ``layer_id`` is its slot in the attention backend and KV
    pool -- one past the trunk's last layer, which is also how the checkpoint names it."""

    def __init__(self, config: "ModelConfig", layer_id: int, *, prefix: str = "mtp") -> None:
        from .model import Qwen4ExpDecoderLayer

        args = config.qwen4_args
        self.layer_id = layer_id
        self.hc_count = args.hc_count
        self.hidden_size = config.hidden_size
        # Both are plus-one norms like every other norm in this architecture; enorm spans
        # the hidden width and hnorm the wide residual, one group per stream.
        self.enorm = GroupedPlusOneRMSNorm(config.hidden_size, config.rms_norm_eps, 1)
        self.hnorm = GroupedPlusOneRMSNorm(
            args.ple_state_width, config.rms_norm_eps, args.hc_count
        )
        self.eh_proj = LinearReplicated(
            2 * config.hidden_size, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.eh_proj",
        )
        self.layer = Qwen4ExpDecoderLayer(config, layer_id, prefix=f"{prefix}.layer")
        assert not self.layer._is_linear, (
            f"MTP block {layer_id} was built as linear attention; the nextn checkpoints "
            "carry attn_q/k/v for it, so the config's layer pattern disagrees"
        )
        assert self.layer.ple is None, f"MTP block {layer_id} must not carry a PLE layer"
        # use_combine=False: a mixer, not a residual. Same shape as the trunk's final one.
        self.hyper_connection_mixer = GatedResidual(
            config, use_combine=False, prefix=f"{prefix}.hyper_connection_mixer"
        )

    def forward(
        self, hidden: torch.Tensor, token_embed: torch.Tensor, batch: "Batch"
    ) -> torch.Tensor:
        """``hidden`` [T, hc*hidden] is the trunk's wide residual at t, ``token_embed``
        [T, hidden] the embedding of the token at t+1; returns [T, hidden] for the
        trunk's lm_head."""
        tokens = hidden.shape[0]
        h = self.hnorm.forward(hidden).view(tokens, self.hc_count, self.hidden_size)
        e = self.enorm.forward(token_embed).unsqueeze(1).expand(-1, self.hc_count, -1)
        stream = self.eh_proj.forward(torch.cat([e, h], dim=-1))
        out = self.layer.forward(stream.reshape(tokens, -1), batch)
        return self.hyper_connection_mixer.mix(out)[0]


__all__ = ["Qwen4ExpMTPHead", "MTP_DRAFT_LAYER_KV"]


# --------------------------------------------------------------------------------------
# Weights.
# --------------------------------------------------------------------------------------

# Head-level tensors, on top of the decoder-block suffixes the trunk's reader knows.
_NEXTN_MAP = {
    "nextn.eh_proj.weight": "eh_proj.weight",
    "nextn.hc_head_down.weight": "hyper_connection_mixer.input_mix_weight_down.weight",
    "nextn.hc_head_up.weight": "hyper_connection_mixer.input_mix_weight_up.weight",
}
# Stored as (1 + w), like every other norm this architecture writes.
_NEXTN_PLUS_ONE = {
    "nextn.enorm.weight": "enorm.weight",
    "nextn.hnorm.weight": "hnorm.weight",
    "nextn.hc_head_norm.weight": "hyper_connection_mixer.hc_norm.weight",
}
# Present only when the head does NOT borrow them; qwen4exp sets
# mtp_use_dedicated_embeddings=false, so a shared export carries neither.
_BORROWED = ("nextn.embed_tokens.weight", "nextn.shared_head_head.weight")


def mtp_draft_layer(model_path: str) -> int | None:
    """The block index a checkpoint's draft head occupies, or ``None`` if it has none."""
    from freetoken.models.gguf.reader import gguf_architecture, load_gguf_metadata

    md = load_gguf_metadata(model_path)
    arch = gguf_architecture(model_path)
    n_nextn = md.get(f"{arch}.nextn_predict_layers")
    n_layer = md.get(f"{arch}.block_count")
    if not n_nextn or not n_layer:
        return None
    return int(n_layer) - int(n_nextn)


def iter_gguf_mtp_weights(model_path: str, prefix: str = "mtp"):
    """Yield ``(name, tensor)`` for the draft head, in this module's naming.

    The block's ordinary tensors ride the trunk's own translation -- the converter renames
    ``mtp.layers.0.*`` onto the trailing block index precisely so they can -- which keeps
    the two in agreement by construction rather than by a second copy of the table. Only
    the head-level tensors above are specific to the head.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    from .gguf import (
        _MERGED,
        _PLUS_ONE_MAP,
        _ROW_CONCAT,
        _SUFFIX_MAP,
        _pad_rows,
        _to_bf16,
    )

    layer = mtp_draft_layer(model_path)
    if layer is None:
        raise ValueError(f"{model_path}: no MTP draft block (no nextn metadata)")

    stem = f"blk.{layer}."
    buf: dict[str, dict[str, torch.Tensor]] = {}
    seen: set[str] = set()

    for t in iter_gguf_tensors(model_path):
        if not t.name.startswith(stem):
            continue
        suffix = t.name[len(stem):]
        seen.add(suffix)
        if suffix in _BORROWED:
            continue                       # the head reads the trunk's copy
        if suffix in _NEXTN_MAP:
            yield f"{prefix}.{_NEXTN_MAP[suffix]}", _to_bf16(t)
            continue
        if suffix in _NEXTN_PLUS_ONE:
            yield f"{prefix}.{_NEXTN_PLUS_ONE[suffix]}", _to_bf16(t) - 1.0
            continue
        if suffix == "ffn_gate_inp_shexp.weight":
            # stored flat; the module holds it as a 1-row linear
            yield f"{prefix}.layer.mlp.shared_expert_gate.weight", _to_bf16(t).reshape(1, -1)
            continue
        if "_exps." in suffix:
            continue                       # routed experts: the bank reader's job
        if suffix in _PLUS_ONE_MAP:
            yield f"{prefix}.layer.{_PLUS_ONE_MAP[suffix]}", _to_bf16(t) - 1.0
            continue
        rel = _SUFFIX_MAP.get(suffix)
        if rel is not None:
            yield f"{prefix}.layer.{rel}", _to_bf16(t)
            continue
        done = False
        for module, parts in _MERGED.items():
            if suffix not in parts:
                continue
            group = buf.setdefault(module, {})
            group[suffix] = _to_bf16(t)
            if len(group) == len(parts):
                yield (f"{prefix}.layer.{module}.weight",
                       torch.cat([group.pop(s) for s in parts], dim=0).contiguous())
            done = True
            break
        if done:
            continue
        for module, (parts, align) in _ROW_CONCAT.items():
            if suffix not in parts:
                continue
            group = buf.setdefault(module, {})
            group[suffix] = _to_bf16(t)
            if len(group) == len(parts):
                rows = [group.pop(s) for s in parts]
                pad = _pad_rows(sum(r.shape[0] for r in rows), align)
                if pad:
                    rows.append(rows[0].new_zeros(pad, rows[0].shape[1]))
                yield (f"{prefix}.layer.{module}.weight",
                       torch.cat(rows, dim=0).contiguous())
            done = True
            break
        if not done:
            raise ValueError(f"{t.name}: unrecognized MTP draft tensor")

    if not seen:
        raise ValueError(f"{model_path}: no tensors under {stem}")


__all__ = [
    "Qwen4ExpMTPHead",
    "MTP_DRAFT_LAYER_KV",
    "iter_gguf_mtp_weights",
    "mtp_draft_layer",
]
