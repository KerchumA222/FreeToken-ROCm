"""Qwen3.8-Flash-Next MTP (``nextn``) draft head.

One trunk-shaped block that predicts the token after next. Where the Qwen3.5/3.6 head reads a plain hidden
state, this one is conditioned on the trunk's wide residual ``R`` ([hc_count*hidden], before the final
mixer) and stays on hyper-connection streams until its own mixer folds them for the LM head::

    h = hnorm(R)                       # per stream, (1 + w) over each hidden slice
    e = enorm(embed_tokens[x])         # [hidden], repeated across the streams
    S = eh_proj(concat(e, h_s))        # per stream s, e first
    S = trunk block(S)                 # QSA attention + MoE, each hyper-connection wrapped
    out = hc_head mixer(S)             # the trunk's final mixer, with the head's own weights
    logits = lm_head(out)              # the target's

Follows llama.cpp ``llama_model_qwen4exp::graph_mtp``. A chained draft step feeds ``S`` (before the mixer) back as
the next ``R``, which is what llama.cpp's ``h_nextn`` of the draft graph is.

llama.cpp writes compress ratio 0 for the draft block, so its attention is dense; this head runs QSA like the
trunk's full-attention layers, with the indexer the sidecar ships. The two agree while a request sees at most
``index_budget + index_ratio - 1`` tokens (every complete block is selected) and the target verifies every draft.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP
from freetoken.layers.linear import LinearReplicated
from freetoken.models.qwen3_5_moe.mtp import Qwen3_5MTP

from .hc import GatedResidual, GroupedPlusOneRMSNorm

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpMTPHead(BaseOP):
    """The ``nextn`` draft block. ``layer_id`` is its slot in the attention backend and KV pool: one past the
    trunk's last layer, which is also how the sidecar names it (``blk.<n_layer>.*``)."""

    def __init__(self, config: "ModelConfig", layer_id: int, *, prefix: str = "mtp") -> None:
        from .model import Qwen4ExpDecoderLayer

        args = config.qwen4_args
        self.layer_id = layer_id
        self.hc_count = args.hc_count
        self.hidden_size = config.hidden_size
        self.enorm = GroupedPlusOneRMSNorm(config.hidden_size, config.rms_norm_eps, 1)
        self.hnorm = GroupedPlusOneRMSNorm(args.ple_state_width, config.rms_norm_eps, args.hc_count)
        self.eh_proj = LinearReplicated(
            2 * config.hidden_size, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.eh_proj",
        )
        self.layer = Qwen4ExpDecoderLayer(config, layer_id, prefix=f"{prefix}.layer")
        # the sidecar carries attn_q/k/v and the indexer for the block, whatever the layer pattern says past the end
        assert not self.layer._is_linear, f"MTP block {layer_id} was built as linear attention; the config's layer pattern disagrees with the weights"
        assert self.layer.ple is None, f"MTP block {layer_id} must not carry a PLE layer"
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False, prefix=f"{prefix}.hyper_connection_mixer")

    def forward_chain(self, hidden: torch.Tensor, token_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``hidden`` [T, hc*hidden] is the trunk's wide residual at t (or the previous draft step's block
        output), ``token_embed`` [T, hidden] the embedding of the token at t+1. Returns the LM-head input
        [T, hidden] and the block output [T, hc*hidden], the next chained step's ``hidden``."""
        tokens = hidden.shape[0]
        h = self.hnorm.forward(hidden).view(tokens, self.hc_count, self.hidden_size)
        e = self.enorm.forward(token_embed).unsqueeze(1).expand(-1, self.hc_count, -1)
        streams = self.eh_proj.forward(torch.cat([e, h], dim=-1).view(tokens * self.hc_count, -1))
        out = self.layer.forward(streams.view(tokens, -1), get_global_ctx().batch)
        return self.hyper_connection_mixer.mix(out)[0], out

    def forward(self, hidden: torch.Tensor, token_embed: torch.Tensor) -> torch.Tensor:
        return self.forward_chain(hidden, token_embed)[0]


# ----------------------------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------------------------

_BORROWED = ("embed_tokens", "shared_head_head")


def mtp_draft_layer(model_path: str) -> int | None:
    """The block index a GGUF's draft head occupies (its ``nextn`` tensors), or ``None`` if it has none."""
    from freetoken.models.gguf.reader import gguf_tensor_index
    from freetoken.models.qwen3_5_moe.gguf import nextn_block

    return nextn_block(gguf_tensor_index(model_path))


def check_head_metadata(path: str, md: dict) -> None:
    """The sidecar must hold one draft block that borrows the target's embedding and LM head."""
    n_nextn = int(md.get("qwen4exp.nextn_predict_layers", 0))
    if n_nextn != 1:
        raise NotImplementedError(f"{path}: qwen4exp MTP has one draft block, the GGUF declares {n_nextn}")
    if not md.get("qwen4exp.nextn_shared_target_tensors", False):
        raise NotImplementedError(f"{path}: a draft head with its own embedding and LM head is not supported")


def iter_gguf_mtp_weights(model_path: str, config: "ModelConfig", prefix: str = "mtp") -> Iterator[tuple[str, torch.Tensor]]:
    """The draft block's dense tensors under ``prefix`` (``Qwen4ExpMTPHead``'s naming), packed like the trunk's.

    ``config`` is the target's: the sidecar carries no PLE or vision keys. The block itself is read by the trunk's
    own layer reader, so the two cannot drift. The routed experts load with ``load_gguf_expert_banks(path, 1, first_layer=<block>)``."""
    from freetoken.models.gguf.reader import gguf_tensor_index
    from freetoken.models.gguf.tensors import float_tensor, fused, matrix

    from .gguf import _iter_layer, _reader

    index = gguf_tensor_index(model_path)
    n = mtp_draft_layer(model_path)
    if n is None:
        raise ValueError(f"{model_path}: no nextn (MTP draft) block")
    t, norm, _ = _reader(index, config)
    blk = f"blk.{n}"
    hidden, args = config.hidden_size, config.qwen4_args
    for leaf in _BORROWED:
        if f"{blk}.nextn.{leaf}.weight" in index:
            raise NotImplementedError(f"{model_path}: nextn.{leaf} is not supported, the head borrows the target's")
    yield f"{prefix}.enorm.weight", norm(f"{blk}.nextn.enorm")
    yield f"{prefix}.hnorm.weight", norm(f"{blk}.nextn.hnorm")
    eh = t(f"{blk}.nextn.eh_proj")
    yield from fused(f"{prefix}.eh_proj", [eh], [matrix(eh)], 2 * hidden)
    for proj, k in (("down", hidden * args.hc_count), ("up", args.hc_lowrank)):
        info = t(f"{blk}.nextn.hc_head_{proj}")
        yield from fused(f"{prefix}.hyper_connection_mixer.input_mix_weight_{proj}", [info], [matrix(info)], k)
    yield f"{prefix}.hyper_connection_mixer.hc_norm.weight", norm(f"{blk}.nextn.hc_head_norm")
    yield from _iter_layer(index, config, None, n, f"{prefix}.layer", linear=False)


class Qwen4ExpMTP(Qwen3_5MTP):
    """``--speculative-algorithm mtp`` for qwen4exp GGUFs: the draft block of a sidecar GGUF (``mtp-*.gguf``,
    ``nextn_shared_target_tensors``) beside the target."""

    def draft_layer(self, path: str) -> int | None:
        return mtp_draft_layer(path)

    def locate(self, model_path: str, draft_path: str | None):
        from freetoken.models.gguf.reader import load_gguf_metadata

        source = super().locate(model_path, draft_path)
        check_head_metadata(source.path, load_gguf_metadata(source.path))
        return source

    def head_config(self, config: "ModelConfig", source):
        """The head keeps its own MoE geometry: a sidecar cut from the base model has 512 experts, a pruned fine-tune beside it fewer."""
        from freetoken.models.gguf.reader import load_gguf_metadata
        from freetoken.speculative.mtp import replace_config

        md = load_gguf_metadata(source.path)
        kv = lambda key: int(md[f"qwen4exp.{key}"])
        if kv("embedding_length") != config.hidden_size:
            raise ValueError(f"{source.path}: draft head is {kv('embedding_length')} wide, the checkpoint {config.hidden_size}")
        return replace_config(
            config,
            num_experts=kv("expert_count"),
            num_experts_per_tok=kv("expert_used_count"),
            moe_intermediate_size=kv("expert_feed_forward_length"),
            shared_expert_intermediate_size=kv("expert_shared_feed_forward_length"),
        )

    def build_head(self, config: "ModelConfig", layer_id: int):
        return Qwen4ExpMTPHead(config, layer_id)

    def iter_weights(self, source, config: "ModelConfig"):
        for name, tensor in iter_gguf_mtp_weights(source.path, config):
            yield name.removeprefix("mtp."), tensor

    def build_token_modules(self, model, config: "ModelConfig", source, *, dtype):
        return self.embed_tokens(model), self.lm_head(model)

    def iter_token_weights(self, source, config: "ModelConfig"):
        return iter(())


QWEN4_EXP_MTP = Qwen4ExpMTP()


__all__ = ["QWEN4_EXP_MTP", "Qwen4ExpMTP", "Qwen4ExpMTPHead", "check_head_metadata", "iter_gguf_mtp_weights", "mtp_draft_layer"]
