"""MTP (``nextn``) speculative decoding, the engine side.

The draft head is one decoder block after the target's last layer (llama.cpp ``graph_mtp``): at
row ``t`` it reads the target's LM-head input there and the embedding of the token at ``t + 1``
and predicts the token at ``t + 2``. It is a separate module beside the target, borrowing its
embedding and LM head, with its own KV layer (layer id ``num_layers``) and, for MoE heads, its own
routed experts in the offload cache.

After every forward the head runs over every row in the same batch context (its KV has to cover
every position, as the target's does) and drafts at each request's last row; on a verify step at
every row, since which row the step stops at is decided afterwards. A draft of depth ``k > 1``
chains the head on its own output for ``k - 1`` more steps before the next verify.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.env import ENV
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.engine.config import EngineConfig
    from freetoken.models.config import ModelConfig
    from freetoken.models.register import ModelSpec

logger = init_logger(__name__)


@dataclass(frozen=True)
class MtpSource:
    """Where a draft head's weights live: GGUF ``path``, block ``blk.<block>``."""

    path: str
    block: int


class MtpAdapter:
    """What a model family supplies for its MTP head; see ``models/qwen3_5_moe/mtp.py``."""

    def locate(self, model_path: str, draft_path: str | None) -> MtpSource:
        raise NotImplementedError

    def build_head(self, config: "ModelConfig", layer_id: int):
        raise NotImplementedError

    def iter_weights(self, source: MtpSource, config: "ModelConfig") -> Iterator[tuple[str, torch.Tensor]]:
        """Dense head tensors keyed relative to the head module (routed experts excluded)."""
        raise NotImplementedError

    def embed_tokens(self, model):
        return model.model.embed_tokens

    def lm_head(self, model):
        return model.lm_head

    def build_token_modules(
        self, model, config, source: MtpSource, *, dtype: torch.dtype
    ):
        """Return the head's embedding and output projection, borrowing target modules by default."""
        return self.embed_tokens(model), self.lm_head(model)

    def iter_token_weights(
        self, source: MtpSource, config: "ModelConfig"
    ) -> Iterator[tuple[str, torch.Tensor]]:
        return iter(())


def with_draft_layer(config: "ModelConfig") -> "ModelConfig":
    """``config`` with one draft block at layer id ``num_layers``, in the full-attention group.

    The checkpoints carry ``attn_q/k/v`` for it whatever a hybrid layer pattern would say for an index
    one past the end. Attributes stashed outside the dataclass fields are carried across."""
    from freetoken.models.config import FullAttentionGroupConfig

    layer_id = config.num_layers
    groups = []
    for group in config.attention_groups or ():
        ids = set(group.layer_ids)
        if isinstance(group, FullAttentionGroupConfig):
            ids.add(layer_id)
        else:
            ids.discard(layer_id)
        groups.append(replace(group, layer_ids=tuple(sorted(ids))))
    if not any(isinstance(g, FullAttentionGroupConfig) for g in groups):
        raise NotImplementedError("an MTP draft block needs a full-attention group to give it a KV layer")
    out = replace(config, attention_groups=tuple(groups), num_draft_layers=1)
    for key, value in config.__dict__.items():
        if key not in out.__dict__:
            object.__setattr__(out, key, value)
    return out


def mtp_adapter(spec: "ModelSpec") -> MtpAdapter:
    from freetoken.models.register import _load_attr

    if spec.mtp is None:
        raise ValueError(f"--speculative-algorithm mtp: {spec.module}.{spec.model_cls} (this checkpoint format) has no MTP draft head support")
    return _load_attr(spec.module, spec.mtp)


def locate_mtp_head(config: "EngineConfig") -> MtpSource:
    return mtp_adapter(config.model_spec).locate(config.model_path, config.speculative_draft_model_path)


def head_quant_config(source: MtpSource, spec: "ModelSpec"):
    """The GGUF quant config over the file that holds the head, so its modules get their own ggml types (a sidecar quantizes differently)."""
    from freetoken.layers.quantization import GGUFConfig, NameMap
    from freetoken.models.gguf.reader import gguf_tensor_index
    from freetoken.models.register import _load_attr

    to_gguf = _load_attr(spec.module, "to_gguf")
    types = {name: info.ggml_type for name, info in gguf_tensor_index(source.path).items()}
    name_map = NameMap(roots=spec.checkpoint_roots, segments=spec.checkpoint_segments, packed=spec.packed_modules_mapping)
    return GGUFConfig(types, to_gguf, name_map=name_map, unquantized=spec.unquantized_modules)


class MtpDrafter:
    """The draft head of a running engine, its hidden store, and the per-step head work."""

    def __init__(self, config: "EngineConfig", model, *, device: torch.device, page_table_ref):
        from freetoken.distributed import get_tp_info
        from freetoken.layers.quantization import finalize_quant
        from freetoken.utils import torch_dtype

        if get_tp_info().size != 1:
            raise NotImplementedError("--speculative-algorithm mtp runs with tensor parallelism 1 only")
        if not config.model_spec.parse_config == "parse_gguf_config":
            raise NotImplementedError("--speculative-algorithm mtp serves the draft heads of GGUF checkpoints only")
        self.adapter = mtp_adapter(config.model_spec)
        self.source = config.mtp_source
        self.model = model
        self.device = device
        self.depth = int(config.speculative_num_draft_tokens)
        mc = config.model_config
        self.layer_id = mc.num_layers
        head_config = replace(mc, quant=head_quant_config(self.source, config.model_spec))
        for key, value in mc.__dict__.items():
            if key not in head_config.__dict__:
                object.__setattr__(head_config, key, value)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.head = self.adapter.build_head(head_config, self.layer_id)
        reference = self.head.state_dict()
        state = {}
        for key, tensor in self.adapter.iter_weights(self.source, mc):
            expected = reference.get(key)
            if expected is None:
                raise ValueError(f"{self.source.path}: draft head tensor {key!r} has no place in the head module")
            state[key] = tensor.to(device=device, dtype=expected.dtype)
        missing = sorted(set(reference) - set(state))
        if missing:
            raise ValueError(f"{self.source.path}: draft head is missing {missing[:8]}")
        num_dense = len(state)
        self.head.load_state_dict(state)
        finalize_quant(self.head)
        self.embed, self.lm_head = self.adapter.build_token_modules(
            model, head_config, self.source, dtype=config.dtype
        )
        token_weights = dict(self.adapter.iter_token_weights(self.source, mc))
        target_embed, target_head = self.adapter.embed_tokens(model), self.adapter.lm_head(model)
        consumed: set[str] = set()
        for module, target, prefix in (
            (self.embed, target_embed, "mtp.embed_tokens"),
            (self.lm_head, target_head, "mtp.shared_head_head"),
        ):
            if module is target:
                continue
            module_weights = {
                key.removeprefix(f"{prefix}."): tensor
                for key, tensor in token_weights.items()
                if key.startswith(f"{prefix}.")
            }
            expected = module.state_dict()
            missing = sorted(set(expected) - set(module_weights))
            unexpected = sorted(set(module_weights) - set(expected))
            if missing or unexpected:
                raise ValueError(
                    f"{self.source.path}: draft {prefix} tensors mismatch; "
                    f"missing={missing[:8]}, unexpected={unexpected[:8]}"
                )
            state = {
                key: tensor.to(device=device, dtype=expected[key].dtype)
                for key, tensor in module_weights.items()
            }
            module.load_state_dict(state)
            finalize_quant(module)
            consumed.update(f"{prefix}.{key}" for key in module_weights)
        extra = sorted(set(token_weights) - consumed)
        if extra:
            raise ValueError(f"{self.source.path}: unexpected MTP token tensors {extra[:8]}")
        self._page_table = page_table_ref
        self.store: torch.Tensor | None = None
        logger.info_rank0(
            f"MTP draft head: block {self.source.block} of {self.source.path} as layer {self.layer_id}, "
            f"{num_dense} dense tensors, up to {self.depth} drafts per step"
        )

    @property
    def has_experts(self) -> bool:
        from freetoken.moe.offload_cache import iter_offload_moe_layers

        return any(True for _ in iter_offload_moe_layers(self.head))

    def alloc_store(self, num_table_rows: int, width: int, dtype: torch.dtype) -> None:
        """One head state per (request slot, verify row): the chained draft's starting point."""
        self.store = torch.zeros(num_table_rows * (self.depth + 1), width, dtype=dtype, device=self.device)

    def store_index(self, table_idx: int, row: int) -> int:
        return table_idx * (self.depth + 1) + row

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """The target's LM head over ``x``, every row (the batch-aware forward gathers prefill rows)."""
        lm = self.lm_head
        if lm.tied_embedding is not None:
            return lm.tied_embedding.quant_method.logits(lm.tied_embedding, x, lm.bias)
        return lm.quant_method.apply(lm, x)

    def _draft(self, out: torch.Tensor, with_conf: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The head's greedy draft at each row and, when asked, its softmax probability at temperature 1."""
        logits = self.logits(out)
        token = logits.argmax(dim=-1).to(torch.int32)
        if not with_conf:
            return token, None
        logits = logits.float()
        return token, torch.exp(logits.max(dim=-1).values - torch.logsumexp(logits, dim=-1))

    def _head(self, hidden: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(LM-head input, state for the next chained step) at every row."""
        embed = self.embed.forward(tokens.to(torch.int64))
        forward_chain = getattr(self.head, "forward_chain", None)
        if forward_chain is not None:
            return forward_chain(hidden, embed)
        out = self.head.forward(hidden, embed)
        return out, out

    def head_pass(self, batch: "Batch", sampled: torch.Tensor) -> torch.Tensor:
        """Run the head over this forward's rows (inside its batch context) and draft at the plan's rows."""
        plan = batch.mtp_plan
        hidden = batch.hidden_states
        assert plan is not None and hidden is not None, "MTP head pass without a plan or captured hidden states"
        rows = hidden.shape[0]
        if plan.next_ids is None:
            nxt = sampled[:rows]
        else:
            nxt = plan.next_ids.to(sampled.dtype)
            sampled_rows = plan.sampled_rows
            if sampled_rows is None:
                sampled_rows = torch.arange(rows, device=nxt.device)
            sampled_tokens = sampled[: sampled_rows.numel()]
            if plan.sampled_src_rows is not None:
                sampled_tokens = sampled.index_select(0, plan.sampled_src_rows)
            nxt.index_copy_(0, sampled_rows, sampled_tokens)
        out, state = self._head(hidden, nxt)
        if plan.draft_rows is not None:
            out, state = out.index_select(0, plan.draft_rows), state.index_select(0, plan.draft_rows)
        if self.store is None:
            self.alloc_store(self._page_table().shape[0], state.shape[-1], state.dtype)
        self.store.index_copy_(0, plan.store_idx, state.to(self.store.dtype))
        token, batch.draft_conf = self._draft(out, ENV.SPEC_DRAFT_MIN_P.value is not None)
        return token

    def draft_chain(self, batch: "Batch", token_pool: torch.Tensor, ctx, attn_backend, mrope: bool) -> torch.Tensor | None:
        """Drafts ``2..k`` of a verify step, chained from each request's stored head state.

        Step ``j`` runs the head at position ``cached_len + j - 2`` on the previous step's state and the
        previous draft, writing a provisional head-KV row there that the verify's own head pass
        overwrites. The drafts go into the token pool where the verify reads them; returns ``[B, k - 1]``.

        With FREETOKEN_SPEC_DRAFT_MIN_P the chain stops once no request would verify a further draft,
        and returns ``[B, depth - 1]`` for the deepest request's depth (None if that is 1)."""
        from freetoken.core import Batch, VerifyRow
        from freetoken.speculative.accept import batch_draft_depth

        spec = batch.spec
        assert spec is not None and self.store is not None
        if spec.k < 2:
            return None
        reqs = spec.reqs
        dev = self.device

        def tensor(xs, dtype=torch.int64):
            return torch.tensor(xs, dtype=dtype, pin_memory=True).to(dev, non_blocking=True)

        tbl = tensor([r.table_idx for r in reqs])
        state = self.store.index_select(0, tensor([self.store_index(r.table_idx, r.pending_slot) for r in reqs]))
        token = tensor(spec.first_drafts, torch.int32)
        page_table = self._page_table()
        drafts = []
        min_p = ENV.SPEC_DRAFT_MIN_P.value
        confs = [[c] for c in (spec.first_conf or [1.0] * len(reqs))] if min_p is not None else None
        for j in range(2, spec.k + 1):
            if min_p is not None and batch_draft_depth(confs, min_p, spec.k) < j:
                break
            rows = [VerifyRow(r, j - 2) for r in reqs]
            step = Batch(reqs=rows, phase="decode")
            step.padded_reqs = rows
            pos = [r.cached_len for r in rows]
            step.positions = tensor(pos, torch.int32)
            if mrope:
                from freetoken.scheduler.scheduler import _make_mrope_positions

                step.mrope_positions = _make_mrope_positions(step, dev)
            pos64 = step.positions.to(torch.int64)
            step.out_loc = page_table[tbl, pos64]
            step.input_ids = token
            attn_backend.prepare_metadata(step)
            with ctx.forward_batch(step):
                out, state = self._head(state, token)
                token, conf = self._draft(out, min_p is not None)
            token_pool[tbl, pos64 + 2] = token
            drafts.append(token)
            if min_p is not None:
                for row, c in zip(confs, conf.tolist()):
                    row.append(c)
        return torch.stack(drafts, dim=1) if drafts else None


__all__ = ["MtpAdapter", "MtpDrafter", "MtpSource", "head_quant_config", "locate_mtp_head", "mtp_adapter", "with_draft_layer"]
