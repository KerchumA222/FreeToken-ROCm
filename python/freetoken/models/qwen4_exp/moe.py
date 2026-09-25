from __future__ import annotations

from typing import TYPE_CHECKING

import os

import torch
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


_ROUTE_TRACE = os.environ.get("FT_ROUTE_TRACE", "")
_route_rows: list = []


def _record_route(moe, x: torch.Tensor, logits: torch.Tensor) -> None:
    """FT_ROUTE_TRACE=<path.pt>: record each decode step's router input and routed ids per
    layer (eager only; it syncs), for offline expert-prediction studies. Written every
    4096 records."""
    from freetoken.core import get_global_ctx

    batch = get_global_ctx().batch
    if not batch.is_decode or torch.cuda.is_current_stream_capturing():
        return
    ids = logits.float().softmax(-1).topk(moe.experts.top_k, dim=-1).indices
    _route_rows.append((moe.experts.layer_id, x.detach().to("cpu", torch.float16), ids.cpu()))
    if len(_route_rows) % 4096 == 0:
        torch.save(_route_rows, _ROUTE_TRACE)


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.
    """

    # (target layer id, that layer's MoE block, predictions) -- see model._wire_expert_lookahead
    _lookahead = None

    def _stage_lookahead(self, x: torch.Tensor) -> None:
        """Predict the experts ``d`` layers ahead from this layer's router input and stage
        the ones not already in the GPU cache for a disk prefetch. Single-row decode only:
        the staging buffer's size is fixed per captured graph."""
        from freetoken.core import get_global_ctx
        from freetoken.kernel.triton.small_gemv import small_gemv

        cache = getattr(self.experts, "offload_cache", None)
        if cache is None or cache.host_tier is None or x.shape[0] != 1:
            return
        if not get_global_ctx().batch.is_decode:
            return
        target, block, k = self._lookahead.target, self._lookahead.block, self._lookahead.k
        logits = small_gemv(x.contiguous(), block.gate.weight, block_n=1, block_k=2048, num_warps=2)
        top = logits.float().softmax(-1).topk(k, dim=-1)
        ids = top.indices.reshape(-1)
        skip = cache.slot_for_id[target].index_select(0, ids) >= 0      # already on the GPU
        if self._lookahead.p_min > 0:
            skip |= top.values.reshape(-1) < self._lookahead.p_min      # unlikely guesses
        cache.stage_prefetch(self.experts.layer_id, target,
                             torch.where(skip, torch.full_like(ids, -1), ids).to(torch.int32))

    def _router(self, x: torch.Tensor) -> torch.Tensor:
        """Router logits. At decode sizes on RDNA, rocBLAS runs this [512, H] GEMV as a
        large Tensile tile (~85 us a layer on an RX 6800); the Triton small-M GEMV reads the
        weight once in a few us."""
        from freetoken.kernel.triton.small_gemv import MAX_ROWS, small_gemv

        w = self.gate.weight
        if torch.version.hip is None or x.shape[0] > MAX_ROWS or w.dtype != x.dtype:
            return self.gate.forward(x)
        return small_gemv(x.contiguous(), w, block_n=1, block_k=2048, num_warps=2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        # Already [T, H]: no view, which would drop q8_1 blocks attached upstream.
        if hidden_states.dim() != 2:
            hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self._router(hidden_states)
        if _ROUTE_TRACE:
            _record_route(self, hidden_states, router_logits)
        if self._lookahead is not None:
            self._stage_lookahead(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)


__all__ = ["Qwen4ExpMoE"]
