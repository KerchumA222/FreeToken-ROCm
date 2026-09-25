"""Grouped expert GEMM over native GGUF quant banks (borrowed ggml MoE kernels).

Ports vLLM/sglang's ``_fused_moe_gguf`` MMVQ path onto FreeToken's offload-cache
interface: the experts are streamed to the GPU as packed GGUF block bytes and
dequantized *inside* ``ggml_moe_a8_vec`` -- no bf16 expert copy is materialized. We
use the MMVQ (vector) kernel for both prefill and decode: it consumes ``topk_ids``
directly (no ``moe_align_block_size`` needed) and on small batches it is the right
choice anyway. ``topk_ids`` already index the streamed cache slots (decode) or the
materialized layer positions (prefill).

Any ggml type the vendored MMVQ kernel covers is accepted (classic Q4_0..Q8_0, all
K-quants, all IQ types -- see the case labels in ``csrc/gguf/gguf_kernel.cu``);
gate_up and down may use different types (Q4_K_M mixes Q4_K/Q6_K). The historical
``fused_experts_gguf_q4_0`` name is kept as the Q4_0-typed wrapper.
"""

from __future__ import annotations

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGML_Q4_0

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, row_bytes(H, gate_up_type)] uint8
    down_q: torch.Tensor,  # [num_slots, H, row_bytes(I, down_type)] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    *,
    gate_up_type: int = GGML_Q4_0,
    down_type: int = GGML_Q4_0,
) -> torch.Tensor:
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    # Grouped GEMMs pay per routed expert (a dequantize and two launches), so they need a
    # few rows per expert to beat the vector kernel: ~12 each (a 630-token prefill over 512
    # experts) was 20% slower, ~50 each 10% faster, ~150 each 70% faster.
    per_expert = num_tokens * topk_ids.shape[1] / gate_up_q.shape[0]
    if (num_tokens > _GROUPED_MIN_TOKENS and per_expert >= _GROUPED_MIN_PER_EXPERT
            and not torch.cuda.is_current_stream_capturing()):
        return _fused_experts_grouped(
            hidden_states, gate_up_q, down_q, topk_weights, topk_ids, act_fn,
            int(gate_up_type), int(down_type),
        )
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]

    # gate_up: [num_tokens*top_k, 2I] -> activation -> [num_tokens*top_k, I]
    gate_up = ggml_moe_a8_vec(
        hidden_states, gate_up_q, topk_ids, top_k, int(gate_up_type), n2, num_tokens
    )
    inter = act_fn(gate_up)
    # down: each of the num_tokens*top_k intermediate rows uses its own expert id.
    out = ggml_moe_a8_vec(
        inter, down_q, topk_ids, 1, int(down_type), h, num_tokens * top_k
    )
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


# Above this many tokens (prefill) the experts run as dequantized batched GEMMs instead of
# the per-(token, expert) vector kernel, which was 27% of a 2k-token Qwen3.8-Flash-Next
# prefill (48 launches of ~50 ms each).
_GROUPED_MIN_TOKENS = 64
_GROUPED_MIN_PER_EXPERT = 16
# A group is at most _GROUP experts (a Flash-Next expert is ~6.5 MB gate_up + 3.3 MB down
# as fp16) and _GROUP_ROWS padded token rows, so the path's peak stays ~150 MB whatever
# the chunk: the engine sizes the prefill chunk from a probe that routes to few experts.
_GROUP = 8
_GROUP_ROWS = 8192


def _dequant_rows(q: torch.Tensor, ggml_type: int, dtype: torch.dtype) -> torch.Tensor:
    """``q`` [n, rows, row_bytes] packed -> [n, rows, cols] ``dtype``."""
    from freetoken.models.gguf.dequant import BLOCK_SHAPE, dequantize

    n, rows, rb = q.shape
    block, size = BLOCK_SHAPE[ggml_type]
    cols = rb // size * block
    flat = q.reshape(n * rows, rb)
    from freetoken.layers.gguf import is_mmvq_type
    from freetoken.models.gguf.dequant import GGML_Q2_0, GGML_Q2_0_SYM

    if is_mmvq_type(ggml_type):
        from freetoken.kernel.gguf import ggml_dequantize

        return ggml_dequantize(flat, ggml_type, n * rows, cols, dtype).view(n, rows, cols)
    if ggml_type in (GGML_Q2_0, GGML_Q2_0_SYM) and flat.is_cuda:
        from freetoken.kernel.triton.q2_0_dequant import dequant_q2_0

        return dequant_q2_0(flat, dtype, ggml_type == GGML_Q2_0_SYM).view(n, rows, cols)
    return dequantize(flat.reshape(-1), ggml_type, dtype).view(n, rows, cols)


def _fused_experts_grouped(x, gate_up_q, down_q, topk_weights, topk_ids, act_fn, gu_type, dn_type):
    """Prefill MoE: route (token, expert) pairs by expert, and for groups of experts of
    similar load dequantize their weights and run two padded batched GEMMs. Each group
    gathers its own rows from ``x`` and adds its weighted outputs into one fp32 [T, H]
    accumulator, so nothing is sized tokens x top_k x hidden."""
    t, k = topk_ids.shape
    h = x.shape[1]
    flat = topk_ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    experts, counts = torch.unique_consecutive(flat.index_select(0, order), return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    # Busiest first, so each group pads to a similar count.
    by_load = torch.argsort(counts, descending=True)
    experts, counts, starts = experts[by_load], counts[by_load], starts[by_load]
    counts_host = counts.tolist()
    tok_sorted = order // k
    w_sorted = topk_weights.reshape(-1).index_select(0, order).float()
    out = torch.zeros(t, h, dtype=torch.float32, device=x.device)
    g = 0
    while g < len(counts_host):
        maxc = counts_host[g]  # the busiest of the group
        n = max(1, min(_GROUP, _GROUP_ROWS // maxc, len(counts_host) - g))
        ids = experts[g : g + n]
        span = torch.arange(maxc, device=x.device)
        pos = starts[g : g + n, None] + span[None, :]
        valid = span[None, :] < counts[g : g + n, None]
        pos = torch.where(valid, pos, torch.zeros_like(pos))  # padding rows compute on row 0
        tok = tok_sorted.index_select(0, pos.reshape(-1))
        xs = x.index_select(0, tok).view(n, maxc, h)
        w_gu = _dequant_rows(gate_up_q.index_select(0, ids), gu_type, x.dtype)
        inter = act_fn(torch.bmm(xs, w_gu.transpose(1, 2)).view(n * maxc, -1))
        del xs, w_gu
        w_dn = _dequant_rows(down_q.index_select(0, ids), dn_type, x.dtype)
        y = torch.bmm(inter.view(n, maxc, -1), w_dn.transpose(1, 2)).view(n * maxc, h)
        del inter, w_dn
        keep = valid.reshape(-1)
        weight = w_sorted.index_select(0, pos.reshape(-1)) * keep
        out.index_add_(0, tok, y.float() * weight[:, None])
        g += n
    return out.to(x.dtype)


def fused_experts_gguf_q4_0(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    return fused_experts_gguf(
        hidden_states, gate_up_q, down_q, topk_weights, topk_ids, activation
    )


__all__ = ["fused_experts_gguf", "fused_experts_gguf_q4_0"]
