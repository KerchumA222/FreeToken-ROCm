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

import os

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
# Groups padded to fewer rows than this run down as W @ a: rocBLAS on gfx1030 runs that
# ~1.4x faster at ~150 rows (x @ W^T catches up by ~300). FT_MOE_DOWN_FLIP=0 turns it off.
_DOWN_FLIP_ROWS = int(os.environ.get("FT_MOE_DOWN_FLIP", "256"))


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
    from freetoken.kernel.triton.moe_prefill import (
        gather_transposed, scatter_add_transposed, silu_and_mul_cols, silu_and_mul_transposed)

    """Prefill MoE: route (token, expert) pairs by expert, and for groups of experts of
    similar load dequantize their weights and run two padded batched GEMMs. Each group
    gathers its own rows from ``x`` and adds its weighted outputs into one fp32 [T, H]
    accumulator, so nothing is sized tokens x top_k x hidden."""
    import numpy as np

    t, k = topk_ids.shape
    h = x.shape[1]
    flat = topk_ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    # One host sync per layer: the per-expert counts. bincount has a fixed output size;
    # unique_consecutive synced once more for its own. The groups' padded row layout is
    # then planned on the host in one pass and gathered once, so each group launches
    # only its GEMM work -- per-group index math was ~0.6 s of GPU idle over a
    # 7.5k-token Flash-Next chunk, the host falling behind small groups.
    c = np.asarray(torch.bincount(flat, minlength=gate_up_q.shape[0]).tolist())
    s = np.cumsum(c) - c
    ex = np.nonzero(c)[0]
    ex = ex[np.argsort(-c[ex], kind="stable")]  # busiest first: similar padding per group
    groups, pos_parts, valid_parts = [], [], []
    g = row = 0
    while g < len(ex):
        maxc = int(c[ex[g]])  # the busiest of the group
        n = max(1, min(_GROUP, _GROUP_ROWS // maxc, len(ex) - g))
        e = ex[g : g + n]
        span = np.arange(maxc)
        valid = span[None, :] < c[e][:, None]
        pos_parts.append(np.where(valid, s[e][:, None] + span[None, :], 0).ravel())  # pad -> row 0
        valid_parts.append(valid.ravel())
        groups.append((g, n, row, maxc))
        g += n
        row += n * maxc

    def dev(a, dtype):
        return torch.from_numpy(np.ascontiguousarray(a)).to(dtype).pin_memory().to(x.device, non_blocking=True)

    ids_all = dev(ex, torch.int64)
    pos_all = dev(np.concatenate(pos_parts), torch.int64)
    valid_all = dev(np.concatenate(valid_parts), torch.float32)
    tok_all = (order // k).index_select(0, pos_all)
    w_all = topk_weights.reshape(-1).index_select(0, order).float().index_select(0, pos_all) * valid_all
    out = torch.zeros(t, h, dtype=torch.float32, device=x.device)
    for g, n, row, maxc in groups:
        ids = ids_all[g : g + n]
        tok = tok_all[row : row + n * maxc]
        weight = w_all[row : row + n * maxc]
        # gate_up as W [n, 2I, H] @ x^T: rocBLAS on gfx1030 runs this batched shape ~2.4x
        # faster than x @ W^T over the dequantized (row = output) layout.
        # The two transposes run in Triton (kernel/triton/moe_prefill.py): torch's strided
        # copies of them were 21% of a long prefill.
        xt = gather_transposed(x, tok.view(n, maxc))
        w_gu = _dequant_rows(gate_up_q.index_select(0, ids), gu_type, x.dtype)
        gu = torch.bmm(w_gu, xt)                                   # [n, 2I, maxc]
        del xt, w_gu
        w_dn = _dequant_rows(down_q.index_select(0, ids), dn_type, x.dtype)
        if act_fn is silu_and_mul and maxc < _DOWN_FLIP_ROWS:
            # Below ~256 columns W [n, H, I] @ a [n, I, c] is ~1.4x faster than a @ W^T;
            # the output comes back [n, H, c] and is scattered in one Triton pass.
            inter = silu_and_mul_cols(gu)
            del gu
            y = torch.bmm(w_dn, inter)                             # [n, H, maxc]
            del inter, w_dn
            scatter_add_transposed(out, y, tok.view(n, maxc), weight.view(n, maxc))
        else:
            if act_fn is silu_and_mul:
                inter = silu_and_mul_transposed(gu)
            else:
                inter = act_fn(gu.transpose(1, 2).contiguous().view(n * maxc, -1))
            del gu
            y = torch.bmm(inter.view(n, maxc, -1), w_dn.transpose(1, 2)).view(n * maxc, h)
            del inter, w_dn
            out.index_add_(0, tok, y.float() * weight[:, None])
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
