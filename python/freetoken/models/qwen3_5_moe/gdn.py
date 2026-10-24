from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.env import ENV
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, GatedRMSNorm, LinearColParallelMerged, LinearReplicated
from freetoken.layers.quantization import QuantConfig

from .gdn_kernels import gdn_decode_fla, gdn_prefill_chunk_fla


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[conv_dim, 1, K]`` (key ``conv1d.weight``)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class Qwen3_5GatedDeltaNet(BaseOP):
    """GatedDeltaNet op using the vendored flash-linear-attention triton kernels
    (``freetoken.kernel.fla``) for the recurrence and a per-request
    recurrent + conv state held in ``ctx.linear_state_pool`` (keyed by ``Req.table_idx``).

    Parameter names match HF (``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a``/
    ``conv1d``/``A_log``/``dt_bias``/``norm``/``out_proj``). Handles prefill (incl. chunked
    continuation) and single-token decode; state is fresh when ``req.cached_len == 0``.
    """

    def __init__(
        self, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_kernel_size, rms_norm_eps, layer_id, *, quant_config: QuantConfig | None = None,
        prefix: str = "", out_heads_tiled: bool = False,
    ):
        self.layer_id = layer_id
        self._out_heads_tiled = out_heads_tiled
        # The fla chunk/decode kernels read+write the recurrent state and the per-chunk h as
        # [V, K] while the LinearStatePool declares it [K, V]; these coincide (and the
        # hybrid-radix snapshot scatter h[h_row]->slot is a plain copy) only when the two head
        # dims are equal. Qwen3.5/3.6 satisfy this (128/128); guard any future config.
        assert head_k_dim == head_v_dim, (
            f"GatedDeltaNet requires head_k_dim == head_v_dim, got {head_k_dim} != {head_v_dim}"
        )
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        # quantized checkpoints quantize qkv|z but not b|a, so the fusion splits into a qkvz GEMM and a ba GEMM with their own schemes (matches sglang / vLLM)
        self._split_in_proj = (
            quant_config is not None and quant_config.scheme_for(f"{prefix}.in_proj_qkvz") is not None
        )

        self._in_proj_split = [self.conv_dim, self.value_dim, num_v_heads, num_v_heads]
        if self._split_in_proj:
            self.in_proj_qkvz = LinearColParallelMerged(
                hidden_size, [self.conv_dim, self.value_dim], has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj_qkvz",
            )
            self.in_proj_ba = LinearColParallelMerged(
                hidden_size, [num_v_heads, num_v_heads], has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj_ba",
            )
        else:
            # Fused input projection (one GEMM instead of four): qkv | z | b | a.
            self.in_proj = LinearColParallelMerged(
                hidden_size, self._in_proj_split, has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj",
            )
        self.conv1d = _DepthwiseConv1d(self.conv_dim, conv_kernel_size)
        # Recurrence-gating params kept in fp32 (exp/softplus is precision-sensitive,
        # and the fla kernel reads them as fp32) -- matches HF/sglang, and avoids a
        # per-call .float() upcast in the decode wrapper. The weight loader exempts
        # *.A_log / *.dt_bias from the model-dtype downcast.
        self.dt_bias = torch.empty(num_v_heads, dtype=torch.float32)
        self.A_log = torch.empty(num_v_heads, dtype=torch.float32)
        self.norm = GatedRMSNorm(head_v_dim, eps=rms_norm_eps)
        self.out_proj = LinearReplicated(
            self.value_dim, hidden_size, has_bias=False,
            quant_config=quant_config, prefix=f"{prefix}.out_proj",
        )

    def _gate_params(self, a: torch.Tensor, b: torch.Tensor):
        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)
        return g, beta

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, kernel] for the fused kernel

    def _conv_prefill(self, conv_in, pool, cu_seqlens, cache_indices, has_initial_state) -> torch.Tensor:
        """Varlen causal conv (fused sgl_kernel) with silu; reads/updates each request's
        conv state in place by ``cache_indices`` slot. ``conv_in`` [total, conv_dim].
        ``cu_seqlens`` / ``cache_indices`` / ``has_initial_state`` come from FLAMetadata."""
        li = pool.local_index(self.layer_id)
        # [conv_dim, total]; channel-last keeps it a view of the token-major input
        x = conv_in.t() if ENV.GDN_CONV_CHANNEL_LAST else conv_in.transpose(0, 1).contiguous()
        out = causal_conv1d_varlen(x, self._conv_weight(), pool.conv_states[li],
                                   cu_seqlens, cache_indices, has_initial_state)
        return out.transpose(0, 1)  # [total, conv_dim]

    def _conv_decode(self, conv_in: torch.Tensor, table_idx: torch.Tensor, pool) -> torch.Tensor:
        """Single-token causal conv update (fused sgl_kernel) by ``table_idx`` slot;
        updates conv state in place, no host loop -> CUDA-graph capturable.
        ``conv_in`` [B, conv_dim] -> silu(conv) [B, conv_dim]."""
        li = pool.local_index(self.layer_id)
        return causal_conv1d_decode(conv_in, pool.conv_states[li], self._conv_weight(), table_idx)

    def _write_track_snapshot(self, pool, li: int, conv_in: torch.Tensor,
                              h: torch.Tensor, fla) -> None:
        """Snapshot this layer's recurrent + conv state at the chunk-aligned track boundary
        into a donatable pool slot, on the forward stream (hybrid-radix extra_buffer path).
        SSM: ``recurrent_states[li, dst] = h[0, h_row]`` -- a DIRECT copy (h is [V,K], the
        state pool is [K,V]; they coincide because GDN requires head_k_dim == head_v_dim).
        Conv: the last (kernel-1) raw conv-input timesteps ending at the boundary."""
        rec = pool.recurrent_states[li]
        rec.index_copy_(0, fla.track_dst, h[0, fla.track_h_row].to(rec.dtype))
        cv = pool.conv_states[li]
        # conv_in [total, conv_dim]; gather the (kernel-1) window per tracked req.
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()  # [nt, conv_dim, K-1]
        cv.index_copy_(0, fla.track_dst, conv_win.to(cv.dtype))

    def _decode_rows(self, conv_in, a, b, idx, pool, li, cu_seqlens, dtype) -> torch.Tensor:
        """One token per request through the decode conv + recurrence, updating slots ``idx`` in place."""
        mixed = self._conv_decode(conv_in, idx, pool)  # [B, conv_dim]
        B = mixed.shape[0]
        qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = qf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
        k = kf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
        v = vf.reshape(1, B, self.num_v_heads, self.head_v_dim).to(dtype)
        return gdn_decode_fla(
            q, k, v, a, b, A_log=self.A_log, dt_bias=self.dt_bias,
            state_source=pool.recurrent_states[li], indices=idx,
            cu_seqlens=cu_seqlens, scale=self.head_k_dim ** -0.5,
        )

    def _verify(self, conv_in, a, b, pool, li, fla, spec, dtype) -> torch.Tensor:
        """A speculative verify: rows are step-major, ``B`` requests per step, each step through the
        decode kernels exactly as a plain decode step would run it. The state after each of the first
        ``k`` steps is copied to that step's snapshot slots, so a partial accept restores it."""
        B = spec.num_reqs
        if ENV.GDN_FUSED_VERIFY and B == 1 and spec.step is None and spec.k >= 1:
            return self._verify_fused(conv_in, a, b, pool, li, fla, spec, dtype)
        steps = range(spec.k + 1) if spec.step is None else (spec.step,)
        cu = fla.cu_seqlens[: B + 1]
        rec, conv = pool.recurrent_states[li], pool.conv_states[li]
        outs = []
        for i, s in enumerate(steps):
            rows = slice(i * B, (i + 1) * B)
            idx = fla.cache_indices[rows]
            outs.append(self._decode_rows(conv_in[rows], a[rows], b[rows], idx, pool, li, cu, dtype))
            if s < spec.k:
                dst = spec.snap_slots[s]
                rec.index_copy_(0, dst, rec.index_select(0, idx))
                conv.index_copy_(0, dst, conv.index_select(0, idx))
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

    def _verify_fused(self, conv_in, a, b, pool, li, fla, spec, dtype) -> torch.Tensor:
        """``_verify`` for one request (FREETOKEN_GDN_FUSED_VERIFY): its ``k + 1`` rows through one conv
        update, the recurrence over rows ``0..k-1`` in one kernel that writes the state after each row
        straight into the snapshot slots (``spec_slot(0..k-1)``, contiguous for request 0), then row ``k``
        as a decode step. The conv snapshots are the window before the verify shifted by ``s + 1`` rows."""
        from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

        k = spec.k
        rows = k + 1
        idx = fla.cache_indices[:1]
        rec, conv = pool.recurrent_states[li], pool.conv_states[li]
        base = pool.spec_slot(0)
        state_len = self._conv_weight().shape[-1] - 1
        prior = conv.index_select(0, idx)[..., :state_len]  # the window before this verify (the kernel keeps it first)
        x = conv_in.view(1, rows, self.conv_dim).transpose(1, 2)  # [1, conv_dim, rows], strided
        mixed = self._conv_decode(x, idx, pool).transpose(1, 2).reshape(rows, self.conv_dim)
        window = torch.cat([prior, x], dim=-1)
        conv[base : base + k, :, :state_len] = torch.stack(
            [window[0, :, s + 1 : s + 1 + state_len] for s in range(k)]
        )
        qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = qf.reshape(1, rows, self.num_k_heads, self.head_k_dim).to(dtype)
        kk = kf.reshape(1, rows, self.num_k_heads, self.head_k_dim).to(dtype)
        v = vf.reshape(1, rows, self.num_v_heads, self.head_v_dim).to(dtype)
        snaps = rec[base : base + k].unsqueeze(0)  # [1, k, ...]: step s's state -> spec_slot(s)
        head = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log, a=a[:k], dt_bias=self.dt_bias, softplus_beta=1.0, softplus_threshold=20.0,
            q=q[:, :k], k=kk[:, :k], v=v[:, :k], b=b[:k],
            initial_state_source=rec, initial_state_indices=idx,
            scale=self.head_k_dim ** -0.5, use_qk_l2norm_in_kernel=True,
            cu_seqlens=fla.cu_seqlens[:2] * k,  # a verify's rows are decode rows: [0, 1] -> [0, k], no host copy
            intermediate_states_buffer=snaps, intermediate_state_indices=idx.new_zeros(1),
        )[0]
        tail = gdn_decode_fla(
            q[:, k:], kk[:, k:], v[:, k:], a[k:], b[k:], A_log=self.A_log, dt_bias=self.dt_bias,
            state_source=rec, indices=idx, cu_seqlens=fla.cu_seqlens[:2], scale=self.head_k_dim ** -0.5,
        )
        return torch.cat([head.reshape(k, *tail.shape[1:]), tail], dim=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype

        # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
        # built once and shared by all GDN layers. The scheduler/graph set it; build it
        # lazily here (cached on the batch) for direct-op callers (tests).
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        if self._split_in_proj:
            qkvz = self.in_proj_qkvz.forward(hidden_states)
            conv_in, z = torch.split(qkvz, [self.conv_dim, self.value_dim], dim=-1)
            ba = self.in_proj_ba.forward(hidden_states)
            b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        else:
            proj = self.in_proj.forward(hidden_states)
            conv_in, z, b, a = torch.split(proj, self._in_proj_split, dim=-1)
        z = z.reshape(total, self.num_v_heads, self.head_v_dim)
        li = pool.local_index(self.layer_id)

        spec = getattr(batch, "spec", None)
        if spec is not None:
            core_out = self._verify(conv_in, a, b, pool, li, fla, spec, dtype)
        elif batch.is_decode:
            # Fused fla decode kernel: gating + in-kernel l2norm + recurrent update +
            # per-request state read/write-by-index, all in one kernel (no gather/scatter,
            # no clone, no external l2norm). q/k stay at num_k_heads (kernel handles GQA).
            core_out = self._decode_rows(conv_in, a, b, fla.cache_indices, pool, li, fla.cu_seqlens, dtype)
        else:
            mixed = self._conv_prefill(
                conv_in, pool, fla.cu_seqlens, fla.cache_indices, fla.has_initial_state)
            # fla chunk handles GQA in-kernel: q/k stay at num_k_heads, v at num_v_heads.
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, total, self.num_v_heads, self.head_v_dim).to(dtype)
            g, beta = self._gate_params(a, b)
            g = g.reshape(1, total, self.num_v_heads)
            beta = beta.float().reshape(1, total, self.num_v_heads)
            # The chunk kernel reads + writes back initial_state[cache_indices] in place;
            # fresh sequences (cached_len==0) must start from a zeroed slot.
            if fla.fresh_state_indices is not None:
                pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
            track = fla.track_dst is not None
            result = gdn_prefill_chunk_fla(
                q, k, v, g, beta,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
                return_h=track,
            )
            if track:
                core_out, h = result
                self._write_track_snapshot(pool, li, conv_in, h, fla)
            else:
                core_out = result

        core_out = core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm.forward(core_out, z).reshape(total, -1)
        if self._out_heads_tiled:
            # grouped value heads (k-head major) -> tiled (k-head minor), the column order of a GGUF ssm_out
            r = self.num_v_heads // self.num_k_heads
            out = out.view(total, self.num_k_heads, r, self.head_v_dim).transpose(1, 2).reshape(total, -1)
        return self.out_proj.forward(out)


__all__ = ["Qwen3_5GatedDeltaNet"]
