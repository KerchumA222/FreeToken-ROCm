"""Numerical checks for the Qwen3.5/3.6 GDN (gated delta-net) and causal-conv1d kernels.

These paths had no coverage at all: the qwen35moe adapter was validated against
synthetic random-weight GGUFs, which cannot detect a numerically wrong kernel (or a
wrong weight mapping) because there is no reference output to disagree with. Both
kernels run in every one of the model's 30 linear-attention layers, so a fault here
corrupts generation everywhere.

The oracle is ``gdn_reference.recurrent_gated_delta_rule``, a verbatim port of HF's
``torch_recurrent_gated_delta_rule`` that was validated bit-for-bit when ported.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HK, HV, DK, DV = 4, 8, 128, 128  # GQA 2, square head dims (as in Qwen3.6-35B-A3B)
SCALE = DK**-0.5


def _corr(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _rel(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def _inputs(t, seed=0):
    torch.manual_seed(seed)
    d = "cuda"
    return (
        torch.randn(1, t, HK, DK, device=d, dtype=torch.bfloat16),
        torch.randn(1, t, HK, DK, device=d, dtype=torch.bfloat16),
        torch.randn(1, t, HV, DV, device=d, dtype=torch.bfloat16),
        -torch.rand(1, t, HV, device=d, dtype=torch.float32) * 0.5,
        torch.rand(1, t, HV, device=d, dtype=torch.float32),
    )


def _expand(q, k):
    rep = HV // HK
    return q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)


def test_gdn_prefill_matches_recurrent_reference():
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla
    from freetoken.models.qwen3_5_moe.gdn_reference import recurrent_gated_delta_rule

    T = 96
    q, k, v, g, beta = _inputs(T)
    state = torch.zeros(1, HV, DK, DV, device="cuda", dtype=torch.float32)
    o = gdn_prefill_chunk_fla(
        q, k, v, g, beta,
        state_source=state,
        indices=torch.zeros(1, device="cuda", dtype=torch.int32),
        cu_seqlens=torch.tensor([0, T], device="cuda", dtype=torch.int64),
        scale=SCALE,
    )
    qe, ke = _expand(q, k)
    o_ref, _ = recurrent_gated_delta_rule(qe, ke, v, g, beta, use_qk_l2norm=True)
    assert _corr(o, o_ref[0]) > 0.999, _corr(o, o_ref[0])
    assert _rel(o, o_ref[0]) < 5e-2


def test_gdn_prefill_writes_state_transposed_as_v_by_k():
    """The recurrent state is stored [heads, V, K], NOT [heads, K, V].

    ``gdn_kernels.py`` documents ``state_source`` as ``[slots, heads, head_k_dim,
    head_v_dim]``, but both the chunked prefill kernel and the decode kernel address it
    as ``o_v * K + o_k``. That is invisible while head_k_dim == head_v_dim (every
    shipped Qwen3.5/3.6 config), so pin the real convention here: anything that builds
    a state by hand -- a test, a checkpoint restore, a cache migration -- has to match
    the kernels, not the docstring.
    """
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla
    from freetoken.models.qwen3_5_moe.gdn_reference import recurrent_gated_delta_rule

    T = 64
    q, k, v, g, beta = _inputs(T, seed=1)
    state = torch.zeros(1, HV, DK, DV, device="cuda", dtype=torch.float32)
    gdn_prefill_chunk_fla(
        q, k, v, g, beta,
        state_source=state,
        indices=torch.zeros(1, device="cuda", dtype=torch.int32),
        cu_seqlens=torch.tensor([0, T], device="cuda", dtype=torch.int64),
        scale=SCALE,
    )
    qe, ke = _expand(q, k)
    _, ref = recurrent_gated_delta_rule(qe, ke, v, g, beta, use_qk_l2norm=True)
    assert _corr(state[0], ref[0].transpose(-1, -2)) > 0.999
    assert _corr(state[0], ref[0]) < 0.5  # the un-transposed reading is NOT the layout


def test_gdn_decode_matches_reference_from_a_nonzero_state():
    """A zero incoming state hides state-indexing faults (and transposes), so this
    starts from a populated recurrent state -- the case every real decode step hits."""
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla
    from freetoken.models.qwen3_5_moe.gdn_reference import recurrent_gated_delta_rule

    torch.manual_seed(2)
    d, B = "cuda", 2
    q = torch.randn(1, B, HK, DK, device=d, dtype=torch.bfloat16)
    k = torch.randn(1, B, HK, DK, device=d, dtype=torch.bfloat16)
    v = torch.randn(1, B, HV, DV, device=d, dtype=torch.bfloat16)
    a = torch.randn(B, HV, device=d, dtype=torch.float32)
    b = torch.randn(B, HV, device=d, dtype=torch.float32)
    A_log = torch.randn(HV, device=d, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=d, dtype=torch.float32)
    state = torch.randn(B, HV, DV, DK, device=d, dtype=torch.float32) * 0.05  # [V, K]

    o = gdn_decode_fla(
        q, k, v, a, b, A_log=A_log, dt_bias=dt_bias,
        state_source=state.clone(),
        indices=torch.arange(B, device=d, dtype=torch.int32),
        cu_seqlens=torch.arange(B + 1, device=d, dtype=torch.int64),
        scale=SCALE,
    )
    # gating per gdn.Qwen3_5GatedDeltaNet; reference takes the state as [K, V]
    rep = HV // HK
    o_ref, _ = recurrent_gated_delta_rule(
        q[0].repeat_interleave(rep, dim=1).unsqueeze(1),
        k[0].repeat_interleave(rep, dim=1).unsqueeze(1),
        v[0].unsqueeze(1),
        (-A_log.exp() * F.softplus(a.float() + dt_bias)).unsqueeze(1),
        b.sigmoid().unsqueeze(1),
        initial_state=state.transpose(-1, -2).contiguous(),
        use_qk_l2norm=True,
    )
    assert _corr(o, o_ref[:, 0]) > 0.999, _corr(o, o_ref[:, 0])
    assert _rel(o, o_ref[:, 0]) < 5e-2


@pytest.mark.parametrize("kernel_size", [4])
def test_causal_conv1d_prefill_and_decode_match_torch(kernel_size):
    from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen

    torch.manual_seed(3)
    d, C, T = "cuda", 512, 40
    x = torch.randn(C, T, device=d, dtype=torch.bfloat16)
    w = torch.randn(C, kernel_size, device=d, dtype=torch.bfloat16) * 0.3
    cu = torch.tensor([0, T], device=d, dtype=torch.int32)
    ci = torch.zeros(1, device=d, dtype=torch.int32)
    hi = torch.zeros(1, device=d, dtype=torch.bool)

    state = torch.zeros(1, C, kernel_size - 1, device=d, dtype=torch.bfloat16)
    out = causal_conv1d_varlen(x.clone(), w, state, cu, ci, hi)
    ref = F.silu(
        F.conv1d(
            F.pad(x.float().unsqueeze(0), (kernel_size - 1, 0)),
            w.float().unsqueeze(1), groups=C,
        )[0]
    )
    assert _corr(out, ref) > 0.999
    assert _rel(out, ref) < 5e-2

    # one decode step continuing from the tail state the prefill just wrote
    xt = torch.randn(1, C, device=d, dtype=torch.bfloat16)
    od = causal_conv1d_decode(xt.clone(), state.clone(), w, ci)
    hist = torch.cat([x[:, -(kernel_size - 1):], xt[0].unsqueeze(-1)], dim=-1).float()
    ref_d = F.silu((hist * w.float()).sum(-1))
    assert _corr(od[0], ref_d) > 0.999
    assert _rel(od[0], ref_d) < 5e-2
