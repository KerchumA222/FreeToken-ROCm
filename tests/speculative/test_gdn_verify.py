"""GatedDeltaNet under an MTP verify, on the GPU kernels: the verify runs each step through the
same decode kernels as plain decoding, so its outputs, the per-step snapshots and a rollback to
any of them are bit-identical to stepping one token at a time."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN, NK, NV, D, KERNEL = 256, 2, 4, 128, 4


@pytest.fixture(autouse=True)
def _single_rank_tp():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _op_and_pool(num_reqs: int, k: int):
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig
    from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
    from freetoken.utils import torch_dtype

    torch.manual_seed(0)
    with torch.device("cuda"), torch_dtype(torch.bfloat16):
        op = Qwen3_5GatedDeltaNet(HIDDEN, NK, NV, D, D, KERNEL, 1e-6, layer_id=0)
    for name, t in op.state_dict().items():
        scale = 0.2 if "conv1d" in name else (1.0 if name.endswith(("norm.weight", "A_log", "dt_bias")) else 0.05)
        t.copy_(torch.randn_like(t, dtype=torch.float32).to(t.dtype) * scale + (1.0 if name.endswith("norm.weight") else 0.0))
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=NK, num_value_heads=NV,
        key_head_dim=D, value_head_dim=D, conv_kernel_dim=KERNEL, output_gate="silu",
    )
    pool = LinearStatePool(group, num_reqs + 2, dtype=torch.bfloat16, device=torch.device("cuda"),
                           tp_size=1, spec_slots=num_reqs * k)
    pool.recurrent_states.copy_(torch.randn_like(pool.recurrent_states) * 0.1)
    pool.conv_states.copy_(torch.randn_like(pool.conv_states, dtype=torch.float32).to(pool.conv_states.dtype))
    return op, pool


def _spec(pool, num_reqs: int, k: int, step=None):
    from freetoken.core import SpecVerify

    snap = [torch.tensor([pool.spec_slot(i * k + s) for i in range(num_reqs)], dtype=torch.int64, device="cuda")
            for s in range(k)]
    return SpecVerify(reqs=[None] * num_reqs, k=k, first_drafts=[0] * num_reqs, snap_slots=snap, step=step)


def _fla(rows: int, slots: torch.Tensor):
    from freetoken.attention.linear import FLAMetadata

    return FLAMetadata(cu_seqlens=torch.arange(rows + 1, dtype=torch.int32, device="cuda"), cache_indices=slots)


def _states(pool):
    return pool.recurrent_states.clone(), pool.conv_states.clone()


@pytest.mark.parametrize("num_reqs, k", [(1, 1), (1, 3), (3, 2)])
def test_verify_steps_match_decode_and_snapshot_every_step(num_reqs, k):
    op, pool = _op_and_pool(num_reqs, k)
    live = torch.arange(1, num_reqs + 1, dtype=torch.int32, device="cuda")
    rows = num_reqs * (k + 1)
    torch.manual_seed(1)
    conv_in = torch.randn(rows, op.conv_dim, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(rows, NV, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(rows, NV, device="cuda", dtype=torch.bfloat16)
    cu = torch.arange(num_reqs + 1, dtype=torch.int32, device="cuda")
    start = _states(pool)

    ref_out, ref_states = [], []
    for s in range(k + 1):
        sl = slice(s * num_reqs, (s + 1) * num_reqs)
        ref_out.append(op._decode_rows(conv_in[sl], a[sl], b[sl], live, pool, 0, cu, torch.bfloat16))
        ref_states.append(_states(pool))
    ref_out = torch.cat(ref_out)

    pool.recurrent_states.copy_(start[0])
    pool.conv_states.copy_(start[1])
    spec = _spec(pool, num_reqs, k)
    out = op._verify(conv_in, a, b, pool, 0, _fla(rows, live.repeat(k + 1)), spec, torch.bfloat16)

    assert torch.equal(out, ref_out)
    rec, conv = pool.recurrent_states, pool.conv_states
    for i in range(num_reqs):
        slot = int(live[i])
        assert torch.equal(rec[:, slot], ref_states[k][0][:, slot])
        for s in range(k):
            snap = pool.spec_slot(i * k + s)
            assert torch.equal(rec[:, snap], ref_states[s][0][:, slot]), (i, s)
            assert torch.equal(conv[:, snap], ref_states[s][1][:, slot]), (i, s)

    # roll each request back to a different step and check it holds exactly that step's state
    for i in range(num_reqs):
        s = i % k
        slot = int(live[i])
        pool.copy_from(pool.spec_slot(i * k + s), slot)
        assert torch.equal(rec[:, slot], ref_states[s][0][:, slot])
        assert torch.equal(conv[:, slot], ref_states[s][1][:, slot])


def test_one_step_of_a_sequential_verify_is_a_decode_step(monkeypatch):
    """FREETOKEN_SPEC_SEQUENTIAL_VERIFY runs each verify step as its own forward: the op's whole
    forward (projections included) is then the same computation as a plain decode step."""
    num_reqs, k = 2, 2
    op, pool = _op_and_pool(num_reqs, k)
    live = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    x = torch.randn(num_reqs, HIDDEN, device="cuda", dtype=torch.bfloat16)
    start = _states(pool)
    ctx = SimpleNamespace(linear_state_pool=pool, batch=None)
    monkeypatch.setattr("freetoken.models.qwen3_5_moe.gdn.get_global_ctx", lambda: ctx)

    ctx.batch = SimpleNamespace(is_decode=True, spec=None, fla_metadata=_fla(num_reqs, live))
    plain = op.forward(x)
    after = _states(pool)

    pool.recurrent_states.copy_(start[0])
    pool.conv_states.copy_(start[1])
    ctx.batch = SimpleNamespace(is_decode=True, spec=_spec(pool, num_reqs, k, step=0), fla_metadata=_fla(num_reqs, live))
    step = op.forward(x)
    assert torch.equal(step, plain)
    assert torch.equal(pool.recurrent_states[:, 1:3], after[0][:, 1:3])
    assert torch.equal(pool.recurrent_states[:, pool.spec_slot(0)], after[0][:, 1])
    assert torch.equal(pool.recurrent_states[:, pool.spec_slot(k)], after[0][:, 2])
