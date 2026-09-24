from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_q2_0_reference_dequantizes_four_codes_per_byte():
    from freetoken.models.gguf.dequant import GGML_Q2_0, dequant_q2_0, row_bytes

    raw = torch.zeros(18, dtype=torch.uint8)
    raw[:2] = torch.tensor([2.0], dtype=torch.float16).view(torch.uint8)
    raw[2:] = 0b11100100
    got = dequant_q2_0(raw, torch.float32).reshape(16, 4)

    assert row_bytes(640, GGML_Q2_0) == 180
    assert torch.equal(got, torch.tensor([-2.0, 0.0, 2.0, 4.0]).expand(16, 4))


def test_q2_0_symmetric_reference_dequantizes_four_codes_per_byte():
    from freetoken.models.gguf.dequant import GGML_Q2_0_SYM, dequant_q2_0_sym, row_bytes

    raw = torch.zeros(18, dtype=torch.uint8)
    raw[:2] = torch.tensor([2.0], dtype=torch.float16).view(torch.uint8)
    raw[2:] = 0b11100100
    got = dequant_q2_0_sym(raw, torch.float32).reshape(16, 4)

    assert row_bytes(640, GGML_Q2_0_SYM) == 180
    assert torch.equal(got, torch.tensor([-6.0, -2.0, 2.0, 6.0]).expand(16, 4))


@pytest.mark.parametrize(
    ("metadata", "expected_type"),
    [({}, 42), ({"freetoken.q2_0.codebook": "symmetric_odd"}, 10042)],
)
def test_q2_0_codebook_metadata_selects_internal_type(monkeypatch, metadata, expected_type):
    from freetoken.models.gguf import reader

    tensor = SimpleNamespace(
        name="blk.0.ffn_down_exps.weight",
        shape=(64, 1),
        tensor_type=42,
        data=np.zeros(18, dtype=np.uint8),
        data_offset=0,
    )
    monkeypatch.setattr(reader, "_reader", lambda _path: SimpleNamespace(tensors=[tensor]))
    monkeypatch.setattr(reader, "load_gguf_metadata", lambda _path: metadata)

    [got] = reader._iter_shard_tensors("model.gguf")

    assert got.ggml_type == expected_type
    assert got.row_bytes == 18


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_q2_0_moe_vec_matches_reference():
    from freetoken.kernel.gguf import ggml_moe_a8_vec
    from freetoken.models.gguf.dequant import GGML_Q2_0_SYM, dequant_q2_0_sym

    rows, cols = 4, 640
    packed = torch.empty((1, rows, cols // 64 * 18), dtype=torch.uint8)
    scales = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float16)
    for row, scale in enumerate(scales):
        blocks = packed[0, row].reshape(-1, 18)
        blocks[:, :2] = scale.reshape(1).view(torch.uint8)
        blocks[:, 2:] = 0b11100100

    x = (torch.sin(torch.arange(cols, dtype=torch.float32)) * 0.1).reshape(1, cols)
    dense = dequant_q2_0_sym(packed, torch.float32).reshape(rows, cols)
    expected = x @ dense.T
    got = ggml_moe_a8_vec(
        x.half().cuda(), packed.cuda(), torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
        1, GGML_Q2_0_SYM, rows, 1,
    ).float().cpu()

    torch.testing.assert_close(got, expected, rtol=0.03, atol=0.03)


def test_gguf_mixed_type_ids_keep_gate_up_down_order():
    from freetoken.layers.quantization.moe.gguf import GgufMoEMethod, _bank_types

    cfg = SimpleNamespace(scheme=SimpleNamespace(weight=SimpleNamespace(elem="Q4_K+Q5_1")))
    method = GgufMoEMethod.__new__(GgufMoEMethod)
    method.cfg = cfg
    assert _bank_types(cfg) == (12, 7)
    assert method.ggml_types == (12, 7)


def test_cpu_gguf_bank_rows_use_each_bank_type_geometry():
    from freetoken.models.gguf.dequant import row_bytes
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    hidden, inter, experts = 2560, 640, 2
    ex = CpuMoeExecutor.__new__(CpuMoeExecutor)
    ex.num_layers = 1
    ex._banks = []
    banks = {
        "gate_up": [torch.empty(experts, 2 * inter, row_bytes(hidden, 12), dtype=torch.uint8)],
        "down": [torch.empty(experts, hidden, row_bytes(inter, 7), dtype=torch.uint8)],
    }

    ptrs, shape = ex._resolve_gguf_banks(banks, (12, 7))

    assert shape == (hidden, inter)
    assert ptrs["gate_up_ptr"] != 0 and ptrs["down_ptr"] != 0
    assert row_bytes(hidden, 12) == 1440
    assert row_bytes(inter, 7) == 480
    # H=2560 happens to make Q4_K's row width equal Q4_0's; a byte-width-only
    # heuristic would accept the Q4_K bytes and then decode them as Q4_0.
    assert row_bytes(hidden, 12) == row_bytes(hidden, 2)
