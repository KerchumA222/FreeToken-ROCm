"""Adjacent packed slots of one ggml type run as a single GEMV after finalize."""

from types import SimpleNamespace

import torch

from freetoken.layers.quantization.linear.gguf import MmvqGgufLinearKernel

Q8_0, Q4_K, F16 = 8, 12, 1


def _layer(types, rows):
    layer = SimpleNamespace(gguf_types=list(types),
                            gguf_slots=tuple(f"weight_{i}" for i in range(len(types))))
    for i, (t, r) in enumerate(zip(types, rows)):
        setattr(layer, f"weight_{i}", torch.full((r, 4), i, dtype=torch.uint8))
    return layer


def test_same_type_slots_merge_into_views():
    layer = _layer([Q8_0, Q8_0, Q8_0, Q8_0], [6, 3, 1, 1])
    MmvqGgufLinearKernel().finalize(layer)
    assert len(layer._gguf_runs) == 1
    name, t = layer._gguf_runs[0]
    buf = getattr(layer, name)
    assert t == Q8_0 and buf.shape == (11, 4)
    assert layer.weight_1.data_ptr() == buf[6:].data_ptr()
    assert torch.equal(buf[:, 0], torch.tensor([0] * 6 + [1] * 3 + [2, 3], dtype=torch.uint8))


def test_type_changes_and_dense_slots_break_runs():
    layer = _layer([Q4_K, Q4_K, Q8_0, F16, F16], [2, 2, 2, 2, 2])
    MmvqGgufLinearKernel().finalize(layer)
    assert [t for _, t in layer._gguf_runs] == [Q4_K, Q8_0, F16, F16]
