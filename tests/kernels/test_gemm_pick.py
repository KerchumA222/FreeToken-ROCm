"""FREETOKEN_PREFILL_GEMM_PICK: every formulation a shape may pick is the same product, and the choice is cached."""

from __future__ import annotations

import pytest
import torch

from freetoken.layers.quantization.ggml_types import dequantize_cpu
from freetoken.third_party.gguf import GGMLQuantizationType as Q

from .test_ggml_types import _activation, _close, _device_weight, _random_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def test_pick_caches_the_fastest_candidate():
    from freetoken.kernel import gemm_pick

    calls = []

    def make(i, delay):
        def run():
            calls.append(i)
            if delay:
                torch.cuda._sleep(5_000_000)
            return torch.full((1,), float(i), device="cuda")
        return run

    key = ("test", object())
    first = gemm_pick.pick(key, (make(0, True), make(1, False)))
    assert first.item() == 1.0 and gemm_pick.choices()[key] == 1
    calls.clear()
    assert gemm_pick.pick(key, (make(0, True), make(1, False))).item() == 1.0
    assert calls == [1]


@pytest.mark.parametrize("shape", [(300, 1024, 336), (1100, 512, 96)])
def test_dense_linear_orientations_match(shape, monkeypatch):
    from freetoken.env import ENV
    from freetoken.kernel import gemm_pick
    from freetoken.kernel.triton.small_gemv import dense_linear

    monkeypatch.setattr(ENV.PREFILL_GEMM_PICK, "value", True)
    m, k, n = shape
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16, generator=gen)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16, generator=gen) / k**0.5
    b = torch.randn(n, device="cuda", dtype=torch.float16, generator=gen)
    ref = torch.nn.functional.linear(x.float(), w.float(), b.float())
    torch.testing.assert_close(gemm_pick.flipped_linear(x, w).float() + b.float(), ref, rtol=2e-2, atol=2e-2)
    for _ in range(2):  # timed first call, cached second
        torch.testing.assert_close(dense_linear(x, w, b).float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("ggml_type", [Q.Q6_K, Q.Q4_K, Q.IQ4_XS, Q.IQ3_S])
@pytest.mark.parametrize("choice", [0, 1, 2])
def test_ggml_prefill_candidates_match_cpu(ggml_type, choice, monkeypatch):
    """MMQ, dequantize + x @ W^T and dequantize + (W @ x^T)^T all agree with the CPU reference at prefill rows."""
    from freetoken.env import ENV
    from freetoken.kernel import gemm_pick, ggml

    monkeypatch.setattr(ENV.PREFILL_GEMM_PICK, "value", True)
    k, rows, m = 512, 160, 300
    raw = _random_rows(int(ggml_type), rows, k, seed=7)
    ref = torch.from_numpy(dequantize_cpu(raw, int(ggml_type), k))
    w = _device_weight(raw, k, int(ggml_type))
    x = _activation((m, k), seed=3)
    key = ("ggml", int(ggml_type), gemm_pick.row_bucket(m), rows, k, torch.float16)
    monkeypatch.setitem(gemm_pick._choice, key, choice)
    out = ggml.mul_mat(w, int(ggml_type), x.to("cuda", torch.float16))
    _close(out, x @ ref.T, 1e-2, f"choice {choice}")
