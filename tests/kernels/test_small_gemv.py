import pytest
import torch

from freetoken.kernel.triton.small_gemv import flash_next_gemv_config, small_gemv


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("n,k", [(336, 10240), (512, 2560), (640, 2560)])
def test_flash_next_decode_gemv_matches_dense_linear(n, k):
    if torch.version.hip is None or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1030"):
        pytest.skip("RX 6800 launch configurations")
    torch.manual_seed(13)
    x = torch.randn((1, k), device="cuda", dtype=torch.float16)
    w = torch.randn((n, k), device="cuda", dtype=torch.float16)
    block_n, block_k, num_warps = flash_next_gemv_config(1, n, k, x.device.index)
    actual = small_gemv(
        x, w, block_n=block_n, block_k=block_k, num_warps=num_warps, block_m=1,
    )
    legacy = small_gemv(
        x, w, block_n=block_n, block_k=block_k, num_warps=num_warps, block_m=4,
    )
    expected = torch.nn.functional.linear(x.float(), w.float()).half()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
    torch.testing.assert_close(legacy, expected, rtol=0.01, atol=0.02)
    default = small_gemv(x, w, block_n=block_n, block_k=block_k, num_warps=num_warps)
    torch.testing.assert_close(default, actual, rtol=0.0, atol=0.0)

    side_weight = torch.randn((k,), device="cuda", dtype=torch.float16)
    side_output = torch.empty((1,), device="cuda", dtype=torch.float32)
    fused = small_gemv(
        x, w, block_n=block_n, block_k=block_k, num_warps=num_warps,
        side_weight=side_weight, side_output=side_output,
    )
    torch.testing.assert_close(fused, actual, rtol=0.01, atol=0.02)
    expected_gate = torch.sigmoid(x.float() @ side_weight.float())
    torch.testing.assert_close(side_output, expected_gate, rtol=1e-4, atol=1e-5)


def test_flash_next_gemv_tuning_selects_measured_rx6800_geometry():
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("requires an AMD GPU")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1030"):
        pytest.skip("RX 6800 launch configurations")
    assert flash_next_gemv_config(1, 336, 10240, 0) == (2, 2048, 4)
    assert flash_next_gemv_config(1, 512, 2560, 0) == (2, 1024, 2)
    assert flash_next_gemv_config(1, 640, 2560, 0) == (2, 1024, 2)
    assert flash_next_gemv_config(4, 256, 2560, 0) == (2, 2048, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_flash_next_batched_router_gemv_matches_dense_linear():
    if torch.version.hip is None or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1030"):
        pytest.skip("RX 6800 launch configurations")
    torch.manual_seed(47)
    x = torch.randn((4, 2560), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 2560), device="cuda", dtype=torch.float16)
    block_n, block_k, num_warps = flash_next_gemv_config(4, 256, 2560, x.device.index)
    actual = small_gemv(x, w, block_n=block_n, block_k=block_k, num_warps=num_warps)
    expected = torch.nn.functional.linear(x.float(), w.float()).half()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)


def test_unquantized_linear_uses_small_gemv_only_with_the_flag(monkeypatch):
    from types import SimpleNamespace

    from freetoken.env import ENV
    from freetoken.kernel.triton import small_gemv as sg
    from freetoken.layers.quantization.linear.unquantized import TorchLinearKernel

    calls = []
    real = sg.dense_linear
    monkeypatch.setattr(sg, "dense_linear", lambda x, w, b=None: calls.append(x.shape) or real(x, w, b))
    torch.manual_seed(0)
    layer = SimpleNamespace(
        weight=torch.randn(320, 2560, device="cuda", dtype=torch.float16) / 50,
        bias=None,
    )
    x = torch.randn(1, 2560, device="cuda", dtype=torch.float16)
    ref = torch.nn.functional.linear(x, layer.weight)
    monkeypatch.setattr(ENV.SMALL_GEMV, "value", False)
    torch.testing.assert_close(TorchLinearKernel().apply(layer, x), ref)
    assert calls == []
    monkeypatch.setattr(ENV.SMALL_GEMV, "value", True)
    out = TorchLinearKernel().apply(layer, x)
    assert len(calls) == (1 if torch.version.hip is not None else 0)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
