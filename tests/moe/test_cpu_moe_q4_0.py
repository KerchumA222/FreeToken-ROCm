"""CPU MoE executor -- native GGUF Q4_0 experts (``--moe-backend cpu``/``hybrid`` for
the gemma4 GGUF checkpoint).

The CPU W4A16 GEMV (``q4_0_dot`` in csrc/cpu_moe/cpu_moe_ext.cpp) reads the *same*
packed Q4_0 banks the GPU offload path streams and dequantizes weights inside the
K-loop. We check it against the reference dequant (models/gguf/dequant.py) + the
production bf16 GPU decode kernel on byte-identical banks: both are W4A16, so the
only spread is weight bf16-rounding + reduction order -> tight relative tolerance.

Part 2 covers CUDA-graph capture/replay (the cudaLaunchHostFunc submit/sync nodes
must recompute from the freshly written pinned routing on each replay).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _pack_q4_0(nibbles: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pack [S, OUT, K] uint8 nibble codes (0..15) + [S, OUT, K//32] fp16 scales into
    the native Q4_0 row layout [S, OUT, K//32*18]: each 32-elem block is a 2-byte fp16
    scale followed by 16 bytes whose byte j holds elem j (low nibble) and elem j+16
    (high nibble) -- the exact layout ``dequant_q4_0`` reads back."""
    S, OUT, K = nibbles.shape
    nb = K // 32
    blk = nibbles.reshape(S, OUT, nb, 32)
    lo = blk[..., :16]      # elems 0..15 -> low nibbles of bytes 0..15
    hi = blk[..., 16:]      # elems 16..31 -> high nibbles of bytes 0..15
    packed = (lo | (hi << 4)).to(torch.uint8)                       # [S, OUT, nb, 16]
    d_bytes = scale.to(torch.float16).view(torch.uint8).reshape(S, OUT, nb, 2)
    row = torch.cat([d_bytes, packed], dim=-1)                      # [S, OUT, nb, 18]
    return row.reshape(S, OUT, nb * 18).contiguous()


def _dequant_bank(packed: torch.Tensor, K: int, dev) -> torch.Tensor:
    """[S, OUT, K//32*18] packed Q4_0 -> [S, OUT, K] bf16 (storage order == elem order)."""
    from freetoken.models.gguf.dequant import GGML_Q4_0, dequantize

    S, OUT, _ = packed.shape
    flat = dequantize(packed.reshape(-1), GGML_Q4_0, torch.bfloat16)
    return flat.reshape(S, OUT, K).to(dev)


def _make_q4_0_cache(L, E, H, I, seed=0):
    """Random but valid native Q4_0 banks for the cpu backend (pinned host tensors)."""
    from freetoken.kernel.pinned import alloc_pinned_tensor

    torch.manual_seed(seed)
    S = L * E

    def rows(OUT, K):
        nib = torch.randint(0, 16, (S, OUT, K), dtype=torch.uint8)
        scale = 0.02 + 0.03 * torch.rand(S, OUT, K // 32)
        packed = _pack_q4_0(nib, scale)                            # [S, OUT, K//32*18]
        pinned = alloc_pinned_tensor(*packed.shape, dtype=torch.uint8)
        pinned.copy_(packed)
        return pinned

    return SimpleNamespace(
        quant_format="q4_0",
        bank_sources={"gate_up": list(rows(2 * I, H).split(E)), "down": list(rows(H, I).split(E))},
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


def _make_q4_k_q5_1_cache(L, E, H, I, seed=0):
    """Valid mixed GGUF rows for Q4_K gate/up and Q5_1 down (types 12 and 7)."""
    from freetoken.kernel.pinned import alloc_pinned_tensor

    torch.manual_seed(seed)
    S = L * E
    q4 = torch.randint(0, 256, (S, 2 * I, H // 256, 144), dtype=torch.uint8)
    q4[..., :4] = torch.tensor([0.012, 0.006], dtype=torch.float16).view(torch.uint8)
    q4[..., 4:16] = torch.tensor([1] * 8 + [0x11] * 4, dtype=torch.uint8)
    q4 = q4.reshape(S, 2 * I, -1).contiguous()

    q5 = torch.randint(0, 256, (S, H, I // 32, 24), dtype=torch.uint8)
    q5[..., :4] = torch.tensor([0.0015, -0.02], dtype=torch.float16).view(torch.uint8)
    q5 = q5.reshape(S, H, -1).contiguous()

    q4_host = alloc_pinned_tensor(*q4.shape, dtype=torch.uint8)
    q5_host = alloc_pinned_tensor(*q5.shape, dtype=torch.uint8)
    q4_host.copy_(q4)
    q5_host.copy_(q5)
    return SimpleNamespace(
        quant_format="q4_0",
        bank_sources={
            "gate_up": list(q4_host.split(E)),
            "down": list(q5_host.split(E)),
        },
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


def _load_real_q4_k_q5_1_cache(model_path: str, layer: int, experts: tuple[int, ...]):
    """Read selected experts from a GGUF into one compact, remapped host bank."""
    import numpy as np

    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.moe.disk_store import GgufExpertStore

    with GgufExpertStore(
        model_path,
        num_experts=512,
        bank_types={"gate_up": 12, "down": 7},
        num_layers=48,
    ) as store:
        banks = {}
        for name in ("gate_up", "down"):
            rows, row_bytes = store.row_shape(name)
            data = np.empty((len(experts), rows, row_bytes), dtype=np.uint8)
            for dst, expert in zip(data, experts, strict=True):
                store.read_expert(name, layer, expert, dst)
            host = alloc_pinned_tensor(*data.shape, dtype=torch.uint8)
            host.copy_(torch.from_numpy(data))
            banks[name] = [host]

    return SimpleNamespace(
        quant_format="q4_0",
        bank_sources=banks,
        num_layers=1,
        num_experts=len(experts),
        decode_target="cpu",
        cpu_executor=None,
    )


def _dequant_gguf_bank(packed: torch.Tensor, ggml_type: int, K: int, dev):
    """Use gguf-py's reference dequantizer to build a small dense GPU oracle."""
    import numpy as np
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize
    from freetoken.models.gguf.dequant import row_bytes

    S, OUT, _ = packed.shape
    rows = packed.cpu().numpy().reshape(-1, row_bytes(K, ggml_type))
    dense = dequantize(rows, GGMLQuantizationType(ggml_type))
    return torch.from_numpy(np.asarray(dense)).reshape(S, OUT, K).to(
        device=dev, dtype=torch.bfloat16
    )


@pytest.mark.parametrize("bs", [1, 3, 8])
def test_cpu_decode_q4_0_matches_dequant_then_gpu(bs):
    """CPU inline-dequant Q4_0 GEMV vs. canonical dequant_q4_0 + bf16 GPU decode."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    torch.manual_seed(400 + bs)
    L, E, H, I, top_k = 3, 16, 2816, 704, 8   # gemma-4-26B-A4B geometry (H,I % 32 == 0)
    layer = 1
    dev = torch.device("cuda")
    cache = _make_q4_0_cache(L, E, H, I)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="gelu_tanh",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    gate_up_layer = _dequant_bank(b["gate_up"][layer], H, dev)  # [E, 2I, H]
    down_layer = _dequant_bank(b["down"][layer], I, dev)        # [E, H, I]
    gpu_out = fused_experts_decode_impl(
        hidden, gate_up_layer, down_layer, w, ids.clone(), "gelu_tanh", False
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 2e-2, f"q4_0 bs={bs} rel err {rel.item()}"


@pytest.mark.slow
def test_cpu_decode_q4_0_matches_ggml_mmvq():
    """Sanity: the CPU W4A16 GEMV lands close to the GPU ggml MMVQ (W4A8) kernel the
    offload path uses -- a looser tol since MMVQ quantizes activations to int8."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_q4_0 import fused_experts_gguf_q4_0

    torch.manual_seed(77)
    L, E, H, I, top_k = 2, 16, 2816, 704, 8
    layer, bs = 1, 4
    dev = torch.device("cuda")
    cache = _make_q4_0_cache(L, E, H, I)

    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="gelu_tanh",
        apply_router_weight_on_input=False, num_threads=0, max_tokens=bs, device=dev,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    gate_up_layer = b["gate_up"][layer].to(dev)
    down_layer = b["down"][layer].to(dev)
    gpu_out = fused_experts_gguf_q4_0(
        hidden, gate_up_layer, down_layer, w, ids.clone(), "gelu_tanh"
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 6e-2, f"q4_0 vs ggml-mmvq rel err {rel.item()}"


def test_cpu_moe_decode_q4_0_cuda_graph_replay():
    """Q4_0 CPU path under capture/replay: the host nodes must recompute the GEMV from
    the freshly written pinned routing on each replay (dep flows through pinned buffers)."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    torch.manual_seed(9)
    L, E, H, I, top_k = 2, 8, 2816, 704, 8
    layer, bs = 1, 4
    cache = _make_q4_0_cache(L, E, H, I)

    dev = torch.device("cuda")
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="gelu_tanh",
        apply_router_weight_on_input=False, num_threads=8, max_tokens=bs, device=dev,
    )
    b = cache.bank_sources
    gate_up_layer = _dequant_bank(b["gate_up"][layer], H, dev)
    down_layer = _dequant_bank(b["down"][layer], I, dev)

    def reference(hidden, ids, w):
        return fused_experts_decode_impl(
            hidden, gate_up_layer, down_layer, w, ids.clone(), "gelu_tanh", False
        ).float()

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.randint(0, E, (bs, top_k), device=dev, dtype=torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    ex.decode(layer, hidden, w, ids)  # eager warmup: materialize buffers + task
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out_cap = ex.decode(layer, hidden, w, ids)
    torch.cuda.synchronize()

    for it in range(3):
        torch.manual_seed(600 + it)
        hidden.copy_(torch.randn(bs, H, dtype=torch.bfloat16) * 0.5)
        ids.copy_(torch.randint(0, E, (bs, top_k), dtype=torch.int32))
        w.copy_(torch.rand(bs, top_k, dtype=torch.float32))
        g.replay()
        torch.cuda.synchronize()
        ref = reference(hidden, ids, w)
        rel = (out_cap.float() - ref).abs().max() / (ref.abs().max() + 1e-6)
        assert rel < 2e-2, f"q4_0 replay {it} rel err {rel.item()}"

    print("cpu moe cuda graph replay (q4_0) OK")


@pytest.mark.parametrize("isa", ["scalar", "avx2"])
def test_cpu_decode_mixed_q4_k_q5_1_matches_dequant_then_gpu(monkeypatch, isa):
    """Exercise both mixed CPU dots against dense weights from gguf-py's dequantizer."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    monkeypatch.setenv("FREETOKEN_CPU_MOE_ISA", isa)
    torch.manual_seed(2026)
    L, E, H, I, top_k, bs = 2, 8, 256, 64, 2, 3
    layer = 1
    dev = torch.device("cuda")
    cache = _make_q4_k_q5_1_cache(L, E, H, I, seed=5)
    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="gelu_tanh", apply_router_weight_on_input=False,
        num_threads=4, max_tokens=bs, device=dev, fmt="q4_0", ggml_types=(12, 7),
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    weights = torch.rand(bs, top_k, device=dev, dtype=torch.float32)
    cpu_out = ex.decode(layer, hidden, weights, ids).float()

    banks = cache.bank_sources
    gate_up = _dequant_gguf_bank(banks["gate_up"][layer], 12, H, dev)
    down = _dequant_gguf_bank(banks["down"][layer], 7, I, dev)
    ref = fused_experts_decode_impl(
        hidden, gate_up, down, weights, ids.clone(), "gelu_tanh", False
    ).float()
    rms_rel = torch.sqrt(torch.mean((cpu_out - ref) ** 2)) / (torch.sqrt(torch.mean(ref ** 2)) + 1e-6)
    assert rms_rel < 0.08, f"mixed Q4_K/Q5_1 RMS relative error {rms_rel.item()}"


@pytest.mark.slow
def test_cpu_decode_real_qwen38_q4_k_q5_1_matches_dequant_then_gpu():
    """Check the mixed dots against bytes from the checkpoint that needs them."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    model = os.environ.get(
        "FT_TEST_QWEN38_FLASH_GGUF",
        os.path.expanduser(
            "~/models/Qwen3.8-Flash-Next-Q4/Qwen3.8-Flash-Next-Q4_K-v3.gguf"
        ),
    )
    if not os.path.exists(model):
        pytest.skip(f"checkpoint not present: {model}")

    torch.manual_seed(2038)
    H, I, top_k = 2560, 640, 2
    cache = _load_real_q4_k_q5_1_cache(model, layer=0, experts=(17, 291))
    dev = torch.device("cuda")
    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=4,
        max_tokens=1,
        device=dev,
        fmt="q4_0",
        ggml_types=(12, 7),
    )

    hidden = torch.randn(1, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.tensor([[0, 1]], device=dev, dtype=torch.int32)
    weights = torch.tensor([[0.6, 0.4]], device=dev, dtype=torch.float32)
    got = ex.decode(0, hidden, weights, ids).float()

    gate_up = _dequant_gguf_bank(cache.bank_sources["gate_up"][0], 12, H, dev)
    down = _dequant_gguf_bank(cache.bank_sources["down"][0], 7, I, dev)
    ref = fused_experts_decode_impl(
        hidden, gate_up, down, weights, ids.clone(), "silu", False
    ).float()
    rms_rel = torch.sqrt(torch.mean((got - ref) ** 2)) / (
        torch.sqrt(torch.mean(ref ** 2)) + 1e-6
    )
    assert rms_rel < 0.08, f"real Q4_K/Q5_1 RMS relative error {rms_rel.item()}"


def test_cpu_decode_mixed_q4_k_q5_1_avx2_matches_scalar(monkeypatch):
    """The vectorized dot must preserve the scalar mixed-format result."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    torch.manual_seed(2039)
    H, I, E, top_k = 256, 64, 8, 2
    cache = _make_q4_k_q5_1_cache(1, E, H, I, seed=11)
    dev = torch.device("cuda")
    # Four routes to expert 1 exercise the multi-dot width-4 chunk, while the
    # singleton expert routes keep the grouped and ordinary expert layouts mixed.
    hidden = torch.randn(4, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.tensor([[1, 6], [1, 2], [1, 6], [1, 2]], device=dev, dtype=torch.int32)
    weights = torch.rand(4, top_k, device=dev, dtype=torch.float32)
    outputs = []
    for isa in ("scalar", "avx2"):
        monkeypatch.setenv("FREETOKEN_CPU_MOE_ISA", isa)
        ex = CpuMoeExecutor(
            cache,
            top_k=top_k,
            activation="gelu_tanh",
            apply_router_weight_on_input=False,
            num_threads=4,
            max_tokens=4,
            device=dev,
            fmt="q4_0",
            ggml_types=(12, 7),
        )
        outputs.append(ex.decode(0, hidden, weights, ids).float())

    torch.testing.assert_close(outputs[1], outputs[0], rtol=5e-3, atol=5e-3)


def test_cpu_mixed_q8_1_quantizer_avx2_matches_scalar():
    """AVX2 Q8_1 keeps GGML bytes and lround tie behavior identical to scalar."""
    from freetoken.kernel import _cpu_moe

    torch.manual_seed(2040)
    tie_block = torch.zeros(32, dtype=torch.bfloat16)
    tie_block[:8] = torch.tensor(
        [1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, -0.0], dtype=torch.bfloat16
    )
    tie_block[8:] = torch.randn(24, dtype=torch.float32).to(torch.bfloat16) * 0.25
    values = torch.cat((tie_block, torch.zeros(32, dtype=torch.bfloat16),
                        torch.randn(96, dtype=torch.float32).to(torch.bfloat16)))
    scalar = _cpu_moe._quantize_q8_1_for_test(values, False)
    avx2 = _cpu_moe._quantize_q8_1_for_test(values, True)
    torch.testing.assert_close(avx2, scalar, rtol=0, atol=0)
    # With amax=1, +/-0.5 map to +/-64 under std::lround's away-from-zero rule.
    assert int(scalar[0, 4 + 2].view(torch.int8)) == 64
    assert int(scalar[0, 4 + 3].view(torch.int8)) == -64


def test_cpu_moe_decode_mixed_q4_k_q5_1_cuda_graph_replay():
    """Mixed-format graph replay must refresh activations and routed experts each time."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    torch.manual_seed(2027)
    L, E, H, I, top_k, bs = 2, 8, 256, 64, 2, 2
    layer = 1
    dev = torch.device("cuda")
    cache = _make_q4_k_q5_1_cache(L, E, H, I, seed=7)
    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="gelu_tanh", apply_router_weight_on_input=False,
        num_threads=4, max_tokens=bs, device=dev, fmt="q4_0", ggml_types=(12, 7),
    )
    banks = cache.bank_sources
    gate_up = _dequant_gguf_bank(banks["gate_up"][layer], 12, H, dev)
    down = _dequant_gguf_bank(banks["down"][layer], 7, I, dev)

    def reference(x, w, ids):
        return fused_experts_decode_impl(
            x, gate_up, down, w, ids.clone(), "gelu_tanh", False
        ).float()

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.randint(0, E, (bs, top_k), device=dev, dtype=torch.int32)
    weights = torch.rand(bs, top_k, device=dev, dtype=torch.float32)
    ex.decode(layer, hidden, weights, ids)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out_cap = ex.decode(layer, hidden, weights, ids)
    torch.cuda.synchronize()

    for it in range(3):
        torch.manual_seed(810 + it)
        hidden.copy_(torch.randn(bs, H, dtype=torch.bfloat16, device=dev) * 0.5)
        ids.copy_(torch.randint(0, E, (bs, top_k), dtype=torch.int32, device=dev))
        weights.copy_(torch.rand(bs, top_k, dtype=torch.float32, device=dev))
        graph.replay()
        torch.cuda.synchronize()
        ref = reference(hidden, weights, ids)
        rms_rel = torch.sqrt(torch.mean((out_cap.float() - ref) ** 2)) / (
            torch.sqrt(torch.mean(ref ** 2)) + 1e-6
        )
        assert rms_rel < 0.08, f"mixed Q4_K/Q5_1 replay {it} RMS rel {rms_rel.item()}"


def test_cpu_decode_mixed_q4_k_q5_1_grouped_cross_token_duplicates():
    """Grouped mixed decode keeps deliberate cross-token expert reuse correct."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    torch.manual_seed(2041)
    L, E, H, I, top_k, bs = 1, 8, 256, 64, 2, 4
    dev = torch.device("cuda")
    cache = _make_q4_k_q5_1_cache(L, E, H, I, seed=13)
    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="gelu_tanh",
        apply_router_weight_on_input=False,
        num_threads=4,
        max_tokens=bs,
        device=dev,
        fmt="q4_0",
        ggml_types=(12, 7),
    )
    banks = cache.bank_sources
    gate_up = _dequant_gguf_bank(banks["gate_up"][0], 12, H, dev)
    down = _dequant_gguf_bank(banks["down"][0], 7, I, dev)
    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.tensor([[1, 2], [3, 1], [4, 2], [3, 5]], device=dev, dtype=torch.int32)
    weights = torch.rand(bs, top_k, device=dev, dtype=torch.float32)
    got = ex.decode(0, hidden, weights, ids).float()
    ref = fused_experts_decode_impl(
        hidden, gate_up, down, weights, ids.clone(), "gelu_tanh", False
    ).float()
    rms_rel = torch.sqrt(torch.mean((got - ref) ** 2)) / (torch.sqrt(torch.mean(ref ** 2)) + 1e-6)
    assert rms_rel < 0.08, f"grouped mixed Q4_K/Q5_1 RMS rel {rms_rel.item()}"


def test_cpu_decode_mixed_q4_k_q5_1_grouped_duplicate_graph_replay():
    """Graph replay refreshes changing duplicate groups and their route reductions."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    torch.manual_seed(2042)
    L, E, H, I, top_k, bs = 1, 8, 256, 64, 2, 4
    dev = torch.device("cuda")
    cache = _make_q4_k_q5_1_cache(L, E, H, I, seed=14)
    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="gelu_tanh",
        apply_router_weight_on_input=False,
        num_threads=4,
        max_tokens=bs,
        device=dev,
        fmt="q4_0",
        ggml_types=(12, 7),
    )
    banks = cache.bank_sources
    gate_up = _dequant_gguf_bank(banks["gate_up"][0], 12, H, dev)
    down = _dequant_gguf_bank(banks["down"][0], 7, I, dev)

    def reference(x, w, routed):
        return fused_experts_decode_impl(
            x, gate_up, down, w, routed.clone(), "gelu_tanh", False
        ).float()

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.tensor([[1, 2], [3, 1], [4, 2], [3, 5]], device=dev, dtype=torch.int32)
    weights = torch.rand(bs, top_k, device=dev, dtype=torch.float32)
    ex.decode(0, hidden, weights, ids)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ex.decode(0, hidden, weights, ids)
    torch.cuda.synchronize()

    routed_cases = (
        ((7, 0), (6, 7), (0, 1), (2, 6)),
        ((4, 4), (5, 1), (4, 5), (1, 4)),
        ((2, 3), (2, 6), (7, 3), (6, 2)),
    )
    for it, routed in enumerate(routed_cases):
        torch.manual_seed(2050 + it)
        hidden.copy_(torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5)
        ids.copy_(torch.tensor(routed, device=dev, dtype=torch.int32))
        weights.copy_(torch.rand(bs, top_k, device=dev, dtype=torch.float32))
        graph.replay()
        torch.cuda.synchronize()
        ref = reference(hidden, weights, ids)
        rms_rel = torch.sqrt(torch.mean((captured.float() - ref) ** 2)) / (
            torch.sqrt(torch.mean(ref ** 2)) + 1e-6
        )
        assert rms_rel < 0.08, f"grouped replay {it} RMS rel {rms_rel.item()}"


@pytest.mark.slow
def test_cpu_decode_real_qwen38_from_bounded_host_slots_graph_replay():
    """Disk-backed host slots may evict between replays without changing results."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.disk_store import GgufExpertStore
    from freetoken.moe.fused import fused_experts_decode_impl
    from freetoken.moe.host_tier import HostExpertCache

    model = os.environ.get(
        "FT_TEST_QWEN38_FLASH_GGUF",
        os.path.expanduser(
            "~/models/Qwen3.8-Flash-Next-Q4/Qwen3.8-Flash-Next-Q4_K-v3.gguf"
        ),
    )
    if not os.path.exists(model):
        pytest.skip(f"checkpoint not present: {model}")

    class Cache:
        pass

    torch.manual_seed(2040)
    H, I, E, L, top_k = 2560, 640, 512, 48, 2
    dev = torch.device("cuda")
    with GgufExpertStore(
        model, E, {"gate_up": 12, "down": 7}, L
    ) as store, HostExpertCache(store, E, capacity=top_k) as tier:
        cache = Cache()
        cache.quant_format = "q4_0"
        cache.bank_sources = {
            name: [bank] * L for name, bank in tier.banks.items()
        }
        cache.num_layers = L
        cache.num_experts = E
        cache.decode_target = "cpu"
        cache.cpu_executor = None
        cache.host_tier = tier
        cache._admit_error = None
        ex = CpuMoeExecutor(
            cache,
            top_k=top_k,
            activation="silu",
            apply_router_weight_on_input=False,
            num_threads=8,
            max_tokens=1,
            device=dev,
            fmt="q4_0",
            ggml_types=(12, 7),
        )

        hidden = torch.randn(1, H, device=dev, dtype=torch.bfloat16) * 0.5
        ids = torch.tensor([[17, 291]], device=dev, dtype=torch.int32)
        weights = torch.tensor([[0.6, 0.4]], device=dev, dtype=torch.float32)
        ex.decode(0, hidden, weights, ids)
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = ex.decode(0, hidden, weights, ids)
        torch.cuda.synchronize()

        for routed in ((17, 291), (42, 300), (291, 42)):
            ids.copy_(torch.tensor([routed], device=dev, dtype=torch.int32))
            graph.replay()
            torch.cuda.synchronize()
            assert cache._admit_error is None

            slots = [tier.slot_of(0, expert) for expert in routed]
            assert all(slot >= 0 for slot in slots)
            gate_up_packed = tier.banks["gate_up"][slots].clone()
            down_packed = tier.banks["down"][slots].clone()
            gate_up = _dequant_gguf_bank(gate_up_packed, 12, H, dev)
            down = _dequant_gguf_bank(down_packed, 7, I, dev)
            local_ids = torch.tensor([[0, 1]], device=dev, dtype=torch.int32)
            ref = fused_experts_decode_impl(
                hidden, gate_up, down, weights, local_ids, "silu", False
            ).float()
            rms_rel = torch.sqrt(torch.mean((captured.float() - ref) ** 2)) / (
                torch.sqrt(torch.mean(ref ** 2)) + 1e-6
            )
            assert rms_rel < 0.08, (
                f"bounded host replay {routed} RMS relative error {rms_rel.item()}"
            )


if __name__ == "__main__":
    for bs in (1, 3, 8):
        test_cpu_decode_q4_0_matches_dequant_then_gpu(bs)
        print(f"q4_0 bs={bs} OK")
    test_cpu_decode_q4_0_matches_ggml_mmvq()
    print("q4_0 vs ggml-mmvq OK")
    test_cpu_moe_decode_q4_0_cuda_graph_replay()
