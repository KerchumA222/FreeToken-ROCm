"""Per-layer expert types: a checkpoint whose expert layers differ in ggml type (Unsloth's
dynamic quants upgrade a few layers) is served from banks sized for the widest layer.

A narrower layer's expert is one contiguous run at the start of its slot, and the MoE
kernels step between experts by the bank's stride rather than the layer's packed size.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")

from freetoken.moe.disk_store import GgufExpertStore  # noqa: E402

E, I, H, L = 4, 64, 32, 3
Q4_0 = int(gguf.GGMLQuantizationType.Q4_0)
Q8_0 = int(gguf.GGMLQuantizationType.Q8_0)
WIDE_LAYER = 1  # gate/up at Q8_0; every other expert tensor Q4_0


def _row_bytes(n: int, qtype: int) -> int:
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(qtype)]
    return n // block * size


@pytest.fixture(scope="module")
def mixed_gguf(tmp_path_factory):
    path = tmp_path_factory.mktemp("gguf_mixed") / "experts.gguf"
    w = gguf.GGUFWriter(str(path), "qwen35moe")
    w.add_uint32("qwen35moe.expert_count", E)
    payload, rng = {}, np.random.default_rng(0)
    for layer in range(L):
        gu_type = Q8_0 if layer == WIDE_LAYER else Q4_0
        for suffix, rows, cols, qtype in (
            ("ffn_gate_exps.weight", E * I, H, gu_type),
            ("ffn_up_exps.weight", E * I, H, gu_type),
            ("ffn_down_exps.weight", E * H, I, Q4_0),
        ):
            data = rng.integers(0, 256, size=(rows, _row_bytes(cols, qtype)), dtype=np.uint8)
            name = f"blk.{layer}.{suffix}"
            payload[name] = data
            w.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType(qtype))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path), payload


def _per_layer():
    return {
        "gate_up": {l: (Q8_0 if l == WIDE_LAYER else Q4_0) for l in range(L)},
        "down": {l: Q4_0 for l in range(L)},
    }


def _expert(payload, layer, e, bank):
    if bank == "down":
        return payload[f"blk.{layer}.ffn_down_exps.weight"].reshape(E, -1)[e]
    g = payload[f"blk.{layer}.ffn_gate_exps.weight"].reshape(E, -1)[e]
    u = payload[f"blk.{layer}.ffn_up_exps.weight"].reshape(E, -1)[e]
    return np.concatenate([g, u])


def test_slots_are_sized_for_the_widest_layer(mixed_gguf):
    path, _ = mixed_gguf
    with GgufExpertStore(path, E, _per_layer()) as store:
        assert store.row_shape("gate_up") == (2 * I, _row_bytes(H, Q8_0))
        assert store.expert_bytes("gate_up") == 2 * I * _row_bytes(H, Q8_0)
        assert store.expert_bytes("gate_up", layer=0) == 2 * I * _row_bytes(H, Q4_0)
        assert store.requantized_layers() == {}


def test_a_single_bank_type_still_flags_the_odd_layer(mixed_gguf):
    path, _ = mixed_gguf
    with GgufExpertStore(path, E, {"gate_up": Q4_0, "down": Q4_0}) as store:
        assert store.requantized_layers() == {"gate_up": (WIDE_LAYER,)}


def test_a_narrow_expert_lands_at_the_start_of_a_wide_slot(mixed_gguf):
    path, payload = mixed_gguf
    with GgufExpertStore(path, E, _per_layer()) as store:
        slot = np.full(store.expert_bytes("gate_up"), 0xAB, dtype=np.uint8)
        store.read_expert("gate_up", 0, 2, slot)
        want = _expert(payload, 0, 2, "gate_up")
        assert np.array_equal(slot[: want.size], want)
        assert (slot[want.size:] == 0xAB).all()


def test_read_layer_places_each_expert_as_one_run(mixed_gguf):
    path, payload = mixed_gguf
    with GgufExpertStore(path, E, _per_layer()) as store:
        rows, width = store.row_shape("gate_up")
        for layer in range(L):
            dst = np.zeros((E, rows, width), dtype=np.uint8)
            store.read_layer("gate_up", layer, dst)
            for e in range(E):
                want = _expert(payload, layer, e, "gate_up")
                assert np.array_equal(dst[e].reshape(-1)[: want.size], want), (layer, e)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GGUF MoE kernels")
def test_kernel_reads_a_narrow_layer_out_of_wide_slots():
    """The strided per-layer view must give exactly the contiguous bank's output."""
    from freetoken.layers.quantization.moe.gguf import _typed_view
    from freetoken.moe.fused_q4_0 import fused_experts_gguf

    torch.manual_seed(0)
    E_, I_, H_, tokens, k = 8, 256, 512, 5, 2
    gu_rb, dn_rb = _row_bytes(H_, Q4_0), _row_bytes(I_, Q4_0)
    wide_gu, wide_dn = _row_bytes(H_, Q8_0), _row_bytes(I_, Q8_0)

    def q4(rows, cols):  # valid Q4_0 blocks: a small fp16 scale, random nibbles
        n = rows * cols // 32
        blocks = np.zeros((n, 18), dtype=np.uint8)
        blocks[:, :2] = np.frombuffer(np.full(n, 0.01, np.float16).tobytes(), np.uint8).reshape(n, 2)
        blocks[:, 2:] = np.random.default_rng(rows).integers(0, 256, (n, 16), dtype=np.uint8)
        return torch.from_numpy(blocks.reshape(rows, cols // 32 * 18))

    gate_up = torch.stack([q4(2 * I_, H_) for _ in range(E_)]).cuda()   # [E, 2I, gu_rb]
    down = torch.stack([q4(H_, I_) for _ in range(E_)]).cuda()          # [E, H, dn_rb]
    wide_gate_up = torch.zeros(E_, 2 * I_, wide_gu, dtype=torch.uint8, device="cuda")
    wide_down = torch.zeros(E_, H_, wide_dn, dtype=torch.uint8, device="cuda")
    wide_gate_up.view(E_, -1)[:, : gate_up[0].numel()] = gate_up.view(E_, -1)
    wide_down.view(E_, -1)[:, : down[0].numel()] = down.view(E_, -1)

    x = torch.randn(tokens, H_, dtype=torch.float16, device="cuda")
    ids = torch.randint(0, E_, (tokens, k), dtype=torch.int32, device="cuda")
    w = torch.softmax(torch.randn(tokens, k, device="cuda"), -1).half()
    ref = fused_experts_gguf(x, gate_up, down, w, ids, "silu", gate_up_type=Q4_0, down_type=Q4_0)
    got = fused_experts_gguf(
        x, _typed_view(wide_gate_up, 2 * I_, gu_rb), _typed_view(wide_down, H_, dn_rb), w, ids,
        "silu", gate_up_type=Q4_0, down_type=Q4_0,
    )
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
