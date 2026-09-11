"""GgufExpertStore addresses individual experts inside a real GGUF file.

Builds a small but structurally faithful GGUF -- three routed-expert tensors per
layer, quantized block bytes, the same [E * rows_per_expert, row_bytes] layout
llama.cpp writes -- and checks that the store's byte ranges reproduce exactly the
slices ``load_q4_0_expert_sources`` copies into the pinned host banks.
"""
from __future__ import annotations

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from freetoken.models.gguf.reader import iter_gguf_tensors  # noqa: E402
from freetoken.moe.disk_store import GgufExpertStore  # noqa: E402

E, I, H, L = 4, 64, 32, 3          # experts, intermediate, hidden, layers
Q4_0 = int(gguf.GGMLQuantizationType.Q4_0)
BLOCK, TYPE_SIZE = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q4_0]


def _blocks(rows: int, n_fast: int, seed: int) -> np.ndarray:
    """Random packed Q4_0 block bytes shaped the way ggml stores them."""
    rb = n_fast // BLOCK * TYPE_SIZE
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(rows, rb), dtype=np.uint8)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("gguf") / "experts.gguf"
    w = gguf.GGUFWriter(str(path), "qwen35moe")
    w.add_uint32("qwen35moe.expert_count", E)
    payload = {}
    for layer in range(L):
        # Each tensor is [E * rows_per_expert, row_bytes] of packed blocks, exactly
        # how llama.cpp lays routed experts out. raw_shape defaults to that byte
        # shape; the writer derives the ggml dims from it.
        for i, (suffix, rows, n_fast) in enumerate((
            ("ffn_gate_exps.weight", E * I, H),
            ("ffn_up_exps.weight", E * I, H),
            ("ffn_down_exps.weight", E * H, I),
        )):
            data = _blocks(rows, n_fast, seed=layer * 10 + i)
            name = f"blk.{layer}.{suffix}"
            payload[name] = data
            w.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path), payload


def test_reads_match_the_bank_slices(model):
    path, payload = model
    with GgufExpertStore(path, E, {"gate_up": Q4_0, "down": Q4_0}) as store:
        assert set(store.banks) == {"gate_up", "down"}
        assert store.layers("gate_up") == tuple(range(L))
        assert store.requantized_layers() == {}

        gu_bytes = H // BLOCK * TYPE_SIZE
        dn_bytes = I // BLOCK * TYPE_SIZE
        assert store.expert_bytes("gate_up") == 2 * I * gu_bytes
        assert store.expert_bytes("down") == H * dn_bytes

        for layer in range(L):
            g = payload[f"blk.{layer}.ffn_gate_exps.weight"].reshape(E, I, gu_bytes)
            u = payload[f"blk.{layer}.ffn_up_exps.weight"].reshape(E, I, gu_bytes)
            d = payload[f"blk.{layer}.ffn_down_exps.weight"].reshape(E, H, dn_bytes)
            for e in range(E):
                buf = bytearray(store.expert_bytes("gate_up"))
                assert store.read_expert("gate_up", layer, e, buf) == 2
                # gate rows first, then up rows -- the order silu_and_mul expects
                want = np.concatenate([g[e].reshape(-1), u[e].reshape(-1)])
                assert np.array_equal(np.frombuffer(bytes(buf), np.uint8), want)

                buf = bytearray(store.expert_bytes("down"))
                assert store.read_expert("down", layer, e, buf) == 1
                assert np.array_equal(
                    np.frombuffer(bytes(buf), np.uint8), d[e].reshape(-1)
                )


def test_extents_are_contiguous_and_tile_the_expert(model):
    path, _ = model
    with GgufExpertStore(path, E, {"gate_up": Q4_0, "down": Q4_0}) as store:
        for bank in ("gate_up", "down"):
            ext = store.extents(bank, 1, 2)
            assert sum(x.nbytes for x in ext) == store.expert_bytes(bank)
            # destination offsets tile the row block end to end, no gaps or overlap
            assert [x.dst_offset for x in ext] == list(
                np.cumsum([0] + [x.nbytes for x in ext[:-1]])
            )


def test_offsets_agree_with_the_mmap_view(model):
    """The store preads; the reader mmaps. They must land on the same bytes."""
    path, _ = model
    with GgufExpertStore(path, E, {"gate_up": Q4_0, "down": Q4_0}) as store:
        seen = 0
        for t in iter_gguf_tensors(path):
            if not t.name.startswith("blk."):
                continue
            import os

            fd = os.open(path, os.O_RDONLY)
            try:
                raw = os.pread(fd, t.rows * t.row_bytes, t.data_offset)
            finally:
                os.close(fd)
            assert np.array_equal(
                np.frombuffer(raw, np.uint8).reshape(t.rows, t.row_bytes),
                t.packed().numpy(),
            ), t.name
            seen += 1
        assert seen == 3 * L


def test_rejects_an_out_of_range_expert(model):
    path, _ = model
    with GgufExpertStore(path, E, {"gate_up": Q4_0, "down": Q4_0}) as store:
        with pytest.raises(IndexError):
            store.extents("gate_up", 0, E)


def test_reports_layers_that_would_be_requantized(model):
    """A layer whose on-disk type differs from the bank type is converted at load,
    so its file bytes are not its bank bytes and it cannot be served from disk."""
    path, _ = model
    with GgufExpertStore(path, E, {"gate_up": int(gguf.GGMLQuantizationType.Q5_1),
                                   "down": Q4_0}) as store:
        rq = store.requantized_layers()
        assert rq == {"gate_up": tuple(range(L))}
