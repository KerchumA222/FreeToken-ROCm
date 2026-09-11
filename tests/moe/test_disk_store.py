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

from tests.moe.conftest import E, I, H, L, Q4_0, BLOCK, TYPE_SIZE, GU_BYTES, DN_BYTES  # noqa: E402


def test_reads_match_the_bank_slices(tiny_gguf):
    path, payload = tiny_gguf
    with GgufExpertStore(path, E, {"gate_up": Q4_0, "down": Q4_0}) as store:
        assert set(store.banks) == {"gate_up", "down"}
        assert store.layers("gate_up") == tuple(range(L))
        assert store.requantized_layers() == {}

        gu_bytes, dn_bytes = GU_BYTES, DN_BYTES
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


def test_extents_are_contiguous_and_tile_the_expert(tiny_gguf):
    path, _ = tiny_gguf
    with GgufExpertStore(path, E, {"gate_up": Q4_0, "down": Q4_0}) as store:
        for bank in ("gate_up", "down"):
            ext = store.extents(bank, 1, 2)
            assert sum(x.nbytes for x in ext) == store.expert_bytes(bank)
            # destination offsets tile the row block end to end, no gaps or overlap
            assert [x.dst_offset for x in ext] == list(
                np.cumsum([0] + [x.nbytes for x in ext[:-1]])
            )


def test_offsets_agree_with_the_mmap_view(tiny_gguf):
    """The store preads; the reader mmaps. They must land on the same bytes."""
    path, _ = tiny_gguf
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


def test_rejects_an_out_of_range_expert(tiny_gguf):
    path, _ = tiny_gguf
    with GgufExpertStore(path, E, {"gate_up": Q4_0, "down": Q4_0}) as store:
        with pytest.raises(IndexError):
            store.extents("gate_up", 0, E)


def test_reports_layers_that_would_be_requantized(tiny_gguf):
    """A layer whose on-disk type differs from the bank type is converted at load,
    so its file bytes are not its bank bytes and it cannot be served from disk."""
    path, _ = tiny_gguf
    with GgufExpertStore(path, E, {"gate_up": int(gguf.GGMLQuantizationType.Q5_1),
                                   "down": Q4_0}) as store:
        rq = store.requantized_layers()
        assert rq == {"gate_up": tuple(range(L))}


def test_read_layer_fills_the_whole_bank(tiny_gguf, expected_expert):
    """Prefill copies a layer at a time, so the store has to produce the layer's
    [E, rows, row_bytes] block -- with each part scattered across experts correctly."""
    import numpy as np

    path, _ = tiny_gguf
    with GgufExpertStore(path, E, {"gate_up": Q4_0, "down": Q4_0}) as store:
        for bank in ("gate_up", "down"):
            rows, row_bytes = store.row_shape(bank)
            dst = np.zeros((E, rows, row_bytes), dtype=np.uint8)
            for layer in range(L):
                dst[:] = 0
                # one read per part, not per expert
                assert store.read_layer(bank, layer, dst) == (2 if bank == "gate_up" else 1)
                for e in range(E):
                    assert np.array_equal(
                        dst[e].reshape(-1), expected_expert(bank, layer, e)
                    ), (bank, layer, e)


def test_read_layer_rejects_a_mis_shaped_destination(tiny_gguf):
    import numpy as np

    path, _ = tiny_gguf
    with GgufExpertStore(path, E, {"gate_up": Q4_0, "down": Q4_0}) as store:
        rows, row_bytes = store.row_shape("down")
        with pytest.raises(ValueError):
            store.read_layer("down", 0, np.zeros((E, rows, row_bytes + 1), np.uint8))
        with pytest.raises(TypeError):
            store.read_layer("down", 0, np.zeros((E, rows, row_bytes), np.int8))


def test_reads_match_the_bank_slices_across_a_split_set(split_gguf):
    """A split GGUF puts each layer's experts in a different file, and every shard's
    ``data_offset`` is relative to its own. Reading an expert therefore has to pair the
    offset with the shard it came from; pairing it with any other file lands on
    well-formed bytes belonging to some other tensor, which is why this is a test rather
    than an assertion."""
    path, payload = split_gguf
    with GgufExpertStore(path, E, {"gate_up": Q4_0, "down": Q4_0}) as store:
        assert store.layers("gate_up") == tuple(range(L))
        assert store.requantized_layers() == {}
        # the parts really are spread over distinct files, or this proves nothing
        shards = {store.extents("down", layer, 0)[0].path for layer in range(L)}
        assert len(shards) == L, f"expected one shard per layer, got {shards}"

        for layer in range(L):
            g = payload[f"blk.{layer}.ffn_gate_exps.weight"].reshape(E, I, GU_BYTES)
            u = payload[f"blk.{layer}.ffn_up_exps.weight"].reshape(E, I, GU_BYTES)
            d = payload[f"blk.{layer}.ffn_down_exps.weight"].reshape(E, H, DN_BYTES)
            for e in range(E):
                buf = bytearray(store.expert_bytes("gate_up"))
                assert store.read_expert("gate_up", layer, e, buf) == 2
                want = np.concatenate([g[e].reshape(-1), u[e].reshape(-1)])
                assert np.array_equal(np.frombuffer(bytes(buf), np.uint8), want), (
                    f"gate_up layer {layer} expert {e}"
                )

                buf = bytearray(store.expert_bytes("down"))
                assert store.read_expert("down", layer, e, buf) == 1
                assert np.array_equal(
                    np.frombuffer(bytes(buf), np.uint8), d[e].reshape(-1)
                ), f"down layer {layer} expert {e}"
