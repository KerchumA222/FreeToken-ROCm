"""Shared fixtures for the MoE disk/host tier tests."""
from __future__ import annotations

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

# experts, intermediate, hidden, layers -- small but structurally faithful
E, I, H, L = 4, 64, 32, 3
Q4_0 = int(gguf.GGMLQuantizationType.Q4_0)
BLOCK, TYPE_SIZE = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q4_0]
GU_BYTES = H // BLOCK * TYPE_SIZE
DN_BYTES = I // BLOCK * TYPE_SIZE


def _blocks(rows: int, n_fast: int, seed: int) -> np.ndarray:
    """Random packed Q4_0 block bytes shaped the way ggml stores them."""
    rb = n_fast // BLOCK * TYPE_SIZE
    return np.random.default_rng(seed).integers(0, 256, size=(rows, rb), dtype=np.uint8)


@pytest.fixture(scope="session")
def tiny_gguf(tmp_path_factory):
    """(path, {tensor_name: packed bytes}) for a model with routed experts.

    Each tensor is ``[E * rows_per_expert, row_bytes]`` of packed blocks, exactly how
    llama.cpp lays routed experts out, so the byte arithmetic under test is the real
    arithmetic.
    """
    path = tmp_path_factory.mktemp("gguf") / "experts.gguf"
    w = gguf.GGUFWriter(str(path), "qwen35moe")
    w.add_uint32("qwen35moe.expert_count", E)
    payload = {}
    for layer in range(L):
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


@pytest.fixture(scope="session")
def expected_expert(tiny_gguf):
    """(bank, layer, expert) -> the exact bytes the pinned host bank would hold."""
    _path, payload = tiny_gguf

    def get(bank: str, layer: int, e: int) -> np.ndarray:
        if bank == "gate_up":
            g = payload[f"blk.{layer}.ffn_gate_exps.weight"].reshape(E, I, GU_BYTES)
            u = payload[f"blk.{layer}.ffn_up_exps.weight"].reshape(E, I, GU_BYTES)
            return np.concatenate([g[e].reshape(-1), u[e].reshape(-1)])
        d = payload[f"blk.{layer}.ffn_down_exps.weight"].reshape(E, H, DN_BYTES)
        return d[e].reshape(-1)

    return get


@pytest.fixture(scope="session")
def split_gguf(tmp_path_factory):
    """The same model as :func:`tiny_gguf`, written as a llama.cpp split set.

    One file per layer, named with the ``-00001-of-0000N.gguf`` convention the reader
    resolves. Every shard carries its own tensor table with its own ``data_offset``
    base, which is what makes a split set able to catch address arithmetic that
    conflates "the offset" with "the file the offset is in".
    """
    d = tmp_path_factory.mktemp("gguf_split")
    payload = {}
    for layer in range(L):
        path = d / f"experts-{layer + 1:05d}-of-{L:05d}.gguf"
        w = gguf.GGUFWriter(str(path), "qwen35moe")
        if layer == 0:  # shard 1 carries the model KV
            w.add_uint32("qwen35moe.expert_count", E)
        for i, (suffix, rows, n_fast) in enumerate((
            ("ffn_gate_exps.weight", E * I, H),
            ("ffn_up_exps.weight", E * I, H),
            ("ffn_down_exps.weight", E * H, I),
        )):
            data = _blocks(rows, n_fast, seed=100 + layer * 10 + i)
            name = f"blk.{layer}.{suffix}"
            payload[name] = data
            w.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
    return str(d / f"experts-00001-of-{L:05d}.gguf"), payload
