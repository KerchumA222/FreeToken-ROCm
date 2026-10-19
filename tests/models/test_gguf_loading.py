"""GGUF reading on the CPU: split files, the Q2_0 codebook re-typing, the dense escape hatch, and (when the files are on disk) the configs
and tokenizers of the GGUFs we serve."""

from __future__ import annotations

import glob
import os
import struct

import numpy as np
import pytest
import torch

from freetoken.layers.quantization.ggml_types import Q2_0_SYM, dequantize_cpu, row_bytes

_ALIGN = 32
_U32, _U16, _STR, _ARR = 4, 2, 8, 9


def _str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, value) -> bytes:
    if isinstance(value, str):
        return _str(key) + struct.pack("<I", _STR) + _str(value)
    if isinstance(value, tuple):  # (u16,)
        return _str(key) + struct.pack("<IH", _U16, value[0])
    if isinstance(value, list):  # array of strings
        return _str(key) + struct.pack("<IIQ", _ARR, _STR, len(value)) + b"".join(_str(v) for v in value)
    return _str(key) + struct.pack("<II", _U32, value)


def _write_gguf(path, kv: dict, tensors: list[tuple[str, int, tuple[int, ...], bytes]]) -> None:
    """A minimal GGUF v3 writer: ``tensors`` are (name, ggml type, ne in ggml order, packed bytes)."""
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + b"".join(_kv(k, v) for k, v in kv.items())
    infos, data = b"", b""
    for name, ggml_type, ne, raw in tensors:
        data += b"\0" * (-len(data) % _ALIGN)
        infos += _str(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", n) for n in ne) + struct.pack("<IQ", ggml_type, len(data))
        data += raw
    body = head + infos
    with open(path, "wb") as f:
        f.write(body + b"\0" * (-len(body) % _ALIGN) + data)


def _q8_0(rows: int, k: int, seed: int) -> bytes:
    raw = np.random.default_rng(seed).integers(0, 256, (rows, k // 32, 34), dtype=np.uint8)
    raw[..., :2] = np.array([np.float16(0.5).view(np.uint16)], dtype=np.uint16).view(np.uint8)
    return raw.tobytes()


def _clear_caches():
    from freetoken.models.gguf import reader

    for fn in (reader._reader, reader.load_gguf_metadata, reader.gguf_split_paths, reader.gguf_tensor_index):
        fn.cache_clear()


@pytest.fixture
def split_gguf(tmp_path):
    """A 3-way split: split 1 holds the metadata and one tensor, each later split repeats general.architecture and the split keys."""
    stem = tmp_path / "m"
    tensors = {
        1: [("blk.0.attn_q.weight", 8, (64, 4), _q8_0(4, 64, 1))],
        2: [("token_embd.weight", 8, (64, 7), _q8_0(7, 64, 2)), ("blk.0.ffn_down.weight", 8, (64, 3), _q8_0(3, 64, 3))],
        3: [("output.weight", 8, (64, 7), _q8_0(7, 64, 4))],
    }
    paths = []
    for no in (1, 2, 3):
        kv = {"general.architecture": "qwen35moe", "split.no": (no - 1,), "split.count": (3,)}
        if no == 1:
            kv["tokenizer.ggml.tokens"] = [f"t{i}" for i in range(7)]
        path = f"{stem}-{no:05d}-of-00003.gguf"
        _write_gguf(path, kv, tensors[no])
        paths.append(path)
    _clear_caches()
    yield paths, tensors
    _clear_caches()


def test_split_index_spans_every_split(split_gguf):
    from freetoken.models.gguf.reader import gguf_architecture, gguf_config_source, gguf_split_paths, gguf_tensor_index

    paths, tensors = split_gguf
    for any_split in paths:
        assert gguf_split_paths(any_split) == tuple(paths)
        assert gguf_config_source(any_split) == paths[0]
        index = gguf_tensor_index(any_split)
        for no, entries in tensors.items():
            for name, ggml_type, ne, raw in entries:
                info = index[name]
                assert (info.path, info.ggml_type, info.ne) == (paths[no - 1], ggml_type, ne)
                assert bytes(info.data()) == raw
    assert gguf_architecture(paths[0]) == "qwen35moe"


def test_split_vocab_and_tied_head_come_from_later_splits(split_gguf):
    from freetoken.models.gguf.config import build_gguf_shim

    paths, _ = split_gguf
    shim = build_gguf_shim(paths[0])
    assert shim.vocab_size == 7  # token_embd sits in split 2
    assert not shim.tie_word_embeddings  # output.weight sits in split 3


@pytest.mark.parametrize("codebook", [None, "symmetric_odd"])
def test_q2_0_codebook_retypes_tensors(tmp_path, codebook):
    from freetoken.models.gguf.reader import gguf_tensor_index

    raw = np.random.default_rng(0).integers(0, 256, (2, 2, 18), dtype=np.uint8).tobytes()
    kv = {"general.architecture": "qwen4exp"} | ({"freetoken.q2_0.codebook": codebook} if codebook else {})
    path = str(tmp_path / "q2.gguf")
    _write_gguf(path, kv, [("blk.0.ffn_down_exps.weight", 42, (128, 2), raw)])
    _clear_caches()
    try:
        assert gguf_tensor_index(path)["blk.0.ffn_down_exps.weight"].ggml_type == (Q2_0_SYM if codebook else 42)
    finally:
        _clear_caches()


def test_q2_0_sym_reference_codebook():
    block = np.zeros(18, dtype=np.uint8)
    block[:2] = np.array([np.float16(0.25).view(np.uint16)], dtype=np.uint16).view(np.uint8)
    block[2] = 0b11_10_01_00  # codes 0, 1, 2, 3
    np.testing.assert_array_equal(dequantize_cpu(block, Q2_0_SYM, 64)[0, :4], [-0.75, -0.25, 0.25, 0.75])
    np.testing.assert_array_equal(dequantize_cpu(block, 42, 64)[0, :4], [-0.25, 0.0, 0.25, 0.5])


def test_dense_escape_hatch(monkeypatch):
    """FT_GGUF_DENSE_MODULES serves the named modules as bf16 Linears, consistently in the quant config and the weight iterator."""
    from freetoken.layers.quantization import GGUFConfig, NameMap
    from freetoken.models.gguf.reader import GgufTensorInfo
    from freetoken.models.gguf.tensors import fused

    k, rows = 64, 5
    raw = np.frombuffer(_q8_0(rows, k, 7), dtype=np.uint8).reshape(rows, -1)
    info = GgufTensorInfo("blk.0.ffn_down.weight", "", 0, raw.size, 8, (k, rows))
    cfg = GGUFConfig({"blk.0.ffn_down.weight": 8}, lambda name, _: ("blk.0.ffn_down",), name_map=NameMap())
    prefix = "model.layers.0.mlp.down_proj"

    monkeypatch.delenv("FT_GGUF_DENSE_MODULES", raising=False)
    assert cfg.scheme_for(prefix) is not None
    packed = dict(fused(prefix, [info], [torch.from_numpy(raw)], k))
    assert packed[f"{prefix}.weight"].dtype == torch.uint8

    monkeypatch.setenv("FT_GGUF_DENSE_MODULES", "mlp.down_proj")
    cfg = GGUFConfig({"blk.0.ffn_down.weight": 8}, lambda name, _: ("blk.0.ffn_down",), name_map=NameMap())
    assert cfg.scheme_for(prefix) is None
    dense = dict(fused(prefix, [info], [torch.from_numpy(raw)], k))
    torch.testing.assert_close(dense[f"{prefix}.weight"], torch.from_numpy(dequantize_cpu(raw, 8, k)).to(torch.bfloat16))
    assert row_bytes(8, k) == raw.shape[1]


# ---------------------------------------------------------------------------------------------------------------------
# The GGUFs we serve, when they are on disk

_ROOT = os.path.expanduser(os.environ.get("FT_GGUF_MODELS", "~/models"))


def _model(pattern: str) -> str:
    found = sorted(glob.glob(os.path.join(_ROOT, pattern)))
    if not found:
        pytest.skip(f"no {pattern} under {_ROOT}")
    return found[0]


_QWEN36 = os.environ.get(
    "FT_GGUF_QWEN36",
    "Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-MTP-Q4exp.gguf",
)
_FLASH_NEXT = [
    "Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/IQ1_M/*-00001-of-*.gguf",
    "Qwen3.8-Flash-Next-IQ2_S-Q2SYM/*-00001-of-*.gguf",
    "Qwen3.8-Flash-Next-UD-IQ3_XXS/UD-IQ3_XXS/*-00001-of-*.gguf",
]


def test_qwen36_moe_config_skips_the_nextn_block():
    from freetoken.models.qwen3_5_moe.gguf import parse_gguf_config
    from freetoken.utils import cached_load_hf_config

    cfg = parse_gguf_config(cached_load_hf_config(_model(_QWEN36)))
    # block_count 41 = 40 decoder layers + 1 nextn (MTP) block, which the decoder must not build
    assert cfg.num_layers == 40
    assert [cfg.is_linear_layer(i) for i in range(8)] == [True, True, True, False] * 2
    assert (cfg.hidden_size, cfg.num_experts, cfg.vocab_size) == (2048, 256, 248320)
    g = cfg.linear_attention_group()
    assert (g.num_key_heads, g.num_value_heads, g.key_head_dim, g.value_head_dim) == (16, 32, 128, 128)


@pytest.mark.parametrize("pattern", _FLASH_NEXT)
def test_flash_next_config_matches_the_hf_release(pattern):
    """The GGUF metadata, read the way the ``qwen4exp`` loader reads it, against the HF config.json of the same model."""
    from types import SimpleNamespace

    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.gguf import parse_gguf_config
    from freetoken.utils import cached_load_hf_config

    import json

    hf_path = os.path.join(_ROOT, "Qwen3.8-Flash-Next-MTP/config.json")
    if not os.path.isfile(hf_path):
        pytest.skip("no Qwen3.8-Flash-Next HF config.json to compare with")
    hf = json.load(open(hf_path)) | {"vision_config": None}

    def ns(d):
        return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) and k != "rope_parameters" else v for k, v in d.items()})

    ref = parse_config(ns(hf))
    got = parse_gguf_config(cached_load_hf_config(_model(pattern)))
    for field in ("hidden_size", "num_layers", "num_qo_heads", "num_kv_heads", "head_dim", "rotary_config", "num_experts_per_tok", "norm_topk_prob", "vocab_size", "shared_expert_intermediate_size", "moe_intermediate_size", "hidden_act"):
        assert getattr(got, field) == getattr(ref, field), field
    assert np.float32(got.rms_norm_eps) == np.float32(ref.rms_norm_eps)  # the GGUF stores it as f32
    # the GSQ-RCO coder is pruned to 256 experts, so check the expert count against the file's own banks
    from freetoken.models.gguf.reader import gguf_tensor_index

    assert got.num_experts == gguf_tensor_index(_model(pattern))["blk.0.ffn_gate_exps.weight"].ne[2]
    assert [got.is_linear_layer(i) for i in range(got.num_layers)] == [ref.is_linear_layer(i) for i in range(ref.num_layers)]
    assert got.linear_attention_group() == ref.linear_attention_group()
    a, b = got.qwen4_args, ref.qwen4_args
    for field in ("hc_count", "hc_lowrank", "ple_layer_ids", "ple_embed_dim", "ple_conv_kernel_size", "ngram_size", "heads_per_ngram",
                  "ngram_vocab_size_base", "make_ngram_vocab_size_divisible_by", "split_ngram_parts", "ngram_boundary_token_id",
                  "index_n_heads", "index_kv_heads", "index_head_dim", "index_budget", "index_ratio"):
        assert getattr(a, field) == getattr(b, field), field
    assert a.gdn_out_heads_tiled  # ssm_out keeps llama.cpp's tiled value-head columns


@pytest.mark.parametrize("pattern", [_QWEN36, *_FLASH_NEXT])
def test_user_defined_tokens_encode_as_single_ids(pattern):
    """<think>, </think>, <tool_call> ... (USER_DEFINED in the GGUF) and the control tokens must each encode to their own id."""
    pytest.importorskip("transformers.integrations.gguf", reason="needs transformers with the GGUF tokenizer converter")
    from freetoken.models.gguf.reader import gguf_config_source, load_gguf_metadata
    from freetoken.models.gguf.tokenizer import load_gguf_tokenizer

    path = gguf_config_source(_model(pattern))
    try:
        tok = load_gguf_tokenizer(path)
    except ImportError as e:
        pytest.skip(f"transformers lacks the GGUF tokenizer converter: {e}")
    md = load_gguf_metadata(path)
    special = [(i, t) for i, (t, ty) in enumerate(zip(md["tokenizer.ggml.tokens"], md["tokenizer.ggml.token_type"])) if int(ty) in (3, 4)]
    assert any(t == "<think>" for _, t in special)
    for i, t in special:
        assert i in tok.encode(f"a{t}b", add_special_tokens=False), t
    ids = tok.encode("<think>\nhi</think>", add_special_tokens=False)
    assert tok.decode(ids, skip_special_tokens=True) == "<think>\nhi</think>"


def test_qwen36_nextn_block_loads_only_when_asked():
    """The decoder iterator skips the nextn (MTP) block; the draft-head iterator yields it as a full-attention block under ``mtp.``,
    packed like the decoder's own blocks, and the GGUF quant config names its modules."""
    from freetoken.models.qwen3_5_moe.gguf import _iter_layer, iter_gguf_mtp_weights, nextn_block, parse_gguf_config, to_gguf
    from freetoken.models.gguf.reader import gguf_tensor_index
    from freetoken.utils import cached_load_hf_config

    path = _model(_QWEN36)
    index = gguf_tensor_index(path)
    cfg = parse_gguf_config(cached_load_hf_config(path))
    n = nextn_block(index)
    assert n == cfg.num_layers == 40

    mtp = dict(iter_gguf_mtp_weights(path))
    own = {k for k in mtp if not k.startswith("mtp.layer.")}
    assert own == {"mtp.enorm.weight", "mtp.hnorm.weight", "mtp.shared_head_norm.weight", "mtp.eh_proj.weight"}
    assert mtp["mtp.eh_proj.weight"].dtype == torch.uint8  # packed Q8_0, K = 2 * hidden
    # Full-attention q_proj includes both query and per-head gate rows; the output
    # projection consumes only the query half after the model splits them.
    expected_projection_shapes = {
        f"blk.{n}.attn_q.weight": (cfg.hidden_size, 2 * cfg.num_qo_heads * cfg.head_dim),
        f"blk.{n}.attn_k.weight": (cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim),
        f"blk.{n}.attn_v.weight": (cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim),
        f"blk.{n}.attn_output.weight": (cfg.num_qo_heads * cfg.head_dim, cfg.hidden_size),
    }
    for name, shape in expected_projection_shapes.items():
        assert tuple(index[name].ne) == shape, name

    attn = next(i for i in range(cfg.num_layers) if not cfg.is_linear_layer(i))
    block = dict(_iter_layer(index, cfg, f"blk.{attn}", "x", linear=False))
    assert {k.replace("mtp.layer.", "x.") for k in mtp if k not in own} == set(block)
    for k, v in block.items():
        assert mtp[k.replace("x.", "mtp.layer.")].shape[0] == v.shape[0], k

    tensors = {name: info.ggml_type for name, info in index.items()}
    assert to_gguf("mtp.eh_proj", tensors) == (f"blk.{n}.nextn.eh_proj",)
    assert to_gguf("mtp.layer.self_attn.q_proj", tensors) == (f"blk.{n}.attn_q",)
    assert to_gguf("mtp.layer.mlp.experts.0.down_proj", tensors) == (f"blk.{n}.ffn_down_exps",)
    assert to_gguf("model.language_model.layers.0.mlp.gate", tensors) == ("blk.0.ffn_gate_inp",)
