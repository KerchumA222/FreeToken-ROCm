from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.laguna.config import parse_config


def _config(**updates):
    layers = 8
    data = {
        "architectures": ["LagunaForCausalLM"],
        "model_type": "laguna",
        "vocab_size": 256,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": layers,
        "num_attention_heads": 2,
        "num_attention_heads_per_layer": [2, 3, 3, 3] * 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "norm_topk_prob": True,
        "moe_routed_scaling_factor": 2.5,
        "tie_word_embeddings": False,
        "gating": "per-head",
        "gating_types": ["per_head"] * layers,
        "sliding_window": 512,
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"] * 2,
        "rope_parameters": {
            "full_attention": {
                "rope_theta": 500000.0,
                "rope_type": "yarn",
                "factor": 128.0,
                "original_max_position_embeddings": 8192,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
                "attention_factor": 1.4852030263919618,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        },
        "mlp_only_layers": [0],
        "mlp_layer_types": ["dense"] + ["sparse"] * (layers - 1),
    }
    data.update(updates)
    return SimpleNamespace(**data)


def test_laguna_parse_config_preserves_layer_geometry_and_cache_groups():
    config = parse_config(_config())

    assert config.num_layers == 8
    assert config.num_qo_heads == 3
    assert config.num_qo_heads_for_layer(0) == 2
    assert config.num_qo_heads_for_layer(1) == 3
    assert config.num_kv_heads == 1 and config.head_dim == 64
    assert config.first_k_dense_replace == 1
    assert config.num_moe_layers == 7
    assert config.num_experts == 4 and config.num_experts_per_tok == 2
    assert config.routed_scaling_factor == 2.5
    full, swa = config.attention_groups
    assert full.layer_ids == (0, 4)
    assert full.rotary_config.rotary_dim == 32
    assert full.rotary_config.base == 500000.0
    assert full.rotary_config.scaling["rope_type"] == "yarn"
    assert swa.layer_ids == (1, 2, 3, 5, 6, 7)
    assert swa.sliding_window == 512 and swa.rotary_config.rotary_dim == 64
    full_kv, swa_kv = config.kv_cache_group_specs()
    assert full_kv.layer_ids == full.layer_ids and full_kv.sliding_window is None
    assert swa_kv.layer_ids == swa.layer_ids and swa_kv.sliding_window == 512


def test_official_laguna_s21_config_topology():
    cfg = _config()
    cfg.num_hidden_layers = 48
    cfg.hidden_size = 3072
    cfg.intermediate_size = 12288
    cfg.vocab_size = 100352
    cfg.num_key_value_heads = 8
    cfg.head_dim = 128
    cfg.max_position_embeddings = 1048576
    cfg.num_experts = 256
    cfg.num_experts_per_tok = 10
    cfg.moe_intermediate_size = 1024
    cfg.shared_expert_intermediate_size = 1024
    cfg.num_attention_heads_per_layer = [48, 72, 72, 72] * 12
    cfg.layer_types = ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"] * 12
    cfg.gating_types = ["per_head"] * 48
    cfg.mlp_layer_types = ["dense"] + ["sparse"] * 47

    config = parse_config(cfg)
    assert config.num_qo_heads == 72
    assert config.num_qo_heads_per_layer == tuple([48, 72, 72, 72] * 12)
    assert config.attention_groups[0].layer_ids == tuple(range(0, 48, 4))
    assert config.attention_groups[1].layer_ids == tuple(
        layer for layer in range(48) if layer % 4
    )
    assert config.attention_groups[0].rotary_config.table_positions == 1048576
    assert config.first_k_dense_replace == 1 and config.num_moe_layers == 47
    assert config.num_experts == 256 and config.num_experts_per_tok == 10


@pytest.mark.parametrize(
    "tp_size, updates, message",
    [
        (2, {}, "query-head count 3 for layer 1"),
        (
            3,
            {"num_attention_heads_per_layer": [6] * 8, "num_key_value_heads": 2},
            "replicated only when tensor parallel size",
        ),
    ],
)
def test_laguna_rejects_unsupported_tp_head_geometry_during_config_adjustment(
    tp_size, updates, message
):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _adjust_config

    config = EngineConfig(
        model_path="/tmp/laguna-test-model",
        tp_info=DistributedInfo(rank=0, size=tp_size),
        dtype=torch.bfloat16,
    )
    object.__setattr__(config, "model_config", parse_config(_config(**updates)))

    with pytest.raises(ValueError, match=message):
        _adjust_config(config)


def test_laguna_config_rejects_bad_per_layer_heads_and_mlp_layout():
    with pytest.raises(ValueError, match="one positive count per layer"):
        parse_config(_config(num_attention_heads_per_layer=[2, 3]))
    with pytest.raises(NotImplementedError, match="dense layer 0"):
        parse_config(_config(mlp_only_layers=[1]))
    with pytest.raises(NotImplementedError, match="per-head"):
        parse_config(_config(gating="per-channel"))


@pytest.mark.parametrize("rope_type", ["longrope", "dynamic"])
def test_laguna_rejects_unimplemented_hf_rope_schemes(rope_type):
    cfg = _config()
    full_rope = dict(cfg.rope_parameters["full_attention"])
    full_rope["rope_type"] = rope_type
    cfg.rope_parameters = {**cfg.rope_parameters, "full_attention": full_rope}

    with pytest.raises(ValueError, match="Unsupported Laguna RoPE type"):
        parse_config(cfg)


@pytest.mark.parametrize("factor", [0.0, -1.0, float("inf"), float("nan")])
def test_laguna_rejects_invalid_hf_yarn_factor(factor):
    cfg = _config()
    full_rope = dict(cfg.rope_parameters["full_attention"])
    full_rope["factor"] = factor
    cfg.rope_parameters = {**cfg.rope_parameters, "full_attention": full_rope}

    with pytest.raises(ValueError, match="positive finite factor"):
        parse_config(cfg)


@pytest.mark.parametrize("attention_factor", [0.0, -1.0, float("inf"), float("nan")])
def test_laguna_rejects_invalid_hf_yarn_attention_factor(attention_factor):
    cfg = _config()
    full_rope = dict(cfg.rope_parameters["full_attention"])
    full_rope["attention_factor"] = attention_factor
    cfg.rope_parameters = {**cfg.rope_parameters, "full_attention": full_rope}

    with pytest.raises(ValueError, match="positive finite attention_factor"):
        parse_config(cfg)


def _laguna_gguf_metadata():
    return {
        "laguna.block_count": 48,
        "laguna.embedding_length": 3072,
        "laguna.feed_forward_length": 12288,
        "laguna.context_length": 262144,
        "laguna.attention.head_count": [48, 72, 72, 72] * 12,
        "laguna.attention.head_count_kv": 8,
        "laguna.attention.key_length": 128,
        "laguna.attention.layer_norm_rms_epsilon": 1e-6,
        "laguna.attention.sliding_window": 512,
        "laguna.rope.dimension_count": 64,
        "laguna.rope.dimension_count_swa": 128,
        "laguna.rope.freq_base": 500000.0,
        "laguna.rope.freq_base_swa": 10000.0,
        "laguna.rope.scaling.type": "yarn",
        "laguna.rope.scaling.factor": 32.0,
        "laguna.rope.scaling.original_context_length": 8192,
        "laguna.rope.scaling.yarn_beta_fast": 32.0,
        "laguna.rope.scaling.yarn_beta_slow": 1.0,
        "laguna.rope.scaling.yarn_attn_factor": 1.0,
        "laguna.leading_dense_block_count": 1,
        "laguna.expert_count": 256,
        "laguna.expert_used_count": 10,
        "laguna.expert_feed_forward_length": 1024,
        "laguna.expert_shared_feed_forward_length": 1024,
        "laguna.expert_weights_norm": True,
        "laguna.expert_weights_scale": 2.5,
        "laguna.expert_gating_func": 2,  # llama.cpp enum: sigmoid
    }


def test_laguna_gguf_config_uses_verified_topology_and_file_yarn_metadata(monkeypatch):
    from freetoken.models.laguna import gguf

    shim = SimpleNamespace(
        metadata=_laguna_gguf_metadata(),
        model_path="synthetic-laguna.gguf",
        vocab_size=100352,
        tie_word_embeddings=False,
        architectures=["LagunaGGUFForCausalLM"],
    )
    monkeypatch.setattr(gguf, "gguf_tensor_index", lambda _path: {})

    config = gguf.parse_gguf_config(shim)

    assert config.num_layers == 48
    assert config.num_qo_heads_per_layer == tuple([48, 72, 72, 72] * 12)
    assert config.attention_groups[0].layer_ids == tuple(range(0, 48, 4))
    assert config.attention_groups[1].layer_ids == tuple(i for i in range(48) if i % 4)
    assert config.first_k_dense_replace == 1 and config.num_moe_layers == 47
    scaling = config.attention_groups[0].rotary_config.scaling
    assert scaling["attention_factor"] is None
    from freetoken.layers.rotary import _rope_frequencies

    _, attention_factor = _rope_frequencies(128, 64, 500000.0, scaling)
    assert attention_factor == pytest.approx(1.0 + 0.1 * math.log(32.0))
    assert config.attention_groups[0].rotary_config.table_positions == 262144
    assert config.routed_scaling_factor == 2.5 and config.num_experts_per_tok == 10


@pytest.mark.parametrize("attention_factor", [0.0, -1.0, float("inf"), float("nan")])
def test_laguna_gguf_config_rejects_invalid_explicit_yarn_attention_factor(
    monkeypatch, attention_factor
):
    from freetoken.models.laguna import gguf

    metadata = _laguna_gguf_metadata()
    metadata["laguna.rope.scaling.yarn_attn_factor"] = attention_factor
    shim = SimpleNamespace(
        metadata=metadata,
        model_path="synthetic-laguna.gguf",
        vocab_size=100352,
        tie_word_embeddings=False,
        architectures=["LagunaGGUFForCausalLM"],
    )
    monkeypatch.setattr(gguf, "gguf_tensor_index", lambda _path: {})

    with pytest.raises(ValueError, match="invalid YaRN scaling metadata"):
        gguf.parse_gguf_config(shim)


def test_laguna_gguf_config_rejects_unverified_head_profile(monkeypatch):
    from freetoken.models.laguna import gguf

    metadata = _laguna_gguf_metadata()
    metadata["laguna.attention.head_count"] = [72] * 48
    shim = SimpleNamespace(metadata=metadata, model_path="synthetic.gguf", vocab_size=8,
                            tie_word_embeddings=False, architectures=["LagunaGGUFForCausalLM"])
    monkeypatch.setattr(gguf, "gguf_tensor_index", lambda _path: {})
    with pytest.raises(NotImplementedError, match="verified Laguna-S-2.1"):
        gguf.parse_gguf_config(shim)


def test_laguna_gguf_tensor_mapping_and_opt_in(monkeypatch):
    from freetoken.models.gguf import config as gguf_config
    from freetoken.models.laguna.gguf import to_gguf

    assert to_gguf("model.layers.0.self_attn.g_proj", {}) == ("blk.0.attn_gate",)
    assert to_gguf("model.layers.1.mlp.e_score_correction_bias", {}) == ("blk.1.exp_probs_b",)
    assert to_gguf("model.layers.1.mlp.shared_expert.down_proj", {}) == ("blk.1.ffn_down_shexp",)
    assert to_gguf("model.layers.1.mlp.experts.0.gate_proj", {}) == ("blk.1.ffn_gate_exps",)

    monkeypatch.setattr(gguf_config, "gguf_architecture", lambda _path: "laguna")
    monkeypatch.delenv("FREETOKEN_ENABLE_LAGUNA_GGUF", raising=False)
    with pytest.raises(ValueError, match="FREETOKEN_ENABLE_LAGUNA_GGUF=1"):
        gguf_config.build_gguf_shim("synthetic.gguf")

    monkeypatch.setenv("FREETOKEN_ENABLE_LAGUNA_GGUF", "1")
    monkeypatch.setattr(gguf_config, "gguf_tensor_names", lambda _path: {"output.weight"})
    monkeypatch.setattr(gguf_config, "load_gguf_metadata", lambda _path: {})
    monkeypatch.setattr(gguf_config, "_vocab_size", lambda _path: 100352)
    shim = gguf_config.build_gguf_shim("synthetic.gguf")
    assert shim.architectures == ["LagunaGGUFForCausalLM"]


def test_laguna_gguf_rejects_wrong_per_head_gate_width(monkeypatch):
    from freetoken.models.laguna import gguf

    shim = SimpleNamespace(
        metadata=_laguna_gguf_metadata(),
        model_path="synthetic-laguna.gguf",
        vocab_size=100352,
        tie_word_embeddings=False,
        architectures=["LagunaGGUFForCausalLM"],
    )
    bad_gate = SimpleNamespace(ne=(3072, 1))
    monkeypatch.setattr(gguf, "gguf_tensor_index", lambda _path: {"blk.0.attn_gate.weight": bad_gate})
    with pytest.raises(ValueError, match="attn_gate: GGUF dimensions"):
        gguf.parse_gguf_config(shim)


def test_laguna_gguf_weight_iterator_maps_attention_dense_shared_and_router_tensors(monkeypatch):
    from freetoken.models.laguna import gguf

    shim = SimpleNamespace(
        metadata=_laguna_gguf_metadata(),
        model_path="synthetic-laguna.gguf",
        vocab_size=100352,
        tie_word_embeddings=True,
        architectures=["LagunaGGUFForCausalLM"],
    )
    index = {"token_embd.weight": SimpleNamespace(ne=(3072, 100352))}
    index["output_norm.weight"] = SimpleNamespace(ne=(3072,))
    for layer, q_heads in enumerate([48, 72, 72, 72] * 12):
        blk = f"blk.{layer}"
        entries = {
            "attn_q": (3072, q_heads * 128),
            "attn_k": (3072, 8 * 128),
            "attn_v": (3072, 8 * 128),
            "attn_output": (q_heads * 128, 3072),
            "attn_gate": (3072, q_heads),
            "attn_q_norm": (128,),
            "attn_k_norm": (128,),
            "attn_norm": (3072,),
            "ffn_norm": (3072,),
        }
        if layer == 0:
            entries.update({
                "ffn_gate": (3072, 12288),
                "ffn_up": (3072, 12288),
                "ffn_down": (12288, 3072),
            })
        else:
            entries.update({
                "ffn_gate_inp": (3072, 256),
                "ffn_gate_shexp": (3072, 1024),
                "ffn_up_shexp": (3072, 1024),
                "ffn_down_shexp": (1024, 3072),
                "ffn_gate_exps": (3072, 1024, 256),
                "ffn_up_exps": (3072, 1024, 256),
                "ffn_down_exps": (1024, 3072, 256),
            })
        index.update({f"{blk}.{name}.weight": SimpleNamespace(ne=shape) for name, shape in entries.items()})
        if layer:
            # llama.cpp names the router correction bias with the "bias" suffix
            index[f"{blk}.exp_probs_b.bias"] = SimpleNamespace(ne=(256,))

    monkeypatch.setattr(gguf, "gguf_tensor_index", lambda _path: index)
    marker = torch.ones(1)
    monkeypatch.setattr(gguf, "matrix", lambda _info: marker)
    monkeypatch.setattr(gguf, "float_tensor", lambda _info: marker)
    monkeypatch.setattr(gguf, "fused", lambda prefix, *_args, **_kwargs: iter(((f"{prefix}.weight", marker),)))
    monkeypatch.setattr("freetoken.utils.cached_load_hf_config", lambda _path: shim)
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    weights = dict(gguf.iter_gguf_weights(
        shim.model_path,
        torch.device("cpu"),
        include_moe_experts=True,
        include_non_moe=True,
    ))
    assert "model.layers.0.mlp.gate_up_proj.weight" in weights
    assert "model.layers.1.mlp.shared_expert.gate_up_proj.weight" in weights
    assert "model.layers.1.mlp.gate.weight" in weights
    assert "model.layers.1.mlp.e_score_correction_bias" in weights
    assert "model.layers.1.mlp.experts.gate_up_proj" not in weights


def test_laguna_attention_projection_geometry_follows_each_layer():
    from freetoken.models.laguna.attention import LagunaAttention

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    config = parse_config(_config())
    layer0 = LagunaAttention(config, 0, prefix="model.layers.0.self_attn")
    layer1 = LagunaAttention(config, 1, prefix="model.layers.1.self_attn")

    assert layer0.num_qo_heads == 2 and layer0.qo_attn_dim == 128
    assert layer0.qkv_proj.output_sizes == (128, 64, 64)
    assert layer0.g_proj.output_sizes == (2,)
    assert layer0.o_proj.full_input_size == 128
    assert layer1.num_qo_heads == 3 and layer1.qo_attn_dim == 192
    assert layer1.qkv_proj.output_sizes == (192, 64, 64)
    assert layer1.g_proj.output_sizes == (3,)
    assert layer1.o_proj.full_input_size == 192
    assert layer1.attn_spec.sliding_window == 512


def test_laguna_output_projection_uses_global_query_width_for_tp(monkeypatch):
    from freetoken.layers import linear
    from freetoken.models.laguna import attention

    tp_info = SimpleNamespace(rank=0, size=2)
    monkeypatch.setattr(linear, "get_tp_info", lambda: tp_info)
    monkeypatch.setattr(attention, "get_tp_info", lambda: tp_info)

    cfg = _config()
    cfg.hidden_size = 3072
    cfg.num_key_value_heads = 8
    cfg.head_dim = 128
    cfg.num_attention_heads_per_layer = [48, 72, 72, 72] * 2
    config = parse_config(cfg)
    layer = attention.LagunaAttention(config, 0)

    assert layer.qo_attn_dim == 24 * 128
    assert layer.o_proj.full_input_size == 48 * 128
    assert layer.o_proj.local_input_size == 24 * 128


def test_laguna_weight_reader_fuses_attention_dense_shared_and_expert_tensors(
    tmp_path, monkeypatch
):
    from safetensors.torch import save_file

    from freetoken.models.laguna import weight

    hf = _config()
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(128, 128),
        "model.layers.0.self_attn.k_proj.weight": torch.zeros(64, 128),
        "model.layers.0.self_attn.v_proj.weight": torch.zeros(64, 128),
        "model.layers.0.self_attn.g_proj.weight": torch.zeros(2, 128),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros(256, 128),
        "model.layers.0.mlp.up_proj.weight": torch.zeros(256, 128),
        "model.layers.1.mlp.experts.e_score_correction_bias": torch.ones(4),
        "model.layers.1.mlp.shared_expert.gate_proj.weight": torch.zeros(32, 128),
        "model.layers.1.mlp.shared_expert.up_proj.weight": torch.zeros(32, 128),
        "model.layers.1.mlp.shared_expert.down_proj.weight": torch.zeros(128, 32),
    }
    for layer in range(1, 8):
        for expert in range(4):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            tensors[f"{prefix}.gate_proj.weight"] = torch.zeros(32, 128)
            tensors[f"{prefix}.up_proj.weight"] = torch.zeros(32, 128)
            tensors[f"{prefix}.down_proj.weight"] = torch.zeros(128, 32)
    path = tmp_path / "weights.safetensors"
    save_file(tensors, str(path))
    monkeypatch.setattr(weight, "iter_weight_files", lambda _path: [str(path)])
    monkeypatch.setattr(weight, "cached_load_hf_config", lambda _path: hf)
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    loaded = dict(
        weight.iter_weights(
            str(tmp_path), torch.device("cpu"),
            include_moe_experts=True,
            include_non_moe=True,
        )
    )
    assert loaded["model.layers.0.self_attn.qkv_proj.weight"].shape == (256, 128)
    assert loaded["model.layers.0.self_attn.g_proj.weight"].shape == (2, 128)
    assert loaded["model.layers.0.mlp.gate_up_proj.weight"].shape == (512, 128)
    assert loaded["model.layers.1.mlp.shared_expert.gate_up_proj.weight"].shape == (64, 128)
    assert loaded["model.layers.1.mlp.experts.gate_up_proj"].shape == (4, 64, 128)
    assert loaded["model.layers.1.mlp.experts.down_proj"].shape == (4, 128, 32)
    assert torch.equal(loaded["model.layers.1.mlp.e_score_correction_bias"], torch.ones(4))


def test_laguna_registry_is_opt_in(monkeypatch):
    from freetoken.models.register import get_model_spec

    monkeypatch.delenv("FREETOKEN_ENABLE_LAGUNA_HF", raising=False)
    monkeypatch.delenv("FREETOKEN_ENABLE_LAGUNA_GGUF", raising=False)
    with pytest.raises(ValueError, match="FREETOKEN_ENABLE_LAGUNA_HF=1"):
        get_model_spec("LagunaForCausalLM")
    with pytest.raises(ValueError, match="FREETOKEN_ENABLE_LAGUNA_GGUF=1"):
        get_model_spec("LagunaGGUFForCausalLM")

    monkeypatch.setenv("FREETOKEN_ENABLE_LAGUNA_HF", "1")
    monkeypatch.setenv("FREETOKEN_ENABLE_LAGUNA_GGUF", "1")
    assert get_model_spec("LagunaForCausalLM").model_cls == "LagunaForCausalLM"
    assert get_model_spec("LagunaGGUFForCausalLM").model_cls == "LagunaForCausalLM"


def test_laguna_architecture_factory_builds_dense_prefix_and_moe_layers(monkeypatch):
    from freetoken.models import create_model
    from freetoken.models.laguna.model import LagunaDenseMLP, LagunaMoE

    monkeypatch.setenv("FREETOKEN_ENABLE_LAGUNA_HF", "1")
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    config = parse_config(_config())
    model = create_model(config)

    assert type(model).__name__ == "LagunaForCausalLM"
    assert isinstance(model.model.layers.op_list[0].mlp, LagunaDenseMLP)
    assert isinstance(model.model.layers.op_list[1].mlp, LagunaMoE)
    assert model.model.layers.op_list[0].self_attn.num_qo_heads == 2
    assert model.model.layers.op_list[1].self_attn.num_qo_heads == 3


def test_laguna_router_selection_bias_does_not_change_selected_weights():
    from freetoken.models.laguna.model import _laguna_route

    logits = torch.tensor([[0.1, 0.8, 1.2, -0.4]], dtype=torch.float32)
    bias = torch.tensor([2.0, -2.0, 0.0, 0.0], dtype=torch.float32)
    weights, ids = _laguna_route(logits, bias, top_k=2, normalize=True, scale=2.5)

    assert ids.tolist() == [[0, 2]]
    expected = logits.sigmoid().gather(-1, ids.long())
    expected = expected / expected.sum(dim=-1, keepdim=True) * 2.5
    torch.testing.assert_close(weights, expected)


def test_laguna_router_cache_bias_moves_selection_but_not_weights(monkeypatch):
    from freetoken.models.laguna.model import LagunaMoE
    from freetoken.moe import route_bias

    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    moe = SimpleNamespace(
        top_k=2,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        gate=SimpleNamespace(weight=torch.tensor([[2.0, 0.0], [1.0, 0.5], [0.9, 1.0], [-1.0, 2.0]])),
        e_score_correction_bias=torch.zeros(4),
        experts=SimpleNamespace(
            layer_id=0,
            offload_cache=SimpleNamespace(slot_for_id=torch.tensor([[-1, -1, 5, 7]], dtype=torch.int32)),
        ),
    )
    scores = (hidden @ moe.gate.weight.t()).sigmoid()

    monkeypatch.setattr(route_bias, "_ROUTE_BIAS", 0.0)
    monkeypatch.setattr(route_bias, "_HOST_BIAS", 0.0)
    _, plain = LagunaMoE._route(moe, hidden)
    assert [sorted(r) for r in plain.tolist()] == [[0, 1], [2, 3]]

    monkeypatch.setattr(route_bias, "_ROUTE_BIAS", 0.2)
    weights, ids = LagunaMoE._route(moe, hidden)
    # row 0: resident 2 overtakes 1 (0.711 + 0.2 > 0.731); row 1 already picks residents
    assert [sorted(r) for r in ids.tolist()] == [[0, 2], [2, 3]]
    expected = scores.gather(-1, ids.long())
    torch.testing.assert_close(weights, expected / expected.sum(-1, keepdim=True) * 2.5)
