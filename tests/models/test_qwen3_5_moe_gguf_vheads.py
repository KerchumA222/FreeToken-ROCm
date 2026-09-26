"""qwen35moe GGUF: value heads are stored tiled and must be un-tiled at load.

llama.cpp's converter (``conversion/qwen.py``, ``_LinearAttentionVReorderBase``) rewrites
every value-head-indexed axis from HF's grouped order into a tiled one so ggml can use a
plain broadcast. FreeToken's fla kernels are HF-convention, so the adapter has to invert
that. These tests pin the inverse against a verbatim copy of the converter's permutation.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.models.qwen3_5_moe.gguf import _untile_v_heads

# Real Qwen3.5-35B-A3B GDN geometry.
NUM_K_HEADS = 16
NUM_V_HEADS = 32
V_PER_K = NUM_V_HEADS // NUM_K_HEADS
HEAD_V_DIM = 128


def _reorder_v_heads(
    tensor: torch.Tensor, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int
) -> torch.Tensor:
    """Verbatim from llama.cpp ``conversion/qwen.py`` -- the forward (HF -> GGUF) direction."""
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1 :]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)


# (shape, dim, head_dim) for every tensor the adapter un-tiles.
CASES = [
    pytest.param((NUM_V_HEADS * HEAD_V_DIM, 2048), 0, HEAD_V_DIM, id="attn_qkv_v_rows"),
    pytest.param((NUM_V_HEADS * HEAD_V_DIM, 2048), 0, HEAD_V_DIM, id="attn_gate_z"),
    pytest.param((NUM_V_HEADS, 2048), 0, 1, id="ssm_beta_alpha"),
    pytest.param((NUM_V_HEADS,), 0, 1, id="ssm_a_and_dt"),
    pytest.param((NUM_V_HEADS * HEAD_V_DIM, 4), 0, HEAD_V_DIM, id="ssm_conv1d_v_channels"),
    pytest.param((2048, NUM_V_HEADS * HEAD_V_DIM), 1, HEAD_V_DIM, id="ssm_out_columns"),
]


@pytest.mark.parametrize("shape,dim,head_dim", CASES)
def test_untile_inverts_the_converter(shape, dim, head_dim):
    hf = torch.randn(*shape)
    gguf = _reorder_v_heads(hf, dim, NUM_K_HEADS, V_PER_K, head_dim)
    # Negative control: with 32 value heads over 16 key heads the two orders differ, so a
    # no-op _untile_v_heads could not pass the round trip below.
    assert not torch.equal(gguf, hf), "converter permutation is a no-op -- test proves nothing"
    assert torch.equal(_untile_v_heads(gguf, dim, NUM_K_HEADS, V_PER_K, head_dim), hf)


def test_untile_is_the_documented_permutation():
    """Value head ``h`` of the un-tiled result must be tiled slot ``(h % r) * num_k + h // r``."""
    gguf = torch.arange(NUM_V_HEADS, dtype=torch.float32)
    hf = _untile_v_heads(gguf, 0, NUM_K_HEADS, V_PER_K, 1)
    expect = torch.tensor(
        [(h % V_PER_K) * NUM_K_HEADS + h // V_PER_K for h in range(NUM_V_HEADS)],
        dtype=torch.float32,
    )
    assert torch.equal(hf, expect)


def test_untile_rejects_a_mismatched_axis():
    with pytest.raises(AssertionError):
        _untile_v_heads(torch.randn(NUM_V_HEADS + 1, 8), 0, NUM_K_HEADS, V_PER_K, 1)


def test_the_loaders_row_permutation_is_the_same_untiling():
    """iter_gguf_weights applies the fix as row indices (_v_head_permutation); it has to be
    this same un-tiling, head for head, or the two descriptions of the layout disagree."""
    from freetoken.models.qwen3_5_moe.gguf import _permute_head_rows, _v_head_permutation

    perm = _v_head_permutation(NUM_V_HEADS, NUM_K_HEADS)
    gguf = torch.randn(NUM_V_HEADS * HEAD_V_DIM, 3)
    assert torch.equal(
        _permute_head_rows(gguf, perm, HEAD_V_DIM),
        _untile_v_heads(gguf, 0, NUM_K_HEADS, V_PER_K, HEAD_V_DIM),
    )
