from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")


def _quantize_q8_0(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    slots, heads, dim = x.shape
    blocks = x.float().view(slots, heads, dim // 32, 32)
    scale_fp32 = blocks.abs().amax(dim=-1) / 127.0
    scale = scale_fp32.to(torch.float16)
    normalized = blocks / scale_fp32.unsqueeze(-1)
    q = (
        torch.sign(normalized) * torch.floor(normalized.abs() + 0.5)
    ).clamp(-127, 127).to(torch.int8)
    q = q.view(slots, heads, dim)
    restored = (
        q.float().view(slots, heads, dim // 32, 32)
        * scale.float().unsqueeze(-1)
    ).view(slots, heads, dim)
    return q, scale, restored


def _run_decode(q, k, v, indptr, indices, q_positions, *, k_scale=None, v_scale=None):
    from freetoken.kernel.triton.attention import decode_paged_attention

    batch, num_q_heads, head_dim = q.shape
    max_splits = 8
    return decode_paged_attention(
        q,
        k,
        v,
        indptr,
        indices,
        q_positions,
        torch.empty(batch, num_q_heads, max_splits, head_dim, device=q.device),
        torch.empty(batch, num_q_heads, max_splits, device=q.device),
        head_dim**-0.5,
        max_kv_splits=max_splits,
        k_scale_cache=k_scale,
        v_scale_cache=v_scale,
    )


def _call_attention_with_cache_format(kind, *, head_dim, int8_cache, with_scales):
    from freetoken.kernel.triton.attention import (
        decode_paged_attention,
        extend_paged_attention,
        paged_attention,
    )

    device = torch.device("cuda")
    dtype = torch.int8 if int8_cache else torch.bfloat16
    q = torch.randn(1, 1, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.zeros(2, 1, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    scales = None
    if with_scales:
        scales = torch.ones(2, 1, head_dim // 32, device=device, dtype=torch.float16)
    indptr = torch.tensor([0, 2], device=device, dtype=torch.int32)
    indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    q_positions = torch.tensor([1], device=device, dtype=torch.int64)

    if kind == "paged":
        return paged_attention(
            q, k_cache, v_cache, indptr, indices,
            torch.tensor([0], device=device, dtype=torch.int32), q_positions,
            head_dim**-0.5, k_scale_cache=scales, v_scale_cache=scales,
        )
    if kind == "decode":
        return decode_paged_attention(
            q, k_cache, v_cache, indptr, indices, q_positions,
            torch.empty(1, 1, 1, head_dim, device=device),
            torch.empty(1, 1, 1, device=device),
            torch.ones(1, device=device, dtype=torch.int32),
            1, head_dim**-0.5,
            k_scale_cache=scales, v_scale_cache=scales,
        )
    if kind == "extend":
        return extend_paged_attention(
            q, k_cache, v_cache,
            torch.tensor([0, 1], device=device, dtype=torch.int32),
            indptr, indices,
            torch.tensor([1], device=device, dtype=torch.int32),
            max_q_len=1, sm_scale=head_dim**-0.5,
            k_scale_cache=scales, v_scale_cache=scales,
        )
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["paged", "decode", "extend"])
def test_q8_0_attention_rejects_int8_cache_without_scales(kind):
    with pytest.raises(ValueError, match="scale caches"):
        _call_attention_with_cache_format(
            kind, head_dim=64, int8_cache=True, with_scales=False
        )


@pytest.mark.parametrize("kind", ["paged", "decode", "extend"])
def test_q8_0_attention_rejects_scales_for_non_int8_cache(kind):
    with pytest.raises(ValueError, match="scale caches"):
        _call_attention_with_cache_format(
            kind, head_dim=64, int8_cache=False, with_scales=True
        )


@pytest.mark.parametrize("kind", ["paged", "decode", "extend"])
def test_q8_0_attention_rejects_head_dim_not_divisible_by_32(kind):
    with pytest.raises(ValueError, match="head_dim divisible by 32"):
        _call_attention_with_cache_format(
            kind, head_dim=48, int8_cache=True, with_scales=True
        )


def test_q8_0_paged_decode_and_extend_attention_read_dequantized_cache():
    from freetoken.kernel.triton.attention import extend_paged_attention, paged_attention

    torch.manual_seed(19)
    device = torch.device("cuda")
    slots, num_kv_heads, num_q_heads, head_dim = 8, 2, 4, 64
    k_source = torch.randn(slots, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_source = torch.randn_like(k_source)
    k_q, k_scale, k_ref = _quantize_q8_0(k_source)
    v_q, v_scale, v_ref = _quantize_q8_0(v_source)
    q = torch.randn(3, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
    indices = torch.arange(slots, dtype=torch.int32, device=device)
    q_to_req = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    q_positions = torch.tensor([1, 3, 3], dtype=torch.int64, device=device)
    sm_scale = head_dim**-0.5

    for window in (None, 3):
        q8_paged = paged_attention(
            q, k_q, v_q, indptr, indices, q_to_req, q_positions, sm_scale,
            sliding_window=window, k_scale_cache=k_scale, v_scale_cache=v_scale,
        )
        ref_paged = paged_attention(
            q, k_ref, v_ref, indptr, indices, q_to_req, q_positions, sm_scale,
            sliding_window=window,
        )
        torch.testing.assert_close(q8_paged.float(), ref_paged.float(), atol=2e-2, rtol=2e-2)

    decode_q = q[[0, 2]]
    decode_positions = q_positions[[0, 2]]
    q8_decode = _run_decode(
        decode_q, k_q, v_q, indptr, indices, decode_positions,
        k_scale=k_scale, v_scale=v_scale,
    )
    ref_decode = _run_decode(decode_q, k_ref, v_ref, indptr, indices, decode_positions)
    torch.testing.assert_close(q8_decode.float(), ref_decode.float(), atol=2e-2, rtol=2e-2)

    q_extend = q
    qo_indptr = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    prefix_lens = torch.tensor([2, 1], dtype=torch.int32, device=device)
    q8_extend = extend_paged_attention(
        q_extend,
        k_q,
        v_q,
        qo_indptr,
        indptr,
        indices,
        prefix_lens,
        max_q_len=2,
        sm_scale=sm_scale,
        k_scale_cache=k_scale,
        v_scale_cache=v_scale,
    )
    ref_extend = extend_paged_attention(
        q_extend,
        k_ref,
        v_ref,
        qo_indptr,
        indptr,
        indices,
        prefix_lens,
        max_q_len=2,
        sm_scale=sm_scale,
    )
    torch.testing.assert_close(q8_extend.float(), ref_extend.float(), atol=2e-2, rtol=2e-2)


def test_q8_0_decode_maps_partial_gqa_blocks_to_their_kv_head():
    torch.manual_seed(37)
    device = torch.device("cuda")
    slots, num_kv_heads, num_q_heads, head_dim = 4, 2, 48, 64
    k_source = torch.randn(
        slots, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v_source = torch.randn_like(k_source)
    k_q, k_scale, k_ref = _quantize_q8_0(k_source)
    v_q, v_scale, v_ref = _quantize_q8_0(v_source)
    q = torch.randn(1, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor([0, slots], dtype=torch.int32, device=device)
    indices = torch.arange(slots, dtype=torch.int32, device=device)
    q_positions = torch.tensor([slots - 1], dtype=torch.int64, device=device)
    sm_scale = head_dim**-0.5

    actual = _run_decode(
        q, k_q, v_q, indptr, indices, q_positions, k_scale=k_scale, v_scale=v_scale
    )
    group = num_q_heads // num_kv_heads
    reference = []
    for kv_head in range(num_kv_heads):
        scores = torch.matmul(
            q[0, kv_head * group : (kv_head + 1) * group].float(),
            k_ref[:, kv_head].float().transpose(0, 1),
        ) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        reference.append(torch.matmul(probs, v_ref[:, kv_head].float()))
    expected = torch.cat(reference, dim=0).unsqueeze(0)

    # GROUP=24 is covered by two 16-head blocks; the second block's 8-lane tail
    # must stay on the same KV head instead of crossing into the next group.
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
