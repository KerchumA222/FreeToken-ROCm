"""Qwen4-Exp per-layer n-gram embedding (PLE) index arithmetic.

The PLE table is enormous and read almost not at all. With ``ngram_size=3`` and
``heads_per_ngram=8`` there are ``(3-1)*8 = 16`` heads, each owning its own ~20M-row
vocabulary, so the table is ~320M rows -- around 100 GB in bf16 for the shipped
model. A token reads exactly 16 rows of ``hidden_size/16`` from it. That ratio is
what makes the table a natural disk-resident structure rather than a resident one:
the bytes are enormous, the traffic is kilobytes per token.

This module is the addressing half -- pure integer arithmetic that turns token ids
into row ids. It has to agree with the reference *exactly*: an off-by-one in a head
vocabulary or a multiplier silently reads the wrong rows of a valid table, which
looks like a quality problem rather than a bug. ``tests/models/test_qwen4_exp.py``
pins every piece of it to ``modeling_qwen4_exp``.

The gather itself is an ordinary quantized embedding lookup (``GGUFEmbedding``).
"""

from __future__ import annotations

import functools
import math

import torch

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


@functools.lru_cache(maxsize=None)
def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> torch.Tensor:
    """Per-position odd multipliers used to mix a window of token ids into one hash.

    Odd by construction, and bounded so that ``id * multiplier`` cannot overflow a
    signed 64-bit integer for any in-range token id -- which is why the hash can be
    computed in int64 without a wider type.
    """
    multiplier_max = ((1 << 63) - 1) // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    out = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        out.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(out, dtype=torch.long)


def head_vocabularies(
    ngram_vocab_size_base: int, ngram_heads: int, ple_layer_index: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``(sizes, offsets, total)`` for one PLE layer's heads.

    Each head gets a distinct prime-sized vocabulary just above
    ``ngram_vocab_size_base`` -- distinct so two heads cannot collide identically,
    prime so the modulo mixes. The count is indexed *globally* across PLE layers, so
    a layer's sizes depend on its position in ``ple_layer_ids``, not just on itself.
    """
    sizes, offsets, total = [], [], 0
    for head_idx in range(ngram_heads):
        global_head_idx = ple_layer_index * ngram_heads + head_idx
        size = _nth_prime_after(ngram_vocab_size_base - 1, global_head_idx + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    return (
        torch.tensor(sizes, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        total,
    )


def padded_vocab_size(total_vocab_size: int, divisor: int) -> int:
    return math.ceil(total_vocab_size / divisor) * divisor


def shift_right_ignore_eos(token_ids: torch.Tensor, shift: int, eos_token_id: int) -> torch.Tensor:
    """Shift right by ``shift``, refusing to read across an EOS.

    The n-gram context must not span a document boundary, so any position whose
    window would reach back past the preceding EOS reads EOS instead.
    """
    if shift == 0:
        return token_ids
    batch_size, seq_len = token_ids.shape
    positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
    eos_positions = torch.where(token_ids == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
    )
    position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
    shifted = token_ids.gather(dim=1, index=gather_positions)
    valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, token_ids.new_full((), eos_token_id))


def ngram_row_ids(
    token_history: torch.Tensor,
    *,
    multipliers: torch.Tensor,
    head_sizes: torch.Tensor,
    head_offsets: torch.Tensor,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
    keep_last: int | None = None,
) -> torch.Tensor:
    """Row ids into the PLE table: ``[batch, keep_last, (ngram_size-1)*heads_per_ngram]``.

    ``token_history`` is the sequence prefixed with the previous ``ngram_size-1``
    tokens, so that the first real token still sees a full window.
    """
    token_history = token_history.long()
    shifted = [
        shift_right_ignore_eos(token_history, shift, eos_token_id)
        for shift in range(ngram_size)
    ]
    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * multipliers[position])
        ids = torch.remainder(mixed.unsqueeze(-1), head_sizes[start:end].view(1, 1, -1))
        blocks.append(ids + head_offsets[start:end].view(1, 1, -1))
    out = torch.cat(blocks, dim=-1)
    return out if keep_last is None else out[:, -keep_last:]


__all__ = [
    "build_layer_multipliers",
    "head_vocabularies",
    "ngram_row_ids",
    "padded_vocab_size",
    "shift_right_ignore_eos",
]
