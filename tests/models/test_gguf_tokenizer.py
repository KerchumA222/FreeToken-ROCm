"""GGUF control tokens keep their file ids after the byte-level BPE conversion."""

from __future__ import annotations

import pytest

tokenizers = pytest.importorskip("tokenizers")

from freetoken.models.gguf.tokenizer import _pin_control_tokens


def _byte_level(text: str) -> str:
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    table = bytes_to_unicode()
    return "".join(table[b] for b in text.encode("utf-8"))


def _converted(vocab_texts, wrongly_added=()):
    """A byte-level BPE built the way the gpt2 GGUF converter builds one: every vocab key byte-encoded."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    vocab = {_byte_level(t): i for i, t in enumerate(vocab_texts)}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(list(wrongly_added))
    return tok


# Laguna-S-2.1's layout: a non-ASCII control token at id 2 that is also bos/eos
_TOKENS = ["a", "b", "〈|EOS|〉", "</assistant>", "<think>", "IsNotNull"]
_TYPES = [1, 1, 3, 3, 4, 1]


def test_non_ascii_control_token_is_pinned_to_its_file_id():
    # naming the raw text as bos/eos made the converter append it past the vocab (Laguna got id 100352)
    tok = _converted(_TOKENS, wrongly_added=["〈|EOS|〉"])
    assert tok.token_to_id("〈|EOS|〉") == len(_TOKENS)

    fixed = _pin_control_tokens(tok, {"tokenizer.ggml.tokens": _TOKENS, "tokenizer.ggml.token_type": _TYPES})

    assert fixed.token_to_id("〈|EOS|〉") == 2
    assert fixed.encode("〈|EOS|〉ab").ids == [2, 0, 1]
    assert fixed.decode([2, 0], skip_special_tokens=False) == "〈|EOS|〉a"
    assert fixed.get_vocab_size(with_added_tokens=True) == len(_TOKENS)
    assert fixed.id_to_token(5) == "IsNotNull"


def test_tokenizer_that_already_resolves_every_control_token_is_unchanged():
    tokens = ["a", "b", "<|eos|>"]
    tok = _converted(tokens)
    fixed = _pin_control_tokens(tok, {"tokenizer.ggml.tokens": tokens, "tokenizer.ggml.token_type": [1, 1, 3]})
    assert fixed is tok
