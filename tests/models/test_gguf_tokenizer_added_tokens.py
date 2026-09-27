"""GGUF tokenizer: control and user-defined tokens encode as single ids.

Qwen's reasoning and tool-call markers (<think>, </think>, <tool_call>) are GGUF
USER_DEFINED tokens (token_type 4). Converted without registration, the byte-level
BPE splits them into '<', 'think', '>', so a chat template's ``<think>\\n`` prefix
reaches the model as plain text and the model ends its turn or garbles tool calls.
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from freetoken.models.gguf import tokenizer as gguf_tokenizer  # noqa: E402

_NORMAL, _CONTROL, _USER_DEFINED = 1, 3, 4


def _metadata() -> dict:
    # A byte-level (GPT-2 alphabet) vocab: printable ASCII maps to itself, "\n" to "Ċ".
    chars = sorted(set("<>/|_abcdefghijklmnopqrstuvwxyz")) + ["Ċ"]
    tokens = chars + ["th", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>"]
    types = [_NORMAL] * (len(chars) + 1) + [_CONTROL, _CONTROL, _USER_DEFINED, _USER_DEFINED, _USER_DEFINED]
    return {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.merges": ["t h"],  # the converter wants at least one
        "tokenizer.ggml.eos_token_id": tokens.index("<|im_end|>"),
        "tokenizer.ggml.padding_token_id": tokens.index("<|im_end|>"),
    }


@pytest.fixture
def tok(monkeypatch):
    meta = _metadata()
    monkeypatch.setattr(gguf_tokenizer, "load_gguf_metadata", lambda path: meta)
    monkeypatch.setattr(gguf_tokenizer, "gguf_architecture", lambda path: "qwen4exp")
    return gguf_tokenizer.load_gguf_tokenizer("unused.gguf"), meta["tokenizer.ggml.tokens"]


@pytest.mark.parametrize("marker", ["<|im_start|>", "<think>", "</think>", "<tool_call>"])
def test_marker_is_one_token(tok, marker):
    tokenizer, tokens = tok
    assert tokenizer.encode(marker, add_special_tokens=False) == [tokens.index(marker)]


def test_think_prefix_matches_the_template(tok):
    tokenizer, tokens = tok
    ids = tokenizer.encode("<|im_start|>assistant\n<think>\n", add_special_tokens=False)
    assert ids[0] == tokens.index("<|im_start|>")
    assert ids[-2:] == [tokens.index("<think>"), tokens.index("Ċ")]


def test_user_defined_tokens_survive_skip_special_decode(tok):
    # The reasoning and tool-call parsers read these markers from the decoded text.
    tokenizer, tokens = tok
    ids = [tokens.index("<think>"), tokens.index("a"), tokens.index("</think>"), tokens.index("<|im_end|>")]
    assert tokenizer.decode(ids, skip_special_tokens=True) == "<think>a</think>"
