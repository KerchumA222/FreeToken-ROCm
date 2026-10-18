"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF model config from the GGUF, which needs the PyPI ``gguf`` package and an architecture transformers knows (``qwen4exp`` is not one).
So we run only its tokenizer half: ``get_gguf_tokenizer`` and ``convert_gguf_tokenizer``.
"""

from __future__ import annotations

from .reader import load_gguf_metadata

# end-of-generation token names from llama.cpp (llama_vocab::impl::load in src/llama-vocab.cpp): a GGUF stores one eos id and no stop list
_EOG_NAMES = frozenset((
    "<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>", "<end_of_turn>",
    "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>", "<end_of_utterance>",
    "<eos>", "<turn|>", "<|tool_response>", "<\uff5cend\u2581of\u2581sentence\uff5c>", "[e~[", "<|ifm|im_end|>",
))


def load_gguf_tokenizer(model_path: str):
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.gguf import convert_gguf_tokenizer, get_gguf_tokenizer

    arch, tok_dict, special = get_gguf_tokenizer(model_path)
    fast, _ = convert_gguf_tokenizer(arch, tok_dict)
    meta = load_gguf_metadata(model_path)
    fast = _pin_control_tokens(fast, meta)
    _set_pre_tokenizer(fast, meta)
    # special: bos/eos/unk/pad as the file names them (None when absent) and the chat template
    return PreTrainedTokenizerFast(
        tokenizer_object=fast,
        **special,
        add_bos_token=bool(meta.get("tokenizer.ggml.add_bos_token")),
        add_eos_token=bool(meta.get("tokenizer.ggml.add_eos_token")),
    )


# llama.cpp's pre-tokenizer regexes (llama_vocab::impl::load in src/llama-vocab.cpp), applied in order, for the
# tokenizer.ggml.pre types the converter's GPT-2 split gets wrong: laguna glues one leading symbol to the word after it
# ("/home", ".yml"), so the GPT-2 split turns paths and code into token sequences the model never saw
_PRE_TOKENIZER_REGEXES = {
    "laguna": (
        r"[^\n]+|[\n]+",
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ),
}


def _set_pre_tokenizer(fast, meta) -> None:
    regexes = _PRE_TOKENIZER_REGEXES.get(meta.get("tokenizer.ggml.pre"))
    if regexes is None:
        return
    from tokenizers import Regex, pre_tokenizers

    fast.pre_tokenizer = pre_tokenizers.Sequence([
        *(pre_tokenizers.Split(Regex(r), behavior="isolated") for r in regexes),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])


_CONTROL, _USER_DEFINED = 3, 4  # llama.cpp token types


def _pin_control_tokens(fast, meta):
    """Give each control / user-defined GGUF token its file id as an added token.

    The gpt2 converter byte-encodes every vocab string, so a control token with non-ASCII text
    (Laguna's ``〈|EOS|〉`` at id 2) is stored as its byte form and the raw text no longer
    resolves: naming it as bos/eos then appends a new id past the embedding table. llama.cpp
    matches these tokens by their raw text, so pin that text to the file id. Tokens the
    converter already resolves are left as they are.
    """
    import json

    from tokenizers import Tokenizer

    tokens = meta.get("tokenizer.ggml.tokens")
    types = meta.get("tokenizer.ggml.token_type")
    if not tokens or types is None:
        return fast
    data = json.loads(fast.to_str())
    added = data.setdefault("added_tokens", [])
    by_content = {a["content"]: a for a in added}
    vocab = data.get("model", {}).get("vocab")
    key_of = {v: k for k, v in vocab.items()} if isinstance(vocab, dict) else {}
    changed = False
    for i, (text, kind) in enumerate(zip(tokens, types)):
        if int(kind) not in (_CONTROL, _USER_DEFINED):
            continue
        entry = by_content.get(text)
        if entry is not None and entry["id"] == i:
            continue
        if entry is None and fast.token_to_id(text) == i:
            continue
        if entry is not None:
            added.remove(entry)
        # tokenizers resolves an added token's id through the model vocab, so the byte-encoded key must become the raw text
        # (an id can be missing outright when two byte-encoded keys collided)
        old_key = key_of.get(i)
        if isinstance(vocab, dict) and old_key != text and text not in vocab:
            if old_key is not None:
                del vocab[old_key]
            vocab[text] = i
        by_content[text] = {"id": i, "content": text, "single_word": False, "lstrip": False, "rstrip": False,
                            "normalized": False, "special": int(kind) == _CONTROL}
        added.append(by_content[text])
        changed = True
    if not changed:
        return fast
    # the converter's own extras (a default <|endoftext|>) land past the file's vocab, where the model has no embedding row
    added[:] = [a for a in added if a["id"] < len(tokens)]
    added.sort(key=lambda a: a["id"])
    return Tokenizer.from_str(json.dumps(data))


def gguf_eos_token_ids(model_path: str) -> set[int]:
    """Stop ids for GGUF generation, picked the way llama.cpp picks its end-of-generation tokens."""
    meta = load_gguf_metadata(model_path)
    tokens = meta["tokenizer.ggml.tokens"]
    ids = {i for i, t in enumerate(tokens) if t in _EOG_NAMES}
    ids |= {int(meta[f"tokenizer.ggml.{k}_token_id"]) for k in ("eos", "eot", "eom") if f"tokenizer.ggml.{k}_token_id" in meta}
    names = {tokens[i]: i for i in ids if i < len(tokens)}
    # llama.cpp's exceptions: <|end|> only ends a message when <|call|> and <|return|> end the turn (gpt-oss), and gemma4's </s> is a normal token
    if "<|end|>" in names and names.keys() & {"<|call|>", "<|calls|>"} and names.keys() & {"<|return|>", "<|flush|>"}:
        ids.discard(names["<|end|>"])
    if "</s>" in names and names.keys() & {"<|tool_response>", "<|plamo:eos|>"}:
        ids.discard(names["</s>"])
    return ids


__all__ = ["load_gguf_tokenizer", "gguf_eos_token_ids"]
