"""Several tokens for one request in one reply (a speculative accept) must stream once."""

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager


class _Tok:
    eos_token_id = 0
    vocab = {1: "Hello", 2: ",", 3: " world", 4: "!"}

    def batch_decode(self, ids_list):
        return ["".join(self.vocab[i] for i in ids) for ids in ids_list]


def _msg(uid, tok):
    return DetokenizeMsg(uid=uid, next_token=tok, finished=False, finish_reason=None,
                         matched_stop=None, stop_strs=None)


def test_two_tokens_same_uid_in_one_batch_stream_once():
    m = DetokenizeManager(_Tok(), eos_token_ids=frozenset({0}))
    out = m.detokenize([_msg(7, 1)])
    out += m.detokenize([_msg(7, 2), _msg(7, 3), _msg(8, 1)])
    out += m.detokenize([_msg(7, 4)])
    assert "".join(o for o, u in zip(out, [7, 7, 7, 8, 7]) if u == 7) == "Hello, world!"
    assert out[3] == "Hello"
