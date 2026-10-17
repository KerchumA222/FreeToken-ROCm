from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager


class Tokenizer:
    def batch_decode(self, sequences):
        pieces = {220: " ", 17: "2", 18: "3"}
        return ["".join(pieces[token] for token in sequence) for sequence in sequences]


def test_batch_decodes_multiple_tokens_for_one_request_once():
    manager = DetokenizeManager(Tokenizer(), eos_token_ids=frozenset())
    messages = [
        DetokenizeMsg(uid=1, next_token=220, finished=False),
        DetokenizeMsg(uid=1, next_token=17, finished=False),
        DetokenizeMsg(uid=1, next_token=220, finished=False),
        DetokenizeMsg(uid=1, next_token=18, finished=False),
    ]

    output = manager.detokenize(messages)

    assert output == [" ", "2", " ", "3"]
    assert manager.decode_map[1].decoded_str == " 2 3"


def test_stop_string_split_across_tokens_is_withheld_and_trimmed():
    class StopTokenizer:
        def batch_decode(self, sequences):
            pieces = {1: "hello S", 2: "TOP world"}
            return ["".join(pieces[token] for token in sequence) for sequence in sequences]

    manager = DetokenizeManager(StopTokenizer(), eos_token_ids=frozenset())
    first = manager.detokenize([
        DetokenizeMsg(uid=7, next_token=1, finished=False, stop_strs=["STOP"])
    ])
    final = manager.detokenize([
        DetokenizeMsg(
            uid=7,
            next_token=2,
            finished=True,
            matched_stop="STOP",
            stop_strs=["STOP"],
        )
    ])

    assert first == ["hello "]
    assert final == [""]
    assert 7 not in manager.decode_map


def test_partial_unicode_is_recovered_without_duplication():
    class PartialUnicodeTokenizer:
        def batch_decode(self, sequences):
            text = {
                (): "",
                (1,): "A",
                (1, 2): "A�",
                (1, 2, 3): "Aé",
            }
            return [text[tuple(sequence)] for sequence in sequences]

    manager = DetokenizeManager(PartialUnicodeTokenizer(), eos_token_ids=frozenset())
    output = []
    for token in (1, 2, 3):
        output.extend(manager.detokenize([
            DetokenizeMsg(uid=8, next_token=token, finished=False)
        ]))

    assert "".join(output) == "Aé"
    assert manager.decode_map[8].decoded_str == "Aé"


def test_discard_removes_aborted_request_state():
    manager = DetokenizeManager(Tokenizer(), eos_token_ids=frozenset())
    manager.detokenize([DetokenizeMsg(uid=9, next_token=220, finished=False)])

    manager.discard(9)

    assert 9 not in manager.decode_map
