"""`--tool-call-parser auto` / `--reasoning-parser auto`: architecture -> parser name.

Three tables are maintained independently -- the model registry, the substring cascade in
``server/args.py``, and the two parser factories -- and nothing links them. Add a model family and
forget the cascade, and the model serves perfectly: it just stops emitting tool calls and stops
separating out its thinking, because it silently fell through to the generic default. Every other
parser test still passes, since they all start from a parser name that is already correct.

So this drives the *live* registry rather than a list: a newly registered architecture is covered
the moment it is added, and has to be dispositioned here to stay green.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from freetoken.models.register import _MODEL_REGISTRY
from freetoken.server.args import parse_args
from freetoken.server.function_call_parser import FunctionCallParser
from freetoken.server.reasoning_parser import ReasoningParser

ARCHITECTURES = sorted(_MODEL_REGISTRY)

# Most tests use an anonymous path so inference has to use checkpoint metadata. Tests for
# versioned model-path markers pass an explicit path to `_inferred`.
ANON_PATH = "/models/anon"

# Families with no thinking format of their own. Everything else must resolve to a real reasoning
# parser, and a new architecture landing here is the bug this file exists to catch.
NO_REASONING_FORMAT = {
    "LlamaForCausalLM",
    "LagunaForCausalLM",
    "LagunaGGUFForCausalLM",
    "MistralForCausalLM",
    "Mistral3ForConditionalGeneration",
    "Qwen2ForCausalLM",
}

# `llama3` is the end of the cascade -- the answer when nothing matched.
GENERIC_TOOL_CALL_FALLBACK = "llama3"
NO_DEDICATED_TOOL_FORMAT = {
    "LlamaForCausalLM",
    "LagunaForCausalLM",
    "LagunaGGUFForCausalLM",
}


class _Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def _inferred(
    architecture: str,
    *,
    model_path: str = ANON_PATH,
    laguna_enabled: bool = False,
) -> tuple[str, str | None]:
    """(tool_call_parser, reasoning_parser) that `auto` picks for this architecture."""
    config = _Config({"architectures": [architecture], "torch_dtype": "bfloat16"})
    laguna = "1" if laguna_enabled else "0"
    with (
        patch("freetoken.utils.cached_load_hf_config", lambda _path: config),
        patch.dict(
            "os.environ",
            {
                "FREETOKEN_ENABLE_LAGUNA_HF": laguna,
                "FREETOKEN_ENABLE_LAGUNA_GGUF": laguna,
            },
        ),
    ):
        args, _run_shell = parse_args(["--model", model_path])
    return args.tool_call_parser, args.reasoning_parser


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_inferred_parser_names_are_names_the_factories_know(architecture):
    """A name the cascade invents but no factory can build fails at request time, not at boot."""
    tool_call, reasoning = _inferred(architecture)

    assert tool_call in FunctionCallParser.ToolCallParserEnum, tool_call
    if reasoning is not None:
        assert reasoning in ReasoningParser.ReasoningParserEnum, reasoning


def test_only_the_families_without_a_thinking_format_get_no_reasoning_parser():
    fell_through = {a for a in ARCHITECTURES if _inferred(a)[1] is None}
    assert fell_through == NO_REASONING_FORMAT


def test_only_the_families_without_a_tool_format_get_the_generic_fallback():
    fell_through = {
        a for a in ARCHITECTURES if _inferred(a)[0] == GENERIC_TOOL_CALL_FALLBACK
    }
    assert fell_through == NO_DEDICATED_TOOL_FORMAT


def test_laguna_parser_inference_is_opt_in():
    assert _inferred("LagunaForCausalLM") == ("llama3", None)
    assert _inferred("LagunaForCausalLM", laguna_enabled=True) == ("glm47", "qwen3")
    assert _inferred("LagunaGGUFForCausalLM", laguna_enabled=True) == ("glm47", "qwen3")


def test_qwen3_5_and_qwen3_6_are_not_shadowed_by_the_generic_qwen_branch():
    """The cascade matches substrings in order, so the specific arm has to come first: a bare
    ``"qwen" -> qwen25`` reached earlier would swallow later Qwen families and lose their format."""
    assert _inferred("Qwen3_5MoeForConditionalGeneration")[0] == "qwen3_coder"
    assert _inferred("Qwen3_5ForConditionalGeneration")[0] == "qwen3_coder"
    assert _inferred("Qwen3_5GGUFForCausalLM") == ("qwen3_coder", "qwen3")
    assert _inferred("Qwen3_5MoeGGUFForCausalLM") == ("qwen3_coder", "qwen3")
    assert _inferred("Qwen3.6-27B") == ("qwen3_coder", "qwen3")
    assert _inferred("Qwen3_6_27B") == ("qwen3_coder", "qwen3")
    assert _inferred("Qwen3MoeForCausalLM")[0] == "qwen25"


@pytest.mark.parametrize(
    "architecture, model_path",
    [
        ("Qwen3_5GGUFForCausalLM", ANON_PATH),
        ("Qwen3_5MoeGGUFForCausalLM", ANON_PATH),
        # Qwen3.6-27B publishes this Qwen3.5 architecture in its config.
        ("Qwen3_5ForConditionalGeneration", "Qwen/Qwen3.6-27B"),
    ],
)
def test_qwen3_5_and_qwen3_6_inference_parses_structured_tool_call(
    architecture, model_path
):
    tool_call_parser, reasoning_parser = _inferred(architecture, model_path=model_path)
    parser = FunctionCallParser(
        [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {
                        "type": "object",
                        "properties": {"filePath": {"type": "string"}},
                    },
                },
            }
        ],
        tool_call_parser=tool_call_parser,
    )

    result = parser.parse_non_stream(
        "<tool_call><function=read><parameter=filePath>/tmp/test_calc.py</parameter>"
        "</function></tool_call>"
    )

    assert tool_call_parser == "qwen3_coder"
    assert reasoning_parser == "qwen3"
    assert result.normal_text == ""
    assert len(result.calls) == 1
    assert result.calls[0].name == "read"
    assert json.loads(result.calls[0].parameters) == {"filePath": "/tmp/test_calc.py"}


def test_deepseek_v41_is_not_shadowed_by_the_v4_branch():
    """V4.1's DSML leads every tag name with a space; the V4 arm (``"v4"`` is a substring of
    ``"v41"``) would hand it a tool parser that never sees a call."""
    assert _inferred("DeepseekV41ForCausalLM") == ("deepseekv41", "deepseekv32")
    assert _inferred("DeepseekV4ForCausalLM")[0] == "deepseekv32"


def test_an_explicit_choice_beats_inference():
    config = _Config(
        {"architectures": ["Qwen3_5ForConditionalGeneration"], "torch_dtype": "bfloat16"}
    )
    model_path = "Qwen/Qwen3.6-27B"
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        off, _ = parse_args(["--model", model_path, "--reasoning-parser", "off"])
        pinned_reasoning, _ = parse_args(
            ["--model", model_path, "--reasoning-parser", "qwen3"]
        )
        pinned_tool, _ = parse_args(
            ["--model", model_path, "--tool-call-parser", "mistral"]
        )
    assert off.reasoning_parser is None
    assert pinned_reasoning.reasoning_parser == "qwen3"
    assert pinned_tool.tool_call_parser == "mistral"
