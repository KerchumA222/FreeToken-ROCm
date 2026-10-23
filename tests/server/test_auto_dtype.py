"""`--dtype auto` on GPUs that only emulate bf16 (RDNA1/2): fp16, unless bf16 is asked for by name."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from freetoken.server import args as server_args
from freetoken.server.args import parse_args


class _Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return dict(self._data)


def _gpu(arch: str | None, hip: str | None = "7.1"):
    props = SimpleNamespace(gcnArchName=f"{arch}:sramecc-:xnack-") if arch else None
    get = (lambda _i: props) if props else (lambda _i: (_ for _ in ()).throw(RuntimeError("no device")))
    return patch.object(torch.version, "hip", hip), patch.object(torch.cuda, "get_device_properties", get)


@pytest.mark.parametrize(
    "arch, hip, lacks",
    [("gfx1030", "7.1", True), ("gfx1010", "7.1", True), ("gfx1100", "7.1", False), ("gfx90a", "7.1", False),
     ("gfx942", "7.1", False), ("gfx1030", None, False), (None, "7.1", False)],
)
def test_bf16_hardware_detection(arch, hip, lacks):
    hip_patch, props_patch = _gpu(arch, hip)
    with hip_patch, props_patch:
        assert server_args._gpu_lacks_bf16_hardware() is lacks


@pytest.mark.parametrize(
    "requested, checkpoint, lacks, expected",
    [
        ("auto", "bfloat16", True, torch.float16),  # emulated bf16: auto falls back to fp16
        ("auto", "bfloat16", False, torch.bfloat16),
        ("auto", "float16", True, torch.float16),
        ("bfloat16", "bfloat16", True, torch.bfloat16),  # asked for by name: honored
        ("float16", "bfloat16", False, torch.float16),
    ],
)
def test_auto_dtype_resolution(requested, checkpoint, lacks, expected):
    config = _Config({"architectures": ["Qwen3MoeForCausalLM"], "torch_dtype": checkpoint})
    with (
        patch("freetoken.utils.cached_load_hf_config", lambda _path: config),
        patch.object(server_args, "_gpu_lacks_bf16_hardware", lambda: lacks),
    ):
        args, _ = parse_args(["--model", "/models/anon", "--dtype", requested])
    assert args.dtype == expected
