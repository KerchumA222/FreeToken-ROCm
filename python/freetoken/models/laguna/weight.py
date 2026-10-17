from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import (
    MergeRule,
    iter_merged_tensors,
    iter_stacked_experts,
    iter_weight_files,
    shard_tensor,
)
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")
_MERGE_RULES = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}


def _canonical_name(raw_name: str) -> str:
    name = raw_name.removeprefix("language_model.")
    # Transformers exposes the correction vector on the router, while Laguna's
    # checkpoint stores it next to the indexed expert bank.
    return name.replace(
        ".mlp.experts.e_score_correction_bias",
        ".mlp.e_score_correction_bias",
    )


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()

    def sharded() -> Iterator[tuple[str, torch.Tensor]]:
        for file in tqdm(
            iter_weight_files(model_path),
            desc="Loading Laguna weights",
            disable=not tp_info.is_primary(),
        ):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as tensors:
                for raw_name in tensors.keys():
                    name = _canonical_name(raw_name)
                    is_expert = _EXPERT_PATTERN.match(name) is not None
                    if is_expert and not include_moe_experts:
                        continue
                    if not is_expert and not include_non_moe:
                        continue
                    raw = tensors.get_tensor(raw_name)
                    value = shard_tensor(
                        name,
                        raw,
                        rank=tp_info.rank,
                        world_size=tp_info.size,
                        num_kv_heads=config.num_kv_heads,
                    )
                    del raw
                    yield name, value

    merged = iter_merged_tensors(sharded(), _MERGE_RULES, model_name="laguna")
    yield from iter_stacked_experts(
        merged,
        num_experts=config.num_experts,
        model_name="laguna",
        expert_pattern=_EXPERT_PATTERN,
    )


__all__ = ["iter_weights"]
