"""Triton codegen limits of RDNA2 (gfx103x)."""

from __future__ import annotations

import functools

import torch
import triton


@functools.cache
def lacks_bf16_dot2() -> bool:
    """Whether the current GPU is gfx103x, which has no ``v_dot2_bf16_bf16``.

    Triton 3.8's AMDGPU backend pairs two bf16 products that are summed into that instruction, and a kernel that does so
    aborts the whole process at compile time ("LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.fdot2.bf16.bf16").
    Multiplying in fp32 first keeps the pairing from forming; fp16 has the instruction and needs nothing."""
    if torch.version.hip is None:
        return False
    try:
        return str(triton.runtime.driver.active.get_current_target().arch).startswith("gfx103")
    except Exception:
        return False


__all__ = ["lacks_bf16_dot2"]
