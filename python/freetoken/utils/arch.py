from __future__ import annotations

import functools
from typing import Tuple


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    return torch.cuda.get_device_capability()


def is_arch_supported(major: int, minor: int = 0) -> bool:
    """capability >= (major, minor). Open-ended: newer archs also pass. Only use this
    for family-portable features (e.g. PDL); arch-specific kernels (sm_90a/sm_100a
    cubins) need the closed is_smXX_family checks below."""
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    return arch >= (major, minor)


def _is_arch_family(major: int) -> bool:
    arch = _get_torch_cuda_version()
    return arch is not None and arch[0] == major


def is_sm90_family() -> bool:
    """Exactly major 9 (Hopper). For sm_90a-only kernels (e.g. FA3)."""
    return _is_arch_family(9)


def is_sm100_family() -> bool:
    """Exactly major 10 (datacenter Blackwell). For sm_100a/103a-only kernels
    (e.g. trtllm-gen) that consumer Blackwell (sm_120/121) cannot run."""
    return _is_arch_family(10)


def is_sm90_supported() -> bool:
    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    return is_arch_supported(10, 0)


def pdl_launch_kwargs(enabled: bool) -> dict[str, bool]:
    """``{"launch_pdl": True}`` when PDL is on, otherwise ``{}``.

    ``launch_pdl`` is an NVIDIA-only launch option: Triton's AMD backend has no
    such field on ``HIPOptions`` and rejects the *name* outright with
    ``KeyError: Keyword argument launch_pdl was specified but unrecognised`` --
    passing ``launch_pdl=False`` is just as fatal as passing True. Since PDL is
    gated on :func:`is_sm90_supported`, which is False on every ROCm device,
    omitting the kwarg whenever it would be False keeps CUDA behaviour identical
    (False is the default there) and keeps the kernel launchable on HIP.
    """
    return {"launch_pdl": True} if enabled else {}
