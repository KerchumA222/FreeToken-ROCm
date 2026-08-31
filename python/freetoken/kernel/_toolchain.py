"""CUDA toolchain/torch consistency checks.

Standalone on purpose: setup.py and the kernel-cache build backend load this
file by path, so it must not import the freetoken package.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess

ALLOW_MISMATCH_ENV = "FREETOKEN_ALLOW_CUDA_MISMATCH"
_TRUE_VALUES = {"1", "true", "yes", "on"}


@functools.cache
def ensure_rocm_env() -> str | None:
    """Point the JIT at the ROCm runtime inside the venv when nothing else has.

    On the Windows/ROCm port there is no system ROCm install: the TheRock wheels unpack
    the runtime into ``site-packages/_rocm_sdk_core`` and nothing sets ``HIP_PATH``. The
    patched ``tvm_ffi`` decides the ENTIRE Windows toolchain from that one variable --
    clang++ vs cl for the host compile, and whether ``-lamdhip64`` reaches the link -- so
    without it every device kernel that is not already cached fails to build, with a link
    error naming HIP symbols rather than anything about the environment. It was set only
    by ``dist/run-server.ps1``, which made the JIT work from the launcher and nowhere
    else: not from the test suite, not from ``ft serve`` run directly.

    ``ROCM_HOME`` is filled in for the same reason (``tvm_ffi``'s ``_find_rocm_home``
    reads it). ``ROCM_PATH`` is deliberately NOT set -- it sends clang to
    ``%ROCM_PATH%/amdgcn/bitcode``, which is not the wheel layout, and every kernel build
    then fails to find the device bitcode. Anything already in the environment wins.
    """
    import torch

    if not getattr(torch.version, "hip", None):
        return None
    existing = os.environ.get("HIP_PATH")
    if existing:
        os.environ.setdefault("ROCM_HOME", existing)
        return existing
    try:
        import _rocm_sdk_core
    except ImportError:
        return None
    root = os.path.dirname(_rocm_sdk_core.__file__)
    if not os.path.exists(os.path.join(root, "lib", "llvm", "bin", "clang.exe")) and not (
        os.path.exists(os.path.join(root, "lib", "llvm", "bin", "clang"))
    ):
        return None
    os.environ["HIP_PATH"] = root
    os.environ.setdefault("ROCM_HOME", root)
    _ensure_rocm_arch()
    return root


def _ensure_rocm_arch() -> None:
    """Name the target GPU family, since the wheel has no ``rocm_agent_enumerator``.

    ``tvm_ffi`` shells out to that binary to detect the arch and raises when it is
    missing; torch's own JIT silently defaults to gfx906 and builds dead kernels, and
    compiles for EVERY visible device -- including the 9800X3D's iGPU (gfx1036), whose
    build failure kills the backend worker after a full model load. The device itself is
    the authority, so read the arch off it rather than hardcoding one as the launcher
    does. Anything already set wins.
    """
    if all(
        os.environ.get(name)
        for name in ("TVM_FFI_ROCM_ARCH_LIST", "PYTORCH_ROCM_ARCH", "TRITON_OVERRIDE_ARCH")
    ):
        return
    import torch

    try:
        if not torch.cuda.is_available():
            return
        # e.g. "gfx1201:xnack-" -> "gfx1201"
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0].strip()
    except Exception:
        return
    if not arch:
        return
    for name in (
        "TVM_FFI_ROCM_ARCH_LIST",
        "PYTORCH_ROCM_ARCH",
        "TRITON_OVERRIDE_ARCH",
        "ROCM_SDK_TARGET_FAMILY",
    ):
        os.environ.setdefault(name, arch)


def _nvcc_path() -> str | None:
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME:
        return os.path.join(CUDA_HOME, "bin", "nvcc")
    return shutil.which("nvcc")


def nvcc_release(nvcc: str) -> tuple[int, int] | None:
    try:
        proc = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release (\d+)\.(\d+)", proc.stdout)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def torch_cuda_major() -> int | None:
    import torch

    cuda = getattr(torch.version, "cuda", None)
    return int(cuda.split(".")[0]) if cuda else None


@functools.cache
def check_nvcc_matches_torch() -> None:
    """Refuse to nvcc-compile kernels across CUDA majors.

    nvcc-built binaries link libcudart.so.<nvcc major>; at runtime only the
    torch wheel's own CUDA runtime is guaranteed to be loadable.
    """
    if os.getenv(ALLOW_MISMATCH_ENV, "").strip().lower() in _TRUE_VALUES:
        return
    torch_major = torch_cuda_major()
    if torch_major is None:
        return
    nvcc = _nvcc_path()
    if nvcc is None:
        return
    release = nvcc_release(nvcc)
    if release is None:
        return
    if release[0] != torch_major:
        import torch

        raise RuntimeError(
            f"nvcc {release[0]}.{release[1]} would build kernels linking "
            f"libcudart.so.{release[0]}, but torch {torch.__version__} ships CUDA "
            f"{torch.version.cuda} (libcudart.so.{torch_major}). Install a CUDA "
            f"{torch_major}.x toolkit, or set {ALLOW_MISMATCH_ENV}=1 to override."
        )
