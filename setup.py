from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDA_HOME,
    ROCM_HOME,
    CppExtension,
)


ROOT = Path(__file__).parent
CSRC = ROOT / "python" / "freetoken" / "kernel" / "csrc"


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _is_rocm() -> bool:
    """ROCm torch, i.e. the extensions link amdhip64 rather than cudart."""
    import torch

    return getattr(torch.version, "hip", None) is not None


def _gpu_runtime_paths() -> tuple[list[str], list[str], list[str], list[str]]:
    """(include_dirs, library_dirs, libraries, extra_compile_args) for the host-side
    GPU runtime the two extensions call into.

    ``_pinned_tensor`` and ``_cpu_moe`` are plain C++ -- no ``__global__`` kernels --
    so torch never hipifies them; ``freetoken/gpu_runtime.h`` aliases the handful of
    ``cuda*`` runtime calls onto their ``hip*`` twins instead. On ROCm that means
    linking ``amdhip64`` and defining ``__HIP_PLATFORM_AMD__`` so the header picks
    the right branch.
    """
    common_includes = [str(CSRC / "include")]

    if _is_rocm():
        if ROCM_HOME is None:
            raise RuntimeError(
                "ROCM_HOME is required to build the freetoken C++ extensions against "
                "a ROCm torch. Set ROCM_HOME (e.g. /opt/rocm) or "
                "FREETOKEN_SKIP_CUDA_EXT=1 to skip them."
            )
        rocm_home = Path(ROCM_HOME)
        library_dirs = [str(rocm_home / "lib")]
        if (rocm_home / "lib64").exists():
            library_dirs.append(str(rocm_home / "lib64"))
        return (
            common_includes + [str(rocm_home / "include")],
            library_dirs,
            ["amdhip64"],
            ["-D__HIP_PLATFORM_AMD__=1"],
        )

    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA runtime API."
        )
    cuda_home = Path(CUDA_HOME)
    library_dirs = [str(cuda_home / "lib64")]
    if (cuda_home / "lib").exists():
        library_dirs.append(str(cuda_home / "lib"))
    return common_includes + [str(cuda_home / "include")], library_dirs, ["cudart"], []


# The extensions are optional: skip them when there is no GPU toolchain to build
# against (FREETOKEN_SKIP_CUDA_EXT=1, or a CPU-only torch). Without them the
# engine falls back to torch's own pinned-memory path and loses --moe-backend cpu.
ext_modules = []
if os.environ.get("FREETOKEN_SKIP_CUDA_EXT") != "1":
    _rocm = _is_rocm()
    if _rocm or CUDA_HOME is not None:
        if not _rocm:
            _check_toolchain()
        gpu_includes, gpu_library_dirs, gpu_libraries, gpu_cflags = _gpu_runtime_paths()
        ext_modules = [
            CppExtension(
                name="freetoken.kernel._pinned_tensor",
                sources=["python/freetoken/kernel/csrc/pinned_tensor.cpp"],
                include_dirs=gpu_includes,
                library_dirs=gpu_library_dirs,
                libraries=gpu_libraries,
                extra_compile_args=["-O3", "-std=c++17", *gpu_cflags],
            ),
            # CPU-compute MoE executor for --moe-backend cpu. Links the GPU runtime
            # for the cudaLaunchHostFunc/hipLaunchHostFunc submit/sync graph nodes;
            # the bf16 GEMV microkernels use per-function target attributes
            # (avx512bf16/avx512f) + a runtime __builtin_cpu_supports dispatch, so
            # the single binary stays portable (scalar fallback) -- no global -march.
            CppExtension(
                name="freetoken.kernel._cpu_moe",
                sources=["python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp"],
                include_dirs=gpu_includes,
                library_dirs=gpu_library_dirs,
                libraries=gpu_libraries,
                extra_compile_args=["-O3", "-std=c++17", "-pthread", *gpu_cflags],
            ),
        ]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
