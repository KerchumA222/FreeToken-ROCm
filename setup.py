from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import warnings
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension, ROCM_HOME


ROOT = Path(__file__).parent
KERNEL_INCLUDE = str(ROOT / "python" / "freetoken" / "kernel" / "csrc" / "include")
IS_WINDOWS = os.name == "nt"


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _is_rocm() -> bool:
    import torch

    return getattr(torch.version, "hip", None) is not None


def _rocm_paths() -> tuple[list[str], list[str], str]:
    candidates: list[Path] = []
    if os.getenv("ROCM_HOME"):
        candidates.append(Path(os.environ["ROCM_HOME"]))
    if ROCM_HOME:
        candidates.append(Path(ROCM_HOME))
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec and spec.submodule_search_locations:
        candidates.append(Path(next(iter(spec.submodule_search_locations))))
    candidates.append(Path("/opt/rocm"))

    for rocm_home in dict.fromkeys(candidates):
        include_dir = rocm_home / "include"
        library_dir = rocm_home / "lib"
        if not (include_dir / "hip" / "hip_runtime.h").exists():
            continue
        if IS_WINDOWS:
            if (library_dir / "amdhip64.lib").exists():
                return [str(include_dir)], [str(library_dir)], "amdhip64"
            continue
        if (library_dir / "libamdhip64.so").exists():
            return [str(include_dir)], [str(library_dir)], "amdhip64"
        versioned = sorted(library_dir.glob("libamdhip64.so.*"))
        if versioned:
            return [str(include_dir)], [str(library_dir)], f":{versioned[-1].name}"

    searched = ", ".join(str(path) for path in dict.fromkeys(candidates))
    raise RuntimeError(
        "A ROCm SDK with HIP headers and libamdhip64 is required to build on ROCm; "
        f"searched: {searched}. Set ROCM_HOME to override."
    )


def _cuda_runtime_paths() -> tuple[list[str], list[str]]:
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA runtime API."
        )
    cuda_home = Path(CUDA_HOME)
    library_dirs = [str(cuda_home / "lib64")]
    if (cuda_home / "lib").exists():
        library_dirs.append(str(cuda_home / "lib"))
    return [str(cuda_home / "include")], library_dirs


def _clang_cxx() -> str | None:
    cxx = shutil.which(os.environ.get("CXX") or "") or ""
    return cxx if "clang" in Path(cxx).stem.lower() else None


def _clang_rt_builtins() -> list[str]:
    cxx = _clang_cxx()
    if not IS_WINDOWS or cxx is None:
        return []
    resource_dir = subprocess.run(
        [cxx, "-print-resource-dir"], capture_output=True, text=True, check=True
    ).stdout.strip()
    lib = Path(resource_dir) / "lib" / "windows" / "clang_rt.builtins-x86_64.lib"
    if not lib.exists():
        raise RuntimeError(
            f"{cxx} is the configured compiler but {lib} is missing; install LLVM's "
            "compiler-rt component (freetoken.kernel._cpu_moe links against it)."
        )
    return [str(lib)]


def _cpu_moe_extensions(
    extra_compile: list[str],
    thread_compile_args: list[str],
    runtime_include_dirs: list[str],
    runtime_library_dirs: list[str],
    runtime_lib: str,
    runtime_link_args: list[str],
) -> list[CppExtension]:
    if IS_WINDOWS and _clang_cxx() is None:
        warnings.warn(
            "freetoken.kernel._cpu_moe is not being built: its runtime ISA dispatch "
            "needs a clang driver, and CXX is unset or MSVC. Set CXX=clang-cl to build "
            "it; without it --moe-backend cpu and hybrid are unavailable.",
            stacklevel=2,
        )
        return []
    compile_args = extra_compile + thread_compile_args
    if _is_rocm():
        compile_args = compile_args + ["-D__HIP_PLATFORM_AMD__=1", "-DUSE_ROCM=1"]
    return [
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=[
                "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            ],
            include_dirs=[KERNEL_INCLUDE, *runtime_include_dirs],
            library_dirs=runtime_library_dirs,
            libraries=[runtime_lib],
            extra_compile_args=compile_args,
            extra_link_args=runtime_link_args + _clang_rt_builtins(),
        )
    ]


IS_ROCM = _is_rocm()
if IS_WINDOWS:
    extra_compile = ["/O2", "/std:c++17"]
    thread_compile_args: list[str] = []
else:
    extra_compile = ["-O3", "-std:c++17"]
    thread_compile_args = ["-pthread"]

if IS_ROCM:
    runtime_include_dirs, runtime_library_dirs, runtime_lib = _rocm_paths()
    runtime_link_args = [] if IS_WINDOWS else [f"-Wl,-rpath,{runtime_library_dirs[0]}"]
else:
    runtime_include_dirs, runtime_library_dirs = _cuda_runtime_paths() if CUDA_HOME else ([], [])
    runtime_lib = "cudart"
    runtime_link_args = []

# CUDA-only _pinned_tensor is optional; skip it when no CUDA toolchain is present
# (ROCm builds use torch's own pinned-memory path instead).
ext_modules: list[CppExtension] = []
if os.environ.get("FREETOKEN_SKIP_CUDA_EXT") != "1" and CUDA_HOME is not None and not IS_ROCM:
    _check_toolchain()
    cuda_include_dirs, cuda_library_dirs = _cuda_runtime_paths()
    ext_modules.append(
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=[KERNEL_INCLUDE, *cuda_include_dirs],
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=extra_compile,
        )
    )

if IS_ROCM or CUDA_HOME is not None:
    if not IS_ROCM and CUDA_HOME is not None:
        _check_toolchain()
    ext_modules.extend(
        _cpu_moe_extensions(
            extra_compile,
            thread_compile_args,
            runtime_include_dirs,
            runtime_library_dirs,
            runtime_lib,
            runtime_link_args,
        )
    )


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
