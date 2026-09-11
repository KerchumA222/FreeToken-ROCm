<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-light.svg">
    <img alt="FreeToken" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo.svg" width=65%>
  </picture>
</div>

> [!IMPORTANT]
> **This fork: native Windows + AMD ROCm port** (`FreeToken-rocm-test`).
> Verified end-to-end on an AMD Radeon RX 9070 XT & RX 9060 XT (gfx1201 / gfx1200 / RDNA4, 16 GB, Windows 11):
> `ft serve` loads dense HF safetensors models, serves OpenAI-compatible chat
> completions (including SSE token streaming) through Triton-on-AMD attention
> kernels and hipcc/tvm-ffi JIT-compiled CUDA-C++ kernels. See
> [Windows ROCm port](#windows-rocm-port) below for requirements, setup and switches.

<p align="center">
| <a href="https://www.flashml.ai/"><b>Download</b></a> | <a href="https://arxiv.org/abs/2608.16157"><b>Paper</b></a> | <a href="https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA"><b>Developer Slack</b></a> | <a href="https://discord.gg/MsA277cJzZ"><b>Community Discord</b></a> | <a href="https://github.com/FlashML-org/FreeToken/blob/main/assets/freetoken-wechatgroup.png"><b>Community WeChat</b></a> |
</p>


Unlock datacenter-class intelligence on the hardware you already own — Run 290B+ frontier MoE models locally on your gaming PC at blistering interactive speeds.

## About

FreeToken is an edge-native Mixture-of-Experts (MoE) serving engine designed for running frontier-scale open-weight models on personal and consumer hardware. It treats heterogeneous edge resources—GPUs, CPUs, host memory, and interconnects—as a unified, elastic inference platform. Its core features include:  

- **Fast Edge-Native Runtime**: Provides efficient MoE serving with bandwidth-adaptive CPU–GPU co-execution ($q^\star$ policy), full-layer double-buffered prefill streaming, global LRU expert caching, graph-compatible execution, and the FTW fast weight format.  
- **Semantic-Aware Caching**: Features semantic anchor checkpoints for recurrent state and KV caches, allowing agentic context edits (e.g., tool calls, thinking blocks) to avoid redundant context recomputation.  
- **Elastic Memory Management**: Supports dynamic, runtime VRAM re-allocation between expert caches and KV memory without engine restarts or weight reloading.  
- **Broad MoE & Ecosystem Support**: Supports frontier open-weight MoE models (e.g., DeepSeek-V4-Flash, Qwen3.6-35B-A3B, GLM-5.2) across various parameter scales and quantization formats (e.g., MXFP4, NVFP4, FP8, BF16), with Anthropic/OpenAI-compatible APIs for seamless integration with real-world coding and tool-calling agents (e.g., Codex, Claude Code, OpenCode, OpenClaw, DeepSeek Harness). 
- **Diverse Consumer Hardware**: Scales across consumer laptops, gaming desktops, and workstation GPUs, with native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs.  

## Getting Started

### Desktop app

Download FreeToken for Windows or Linux at [flashml.ai](https://www.flashml.ai/). It sets the engine up for you and gives you a GUI for running models, chatting, and tuning the engine.

<div align="center">
  <img alt="FreeToken Desktop" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/desktop-console.png" width=92%>
</div>

### CLI

Install FreeToken with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv pip install "freetoken[accel]"
```

Or build from source:

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

For More details:

- [Install FreeToken](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
- [Quick start](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)
- [Supported models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- [CLI reference](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)
- [Repairing old FTW checkpoints](https://github.com/FlashML-org/FreeToken/blob/main/docs/ftw-hotfix.md)

## Windows ROCm Port

This fork brings FreeToken up on **Windows 11 + AMD ROCm** with no NVIDIA toolchain.
Bring-up target: RX 9070 XT (`gfx1201`). Everything below was verified live: model load,
prefill, decode (~57 tok/s bf16 3B), SSE token streaming, and the bundled mini web UI.

### Measured performance (RX 9070 XT, Windows, stock settings)

| Model | Precision | Fit | Decode speed | Notes |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | BF16 | full VRAM | **~72 tok/s** | stable, coherent; SSE first-token ~0.4 s |
| Qwen2.5-7B-Instruct | BF16 | `--num-pages 4096` | **~16 tok/s** | weights leave only ~1.5 GB headroom |
| gpt-oss-20b (MoE) | MXFP4 experts | `--moe-backend fused --num-pages 4096 --cuda-graph-max-bs 0` | **~12 tok/s** | stable in eager mode; see graph bug below |
| gpt-oss GGUF files | Q8_0 | - | - | rejected: no MoE GGUF adapter yet (dense llama/qwen2/mistral/qwen3/gemma4 supported) |

**RX 9060 XT (gfx1200, 16 GB) — packed GGUF, verified end-to-end:**

| Model | Precision | Config | Decode speed | Notes |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct GGUF | Q4_K_M (packed) | **defaults** (graphs on, HIP MMVQ replayed) | **~93.6 tok/s** | weights stay packed (~1.9 GB VRAM); HIP kernels replay fine on gfx1200 — the graph-replay crash is gfx1201-specific, and the auto backend now falls back to Triton only there |
| Qwen2.5-3B-Instruct GGUF (2-shard split) | Q4_K_M (packed) | defaults | ~93-95 tok/s | llama.cpp `-NNNNN-of-NNNNN.gguf` split sets load natively (pass any shard) |
| Qwen2.5-3B-Instruct GGUF | Q4_K_M (packed) | `--cuda-graph-max-bs 0` (eager HIP kernels) | ~24.4 tok/s | launch-overhead-bound; use if graphs must be off |
| Qwen2.5-3B-Instruct GGUF | Q4_K_M (packed) | `FT_GGUF_BACKEND=triton` | ~6.6-7.6 tok/s | all-Triton fallback, useful for kernel triage |
| **gpt-oss-20b GGUF** (MoE) | MXFP4 experts (packed) | `--moe-backend fused --num-pages 4096` (graphs ON) | **~61 tok/s** | harmony reasoning parsed correctly; graph capture AND replay of the MXFP4 MoE kernels work on gfx1200 (the graph-replay crash is gfx1201-only) |
| gpt-oss-20b GGUF (MoE) | MXFP4 experts (packed) | same, `--cuda-graph-max-bs 0` (eager) | ~20.7 tok/s | fallback if graphs misbehave |
| **Gemma-4-26B-A4B QAT GGUF** (MoE) | Q4_0 experts in pinned-RAM banks | `--moe-backend offload --moe-cache-size 2048 --num-pages 4096` (graphs ON) | **~62-73 tok/s** | upstream's `gemma4` GGUF adapter, first GPU run, zero code changes; ~12 GiB expert banks pinned in host RAM, GPU DMA-fetches misses; prefill ~690 tok/s; eager ~12 tok/s. llama.cpp b10630 same file: 84.8 tok/s all-in-VRAM (Vulkan), but only **15.0 tok/s** in its experts-in-RAM mode (`--n-cpu-moe`) — FreeToken's offload decode is ~4x faster in the config the offload design targets |

Decode is memory-bandwidth-bound: BF16 3B moves ~6 GB/token against ~640 GB/s,
so ~72 tok/s is near ceiling for this precision on one card. Quantized GGUF
support (planned adapters) is the main lever for large-model speed.

#### Known RDNA4 issue: MoE kernels inside CUDA-graph replay

`mxfp4_splitk_gemv` / swiglu Triton kernels run correctly eagerly but crash the
worker when executed via CUDA-graph replay on gfx1201 (dense models' graphs are
unaffected). Workaround until fixed upstream: `--cuda-graph-max-bs 0` on MoE
models. The offload backend additionally fails at capture time (`PAL failed to
finalize a command buffer`), so use `--moe-backend fused` on Windows for now.

### Quick install (automated)

A ready-made distribution kit lives in [dist/](dist/) and
[PORT_REQUIREMENTS.md](PORT_REQUIREMENTS.md):

```powershell
# once: clone + install deps, patches and freetoken
git clone https://github.com/Maxritz/FreeToken-rocm-test.git
cd FreeToken-rocm-test
powershell -ExecutionPolicy Bypass -File dist/install.ps1

# every session: engine + web UI
powershell -File dist/run-server.ps1 -Model <path-to-model>
# then open http://localhost:1420   (stop: dist/stop-server.ps1)
```

A fully portable bundle (embeddable Python + wheels, no clone needed) can be built with
`dist/make-bundle.ps1`; users then run its bundled `install.ps1` instead.

### Requirements

- Windows 11, Python 3.12, VS Build Tools (for `vcvarsall.bat` + MSVC CRT link libs)
- AMD ROCm runtime - **TheRock nightly** (`10.1.0a20260817`, HIP 7.16) until ROCm 10.1
  ships formally; set `HIP_PATH=<your-rocm-root>`
- Wheels fetched from https://rocm.nightlies.amd.com/whl-multi-arch/
- Pip stack: torch `2.15.0a0+rocm10.1.0a20260816` + `amd-torch-device-gfx1201`
  (install with `--no-deps`), `triton-windows >= 3.7.1.post27`,
  `apache-tvm-ffi == 0.1.13.post3`
- Install FreeToken itself without CUDA extensions:

```powershell
$env:FREETOKEN_SKIP_CUDA_EXT = "1"
pip install -e <path-to-this-repo> --no-deps --no-build-isolation
```

### Environment switches

| Switch | Example value | Purpose |
|---|---|---|
| `HIP_PATH` | `<repo>\.venv\Lib\site-packages\_rocm_sdk_core` | locates `hipcc`, HIP libs for JIT builds and linking. Use the venv SDK, NOT a machine-wide HIP SDK install — mixed toolchains break JIT builds |
| `TRITON_OVERRIDE_ARCH` | `gfx1200` | forces Triton codegen target (`gfx1200` = RX 9060 XT, `gfx1201` = RX 9070 XT) |
| `TVM_FFI_ROCM_ARCH_LIST` | `gfx1200` | tvm-ffi emits `--offload-arch=<arch>` (else gfx906 default -> broken kernels) |
| `ROCM_SDK_TARGET_FAMILY` | `gfx1200` | device family for the rocm-sdk wheel runtime (nightly-only) |
| `PYTORCH_ROCM_ARCH` | `gfx1200` | arch for torch `cpp_extension` JIT builds (the packed-GGUF HIP kernels) |
| `CC` | `<rocm-root>\lib\llvm\bin\clang-cl.exe` | host compiler for JIT stubs. Must be `clang-cl` (MSVC driver), NOT `clang` — triton-windows passes MSVC-style args |
| `HIP_DEVICE_LIB_PATH` | `<rocm-root>\lib\llvm\amdgcn\bitcode` | ROCm device bitcode for direct-clang HIP compiles |
| `TVM_FFI_CACHE_DIR` | `<repo>\.tvm-ffi-cache` | JIT build dir; MUST be space-free (default `~/.cache` breaks ninja when the Windows username contains a space) |
| `ROCM_HOME`/`ROCM_PATH` | `<rocm-root>` | toolkit home for tvm-ffi / torch; also prepend `<rocm-root>\bin` to `PATH` so the venv `hipcc` wins over any system ROCm |
| `FT_GGUF_BACKEND` | unset / `triton` / `hip` | packed-GGUF matmul backend; unset (recommended) = HIP kernels everywhere except under graph capture on gfx1201, where the driver crashes replaying them. Defaults give ~93.6 tok/s on gfx1200 |
| `FREETOKEN_SKIP_CUDA_EXT` | `1` | build-time: install without nvcc/CUDA extensions |
| `--num-pages N` | e.g. `4096` | caps KV cache pages so large dense models fit in VRAM |

Launch recipe (what `dist/run-server.ps1` does):

```bat
call "<vs>\VC\Auxiliary\Build\vcvarsall.bat" x64
set HIP_PATH=<rocm-root>
set TVM_FFI_ROCM_ARCH_LIST=gfx1201
set TRITON_OVERRIDE_ARCH=gfx1201
ft serve --model <model_path>
```

`vcvarsall` is required so the linker finds the MSVC CRT when producing JIT DLLs.

### Site-packages patches this fork relies on

Three upstream packages need small patches until merged upstream - applied automatically
by `dist/patch_upstream.py`, documented in [DIAGNOSTICS.md](DIAGNOSTICS.md):

1. **tvm_ffi/cpp/extension.py** - on Windows+HIP: use `hipcc` flags (no `-fPIC`,
   no MSVC-style `-Xcompiler` args), emit `--offload-arch`, link `amdhip64.lib`,
   and build *host* C++ with HIP `clang++` instead of `cl.exe` (MSVC rejects the
   `RuntimeCheck` pack+default-arg idiom).
2. **triton/backends/amd/compiler.py** - add `launch_pdl: bool = False` to
   `HIPOptions` so NVIDIA-only launch kwargs are accepted-and-ignored.
3. **uvicorn/loops/asyncio.py** - return `SelectorEventLoop` (not `ProactorEventLoop`)
   on win32; `zmq.asyncio` requires `add_reader`.

Engine-side patches included in this fork: HIP compat shim for CUDA-flavored kernel
headers (`hip_compat.cuh`), PTX inline-asm gated to NVIDIA with libdevice fallbacks,
WMMA-safe `BLOCK_H` padding in the grouped attention kernel, TCP loopback ZMQ
addresses with deterministic ports, Windows selector event-loop policy,
graceful CUDA-extension skipping, and a `webui/` one-file chat client.

## Status: Windows 11 + RX 9070 XT (ROCm port work log)

This fork runs natively on Windows 11 against an AMD Radeon RX 9070 XT (gfx1201,
RDNA4) using TheRock nightly ROCm runtime (`HIP_PATH`, `TVM_FFI_ROCM_ARCH_LIST`,
`TRITON_OVERRIDE_ARCH=gfx1201`, MSVC vcvars). Measured so far:
Qwen2.5-3B BF16 ~72 tok/s, Qwen2.5-7B BF16 ~15.8 tok/s, gpt-oss-20b fused-MoE
~12.2 tok/s (RDNA4 graph-replay MoE crash worked around via eager mode).

Current effort: **packed GGUF loading** for llama.cpp quant types — weights stay
quantized in VRAM (no bf16 expansion). Approach and state:

- Vendored llama.cpp quant kernels (dequant/GEMV/MMQ/MoE) already cover all
  classic, K-quant, and IQ types; the bottleneck was Python-side dispatch sets
  in `layers/gguf.py`, since widened (MMVQ = all kernel-covered types, MMQ =
  classic+K, chunked-GEMV fallback for IQ at prefill batch sizes).
- `models/gguf/dense.py` rewritten: header-only type scan, packed `.qweight`
  emission, per-layer-correct module construction (real Q4_K_M files mix
  Q4_K/Q6_K per layer), fused groups load as per-slot splits when fully
  quantized.
- Full 24-type tables in `models/gguf/dequant.py`, verified against gguf-py
  `GGML_QUANT_SIZES`; plus MXFP4 and ROCmFPX (types 100–108) dequant support.
- Adapter hooks wired into llama / qwen2 / mistral / qwen3 families; static
  validation passes against a real Mistral-7B Q4_K_M checkpoint.
- **DONE (2026-08-24): MoE GGUF adapters + the offload-decode TDR root cause.**
  Two new GGUF families: **gpt-oss** (llama.cpp MXFP4 experts repacked at load
  into the HF `mxfp4_triton` layout — `models/gpt_oss/gguf.py`, `--moe-backend
  fused`) and **qwen35moe** (Qwen3.5/3.6 hybrid GDN MoE — `models/qwen3_5_moe/gguf.py`,
  experts stay packed in generalized `q4_0`-schema banks accepting any
  MMVQ-covered ggml type, mixed-type banks requantized to `FT_GGUF_BANK_PROMOTE`
  [default Q5_1] at load, `--moe-backend offload`). Both verified end-to-end on
  gfx1200 with synthetic tiny models (real-tokenizer, random-weight GGUFs).
  **Root-caused the historical RDNA4 offload-decode `unspecified launch failure`**:
  `kernel/pinned.py::host_register` was a silent no-op without the never-built
  `_pinned_tensor` extension, so "pinned" expert banks stayed pageable; AND the
  `device_ptr` host-VA-identity probe tested `hipHostMalloc` memory (unified)
  while `hipHostRegister`ed banks map to different device VAs on Windows/WDDM —
  the fused gather then dereferenced host VAs from the GPU. Fixed with a ctypes
  HIP fallback (register + `hipHostGetDevicePointer` translation + a
  registered-memory identity probe). Also fixed: MXFP4 dequant was 2x too large
  (missing E8M0-half), GDN/fla Triton kernels verified on gfx1200, and sharded
  (`-NNNNN-of-NNNNN.gguf`) GGUF loading. Known limit: a 35B-A3B Q4_K_M needs
  ~19 GB of *locked* host RAM for offload banks — not reliable on a 32 GB
  machine; use a Q3-class file there.
  **gpt-oss-20b MXFP4 GGUF verified end-to-end on gfx1200: ~61 tok/s** with CUDA
  graphs + fused MoE (~20.7 eager), harmony reasoning/final channels parsed. Two
  more fixes landed for it: GGUF control tokens are now registered as special
  added tokens at tokenizer conversion (they encoded as raw bytes before, so
  chat-template markers reached the model as byte soup — affects every GGUF
  arch), and `--reasoning-parser gpt_oss` works against the GGUF tokenizer.
- **DONE (2026-08-23): first GPU end-to-end GGUF run** — Qwen2.5-3B-Instruct
  Q4_K_M on an RX 9060 XT (gfx1200, Windows 11, ROCm 10.1.0a20260806 wheels):
  server READY, coherent chat completions, **~93.6 tok/s** decode with the HIP
  MMVQ kernels under CUDA-graph replay (the auto backend is now arch-aware:
  Triton-under-capture only on gfx1201 where replay crashes; ~24.4 tok/s eager),
  web UI working. **Sharded GGUF** (llama.cpp `-NNNNN-of-NNNNN.gguf` split sets)
  loads natively — pass any shard, metadata reads from shard 1, tensors stream
  across shards (`models/gguf/reader.py::gguf_shard_paths`). Fixes that landed
  for this run:
  torch `cpp_extension` hipify None-path guard (patch 4 in `dist/patch_upstream.py`),
  `--offload-arch` emission in the tvm-ffi Windows HIP branch, a `thrust/complex.h`
  shim in `csrc/gguf/jit_shim/` (TheRock wheels ship no rocThrust), `clang-cl`
  as `CC` for triton-windows, space-free `TVM_FFI_CACHE_DIR`, venv-SDK-first
  toolchain resolution, and gfx-arch autodetection in the dist scripts.
- Known RDNA4 issues parked upstream: Triton wave64 cross-lane reduction bug;
  Triton MXFP4 MoE crash under CUDA-graph replay.

## Citation

If you use FreeToken for your research, please cite our [paper](https://arxiv.org/abs/2608.16157):

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## Acknowledgment

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from the following projects:
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

[Apache License 2.0](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE).
