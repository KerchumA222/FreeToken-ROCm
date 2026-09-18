// Minimal AT_DISPATCH helper for the vendored GGUF kernels (borrowed from
// sgl-kernel csrc/quantization/gguf, which are ports of llama.cpp). The donor
// pulls these macros from its large include/utils.h; we only need the float
// dispatch, so vendor just that to keep the JIT compile self-contained.
#pragma once

#include <ATen/Dispatch.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Warp-shuffle wrappers the donor pulls from sgl-kernel's utils.h (CUDA variants).
//
// HIP's __shfl_xor_sync static_asserts sizeof(mask) == 8 -- its lane masks are
// 64-bit because CDNA is wave64 -- while the donor passes CUDA's 32-bit
// uint32_t(-1). Widen the mask on ROCm. Every call site passes all-ones, so the
// extra bits are inert on a wave32 part like gfx1151.
#ifdef __HIP_PLATFORM_AMD__
#define SGLANG_SHFL_MASK_T unsigned long long
#else
#define SGLANG_SHFL_MASK_T unsigned int
#endif
#ifndef SGLANG_SHFL_XOR_SYNC
#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) \
  __shfl_xor_sync((SGLANG_SHFL_MASK_T)(mask), (var), (lane_mask))
#endif
#ifndef SGLANG_SHFL_XOR_SYNC_WIDTH
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) \
  __shfl_xor_sync((SGLANG_SHFL_MASK_T)(mask), (var), (lane_mask), (width))
#endif

#define DISPATCH_CASE_FLOAT_TYPES(...)                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOAT_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOAT_TYPES(__VA_ARGS__))
