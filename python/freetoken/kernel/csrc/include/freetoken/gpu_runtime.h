#pragma once

// Host-side GPU runtime surface for the plain-C++ extensions
// (``_pinned_tensor``, ``_cpu_moe``). Those files use only the CUDA *runtime*
// API -- no ``__global__`` kernels -- so torch's hipify never rewrites them and
// they cannot include <cuda_runtime_api.h> on a ROCm box. Every call they make
// has an identical hip* counterpart, so alias the names here and keep one
// source tree valid for either toolchain.
//
// The driver-API stream-memory-op path in cpu_moe_ext.cpp is resolved with
// dlopen("libcuda.so.1") at runtime and already falls back to
// cudaLaunchHostFunc when that fails, which is what happens on ROCm. Nothing to
// shim there.

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__) ||          \
    defined(USE_ROCM)

#include <hip/hip_runtime_api.h>

using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaHostFn_t = hipHostFn_t;

#ifndef CUDART_CB
#define CUDART_CB
#endif

inline constexpr auto cudaSuccess = hipSuccess;
inline constexpr auto cudaHostAllocPortable = hipHostMallocPortable;
inline constexpr auto cudaHostAllocMapped = hipHostMallocMapped;
inline constexpr auto cudaHostRegisterPortable = hipHostRegisterPortable;
inline constexpr auto cudaHostRegisterMapped = hipHostRegisterMapped;
inline constexpr auto cudaDevAttrUnifiedAddressing =
    hipDeviceAttributeUnifiedAddressing;
inline constexpr auto cudaDevAttrCanUseHostPointerForRegisteredMem =
    hipDeviceAttributeCanUseHostPointerForRegisteredMem;

inline auto cudaFreeHost(void *ptr) -> cudaError_t { return hipHostFree(ptr); }

// cudaMallocHost takes no flags; hipHostMalloc requires them.
inline auto cudaMallocHost(void **ptr, std::size_t size) -> cudaError_t {
  return hipHostMalloc(ptr, size, hipHostMallocDefault);
}

inline auto cudaHostAlloc(void **ptr, std::size_t size, unsigned int flags)
    -> cudaError_t {
  return hipHostMalloc(ptr, size, flags);
}

inline auto cudaHostRegister(void *ptr, std::size_t size, unsigned int flags)
    -> cudaError_t {
  return hipHostRegister(ptr, size, flags);
}

inline auto cudaHostUnregister(void *ptr) -> cudaError_t {
  return hipHostUnregister(ptr);
}

inline auto cudaHostGetDevicePointer(void **dev, void *host, unsigned int flags)
    -> cudaError_t {
  return hipHostGetDevicePointer(dev, host, flags);
}

inline auto cudaGetDevice(int *device) -> cudaError_t {
  return hipGetDevice(device);
}

inline auto cudaDeviceGetAttribute(int *value, hipDeviceAttribute_t attr,
                                   int device) -> cudaError_t {
  return hipDeviceGetAttribute(value, attr, device);
}

inline auto cudaGetErrorString(cudaError_t err) -> const char * {
  return hipGetErrorString(err);
}

inline auto cudaDriverGetVersion(int *version) -> cudaError_t {
  return hipDriverGetVersion(version);
}

inline auto cudaStreamSynchronize(cudaStream_t stream) -> cudaError_t {
  return hipStreamSynchronize(stream);
}

inline auto cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn,
                               void *user_data) -> cudaError_t {
  return hipLaunchHostFunc(stream, fn, user_data);
}

#else // CUDA

#include <cuda_runtime_api.h>

#endif
