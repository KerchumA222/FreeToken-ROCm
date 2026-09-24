// CPU-compute MoE executor for the "cpu" offload backend.
//
// Decode ships activations to the CPU, computes the routed experts here (reading
// the pinned host expert banks at full RAM bandwidth), and ships the results
// back. To keep the whole decode path inside a single CUDA graph we expose
// submit/sync as host nodes via cudaLaunchHostFunc -- the callbacks only touch a
// CPU worker pool + pinned host buffers and never call any CUDA API.
//
// One task is in flight at a time (per MoE layer): submit() wakes the pool,
// sync() blocks the host-func thread until the pool drains. The heavy GEMV runs
// on the persistent worker threads, not the host-func thread.
//
// Weight formats: bf16, NVFP4, MXFP4, ds_fp4, Q4_0, and mixed Q4_K/Q5_1 expert
// banks (see WFmt and the per-format bank schemas). Compute is FP32-accumulate;
// the intermediate is stored bf16 to match the GPU decode path. ISA is chosen
// once at construction (AVX-512-BF16 dpbf16 -> AVX-512F widening -> AVX2+FMA ->
// scalar).

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <freetoken/gpu_runtime.h>  // CUDA runtime API, or its hip* twin on ROCm
#include <torch/extension.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#define CPU_MOE_HAS_AFFINITY 1
#else
#define CPU_MOE_HAS_AFFINITY 0
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define CPU_MOE_X86 1
#else
#define CPU_MOE_X86 0
#endif

namespace {

using bf16_t = uint16_t;

inline float bf16_to_f32(bf16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline bf16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  // round-to-nearest-even
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<bf16_t>(u >> 16);
}

// ACT_SWIGLUOAI is the clamped (up + 1) swiglu (gpt-oss "swigluoai" /
// MiniMax-M3): gate/up are combined jointly with the runtime alpha/limit
// scalars, so it is handled in the do_pass1 epilogue (act_apply never sees it;
// the mxfp4 kernel additionally fuses its own copy of the same math).
// ACT_SWIGLU_CLAMP (GLM-5.3 "swiglu_limit") is the same clamped form WITHOUT
// the (up + 1) bias: clamp(gate, max=lim) * sigmoid(alpha*gate) * clamp(up, +-lim).
enum ActKind {
  ACT_SILU = 0,
  ACT_GELU = 1,
  ACT_GELU_TANH = 2,
  ACT_SWIGLUOAI = 3,
  ACT_SWIGLU_CLAMP = 4,
};

inline float act_apply(int act, float x) {
  if (act == ACT_SILU) return x / (1.0f + std::exp(-x));
  if (act == ACT_GELU)
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
  // gelu_tanh
  const float k0 = 0.79788456080286535588f;  // sqrt(2/pi)
  const float inner = k0 * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

// ------------------------------- dot products -------------------------------
// dot(weight[bf16], act[bf16], n) -> fp32. The selected impl is a function
// pointer chosen at runtime; the per-row call overhead is negligible vs n.

using dot_fn = float (*)(const bf16_t*, const bf16_t*, int);

float dot_scalar(const bf16_t* w, const bf16_t* x, int n) {
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) acc += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return acc;
}

// Software prefetch distance (bytes) ahead of the current weight row stream. The
// weight stream is the bandwidth bottleneck (read once, never reused); nudging the
// HW prefetcher with a few cache lines of lookahead raises sustained throughput.
constexpr int PF_AHEAD = 512;

#if CPU_MOE_X86
__attribute__((target("avx512f")))
float dot_avx512f(const bf16_t* w, const bf16_t* x, int n) {
  // 4 independent accumulators -> more in-flight loads (memory-level parallelism),
  // which is what lifts a bandwidth-bound GEMV toward peak.
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 64 <= n; i += 64) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 64; j += 16) {
      __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i + j));
      __m256i xi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i + j));
      __m512 wf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(wi), 16));
      __m512 xf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(xi), 16));
      __m512& acc = (j == 0) ? a0 : (j == 16) ? a1 : (j == 32) ? a2 : a3;
      acc = _mm512_fmadd_ps(wf, xf, acc);
    }
  }
  for (; i + 16 <= n; i += 16) {
    __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
    __m256i xi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
    __m512 wf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(wi), 16));
    __m512 xf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(xi), 16));
    a0 = _mm512_fmadd_ps(wf, xf, a0);
  }
  float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}

#if (defined(__GNUC__) && __GNUC__ >= 10) || defined(__clang__)
#define CPU_MOE_HAS_AVX512BF16 1
__attribute__((target("avx512bf16,avx512f")))
static inline __m512bh load_bh(const bf16_t* p) {
  __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(p));
  __m512bh out;
  std::memcpy(&out, &raw, sizeof(out));
  return out;
}

__attribute__((target("avx512bf16,avx512f")))
float dot_avx512bf16(const bf16_t* w, const bf16_t* x, int n) {
  // 4 accumulators (128 bf16/iter) for memory-level parallelism + a prefetch nudge.
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
    a1 = _mm512_dpbf16_ps(a1, load_bh(w + i + 32), load_bh(x + i + 32));
    a2 = _mm512_dpbf16_ps(a2, load_bh(w + i + 64), load_bh(x + i + 64));
    a3 = _mm512_dpbf16_ps(a3, load_bh(w + i + 96), load_bh(x + i + 96));
  }
  for (; i + 32 <= n; i += 32) {
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
  }
  float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}
#endif  // avx512bf16 available

// --------- AVX2 + FMA (256-bit fallback for CPUs without AVX-512) ----------
// Covers Intel 12-14th gen / Arrow Lake (AVX-512 fused off) and AMD Zen<4. The
// bf16->fp32 widen is a zero-extend + <<16; with 4 independent accumulators the
// GEMV is memory-bandwidth bound, same as the AVX-512 path (just half the width).
__attribute__((target("avx2,fma")))
inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  lo = _mm_add_ps(lo, _mm256_extractf128_ps(v, 1));
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
  return _mm_cvtss_f32(lo);
}

__attribute__((target("avx2,fma")))
float dot_avx2(const bf16_t* w, const bf16_t* x, int n) {
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 32; j += 8) {
      __m128i wi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + j));
      __m128i xi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i + j));
      __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(wi), 16));
      __m256 xf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(xi), 16));
      __m256& acc = (j == 0) ? a0 : (j == 8) ? a1 : (j == 16) ? a2 : a3;
      acc = _mm256_fmadd_ps(wf, xf, acc);
    }
  }
  for (; i + 8 <= n; i += 8) {
    __m128i wi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i));
    __m128i xi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i));
    __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(wi), 16));
    __m256 xf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(xi), 16));
    a0 = _mm256_fmadd_ps(wf, xf, a0);
  }
  float s = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}
#endif  // CPU_MOE_X86

// --------------------------- NVFP4 (W4A16) dequant ---------------------------
// Weights: e2m1 4-bit codes (2/byte, low nibble first), per-16 block scale in
// fp8-e4m3, per-output-row global scale in fp16. Dequant matches the GPU kernels
// (freetoken/kernel/triton/nvfp4_dequant.py): w = E2M1[code] * e4m3(scale) * global.
// Activations stay bf16 (W4A16); the GEMV dequantizes weights inside the K-loop.

const float kE2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                         -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// e2m1 * 2 as exact int8 (all e2m1 values are multiples of 0.5). Used by the AVX-VNNI
// W4A8 path: nibble -> int8 weight via PSHUFB LUT, then VPDPBUSD against int8 activations;
// the *2 is undone by a 0.5 folded into the final scale. Mirrors ggml's kvalues_mxfp4.
alignas(16) const int8_t kE2M1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline float fp16_to_f32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) {
      f = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FFu;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (man << 13);
  } else {
    f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

inline uint16_t f32_to_fp16(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
  const uint32_t exp = (bits >> 23) & 0xffu;
  uint32_t man = bits & 0x7fffffu;
  if (exp == 0xffu) return (uint16_t)(sign | 0x7c00u | (man ? 0x0200u : 0));
  int he = (int)exp - 127 + 15;
  if (he >= 31) return (uint16_t)(sign | 0x7c00u);
  if (he <= 0) {
    if (he < -10) return sign;
    man |= 0x800000u;
    const int shift = 14 - he;
    uint32_t hm = man >> shift;
    const uint32_t rem = man & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (hm & 1u))) ++hm;
    return (uint16_t)(sign | hm);
  }
  uint32_t hm = man >> 13;
  const uint32_t rem = man & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) {
    if (++hm == 0x400u) {
      hm = 0;
      if (++he >= 31) return (uint16_t)(sign | 0x7c00u);
    }
  }
  return (uint16_t)(sign | ((uint16_t)he << 10) | (uint16_t)hm);
}

// e4m3 (OCP "fn": finite, max-normal 448, exp bias 7). Decoded into a 256-entry LUT.
inline float e4m3_decode(uint8_t v) {
  const float sign = (v & 0x80u) ? -1.0f : 1.0f;
  const uint32_t exp = (v >> 3) & 0xFu;
  const uint32_t man = v & 0x7u;
  if (exp == 0) return sign * (man / 8.0f) * 0.015625f;  // 2^(1-7) = 2^-6
  return sign * (1.0f + man / 8.0f) * std::ldexp(1.0f, (int)exp - 7);
}

// Activations pre-deinterleaved to fp32 (xe[m]=x[2m], xo[m]=x[2m+1]); see the
// ds_fp4 dot below for why (drops the hot loop to ~1.5 shuffle ops / 16 weights).
using nvdot_fn = float (*)(const uint8_t*, const uint8_t*, float, const float*, const float*,
                           int, const float*, const float*);

float dot_nvfp4_scalar(const uint8_t* packed, const uint8_t* scale, float global,
                       const float* xe, const float* xo, int K, const float* e2m1,
                       const float* e4m3) {
  float acc = 0.0f;
  const int nb = K / 16;
  for (int b = 0; b < nb; ++b) {
    const float bs = e4m3[scale[b]];
    const uint8_t* pk = packed + (size_t)b * 8;
    const float* xeb = xe + (size_t)b * 8;  // 16 K -> 8 even + 8 odd
    const float* xob = xo + (size_t)b * 8;
    float bsum = 0.0f;
    for (int j = 0; j < 8; ++j) {
      const uint8_t byte = pk[j];
      bsum += e2m1[byte & 0xF] * xeb[j];
      bsum += e2m1[byte >> 4] * xob[j];
    }
    acc += bs * bsum;
  }
  return acc * global;
}

// ---- NVFP4 W4A8 (int8 activations) dot: nibble->int8 via LUT, per-16 act scale ----
// asi8: int8 activations laid out per-16 block as [even(8), odd(8)]; asb[b] = per-block
// activation scale (absmax/127). Result folds the e2m1*2 -> *0.5 into the scale.
using nvi8dot_fn = float (*)(const uint8_t*, const uint8_t*, float, const int8_t*, int,
                             const float*, const float*);

[[maybe_unused]] float dot_nvfp4_i8_scalar(const uint8_t* packed, const uint8_t* scale,
                          float global, const int8_t* asi8, int K, const float* e4m3,
                          const float* asb) {
  float acc = 0.0f;
  const int nb = K / 16;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16;       // even(8)
    const int8_t* ao = ae + 8;                      // odd(8)
    int isum = 0;
    for (int j = 0; j < 8; ++j) {
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j];
      isum += (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    }
    acc += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return acc * (0.5f * global);
}

#if CPU_MOE_X86
// AVX2 e2m1 nibble decode: codes (0..15) in 8 int32 lanes -> fp32. AVX2 vpermps is
// only 8-wide, so instead of a 16-entry LUT we use the e2m1 sign/magnitude symmetry:
// value = (code&8 ? - : +) * mag8[code&7], mag8 = e2m1[0..7]. The sign is bit 3 of
// the code shifted into the fp32 sign bit (bit 31). Bit-identical to the e2m1 LUT.
__attribute__((target("avx2,fma")))
inline __m256 e2m1_decode8(__m256i codes, __m256 mag8) {
  __m256 mag = _mm256_permutevar8x32_ps(mag8, _mm256_and_si256(codes, _mm256_set1_epi32(7)));
  __m256i sgn = _mm256_slli_epi32(_mm256_and_si256(codes, _mm256_set1_epi32(8)), 28);
  return _mm256_xor_ps(mag, _mm256_castsi256_ps(sgn));
}

// Two 16-K blocks (16 packed bytes) per iter: lo nibbles -> even-K, hi -> odd-K,
// gathered via two vpermps. The per-16 e4m3 scale differs across the two blocks, so
// it is applied per lane (low 8 lanes = block b, high 8 = block b+1).
__attribute__((target("avx512f")))
inline __m512 nvfp4_blk2(const uint8_t* pk, const float* xeb, const float* xob, __m512 lut,
                         __m512i loma, float s0, float s1) {
  __m512i wi = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pk)));
  __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
  __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
  __m512 prod = _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xeb), _mm512_mul_ps(vhi, _mm512_loadu_ps(xob)));
  // lanes 0-7 (block b) -> s0, lanes 8-15 (block b+1) -> s1  (pure AVX512F mask move)
  __m512 scv = _mm512_mask_mov_ps(_mm512_set1_ps(s0), 0xFF00, _mm512_set1_ps(s1));
  return _mm512_mul_ps(prod, scv);
}

__attribute__((target("avx512f")))
float dot_nvfp4_avx512(const uint8_t* packed, const uint8_t* scale, float global,
                       const float* xe, const float* xo, int K, const float* e2m1,
                       const float* e4m3) {
  const __m512 lut = _mm512_loadu_ps(e2m1);  // 16 e2m1 values
  const __m512i loma = _mm512_set1_epi32(0xF);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  const int nb = K / 16;  // 8 packed bytes + one e4m3 scale per block
  int b = 0;
  for (; b + 4 <= nb; b += 4) {  // two blk2 calls -> 4 blocks, 2 accumulators
    acc0 = _mm512_add_ps(acc0, nvfp4_blk2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                          xo + (size_t)b * 8, lut, loma, e4m3[scale[b]], e4m3[scale[b + 1]]));
    acc1 = _mm512_add_ps(acc1, nvfp4_blk2(packed + (size_t)(b + 2) * 8, xe + (size_t)(b + 2) * 8,
                                          xo + (size_t)(b + 2) * 8, lut, loma, e4m3[scale[b + 2]], e4m3[scale[b + 3]]));
  }
  for (; b + 2 <= nb; b += 2)
    acc0 = _mm512_add_ps(acc0, nvfp4_blk2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                          xo + (size_t)b * 8, lut, loma, e4m3[scale[b]], e4m3[scale[b + 1]]));
  float s = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
  for (; b < nb; ++b) {  // odd final 16-K block
    const uint8_t* pk = packed + (size_t)b * 8;
    const float* xeb = xe + (size_t)b * 8;
    const float* xob = xo + (size_t)b * 8;
    float bsum = 0.0f;
    for (int j = 0; j < 8; ++j) {
      bsum += e2m1[pk[j] & 0xF] * xeb[j] + e2m1[pk[j] >> 4] * xob[j];
    }
    s += e4m3[scale[b]] * bsum;
  }
  return s * global;
}

// AVX2: one 16-K block (8 packed bytes) per call, 8 even + 8 odd lanes.
__attribute__((target("avx2,fma")))
inline __m256 nvfp4_blk_avx2(const uint8_t* pk, const float* xeb, const float* xob,
                             __m256 mag8, float sc) {
  __m256i wi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk)));
  __m256 vlo = e2m1_decode8(_mm256_and_si256(wi, _mm256_set1_epi32(0xF)), mag8);
  __m256 vhi = e2m1_decode8(_mm256_srli_epi32(wi, 4), mag8);
  __m256 prod = _mm256_fmadd_ps(vlo, _mm256_loadu_ps(xeb), _mm256_mul_ps(vhi, _mm256_loadu_ps(xob)));
  return _mm256_mul_ps(prod, _mm256_set1_ps(sc));
}

__attribute__((target("avx2,fma")))
float dot_nvfp4_avx2(const uint8_t* packed, const uint8_t* scale, float global,
                     const float* xe, const float* xo, int K, const float* e2m1,
                     const float* e4m3) {
  const __m256 mag8 = _mm256_loadu_ps(e2m1);  // e2m1[0..7] magnitudes
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  const int nb = K / 16;
  int b = 0;
  for (; b + 2 <= nb; b += 2) {
    acc0 = _mm256_add_ps(acc0, nvfp4_blk_avx2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                              xo + (size_t)b * 8, mag8, e4m3[scale[b]]));
    acc1 = _mm256_add_ps(acc1, nvfp4_blk_avx2(packed + (size_t)(b + 1) * 8, xe + (size_t)(b + 1) * 8,
                                              xo + (size_t)(b + 1) * 8, mag8, e4m3[scale[b + 1]]));
  }
  for (; b < nb; ++b)
    acc0 = _mm256_add_ps(acc0, nvfp4_blk_avx2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                              xo + (size_t)b * 8, mag8, e4m3[scale[b]]));
  return hsum256(_mm256_add_ps(acc0, acc1)) * global;
}

// AVX-VNNI W4A8: decode 8 packed bytes (16 nibbles) of one 16-block to int8 [lo(8),hi(8)]
// via PSHUFB against the e2m1*2 LUT (replaces the 2 vpermps fp32 expands -- ~4x less
// port-5 traffic). lo=even-K weights, hi=odd-K, matching the [even(8),odd(8)] act layout.
__attribute__((target("avx2,avxvnni,fma")))
inline __m128i nvfp4_decode_block_i8(const uint8_t* pk, __m128i lut) {
  __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk));   // 8 bytes
  __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
  __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
  return _mm_shuffle_epi8(lut, _mm_unpacklo_epi64(lo, hi));            // [lo(8),hi(8)] -> int8
}

// Two 16-blocks per VPDPBUSD (32 int8). Sign trick (ggml mul_add_epi8): |w|*(sign(w)*a)=w*a,
// so u8*s8 needs no offset/correction term. Per-block scale (e4m3 * act-scale) folded in fp32
// (lanes 0-3 -> block b, 4-7 -> block b+1). Bit-faithful weight; only the int8 activation quant
// (W4A8) differs from the bf16 reference.
__attribute__((target("avx2,avxvnni,fma")))
float dot_nvfp4_i8_vnni(const uint8_t* packed, const uint8_t* scale, float global,
                        const int8_t* asi8, int K, const float* e4m3, const float* asb) {
  const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2));
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 16;
  int b = 0;
  for (; b + 2 <= nb; b += 2) {
    __m128i wb = nvfp4_decode_block_i8(packed + (size_t)b * 8, lut);
    __m128i wb1 = nvfp4_decode_block_i8(packed + (size_t)(b + 1) * 8, lut);
    __m256i w = _mm256_set_m128i(wb1, wb);                              // [blk b | blk b+1]
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(asi8 + (size_t)b * 16));
    __m256i aw = _mm256_sign_epi8(w, w);                               // |w| (u8 operand)
    __m256i sa = _mm256_sign_epi8(a, w);                               // sign(w)*a (s8 operand)
    __m256i di = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, sa);
    __m256 scv = _mm256_blend_ps(_mm256_set1_ps(e4m3[scale[b]] * asb[b]),
                                 _mm256_set1_ps(e4m3[scale[b + 1]] * asb[b + 1]), 0xF0);
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(di), scv, accF);
  }
  float s = hsum256(accF);
  for (; b < nb; ++b) {  // tail (odd block count)
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16; const int8_t* ao = ae + 8;
    int isum = 0;
    for (int j = 0; j < 8; ++j)
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j] + (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    s += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return s * (0.5f * global);
}

#if (defined(__GNUC__) && __GNUC__ >= 10) || defined(__clang__)
#define CPU_MOE_HAS_AVX512VNNI 1

// Software-prefetch distance for the W4A8 weight stream, in 16-K blocks (8 packed
// bytes each). Returns -1 when FREETOKEN_CPU_MOE_PF_BLOCKS is unset: the kernel then
// uses the built-in default min(512 blocks = 4 KB, 2 rows) -- 4 KB is the empirical
// optimum on large-row machines (Emerald Rapids sweep), while the 2-row cap keeps a small-row
// model's overshoot bounded (the executor works in 32-row tiles, so a fixed byte
// distance otherwise prefetches another worker's tile: duplicated DRAM traffic that
// regresses at the bandwidth ceiling). An EXPLICIT env value is honored verbatim
// (no clamp; 0 disables): the per-machine optimum can sit past the safe default (+20%
// at 4 KB on a 24-thread Ice Lake with 256B rows), so the escape hatch must reach it.
// Prefetch never faults, so overshooting a row/bank tail is safe.
static int nvfp4_pf_blocks() {
  static const int v = [] {
    const char* s = getenv("FREETOKEN_CPU_MOE_PF_BLOCKS");
    return (s && s[0]) ? atoi(s) : -1;
  }();
  return v;
}

// AVX-512 VNNI W4A8: FOUR 16-K blocks per VPDPBUSD (64 int8) -- 2x the AVX-VNNI
// (256-bit) path. Decode 32 packed bytes -> 64 int8 with a single _mm512_shuffle_epi8
// (e2m1*2 LUT replicated to all 4 128-bit lanes). AVX-512 has no _mm512_sign_epi8, so
// the u8*s8 sign trick (|w| as u8, sign(w)*a as s8) uses abs_epi8 + a masked negate.
// Bit-faithful weight; only the int8 activation quant (W4A8) differs from bf16.
// One 4-block group (64 int8) -> scaled fp32 partial (16 lanes). Isolated as a helper so
// the caller can run several independent chains into separate accumulators (the decode ->
// dpbusd -> scale chain is long, so a single accumulator leaves the core latency-bound).
__attribute__((target("avx512f,avx512bw,avx512vnni,avx2")))
static inline __m512 nvfp4_i8_grp4(const uint8_t* packed, const uint8_t* scale,
                                   const int8_t* asi8, const float* e4m3, const float* asb,
                                   int b, __m512i lut, __m512i idx, __m512i mask0F,
                                   __m512i idxsc) {
  const __mmask64 hi_half = 0xFF00FF00FF00FF00ULL;  // bytes 8-15 of each 128b lane -> hi nibbles
  __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + (size_t)b * 8));
  __m512i src = _mm512_permutexvar_epi64(idx, _mm512_castsi256_si512(raw));
  __m512i lo = _mm512_and_si512(src, mask0F);
  __m512i hi = _mm512_and_si512(_mm512_srli_epi16(src, 4), mask0F);
  __m512i comb = _mm512_mask_blend_epi8(hi_half, lo, hi);
  __m512i w = _mm512_shuffle_epi8(lut, comb);  // 64 int8 weights (e2m1*2)
  // u8*s8 sign trick without _mm512_sign_epi8: aw=|w|, sa = (w<0 ? -a : a) (a moot at w==0)
  __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(asi8 + (size_t)b * 16));
  __m512i aw = _mm512_abs_epi8(w);
  __mmask64 neg = _mm512_movepi8_mask(w);
  __m512i sa = _mm512_mask_sub_epi8(a, neg, _mm512_setzero_si512(), a);
  __m512i di = _mm512_dpbusd_epi32(_mm512_setzero_si512(), aw, sa);  // 16 int32 (groups of 4)
  // int32[0..3]->blk b, [4..7]->b+1, [8..11]->b+2, [12..15]->b+3. The 4 block scales
  // (e4m3 LUT x per-block act scale) are computed vectorized: a 4-byte load + epu8->epi32
  // widen + one 4-lane LUT gather + one mul replaces 8 scalar loads + 4 scalar muls +
  // a set_ps assembly, which otherwise dominates the per-group op count.
  int sc_raw;
  memcpy(&sc_raw, scale + b, 4);
  __m128i sc4 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(sc_raw));
  __m128 s4 = _mm_mul_ps(_mm_i32gather_ps(e4m3, sc4, 4), _mm_loadu_ps(asb + b));
  __m512 scv = _mm512_permutexvar_ps(idxsc, _mm512_castps128_ps512(s4));
  return _mm512_mul_ps(_mm512_cvtepi32_ps(di), scv);
}

__attribute__((target("avx512f,avx512bw,avx512vnni,avx2")))
float dot_nvfp4_i8_avx512vnni(const uint8_t* packed, const uint8_t* scale, float global,
                              const int8_t* asi8, int K, const float* e4m3, const float* asb) {
  const __m512i lut = _mm512_broadcast_i32x4(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2)));
  const __m512i idx = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);  // block i -> 128b lane i
  const __m512i mask0F = _mm512_set1_epi8(0x0F);
  const __m512i idxsc = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
  const int nb = K / 16;
  // Built-in default: min(4 KB, 2 rows) -- see nvfp4_pf_blocks(). An explicitly set
  // env value is used verbatim so a per-machine sweep can reach operating points past
  // the conservative default.
  const int pfb = nvfp4_pf_blocks();
  const int pf = (pfb < 0) ? std::min(512, 2 * nb) : pfb;
  int b = 0;
  // Four independent 4-block groups per iter -> four accumulator chains hide the ~10-op
  // decode->dpbusd->scale latency (1-2 chains leave the loop latency-bound: the per-core
  // rate sat at ~60% of the core's achievable DRAM stream rate).
  for (; b + 16 <= nb; b += 16) {
    // Prefetch the weight stream ahead: the interleaved decode lowers the L1-miss
    // concurrency the HW prefetcher sustains on its own (134 -> 167 GB/s on Emerald Rapids).
    if (pf > 0) {
      _mm_prefetch(reinterpret_cast<const char*>(packed + ((size_t)b + (size_t)pf) * 8),
                   _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(packed + ((size_t)b + (size_t)pf) * 8 + 64),
                   _MM_HINT_T0);
    }
    acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx,
                                             mask0F, idxsc));
    acc1 = _mm512_add_ps(acc1, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 4, lut, idx,
                                             mask0F, idxsc));
    acc2 = _mm512_add_ps(acc2, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 8, lut, idx,
                                             mask0F, idxsc));
    acc3 = _mm512_add_ps(acc3, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 12, lut, idx,
                                             mask0F, idxsc));
  }
  for (; b + 4 <= nb; b += 4)
    acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx,
                                             mask0F, idxsc));
  float s = _mm512_reduce_add_ps(
      _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
  for (; b < nb; ++b) {  // tail (<4 remaining 16-K blocks)
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16; const int8_t* ao = ae + 8;
    int isum = 0;
    for (int j = 0; j < 8; ++j)
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j] + (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    s += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return s * (0.5f * global);
}
#endif  // avx512vnni available
#endif

// =====================================================================================
// CUDA stream memory operations (driver API, resolved via dlopen -- no link-time or
// toolchain dependence). The GPU side of the flag handshake: submit = WRITE_VALUE
// (done[slot]=0 then ready[slot]=1), sync = WAIT_VALUE(done[slot] >= 1). The wait is
// executed by the GPU front-end (no SM-resident kernel), so GPU "utilization" stays
// truthful during CPU compute windows -- a resident spin kernel pinned it at 99%,
// which laptop CPU/GPU dynamic power schedulers answered by clamping the CPU's max
// frequency (GEMV workers -1.5x: the reported edge regression). Availability is
// probed functionally at startup (memops_probe); anything unsupported (Windows WDDM,
// vGPU, old drivers) falls back to the cudaLaunchHostFunc path.
#if defined(_WIN32)
static void* cumemop_dlopen() { return (void*)::LoadLibraryA("nvcuda.dll"); }
static void* cumemop_dlsym(void* h, const char* n) {
  return (void*)::GetProcAddress((HMODULE)h, n);
}
#else
static void* cumemop_dlopen() {
  void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (h == nullptr) h = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
  return h;
}
static void* cumemop_dlsym(void* h, const char* n) { return dlsym(h, n); }
#endif

using cuMemOp64_fn = int (*)(void* stream, unsigned long long addr, unsigned long long value,
                             unsigned int flags);
static cuMemOp64_fn g_cu_write64 = nullptr;
static cuMemOp64_fn g_cu_wait64 = nullptr;
static constexpr unsigned int kCuWaitValueGeq = 0x0;   // CU_STREAM_WAIT_VALUE_GEQ
static constexpr unsigned int kCuWriteDefault = 0x0;   // CU_STREAM_WRITE_VALUE_DEFAULT

// ROCm has the same capability under different names, and the extension already
// links amdhip64 -- so resolve it at link time rather than dlopen'ing anything.
// Without this the probe fails on every ROCm box (libcuda.so.1 is a CUDA driver
// library and simply is not there), and cpu/hybrid decode silently falls back to
// the cudaLaunchHostFunc path at ~30-50us per call, twice per MoE layer per step.
// hipStreamWaitValue64 takes a trailing mask that the CUDA entry point does not, so
// it needs its own thunk rather than a straight function-pointer assignment.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__) || defined(USE_ROCM)
#define FT_HIP_STREAM_MEMOPS 1
static int ft_hip_write64(void* stream, unsigned long long addr, unsigned long long value,
                          unsigned int flags) {
  return static_cast<int>(hipStreamWriteValue64(
      static_cast<hipStream_t>(stream), reinterpret_cast<void*>(addr),
      static_cast<uint64_t>(value), flags));
}
static int ft_hip_wait64(void* stream, unsigned long long addr, unsigned long long value,
                         unsigned int flags) {
  return static_cast<int>(hipStreamWaitValue64(
      static_cast<hipStream_t>(stream), reinterpret_cast<void*>(addr),
      static_cast<uint64_t>(value), flags, ~0ULL));
}
// hipStreamWaitValueGte == CU_STREAM_WAIT_VALUE_GEQ == 0x0 and
// hipStreamWriteValueDefault == CU_STREAM_WRITE_VALUE_DEFAULT == 0x0, so the flag
// constants above carry over unchanged.
static_assert(hipStreamWaitValueGte == 0x0, "unexpected hipStreamWaitValueGte");
#endif

static bool cumemop_resolve() {
  static bool resolved = [] {
#ifdef FT_HIP_STREAM_MEMOPS
    g_cu_write64 = &ft_hip_write64;
    g_cu_wait64 = &ft_hip_wait64;
    return true;
#else
    void* h = cumemop_dlopen();
    if (h == nullptr) return false;
    // 11.7+ made the v2 entry points the default; older drivers export only the v1
    // names with the same signature.
    g_cu_write64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWriteValue64_v2"));
    if (g_cu_write64 == nullptr)
      g_cu_write64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWriteValue64"));
    g_cu_wait64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWaitValue64_v2"));
    if (g_cu_wait64 == nullptr)
      g_cu_wait64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWaitValue64"));
    return g_cu_write64 != nullptr && g_cu_wait64 != nullptr;
#endif
  }();
  return resolved;
}

// Functional probe on a scratch pinned int64: enqueue WRITE(7) + WAIT(>=7) + sync.
// Returns true only if the whole memop path works on THIS stream/device/driver.
static bool cumemops_probe(uintptr_t stream, uintptr_t scratch_addr) {
  if (!cumemop_resolve()) return false;
  auto* s = reinterpret_cast<void*>(stream);
  if (g_cu_write64(s, (unsigned long long)scratch_addr, 7ULL, kCuWriteDefault) != 0) return false;
  if (g_cu_wait64(s, (unsigned long long)scratch_addr, 7ULL, kCuWaitValueGeq) != 0) return false;
  return cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)) == cudaSuccess;
}

// GPU side of the flag handshake (see the block comment above): enqueued on the
// caller's (possibly capturing) stream; the WAIT immediate is the constant 1,
// replay-safe under CUDA graphs.
// The startup probe validates EAGER memops; a driver could still reject them at graph
// capture time. Those enqueue errors would otherwise be swallowed here and surface only
// as a later EndCapture failure -- log the first CUresult so triage is one step.
static void cumemop_check(int rc, const char* what) {
  static std::atomic<bool> warned{false};
  if (rc != 0 && !warned.exchange(true)) {
    std::fprintf(stderr,
                 "[freetoken/cpu_moe] %s failed with CUresult=%d (first occurrence; "
                 "subsequent errors are not repeated). If this happened during CUDA "
                 "graph capture, the driver lacks capture support for stream memops -- "
                 "set FREETOKEN_CPU_MOE_FLAG_SYNC=0.\n",
                 what, rc);
  }
}

static void cumemop_submit(uintptr_t stream, uintptr_t done_addr, uintptr_t ready_addr,
                           int64_t slot) {
  auto* s = reinterpret_cast<void*>(stream);
  // Order matters and is preserved by the front end: reset done BEFORE raising ready,
  // so the coordinator's completion write for THIS step can never be wiped.
  cumemop_check(g_cu_write64(s, (unsigned long long)(done_addr + (size_t)slot * 8), 0ULL,
                             kCuWriteDefault),
                "cuStreamWriteValue64(done)");
  cumemop_check(g_cu_write64(s, (unsigned long long)(ready_addr + (size_t)slot * 8), 1ULL,
                             kCuWriteDefault),
                "cuStreamWriteValue64(ready)");
}

static void cumemop_sync(uintptr_t stream, uintptr_t done_addr, int64_t slot) {
  cumemop_check(g_cu_wait64(reinterpret_cast<void*>(stream),
                            (unsigned long long)(done_addr + (size_t)slot * 8), 1ULL,
                            kCuWaitValueGeq),
                "cuStreamWaitValue64(done)");
}

struct DotChoice {
  dot_fn fn;
  const char* name;
};

// SIMD tiers, ascending. Each format picks the highest tier <= the one chosen by
// pick_isa() that it implements (fp4 formats have no bf16-specific tier, so the
// avx512bf16 tier maps to their avx512 kernel).
enum IsaTier { ISA_SCALAR = 0, ISA_AVX2 = 1, ISA_AVX512 = 2, ISA_AVX512BF16 = 3 };

// Best tier the CPU+build supports, optionally capped DOWN by
// FREETOKEN_CPU_MOE_ISA={scalar,avx2,avx512,avx512bf16} (A/B testing on a machine
// that supports more). FREETOKEN_CPU_MOE_SCALAR=1 forces scalar (legacy alias).
inline IsaTier pick_isa() {
#if CPU_MOE_X86
  if (getenv("FREETOKEN_CPU_MOE_SCALAR")) return ISA_SCALAR;
  IsaTier best = ISA_SCALAR;
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) best = ISA_AVX2;
  if (best >= ISA_AVX2 && __builtin_cpu_supports("avx512f")) best = ISA_AVX512;
#ifdef CPU_MOE_HAS_AVX512BF16
  if (best >= ISA_AVX512 && __builtin_cpu_supports("avx512bf16")) best = ISA_AVX512BF16;
#endif
  if (const char* f = getenv("FREETOKEN_CPU_MOE_ISA")) {
    IsaTier want = best;
    if (!std::strcmp(f, "scalar")) want = ISA_SCALAR;
    else if (!std::strcmp(f, "avx2")) want = ISA_AVX2;
    else if (!std::strcmp(f, "avx512")) want = ISA_AVX512;
    else if (!std::strcmp(f, "avx512bf16")) want = ISA_AVX512BF16;
    if (want < best) best = want;  // cap downward; never force above hw/build support
  }
  return best;
#else
  return ISA_SCALAR;
#endif
}

DotChoice select_dot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
#ifdef CPU_MOE_HAS_AVX512BF16
  if (t >= ISA_AVX512BF16) return {dot_avx512bf16, "avx512bf16"};
#endif
  if (t >= ISA_AVX512) return {dot_avx512f, "avx512f"};
  if (t >= ISA_AVX2) return {dot_avx2, "avx2"};
#endif
  (void)t;
  return {dot_scalar, "scalar"};
}

nvdot_fn select_nvdot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX512) return dot_nvfp4_avx512;
  if (t >= ISA_AVX2) return dot_nvfp4_avx2;
#endif
  (void)t;
  return dot_nvfp4_scalar;
}

// AVX-VNNI (VEX-256 VPDPBUSD) availability: Alder/Raptor Lake, Sapphire Rapids+, Zen5.
// Distinct from AVX-512 VNNI. Opt out with FREETOKEN_CPU_MOE_NO_VNNI=1 (A/B the W4A8 path).
inline bool cpu_has_avxvnni() {
#if CPU_MOE_X86
  const char* no = getenv("FREETOKEN_CPU_MOE_NO_VNNI");
  if (no && no[0] && no[0] != '0') return false;  // ignore unset/empty/"0"
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("avxvnni");
#else
  return false;
#endif
}

// AVX-512 VNNI (512-bit VPDPBUSD): Cascade Lake+, Ice Lake, Sapphire/Emerald Rapids,
// Zen4+. 2x the 256-bit AVX-VNNI width. FREETOKEN_CPU_MOE_NO_AVX512VNNI=1 forces the
// 256-bit path (A/B the two W4A8 kernels on the same box); FREETOKEN_CPU_MOE_NO_VNNI=1
// still disables the whole W4A8 family (back to the faithful fp32 nvdot), so it is
// honored here too.
inline bool cpu_has_avx512vnni() {
#if CPU_MOE_X86 && defined(CPU_MOE_HAS_AVX512VNNI)
  const char* no = getenv("FREETOKEN_CPU_MOE_NO_AVX512VNNI");
  if (no && no[0] && no[0] != '0') return false;
  const char* no_vnni = getenv("FREETOKEN_CPU_MOE_NO_VNNI");
  if (no_vnni && no_vnni[0] && no_vnni[0] != '0') return false;
  return __builtin_cpu_supports("avx512vnni");
#else
  return false;
#endif
}

// Best W4A8 (int8-activation) nvfp4 dot, or nullptr if no SIMD VNNI (caller keeps the
// faithful fp32 nvdot path). The scalar i8 dot exists only as a correctness reference.
nvi8dot_fn select_nvi8dot() {
#if CPU_MOE_X86
#if defined(CPU_MOE_HAS_AVX512VNNI)
  if (cpu_has_avx512vnni()) return dot_nvfp4_i8_avx512vnni;
#endif
  if (cpu_has_avxvnni()) return dot_nvfp4_i8_vnni;
#endif
  return nullptr;
}

// ----------------------- DeepSeek-V4 ds_fp4 (W4A8) ---------------------------
// Row-major e2m1 (2/byte, low nibble first) + e8m0 per-32 block scale, no global
// (w = E2M1[code] * 2^(e8m0-127)); activations are FP8-e4m3 round-tripped (per-128
// block, ue8m0 scale) before each GEMM. Matches kernel/triton/dsv4 (fused_moe +
// fp8_linear): silu(clamp(gate,max=lim)) * clamp(up,-lim,lim), router weight on the
// down output.

// Activations are pre-deinterleaved to fp32 (xe[m]=x[2m], xo[m]=x[2m+1]) once per
// token/route and reused across all output rows. This drops the hot dot to one
// vpmovzxbd + two vpermps per 32 weights (1.5 shuffle ops / 16 vs 4 for a bf16,
// dup-permute, per-element gather), so the row-major fp4 GEMV stops being port-5
// bound and approaches the bf16 memory-bandwidth ceiling.
using dsdot_fn = float (*)(const uint8_t*, const uint8_t*, const float*, const float*, int,
                           const float*, const float*);

float dot_dsfp4_scalar(const uint8_t* packed, const uint8_t* scale, const float* xe,
                       const float* xo, int K, const float* e2m1, const float* e8m0) {
  float acc = 0.0f;
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const float sc = e8m0[scale[b]];
    const uint8_t* pk = packed + (size_t)b * 16;  // 16 bytes = 32 codes
    const float* xeb = xe + (size_t)b * 16;
    const float* xob = xo + (size_t)b * 16;
    float bsum = 0.0f;
    for (int j = 0; j < 16; ++j) {
      const uint8_t byte = pk[j];
      bsum += e2m1[byte & 0xF] * xeb[j];   // low nibble  -> even-K activation
      bsum += e2m1[byte >> 4] * xob[j];    // high nibble -> odd-K activation
    }
    acc += sc * bsum;
  }
  return acc;
}

#if CPU_MOE_X86
// One 32-block: 16 bytes -> 16 low + 16 high nibble values via two vpermps, times
// the pre-split even/odd fp32 activations, folded by the per-32 e8m0 scale.
__attribute__((target("avx512f")))
inline __m512 dsfp4_blk(const uint8_t* pk, const float* xeb, const float* xob, __m512 lut,
                        __m512i loma, float sc) {
  __m512i wi = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pk)));
  __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
  __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
  __m512 prod = _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xeb), _mm512_mul_ps(vhi, _mm512_loadu_ps(xob)));
  return _mm512_mul_ps(prod, _mm512_set1_ps(sc));
}

__attribute__((target("avx512f")))
float dot_dsfp4_avx512(const uint8_t* packed, const uint8_t* scale, const float* xe,
                       const float* xo, int K, const float* e2m1, const float* e8m0) {
  const __m512 lut = _mm512_loadu_ps(e2m1);
  const __m512i loma = _mm512_set1_epi32(0xF);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  const int nb = K / 32;  // 16 packed bytes + one e8m0 scale per block
  int b = 0;
  for (; b + 2 <= nb; b += 2) {  // two independent accumulators hide FMA latency
    acc0 = _mm512_add_ps(acc0, dsfp4_blk(packed + (size_t)b * 16, xe + (size_t)b * 16,
                                         xo + (size_t)b * 16, lut, loma, e8m0[scale[b]]));
    acc1 = _mm512_add_ps(acc1, dsfp4_blk(packed + (size_t)(b + 1) * 16, xe + (size_t)(b + 1) * 16,
                                         xo + (size_t)(b + 1) * 16, lut, loma, e8m0[scale[b + 1]]));
  }
  for (; b < nb; ++b)
    acc0 = _mm512_add_ps(acc0, dsfp4_blk(packed + (size_t)b * 16, xe + (size_t)b * 16,
                                         xo + (size_t)b * 16, lut, loma, e8m0[scale[b]]));
  return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

// AVX2: a 32-K block is 16 bytes -> two 8-lane halves (8 even + 8 odd each).
__attribute__((target("avx2,fma")))
inline __m256 dsfp4_half_avx2(const uint8_t* pk, const float* xeb, const float* xob, __m256 mag8) {
  __m256i wi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk)));
  __m256 vlo = e2m1_decode8(_mm256_and_si256(wi, _mm256_set1_epi32(0xF)), mag8);
  __m256 vhi = e2m1_decode8(_mm256_srli_epi32(wi, 4), mag8);
  return _mm256_fmadd_ps(vlo, _mm256_loadu_ps(xeb), _mm256_mul_ps(vhi, _mm256_loadu_ps(xob)));
}

__attribute__((target("avx2,fma")))
float dot_dsfp4_avx2(const uint8_t* packed, const uint8_t* scale, const float* xe,
                     const float* xo, int K, const float* e2m1, const float* e8m0) {
  const __m256 mag8 = _mm256_loadu_ps(e2m1);
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 16;
    const float* xeb = xe + (size_t)b * 16;
    const float* xob = xo + (size_t)b * 16;
    const __m256 sc = _mm256_set1_ps(e8m0[scale[b]]);
    acc0 = _mm256_fmadd_ps(dsfp4_half_avx2(pk, xeb, xob, mag8), sc, acc0);
    acc1 = _mm256_fmadd_ps(dsfp4_half_avx2(pk + 8, xeb + 8, xob + 8, mag8), sc, acc1);
  }
  return hsum256(_mm256_add_ps(acc0, acc1));
}
#endif

dsdot_fn select_dsdot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX512) return dot_dsfp4_avx512;
  if (t >= ISA_AVX2) return dot_dsfp4_avx2;
#endif
  (void)t;
  return dot_dsfp4_scalar;
}

// ------------------------- mxfp4 (gpt-oss) GEMV -----------------------------
// Transposed split-K layout: blk[Kpairs, N2] (N innermost), scl[Kpairs/16, N2]
// e8m0 per 32-K. Computes out[c] = sum_kb (E2M1[lo]*x[2kb] + E2M1[hi]*x[2kb+1])
// * 2^(e8m0-127) for a contiguous column tile (blk/scl already offset to col 0 of
// the tile). Vectorized over N (16 columns / __m512), K stays the outer (cache-
// sequential) loop. Used by both gate_up (K=H) and down (K=I).
using mxgemv_fn = void (*)(float*, const uint8_t*, const uint8_t*, const bf16_t*, int, int,
                           int, const float*, const float*);

void mxfp4_gemv_scalar(float* out, const uint8_t* blk, const uint8_t* scl, const bf16_t* x,
                       int Kpairs, int N2, int ncol, const float* e2m1, const float* e8m0) {
  for (int c = 0; c < ncol; ++c) out[c] = 0.0f;
  for (int kb = 0; kb < Kpairs; ++kb) {
    const uint8_t* w = blk + (size_t)kb * N2;
    const uint8_t* s = scl + (size_t)(kb >> 4) * N2;
    const float xl = bf16_to_f32(x[2 * kb]);
    const float xh = bf16_to_f32(x[2 * kb + 1]);
    for (int c = 0; c < ncol; ++c) {
      const uint8_t byte = w[c];
      out[c] += (e2m1[byte & 0xF] * xl + e2m1[byte >> 4] * xh) * e8m0[s[c]];
    }
  }
}

#if CPU_MOE_X86
__attribute__((target("avx512f")))
void mxfp4_gemv_avx512(float* out, const uint8_t* blk, const uint8_t* scl, const bf16_t* x,
                       int Kpairs, int N2, int ncol, const float* e2m1, const float* e8m0) {
  (void)e8m0;  // e8m0[c]=2^(c-127) computed via bit construction (no gather)
  const __m512 lut = _mm512_loadu_ps(e2m1);
  const __m512i loma = _mm512_set1_epi32(0xF);
  // K-outer / N-inner: each kb cache line is read once and all live column chunks
  // (up to 4 -> 64 cols) accumulate from registers, so DRAM/L2 stream the tile once.
  int c0 = 0;
  for (; c0 + 16 <= ncol; c0 += 64) {
    const int nchunk = std::min(4, (ncol - c0) / 16);
    __m512 acc[4];
    for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm512_setzero_ps();
    for (int kblk = 0; kblk < Kpairs; kblk += 16) {  // 16 K-pairs = 32 K = one scale row
      __m512 sc[4];
      for (int ci = 0; ci < nchunk; ++ci) {
        __m128i sraw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
            scl + (size_t)(kblk >> 4) * N2 + c0 + ci * 16));
        sc[ci] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu8_epi32(sraw), 23));
      }
      __m512 blk_acc[4];
      for (int ci = 0; ci < nchunk; ++ci) blk_acc[ci] = _mm512_setzero_ps();
      for (int kk = 0; kk < 16; ++kk) {
        const int kb = kblk + kk;
        const uint8_t* wbase = blk + (size_t)kb * N2 + c0;
        // The transposed layout strides K by N2 bytes; prefetch ahead so the strided
        // reads are not exposed to DRAM latency (the HW streamer misses big strides).
        constexpr int PFD = 8;
        if (kb + PFD < Kpairs)
          _mm_prefetch(reinterpret_cast<const char*>(blk + (size_t)(kb + PFD) * N2 + c0),
                       _MM_HINT_T0);
        const __m512 xl = _mm512_set1_ps(bf16_to_f32(x[2 * kb]));
        const __m512 xh = _mm512_set1_ps(bf16_to_f32(x[2 * kb + 1]));
        for (int ci = 0; ci < nchunk; ++ci) {
          __m512i wi = _mm512_cvtepu8_epi32(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(wbase + ci * 16)));
          __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
          __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
          blk_acc[ci] = _mm512_fmadd_ps(vlo, xl, blk_acc[ci]);
          blk_acc[ci] = _mm512_fmadd_ps(vhi, xh, blk_acc[ci]);
        }
      }
      for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm512_fmadd_ps(blk_acc[ci], sc[ci], acc[ci]);
    }
    for (int ci = 0; ci < nchunk; ++ci) _mm512_storeu_ps(out + c0 + ci * 16, acc[ci]);
  }
  for (int c = c0; c < ncol; ++c) {  // tail columns (< 16)
    float o = 0.0f;
    for (int kb = 0; kb < Kpairs; ++kb) {
      const uint8_t byte = blk[(size_t)kb * N2 + c];
      uint32_t bits = (uint32_t)scl[(size_t)(kb >> 4) * N2 + c] << 23;
      float sc;
      std::memcpy(&sc, &bits, 4);
      o += (e2m1[byte & 0xF] * bf16_to_f32(x[2 * kb]) +
            e2m1[byte >> 4] * bf16_to_f32(x[2 * kb + 1])) * sc;
    }
    out[c] = o;
  }
}

__attribute__((target("avx2,fma")))
void mxfp4_gemv_avx2(float* out, const uint8_t* blk, const uint8_t* scl, const bf16_t* x,
                     int Kpairs, int N2, int ncol, const float* e2m1, const float* e8m0) {
  (void)e8m0;  // e8m0[s]=2^(s-127) built via s<<23 (no gather)
  const __m256 mag8 = _mm256_loadu_ps(e2m1);
  int c0 = 0;
  for (; c0 + 8 <= ncol; c0 += 32) {  // up to 4 chunks of 8 = 32 cols
    const int nchunk = std::min(4, (ncol - c0) / 8);
    __m256 acc[4];
    for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm256_setzero_ps();
    for (int kblk = 0; kblk < Kpairs; kblk += 16) {  // 16 K-pairs = one scale row
      __m256 sc[4];
      for (int ci = 0; ci < nchunk; ++ci) {
        __m128i sraw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
            scl + (size_t)(kblk >> 4) * N2 + c0 + ci * 8));
        sc[ci] = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu8_epi32(sraw), 23));
      }
      __m256 blk_acc[4];
      for (int ci = 0; ci < nchunk; ++ci) blk_acc[ci] = _mm256_setzero_ps();
      for (int kk = 0; kk < 16; ++kk) {
        const int kb = kblk + kk;
        const uint8_t* wbase = blk + (size_t)kb * N2 + c0;
        constexpr int PFD = 8;
        if (kb + PFD < Kpairs)
          _mm_prefetch(reinterpret_cast<const char*>(blk + (size_t)(kb + PFD) * N2 + c0),
                       _MM_HINT_T0);
        const __m256 xl = _mm256_set1_ps(bf16_to_f32(x[2 * kb]));
        const __m256 xh = _mm256_set1_ps(bf16_to_f32(x[2 * kb + 1]));
        for (int ci = 0; ci < nchunk; ++ci) {
          __m256i wi = _mm256_cvtepu8_epi32(
              _mm_loadl_epi64(reinterpret_cast<const __m128i*>(wbase + ci * 8)));
          __m256 vlo = e2m1_decode8(_mm256_and_si256(wi, _mm256_set1_epi32(0xF)), mag8);
          __m256 vhi = e2m1_decode8(_mm256_srli_epi32(wi, 4), mag8);
          blk_acc[ci] = _mm256_fmadd_ps(vlo, xl, blk_acc[ci]);
          blk_acc[ci] = _mm256_fmadd_ps(vhi, xh, blk_acc[ci]);
        }
      }
      for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm256_fmadd_ps(blk_acc[ci], sc[ci], acc[ci]);
    }
    for (int ci = 0; ci < nchunk; ++ci) _mm256_storeu_ps(out + c0 + ci * 8, acc[ci]);
  }
  for (int c = c0; c < ncol; ++c) {  // tail columns (< 8); none when ncol%8==0
    float o = 0.0f;
    for (int kb = 0; kb < Kpairs; ++kb) {
      const uint8_t byte = blk[(size_t)kb * N2 + c];
      uint32_t bits = (uint32_t)scl[(size_t)(kb >> 4) * N2 + c] << 23;
      float sc;
      std::memcpy(&sc, &bits, 4);
      o += (e2m1[byte & 0xF] * bf16_to_f32(x[2 * kb]) +
            e2m1[byte >> 4] * bf16_to_f32(x[2 * kb + 1])) * sc;
    }
    out[c] = o;
  }
}
#endif

mxgemv_fn select_mxgemv() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX512) return mxfp4_gemv_avx512;
  if (t >= ISA_AVX2) return mxfp4_gemv_avx2;
#endif
  (void)t;
  return mxfp4_gemv_scalar;
}

// Round a clamped |x|<=448 to nearest float8-e4m3 (RNE), back to fp32. Matches
// torch.float8_e4m3fn / triton .to(float8e4nv).
inline float e4m3_round(float x) {
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  const float a = std::fabs(x);
  if (a == 0.0f) return 0.0f;
  if (a >= 448.0f) return sign * 448.0f;
  int e;
  std::frexp(a, &e);  // a in [2^(e-1), 2^e)
  float step = std::ldexp(1.0f, e - 4);
  const float min_step = std::ldexp(1.0f, -9);  // e4m3 subnormal step (2^-9)
  if (step < min_step) step = min_step;
  float r = std::nearbyint(a / step) * step;
  if (r > 448.0f) r = 448.0f;
  return sign * r;
}

// IEEE ceil(log2(v)) for v>0 (matches dsv4 _log2_ceil / fast_round_scale).
inline int ceil_log2_pos(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  const int exp = (int)((bits >> 23) & 0xFF);
  const int man = (int)(bits & 0x7FFFFF);
  return exp - 127 + (man != 0 ? 1 : 0);
}

// Split an interleaved bf16 row into fp32 even/odd halves (even[m]=src[2m]).
// bf16->fp32 is exact, so this only reorders -- done once per token/route and
// reused across every output row of the GEMV.
inline void deinterleave_bf16_f32(const bf16_t* src, float* even, float* odd, int K) {
  for (int m = 0; m < K / 2; ++m) {
    even[m] = bf16_to_f32(src[2 * m]);
    odd[m] = bf16_to_f32(src[2 * m + 1]);
  }
}

// DeepSeek-V4 activation FP8 round-trip (bf16 in/out): per 128-block,
// s = 2^ceil(log2(max(|x|,1e-4)/448)); y = round_e4m3(clamp(x/s,+-448)) * s.
void fp8_roundtrip_bf16(const bf16_t* src, bf16_t* dst, int K) {
  for (int b0 = 0; b0 < K; b0 += 128) {
    const int b1 = std::min(K, b0 + 128);
    float amax = 1e-4f;
    for (int i = b0; i < b1; ++i) amax = std::max(amax, std::fabs(bf16_to_f32(src[i])));
    const float s = std::ldexp(1.0f, ceil_log2_pos(amax * (1.0f / 448.0f)));
    const float inv_s = 1.0f / s;
    for (int i = b0; i < b1; ++i) {
      float q = bf16_to_f32(src[i]) * inv_s;
      q = std::min(448.0f, std::max(-448.0f, q));
      dst[i] = f32_to_bf16(e4m3_round(q) * s);
    }
  }
}

// --------------------------------- executor ---------------------------------

struct CpuMoeExecutor;

struct MoeTask {
  CpuMoeExecutor* exec;
  int layer_id;
  int num_tokens;
  const bf16_t* x;     // [num_tokens, H]
  const int32_t* ids;  // [num_tokens, top_k]  (raw expert ids; <0 = skip)
  const float* w;      // [num_tokens, top_k]
  bf16_t* y;           // [num_tokens, H]
};

// Output-row tiling. Small enough to give every worker independent work even at
// batch size 1; large enough to amortize the atomic work-grab.
//
// Bandwidth notes (Sapphire Rapids 8480+, 13 cores): the two passes already read
// every expert weight byte exactly once per token (each output row block is owned
// by one worker), and x stays hot in L1 across a (token,expert)'s rows -- so the
// kernel is single-read bandwidth-optimal at bs=1 (~205 GB/s vs ~55 GB/s PCIe).
// One worker per *physical* core, pinned, is the sweet spot; SMT oversubscription
// thrashes the spin-barrier. Deferred (not worth it here / for this workload):
//   - AMX-bf16: a GEMM tile engine; decode is M=1 GEMV so tiles sit idle. It would
//     only pay off in a grouped/batched (dedup) path.
//   - expert dedup for bs>1: read each distinct expert once and GEMM its tokens.
//     Helps locality+bytes when bs is large; decode batches here are tiny (<=4).
//   - NUMA: a single node is assumed. Multi-socket machines would split each
//     expert's K dimension per node (banks are already per-row contiguous).
constexpr int IBLK = 32;
constexpr int HBLK = 32;

// -------------------------------- Q4_0 (W4A8) --------------------------------
// Native GGUF Q4_0 experts (gemma4 GGUF): per-32 block = fp16 scale d + 16 packed
// bytes; byte j holds element j in its low nibble and j+16 in its high nibble, so a
// block's storage order is [lo0..lo15, hi0..hi15] and w = (nibble - 8) * d. Matches
// the reference dequant (models/gguf/dequant.py) and the packed banks the GPU offload
// path streams.
//
// llama.cpp ggml_vec_dot_q4_0_q8_0: W4A8. The activation is pre-quantized to Q8_0
// (per-32-block int8 ``aq`` + fp32 scale ``asb``); each block unpacks its 16 bytes to
// 32 int8 weights in [-8,7] (bytes_from_nibbles_32: low nibbles -> elems 0..15, high
// -> 16..31) and runs an integer block dot -- VPDPBUSD (AVX-VNNI) or VPMADDUBSW+VPMADDWD
// (AVX2) with the ggml sign trick |w|*(sign(w)*a)=w*a, or a scalar int loop -- then
// scales the block sum by wd*xd in fp32. No fp weight dequant / shuffle chain. The GPU
// offload path (ggml_moe_a8_vec / MMVQ) is also W4A8, so cpu and hybrid stay close.
using q4dot_fn = float (*)(const uint8_t*, const int8_t*, const float*, int);

float q4_0_dot_i8_scalar(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  float acc = 0.0f;
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    const uint8_t* q = blk + 2;  // 16 nibble bytes
    const int8_t* a = aq + (size_t)b * 32;
    int isum = 0;
    for (int j = 0; j < 16; ++j) {
      isum += ((int)(q[j] & 0x0F) - 8) * (int)a[j];       // elem j
      isum += ((int)(q[j] >> 4) - 8) * (int)a[16 + j];    // elem 16+j
    }
    acc += fp16_to_f32(dh) * asb[b] * (float)isum;
  }
  return acc;
}

#if CPU_MOE_X86
// fp16 block scale -> fp32 via HW F16C (single value in lane 0).
__attribute__((target("f16c")))
static inline float q4_scale(uint16_t h) {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}

// Unpack one Q4_0 block's 16 bytes -> 32 int8 weights in [-8,7] (elems 0..15 = low
// nibbles, 16..31 = high nibbles). ``eight`` = _mm256_set1_epi8(8).
__attribute__((target("avx2")))
static inline __m256i q4_unpack32(const uint8_t* blk, __m128i mask, __m256i eight) {
  const __m128i qb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
  const __m128i lo = _mm_and_si128(qb, mask);
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask);
  return _mm256_sub_epi8(_mm256_set_m128i(hi, lo), eight);
}

// AVX2 W4A8 (llama.cpp non-VNNI mul_sum_i8_pairs): integer block dot via VPMADDUBSW +
// VPMADDWD (sign trick), scaled by wd*xd. |aw*sa| pair sums <= 8*127*2 < 32767 -> no
// int16 saturation. This is the fast path on AVX2 CPUs without AVX-VNNI (and the
// avx512-tier fallback, since the block dot is 256-bit either way).
__attribute__((target("avx2,fma,f16c")))
float q4_0_dot_i8_avx2(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi8(8);
  const __m256i ones16 = _mm256_set1_epi16(1);
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    _mm_prefetch(reinterpret_cast<const char*>(blk) + 512, _MM_HINT_T0);
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    __m256i wq = q4_unpack32(blk, mask, eight);
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 32));
    __m256i aw = _mm256_sign_epi8(wq, wq);              // |wq|          (unsigned operand)
    __m256i sa = _mm256_sign_epi8(a, wq);               // sign(wq) * a  (signed operand)
    __m256i d32 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sa), ones16);  // 8 int32
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d32), _mm256_set1_ps(q4_scale(dh) * asb[b]), accF);
  }
  return hsum256(accF);
}

// AVX-VNNI W4A8: one VPDPBUSD per block (the fast path on modern CPUs).
__attribute__((target("avx2,avxvnni,fma,f16c")))
float q4_0_dot_i8_vnni(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi8(8);
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    _mm_prefetch(reinterpret_cast<const char*>(blk) + 512, _MM_HINT_T0);
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    __m256i wq = q4_unpack32(blk, mask, eight);
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 32));
    __m256i aw = _mm256_sign_epi8(wq, wq);   // |wq|            (unsigned operand)
    __m256i sa = _mm256_sign_epi8(a, wq);    // sign(wq) * a    (signed operand)
    __m256i di = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, sa);
    // All 32 elems of the block share wd*xd; distribute over di's 8 partial sums and
    // reduce at the end (equivalent to scale * block_total).
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(di), _mm256_set1_ps(q4_scale(dh) * asb[b]), accF);
  }
  return hsum256(accF);
}
#endif  // CPU_MOE_X86

// All tiers are W4A8 (int8 activations pre-quantized to Q8_0). AVX-VNNI is orthogonal to
// the ISA tier (gated by cpu_has_avxvnni() / FREETOKEN_CPU_MOE_NO_VNNI), so it wins when
// present; otherwise the 256-bit VPMADDUBSW kernel covers both the avx2 and avx512 tiers.
q4dot_fn select_q4dot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (cpu_has_avxvnni()) return q4_0_dot_i8_vnni;
  if (t >= ISA_AVX2) return q4_0_dot_i8_avx2;
#endif
  (void)t;
  return q4_0_dot_i8_scalar;
}

// GGML's Q4_K x Q8_K and Q5_1 x Q8_1 block layouts and scalar dot equations, following
// llama.cpp (MIT): ggml/src/ggml-cpu/quants.c and ggml/src/ggml-quants.c.
struct Q8KBlock {
  float d;
  int8_t qs[256];
  int16_t bsums[16];
};
struct Q8_1Block {
  uint16_t d;
  uint16_t s;
  int8_t qs[32];
};
static_assert(sizeof(Q8KBlock) == 292, "GGML Q8_K activation block layout");
static_assert(sizeof(Q8_1Block) == 36, "GGML Q8_1 activation block layout");

using q8_1_quant_fn = void (*)(const bf16_t*, int, Q8_1Block*);

void quant_q8_1_scalar(const bf16_t* x, int K, Q8_1Block* out) {
  for (int b = 0; b < K / 32; ++b) {
    Q8_1Block& block = out[b];
    const bf16_t* xb = x + (size_t)b * 32;
    float xf[32], amax = 0.0f;
    for (int j = 0; j < 32; ++j) {
      xf[j] = bf16_to_f32(xb[j]);
      amax = std::max(amax, std::fabs(xf[j]));
    }
    const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
    const float inv = amax > 0.0f ? 1.0f / d : 0.0f;
    int sum = 0;
    for (int j = 0; j < 32; ++j) {
      const int q = std::max(-127, std::min(127, (int)std::lround(xf[j] * inv)));
      block.qs[j] = (int8_t)q;
      sum += q;
    }
    block.d = f32_to_fp16(d);
    block.s = f32_to_fp16(d * (float)sum);
  }
}

#if CPU_MOE_X86
__attribute__((target("avx2")))
static inline __m256 bf16x8_to_f32_avx2(const bf16_t* p) {
  // Widen the eight BF16 words, then shift each lane into the BF16 float
  // position. The AVX2 intrinsic takes an eight-word __m128i source.
  const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  const __m256i lanes = _mm256_cvtepu16_epi32(raw);
  return _mm256_castsi256_ps(_mm256_slli_epi32(lanes, 16));
}

__attribute__((target("avx2")))
static inline float hmax8_avx2(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 m = _mm_max_ps(lo, hi);
  m = _mm_max_ps(m, _mm_movehl_ps(m, m));
  m = _mm_max_ss(m, _mm_movehdup_ps(m));
  return _mm_cvtss_f32(m);
}

__attribute__((target("avx2")))
void quant_q8_1_avx2(const bf16_t* x, int K, Q8_1Block* out) {
  const __m256 sign = _mm256_set1_ps(-0.0f);
  const __m256 half = _mm256_set1_ps(0.5f);
  for (int b = 0; b < K / 32; ++b) {
    Q8_1Block& block = out[b];
    const bf16_t* xb = x + (size_t)b * 32;
    __m256 v[4];
    for (int j = 0; j < 4; ++j) {
      v[j] = bf16x8_to_f32_avx2(xb + j * 8);
    }
    __m256 maxv = _mm256_andnot_ps(sign, v[0]);
    maxv = _mm256_max_ps(maxv, _mm256_andnot_ps(sign, v[1]));
    maxv = _mm256_max_ps(maxv, _mm256_andnot_ps(sign, v[2]));
    maxv = _mm256_max_ps(maxv, _mm256_andnot_ps(sign, v[3]));
    const float amax = hmax8_avx2(maxv);
    const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
    const float inv = amax > 0.0f ? 1.0f / d : 0.0f;
    const __m256 mul = _mm256_set1_ps(inv);
    __m256i qi[4];
    const __m256i qmin = _mm256_set1_epi32(-127);
    const __m256i qmax = _mm256_set1_epi32(127);
    for (int j = 0; j < 4; ++j) {
      const __m256 scaled = _mm256_mul_ps(v[j], mul);
      // std::lround rounds halfway cases away from zero. Adding a signed
      // 0.5 followed by truncation reproduces that rule (unlike round_ps).
      const __m256 signed_half = _mm256_or_ps(half, _mm256_and_ps(sign, scaled));
      qi[j] = _mm256_max_epi32(
          qmin, _mm256_min_epi32(qmax, _mm256_cvttps_epi32(
              _mm256_add_ps(scaled, signed_half))));
    }
    const __m256i q01 = _mm256_packs_epi32(qi[0], qi[1]);
    const __m256i q23 = _mm256_packs_epi32(qi[2], qi[3]);
    __m256i packed = _mm256_packs_epi16(q01, q23);
    packed = _mm256_permutevar8x32_epi32(
        packed, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(block.qs), packed);
    __m256i sums = _mm256_add_epi32(_mm256_add_epi32(qi[0], qi[1]),
                                    _mm256_add_epi32(qi[2], qi[3]));
    int32_t sum_lanes[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum_lanes), sums);
    int sum = 0;
    for (int j = 0; j < 8; ++j) sum += sum_lanes[j];
    block.d = f32_to_fp16(d);
    block.s = f32_to_fp16(d * (float)sum);
  }
}
#endif

q8_1_quant_fn select_q8_1_quant() {
#if CPU_MOE_X86
  // Keep quantizer A/B tests independent of FREETOKEN_CPU_MOE_ISA, which also
  // changes the mixed-format dot kernels. The override is intentionally narrow.
  if (const char* f = getenv("FREETOKEN_CPU_MOE_Q8_1")) {
    if (!std::strcmp(f, "scalar")) return quant_q8_1_scalar;
    if (!std::strcmp(f, "avx2") && __builtin_cpu_supports("avx2")) return quant_q8_1_avx2;
  }
  if (pick_isa() >= ISA_AVX2) return quant_q8_1_avx2;
#endif
  return quant_q8_1_scalar;
}

using q4kdot_fn = float (*)(const uint8_t*, const Q8KBlock*, int);
using q5_1dot_fn = float (*)(const uint8_t*, const Q8_1Block*, int);
using q4kdot_multi_fn = void (*)(const uint8_t*, const Q8KBlock* const*, int, int, float*);
using q5_1dot_multi_fn = void (*)(const uint8_t*, const Q8_1Block* const*, int, int, float*);

float q4_k_dot_q8_k_scalar(const uint8_t* w, const Q8KBlock* a, int K) {
  float acc = 0.0f;
  for (int b = 0; b < K / 256; ++b) {
    const uint8_t* wb = w + (size_t)b * 144;
    uint16_t dh, dminh;
    std::memcpy(&dh, wb, 2);
    std::memcpy(&dminh, wb + 2, 2);
    const uint8_t* scales = wb + 4;
    const uint8_t* qs = wb + 16;
    const Q8KBlock& ab = a[b];
    float block_dot = 0.0f;
    float block_min = 0.0f;
    for (int g = 0; g < 8; ++g) {
      int sc, m;
      if (g < 4) {
        sc = scales[g] & 0x3f;
        m = scales[g + 4] & 0x3f;
      } else {
        sc = (scales[g + 4] & 0x0f) | ((scales[g - 4] >> 6) << 4);
        m = (scales[g + 4] >> 4) | ((scales[g] >> 6) << 4);
      }
      int dot = 0;
      for (int j = 0; j < 32; ++j) {
        const uint8_t packed = qs[(g / 2) * 32 + j];
        const int q = (g & 1) ? (packed >> 4) : (packed & 0x0f);
        const int i = g * 32 + j;
        dot += q * (int)ab.qs[i];
      }
      const int sum_a = (int)ab.bsums[g * 2] + (int)ab.bsums[g * 2 + 1];
      block_dot += (float)sc * (float)dot;
      block_min += (float)m * (float)sum_a;
    }
    acc += ab.d * (fp16_to_f32(dh) * block_dot - fp16_to_f32(dminh) * block_min);
  }
  return acc;
}

float q5_1_dot_q8_1_scalar(const uint8_t* w, const Q8_1Block* a, int K) {
  float acc = 0.0f;
  for (int b = 0; b < K / 32; ++b) {
    const uint8_t* wb = w + (size_t)b * 24;
    uint16_t dh, mh;
    std::memcpy(&dh, wb, 2);
    std::memcpy(&mh, wb + 2, 2);
    const uint8_t* qh = wb + 4;
    const uint8_t* qs = wb + 8;
    int dot = 0;
    for (int j = 0; j < 16; ++j) {
      const int q0 = (qs[j] & 0x0f) | (((qh[j / 8] >> (j & 7)) & 1) << 4);
      const int q1 = (qs[j] >> 4) | (((qh[(j + 16) / 8] >> ((j + 16) & 7)) & 1) << 4);
      dot += (q0 * (int)a[b].qs[j]) + (q1 * (int)a[b].qs[16 + j]);
    }
    acc += fp16_to_f32(dh) * fp16_to_f32(a[b].d) * (float)dot +
           fp16_to_f32(mh) * fp16_to_f32(a[b].s);
  }
  return acc;
}

void q4_k_dot_q8_k_multi_scalar(const uint8_t* w, const Q8KBlock* const* a, int width,
                                int K, float* out) {
  for (int r = 0; r < width; ++r) out[r] = q4_k_dot_q8_k_scalar(w, a[r], K);
}

void q5_1_dot_q8_1_multi_scalar(const uint8_t* w, const Q8_1Block* const* a, int width,
                                int K, float* out) {
  for (int r = 0; r < width; ++r) out[r] = q5_1_dot_q8_1_scalar(w, a[r], K);
}

#if CPU_MOE_X86
__attribute__((target("avx2,fma")))
static inline __m256i q5_1_nibbles_avx2(const uint8_t* p) {
  const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  const __m256i both = _mm256_inserti128_si256(
      _mm256_castsi128_si256(packed), _mm_srli_epi16(packed, 4), 1);
  return _mm256_and_si256(both, _mm256_set1_epi8(0x0f));
}

__attribute__((target("avx2,fma")))
static inline __m256i q5_1_high_bits_avx2(const uint8_t* p) {
  uint32_t bits;
  std::memcpy(&bits, p, sizeof(bits));
  const __m256i lanes = _mm256_set_epi64x(
      0x0303030303030303LL, 0x0202020202020202LL,
      0x0101010101010101LL, 0x0000000000000000LL);
  __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int)bits), lanes);
  bytes = _mm256_or_si256(bytes, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL));
  return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

__attribute__((target("avx2,fma")))
static inline __m256 qx_qy_dot8_avx2(__m256i qx, __m256i qy) {
  const __m256i pair16 = _mm256_maddubs_epi16(qx, qy);
  const __m256i quad32 = _mm256_madd_epi16(pair16, _mm256_set1_epi16(1));
  return _mm256_cvtepi32_ps(quad32);
}

__attribute__((target("avx2,fma")))
float q5_1_dot_q8_1_avx2(const uint8_t* w, const Q8_1Block* a, int K) {
  __m256 acc = _mm256_setzero_ps();
  float offset = 0.0f;
  for (int b = 0; b < K / 32; ++b) {
    const uint8_t* wb = w + (size_t)b * 24;
    uint16_t dh, mh;
    std::memcpy(&dh, wb, 2);
    std::memcpy(&mh, wb + 2, 2);
    __m256i qx = q5_1_nibbles_avx2(wb + 8);
    const __m256i hi = _mm256_and_si256(
        q5_1_high_bits_avx2(wb + 4), _mm256_set1_epi8(0x10));
    qx = _mm256_or_si256(qx, hi);
    const __m256i qy = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(a[b].qs));
    const float scale = fp16_to_f32(dh) * fp16_to_f32(a[b].d);
    acc = _mm256_fmadd_ps(qx_qy_dot8_avx2(qx, qy), _mm256_set1_ps(scale), acc);
    offset += fp16_to_f32(mh) * fp16_to_f32(a[b].s);
  }
  return hsum256(acc) + offset;
}

__attribute__((target("avx2,fma")))
float q4_k_dot_q8_k_avx2(const uint8_t* w, const Q8KBlock* a, int K) {
  float acc = 0.0f;
  const __m256i mask = _mm256_set1_epi8(0x0f);
  for (int b = 0; b < K / 256; ++b) {
    const uint8_t* wb = w + (size_t)b * 144;
    uint16_t dh, dminh;
    std::memcpy(&dh, wb, 2);
    std::memcpy(&dminh, wb + 2, 2);
    const uint8_t* scales = wb + 4;
    const uint8_t* qs = wb + 16;
    const Q8KBlock& ab = a[b];
    float block_dot = 0.0f;
    float block_min = 0.0f;
    for (int g = 0; g < 8; ++g) {
      int sc, m;
      if (g < 4) {
        sc = scales[g] & 0x3f;
        m = scales[g + 4] & 0x3f;
      } else {
        sc = (scales[g + 4] & 0x0f) | ((scales[g - 4] >> 6) << 4);
        m = (scales[g + 4] >> 4) | ((scales[g] >> 6) << 4);
      }
      const __m256i packed = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(qs + (g / 2) * 32));
      const __m256i qx = (g & 1)
          ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask)
          : _mm256_and_si256(packed, mask);
      const __m256i qy = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(ab.qs + g * 32));
      block_dot += (float)sc * hsum256(qx_qy_dot8_avx2(qx, qy));
      block_min += (float)m *
          ((float)ab.bsums[g * 2] + (float)ab.bsums[g * 2 + 1]);
    }
    acc += ab.d * (fp16_to_f32(dh) * block_dot - fp16_to_f32(dminh) * block_min);
  }
  return acc;
}

__attribute__((target("avx2,fma")))
void q5_1_dot_q8_1_multi_avx2(const uint8_t* w, const Q8_1Block* const* a, int width,
                              int K, float* out) {
  __m256 acc[4] = {
      _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
  float offset[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int b = 0; b < K / 32; ++b) {
    const uint8_t* wb = w + (size_t)b * 24;
    uint16_t dh, mh;
    std::memcpy(&dh, wb, 2);
    std::memcpy(&mh, wb + 2, 2);
    // Decode this packed weight block once and reuse it for every route.
    __m256i qx = q5_1_nibbles_avx2(wb + 8);
    const __m256i hi = _mm256_and_si256(
        q5_1_high_bits_avx2(wb + 4), _mm256_set1_epi8(0x10));
    qx = _mm256_or_si256(qx, hi);
    const float scale = fp16_to_f32(dh);
    const float min_scale = fp16_to_f32(mh);
    for (int r = 0; r < width; ++r) {
      const __m256i qy = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(a[r][b].qs));
      acc[r] = _mm256_fmadd_ps(qx_qy_dot8_avx2(qx, qy),
                               _mm256_set1_ps(scale * fp16_to_f32(a[r][b].d)), acc[r]);
      offset[r] += min_scale * fp16_to_f32(a[r][b].s);
    }
  }
  for (int r = 0; r < width; ++r) out[r] = hsum256(acc[r]) + offset[r];
}

__attribute__((target("avx2,fma")))
void q4_k_dot_q8_k_multi_avx2(const uint8_t* w, const Q8KBlock* const* a, int width,
                              int K, float* out) {
  // Narrow groups preserve the per-group reduction order; width four amortizes two
  // reductions over four Q4_K groups without changing the single-route path.
  if (width < 4) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const __m256i mask = _mm256_set1_epi8(0x0f);
    for (int b = 0; b < K / 256; ++b) {
      const uint8_t* wb = w + (size_t)b * 144;
      uint16_t dh, dminh;
      std::memcpy(&dh, wb, 2);
      std::memcpy(&dminh, wb + 2, 2);
      const uint8_t* scales = wb + 4;
      const uint8_t* qs = wb + 16;
      float block_dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float block_min[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int g = 0; g < 8; ++g) {
        int sc, m;
        if (g < 4) {
          sc = scales[g] & 0x3f;
          m = scales[g + 4] & 0x3f;
        } else {
          sc = (scales[g + 4] & 0x0f) | ((scales[g - 4] >> 6) << 4);
          m = (scales[g + 4] >> 4) | ((scales[g] >> 6) << 4);
        }
        const __m256i packed = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(qs + (g / 2) * 32));
        const __m256i qx = (g & 1)
            ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask)
            : _mm256_and_si256(packed, mask);
        for (int r = 0; r < width; ++r) {
          const __m256i qy = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(a[r][b].qs + g * 32));
          block_dot[r] += (float)sc * hsum256(qx_qy_dot8_avx2(qx, qy));
          block_min[r] += (float)m *
              ((float)a[r][b].bsums[g * 2] + (float)a[r][b].bsums[g * 2 + 1]);
        }
      }
      const float d = fp16_to_f32(dh), dmin = fp16_to_f32(dminh);
      for (int r = 0; r < width; ++r)
        acc[r] += a[r][b].d * (d * block_dot[r] - dmin * block_min[r]);
    }
    for (int r = 0; r < width; ++r) out[r] = acc[r];
    return;
  }

  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const __m256i mask = _mm256_set1_epi8(0x0f);
  for (int b = 0; b < K / 256; ++b) {
    const uint8_t* wb = w + (size_t)b * 144;
    uint16_t dh, dminh;
    std::memcpy(&dh, wb, 2);
    std::memcpy(&dminh, wb + 2, 2);
    const uint8_t* scales = wb + 4;
    const uint8_t* qs = wb + 16;
    const float d = fp16_to_f32(dh), dmin = fp16_to_f32(dminh);
    __m256 block_dot[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    float block_min[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int g = 0; g < 8; ++g) {
      int sc, m;
      if (g < 4) {
        sc = scales[g] & 0x3f;
        m = scales[g + 4] & 0x3f;
      } else {
        sc = (scales[g + 4] & 0x0f) | ((scales[g - 4] >> 6) << 4);
        m = (scales[g + 4] >> 4) | ((scales[g] >> 6) << 4);
      }
      // Decode this packed weight group once and reuse it for every route.
      const __m256i packed = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(qs + (g / 2) * 32));
      const __m256i qx = (g & 1)
          ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask)
          : _mm256_and_si256(packed, mask);
      for (int r = 0; r < width; ++r) {
        const __m256i qy = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a[r][b].qs + g * 32));
        block_dot[r] = _mm256_fmadd_ps(qx_qy_dot8_avx2(qx, qy),
                                       _mm256_set1_ps((float)sc), block_dot[r]);
        block_min[r] += (float)m *
            ((float)a[r][b].bsums[g * 2] + (float)a[r][b].bsums[g * 2 + 1]);
        if ((g & 3) == 3) {
          acc[r] += a[r][b].d * d * hsum256(block_dot[r]);
          block_dot[r] = _mm256_setzero_ps();
        }
      }
    }
    for (int r = 0; r < width; ++r)
      acc[r] -= a[r][b].d * dmin * block_min[r];
  }
  for (int r = 0; r < width; ++r) out[r] = acc[r];
}

#endif

q4kdot_fn select_q4kdot() {
#if CPU_MOE_X86
  if (pick_isa() >= ISA_AVX2) return q4_k_dot_q8_k_avx2;
#endif
  return q4_k_dot_q8_k_scalar;
}

q5_1dot_fn select_q5_1dot() {
#if CPU_MOE_X86
  if (pick_isa() >= ISA_AVX2) return q5_1_dot_q8_1_avx2;
#endif
  return q5_1_dot_q8_1_scalar;
}

q4kdot_multi_fn select_q4kdot_multi() {
#if CPU_MOE_X86
  if (pick_isa() >= ISA_AVX2) return q4_k_dot_q8_k_multi_avx2;
#endif
  return q4_k_dot_q8_k_multi_scalar;
}

q5_1dot_multi_fn select_q5_1dot_multi() {
#if CPU_MOE_X86
  if (pick_isa() >= ISA_AVX2) return q5_1_dot_q8_1_multi_avx2;
#endif
  return q5_1_dot_q8_1_multi_scalar;
}

enum WFmt { WF_BF16 = 0, WF_NVFP4 = 1, WF_MXFP4 = 2, WF_DSFP4 = 3, WF_Q4_0 = 4 };

// Each ctor pointer arg is the address of a CPU int64 array of length
// num_layers (one base address per layer, built by cpu_executor.py's
// _make_table), not a single flat bank. tbl_at resolves
// tbl[layer_id] once per task/pass; a null table (bank unused by this fmt, ptr
// arg 0) resolves to nullptr without dereferencing.
inline const void* tbl_at(const uint64_t* tbl, int layer_id) {
  return tbl ? reinterpret_cast<const void*>(tbl[layer_id]) : nullptr;
}

struct CpuMoeExecutor {
  int num_threads;
  int num_layers, num_experts, top_k;
  int H, I;
  int act, apply_on_input;
  int fmt;                // WFmt
  bool needs_di = false;  // pre-deinterleave activations to fp32 (nvfp4/ds_fp4)
  // Per-layer pointer tables (one base address per layer, see tbl_at). gate_up_tbl
  // doubles as the bf16 gate_up table and the nvfp4/mxfp4/q4_0/ds_fp4 packed-gate_up
  // table (down_tbl likewise for down); which reinterpretation applies is picked by
  // fmt at each resolve site (see gemm1_dot/gemm2_dot/do_pass1_mxfp4/do_pass1_dsfp4).
  const uint64_t* gate_up_tbl;   // bf16: [E,2I,H] rows; else: packed e2m1/mxfp4-blocks
  const uint64_t* down_tbl;      // bf16: [E,H,I] rows; else: packed e2m1/mxfp4-blocks
  const uint64_t* gu_scale_tbl;  // nvfp4/mxfp4/ds_fp4: [E,2I,*] block scales
  const uint64_t* gu_global_tbl; // nvfp4: [E,2I] fp16 row globals
  const uint64_t* dn_scale_tbl;  // nvfp4/mxfp4/ds_fp4: [E,H,*] block scales
  const uint64_t* dn_global_tbl; // nvfp4: [E,H] fp16 row globals
  const uint64_t* gu_bias_tbl;   // mxfp4: [E,2I] bf16 biases
  const uint64_t* dn_bias_tbl;   // mxfp4: [E,H] bf16 biases
  float swiglu_alpha;
  float swiglu_limit;          // +inf == no clamp
  dot_fn dot;
  nvdot_fn nvdot;
  nvi8dot_fn nvi8dot = nullptr;  // AVX-VNNI W4A8 nvfp4 dot (nullptr -> use fp32 nvdot)
  bool use_vnni = false;         // nvfp4 + AVX-VNNI: decode via int8 VPDPBUSD (W4A8)
  bool use_q4a8 = false;       // q4_0: always W4A8 (llama.cpp Q4_0 x Q8_0); int8 pre-quant
  bool use_q4k_q5_1 = false;   // Q4_K gate/up x Q8_K, Q5_1 down x Q8_1
  dsdot_fn dsdot;
  mxgemv_fn mxgemv;
  q4dot_fn q4dot;
  q4kdot_fn q4kdot;
  q5_1dot_fn q5_1dot;
  q4kdot_multi_fn q4kdot_multi;
  q5_1dot_multi_fn q5_1dot_multi;
  q8_1_quant_fn q8_1_quant;
  const char* q8_1_isa = "scalar";
  // ds_fp4: the caller already FP8-round-tripped the input activations on the GPU
  // (same reference grid), so submit() must not repeat it on the host-callback
  // thread. That scalar per-element pass is single-threaded ON THE DECODE CRITICAL
  // PATH (~0.3ms/layer at H=4096, every worker and the GPU waiting on it); moving
  // it to a captured GPU elementwise kernel removes it while keeping the official
  // W4A8 numerics bit-exact. Set via set_input_prequant (see cpu_executor.py).
  bool input_prequant = false;
  // Q4_0 packed-row byte strides (H/32*18 for gate_up over K=H, I/32*18 for down over K=I).
  int q4_gu_row_bytes = 0, q4_dn_row_bytes = 0;
  std::vector<Q8KBlock> xq8k_scratch;
  std::vector<Q8_1Block> gq8_1_scratch;
  // Q4_K/Q5_1 batched dedup metadata. Routes are grouped by their final expert
  // (or host-slot) id only when a batch actually contains a duplicate.
  bool grouped_q4k_q5_1 = false;
  std::vector<int> grouped_experts;
  std::vector<int64_t> grouped_offsets;
  std::vector<int64_t> grouped_routes;
  std::vector<float> route_out;
  float e2m1_lut[16];
  float e4m3_lut[256];
  float e8m0_lut[256];         // mxfp4 block scale: 2^(s-127), s clamped to [0,254]
  const char* isa;

  std::vector<bf16_t> g_scratch;   // [max_tokens * top_k * I] intermediate
  std::vector<bf16_t> xq_scratch;  // [max_tokens * H] ds_fp4 fp8-roundtripped input
  // ds_fp4 activations pre-deinterleaved to fp32 (even/odd K) for the row-major dot.
  std::vector<float> xe_scratch, xo_scratch;  // [max_tokens * H/2]   (input)
  std::vector<float> ge_scratch, go_scratch;  // [max_tokens*top_k*I/2] (intermediate)
  // AVX-VNNI W4A8: per-16-block int8 activations [even(8),odd(8)] + per-block scale.
  std::vector<int8_t> xi8_scratch, gi8_scratch;  // [max_tokens*H], [max_tokens*top_k*I]
  std::vector<float> xas_scratch, gas_scratch;   // [max_tokens*H/16], [..*top_k*I/16]
  std::string isa_str;

  std::vector<std::thread> workers;
  std::mutex task_mtx;
  std::condition_variable task_cv;
  std::mutex sync_mtx;
  std::condition_variable sync_cv;

  bool stop = false;
  uint64_t cur_gen = 0;
  MoeTask* cur_task = nullptr;
  std::atomic<uint64_t> submitted{0};
  std::atomic<uint64_t> completed{0};

  std::atomic<int64_t> p1_next{0};
  std::atomic<int64_t> p2_next{0};
  std::atomic<int64_t> prt_next{0};  // ds_fp4 intermediate fp8 round-trip phase
  int64_t p1_total = 0, p2_total = 0, prt_total = 0;
  int n_iblk = 0, n_hblk = 0;
  std::atomic<int> done_count{0};
  std::atomic<int> bar_count{0};
  std::atomic<int> bar_sense{0};

  std::vector<MoeTask*> owned_tasks;  // persistent task descriptors (graph-stable)
  std::vector<int> core_ids;          // worker tid -> logical CPU to pin to (may be empty)

  // ---- Flag-based GPU<->CPU handshake (replaces the per-layer cudaLaunchHostFunc pair) ----
  // A tiny GPU kernel bumps ready_flags[slot] at submit; this coordinator thread busy-polls
  // it, runs the slot's task on the worker pool, and sets done_flags[slot], which a GPU
  // spin-wait kernel polls at sync. This removes the ~2x30-50us host-func dispatch round
  // trips per MoE layer per decode step that otherwise idle the GPU (~6 ms/step on a
  // 75-layer model). One slot per (layer, decode batch size) pair -- the Python side
  // allocates slots as tasks are created. Flags live in mapped-pinned host memory (UVA:
  // the same pointers are used by the GPU kernels and by this thread).
  std::thread coord_thread;
  std::atomic<bool> coord_stop{false};
  volatile int64_t* ready_flags = nullptr;  // GPU increments, this thread polls
  volatile int64_t* done_flags = nullptr;   // this thread sets, GPU spin-waits
  int coord_num_slots = 0;
  std::vector<MoeTask*> flag_task;           // slot -> task (registered lazily)
  std::vector<int64_t> flag_served;          // slot -> completed dispatch count (tests/debug)
  std::mutex flag_task_mtx;

  // Portable ordering for the flag handshake: "ready observed => the DMA'd inputs that
  // preceded the bump are visible" and "y stores are visible before done". Plain
  // volatile loads lean on x86 TSO; acquire/release makes it hold on aarch64 too
  // (GH200/Jetson) at zero x86 cost. (MSVC branch is x86-only today: compiler barrier
  // + TSO.)
  static int64_t flag_load_acquire(const volatile int64_t* p) {
#if defined(_MSC_VER)
    const int64_t v = *p;
    _ReadWriteBarrier();
    return v;
#else
    return __atomic_load_n(const_cast<const int64_t*>(p), __ATOMIC_ACQUIRE);
#endif
  }

  static void flag_store_release(volatile int64_t* p, int64_t v) {
#if defined(_MSC_VER)
    _ReadWriteBarrier();
    *p = v;
#else
    __atomic_store_n(const_cast<int64_t*>(p), v, __ATOMIC_RELEASE);
#endif
  }

  CpuMoeExecutor(int num_threads_, int num_layers_, int num_experts_, int top_k_,
                 int hidden_size, int inter_size, int max_tokens, int activation_id,
                 int apply_router_weight_on_input, int weight_format,
                 uintptr_t gate_up_ptr, uintptr_t down_ptr, uintptr_t gate_up_scale_ptr,
                 uintptr_t gate_up_global_ptr, uintptr_t down_scale_ptr,
                 uintptr_t down_global_ptr, uintptr_t gate_up_bias_ptr,
                 uintptr_t down_bias_ptr, double swiglu_alpha_, double swiglu_limit_,
                 std::vector<int> core_ids_, int ggml_gate_up_type_, int ggml_down_type_)
      : num_threads(num_threads_ > 0 ? num_threads_ : 1),
        num_layers(num_layers_),
        num_experts(num_experts_),
        top_k(top_k_),
        H(hidden_size),
        I(inter_size),
        act(activation_id),
        apply_on_input(apply_router_weight_on_input),
        fmt(weight_format),
        gate_up_tbl(reinterpret_cast<const uint64_t*>(gate_up_ptr)),
        down_tbl(reinterpret_cast<const uint64_t*>(down_ptr)),
        gu_scale_tbl(reinterpret_cast<const uint64_t*>(gate_up_scale_ptr)),
        gu_global_tbl(reinterpret_cast<const uint64_t*>(gate_up_global_ptr)),
        dn_scale_tbl(reinterpret_cast<const uint64_t*>(down_scale_ptr)),
        dn_global_tbl(reinterpret_cast<const uint64_t*>(down_global_ptr)),
        gu_bias_tbl(reinterpret_cast<const uint64_t*>(gate_up_bias_ptr)),
        dn_bias_tbl(reinterpret_cast<const uint64_t*>(down_bias_ptr)),
        swiglu_alpha(static_cast<float>(swiglu_alpha_)),
        swiglu_limit(static_cast<float>(swiglu_limit_)),
        core_ids(std::move(core_ids_)) {
    DotChoice c = select_dot();
    dot = c.fn;
    nvdot = select_nvdot();
    dsdot = select_dsdot();
    mxgemv = select_mxgemv();
    q4dot = select_q4dot();
    q4kdot = select_q4kdot();
    q5_1dot = select_q5_1dot();
    q4kdot_multi = select_q4kdot_multi();
    q5_1dot_multi = select_q5_1dot_multi();
    q8_1_quant = select_q8_1_quant();
#if CPU_MOE_X86
    q8_1_isa = q8_1_quant == quant_q8_1_avx2 ? "avx2" : "scalar";
#endif
    if (weight_format == WF_Q4_0) {
      if (ggml_gate_up_type_ == 2 && ggml_down_type_ == 2) {
        if (H % 32 != 0 || I % 32 != 0)
          throw std::runtime_error("Q4_0 CPU MoE requires H and I to be multiples of 32");
        q4_gu_row_bytes = (H / 32) * 18;
        q4_dn_row_bytes = (I / 32) * 18;
      } else if (ggml_gate_up_type_ == 12 && ggml_down_type_ == 7) {
        if (H % 256 != 0 || I % 32 != 0)
          throw std::runtime_error("Q4_K/Q5_1 CPU MoE requires H%256=0 and I%32=0");
        use_q4k_q5_1 = true;
        q4_gu_row_bytes = (H / 256) * 144;
        q4_dn_row_bytes = (I / 32) * 24;
      } else {
        throw std::runtime_error("CPU GGUF MoE supports Q4_0/Q4_0 or Q4_K/Q5_1");
      }
    }
    isa = c.name;
    // nvfp4 (AVX-VNNI only): W4A8 int8 decode when the CPU supports it. q4_0 is always
    // W4A8 (activations pre-quantized to Q8_0); select_q4dot picks VPDPBUSD / VPMADDUBSW
    // / scalar for the tier, so the tag reflects which of those q4dot resolved to.
    nvi8dot = select_nvi8dot();
    use_vnni = (weight_format == WF_NVFP4) && (nvi8dot != nullptr);
    use_q4a8 = (weight_format == WF_Q4_0) && !use_q4k_q5_1;
    const char* q4tag = use_q4a8 ? (cpu_has_avxvnni() ? "+vnni(q4_0-w4a8)" : "+q4_0-w4a8") : "";
    const char* vnni_tag =
        cpu_has_avx512vnni() ? "+avx512vnni(nvfp4-w4a8)" : "+vnni(nvfp4-w4a8)";
    isa_str = std::string(c.name) + (use_vnni ? vnni_tag : "") + q4tag;
    isa = isa_str.c_str();
    for (int i = 0; i < 16; ++i) e2m1_lut[i] = kE2M1[i];
    for (int i = 0; i < 256; ++i) e4m3_lut[i] = e4m3_decode((uint8_t)i);
    // e8m0 (mxfp4 block scale) = 2^(s-127); the GPU GEMV clamps s to [0,254].
    for (int i = 0; i < 256; ++i) e8m0_lut[i] = std::ldexp(1.0f, std::min(i, 254) - 127);
    g_scratch.assign(static_cast<size_t>(max_tokens) * top_k * I, 0);
    // Row-major fp4 (nvfp4/ds_fp4) pre-deinterleaves activations to fp32 even/odd.
    needs_di = (fmt == WF_NVFP4 || fmt == WF_DSFP4);
    if (needs_di) {
      if (fmt == WF_DSFP4) xq_scratch.assign(static_cast<size_t>(max_tokens) * H, 0);
      xe_scratch.assign(static_cast<size_t>(max_tokens) * (H / 2), 0);
      xo_scratch.assign(static_cast<size_t>(max_tokens) * (H / 2), 0);
      ge_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 2), 0);
      go_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 2), 0);
      if (use_vnni) {
        xi8_scratch.assign(static_cast<size_t>(max_tokens) * H, 0);
        xas_scratch.assign(static_cast<size_t>(max_tokens) * (H / 16), 0);
        gi8_scratch.assign(static_cast<size_t>(max_tokens) * top_k * I, 0);
        gas_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 16), 0);
      }
    }
    // q4_0 W4A8: per-32-block Q8_0 activations (int8 + fp32 scale) for input + intermediate.
    if (use_q4a8) {
      xi8_scratch.assign(static_cast<size_t>(max_tokens) * H, 0);
      xas_scratch.assign(static_cast<size_t>(max_tokens) * (H / 32), 0);
      gi8_scratch.assign(static_cast<size_t>(max_tokens) * top_k * I, 0);
      gas_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 32), 0);
    }
    if (use_q4k_q5_1) {
      xq8k_scratch.resize(static_cast<size_t>(max_tokens) * (H / 256));
      gq8_1_scratch.resize(static_cast<size_t>(max_tokens) * top_k * (I / 32));
    }
    for (int t = 0; t < num_threads; ++t)
      workers.emplace_back([this, t] { worker_loop(t); });
  }

  // Quantize a bf16 activation row to Q8_0 (llama.cpp): per-32-block symmetric int8 in
  // natural order + one fp32 scale (amax/127) per block. Done once per token/route,
  // amortized over every output row that the q4_0 W4A8 GEMV reads. K % 32 == 0.
  void quant_q8_0(const bf16_t* x, int K, int8_t* aq, float* asb) {
    const int nb = K / 32;
    for (int b = 0; b < nb; ++b) {
      const bf16_t* xb = x + (size_t)b * 32;
      float xf[32], amax = 0.0f;
      for (int j = 0; j < 32; ++j) {
        xf[j] = bf16_to_f32(xb[j]);
        amax = std::max(amax, std::fabs(xf[j]));
      }
      const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
      asb[b] = d;
      const float inv = amax > 0.0f ? 1.0f / d : 0.0f;
      int8_t* o = aq + (size_t)b * 32;
      for (int j = 0; j < 32; ++j)
        o[j] = (int8_t)std::max(-127, std::min(127, (int)std::lround(xf[j] * inv)));
    }
  }

  void quant_q8_k(const bf16_t* x, int K, Q8KBlock* out) {
    for (int b = 0; b < K / 256; ++b) {
      Q8KBlock& block = out[b];
      const bf16_t* xb = x + (size_t)b * 256;
      float maxv = 0.0f;
      for (int j = 0; j < 256; ++j) {
        const float v = bf16_to_f32(xb[j]);
        if (std::fabs(v) > std::fabs(maxv)) maxv = v;
      }
      if (maxv == 0.0f) {
        block.d = 0.0f;
        std::memset(block.qs, 0, sizeof(block.qs));
        std::memset(block.bsums, 0, sizeof(block.bsums));
        continue;
      }
      const float inv = -127.0f / maxv;
      block.d = 1.0f / inv;
      for (int g = 0; g < 16; ++g) {
        int sum = 0;
        for (int j = 0; j < 16; ++j) {
          const int q = std::max(-127, std::min(127, (int)std::lround(
              bf16_to_f32(xb[g * 16 + j]) * inv)));
          block.qs[g * 16 + j] = (int8_t)q;
          sum += q;
        }
        block.bsums[g] = (int16_t)sum;
      }
    }
  }

  void quant_q8_1(const bf16_t* x, int K, Q8_1Block* out) {
    q8_1_quant(x, K, out);
  }

  const char* q8_1_quant_name() const { return q8_1_isa; }

  // Quantize the pre-deinterleaved fp32 even/odd activations to per-16-block int8 in the
  // [even(8),odd(8)] layout the VNNI dot expects. Done once per token/route (amortized over
  // every output row), so a scalar pass is fine relative to the GEMV.
  void quant_i8_pg16(const float* xe, const float* xo, int K, int8_t* asi8, float* asb) {
    const int nb = K / 16;
    for (int b = 0; b < nb; ++b) {
      const float* xeb = xe + (size_t)b * 8;
      const float* xob = xo + (size_t)b * 8;
      float amax = 0.0f;
      for (int j = 0; j < 8; ++j)
        amax = std::max(amax, std::max(std::fabs(xeb[j]), std::fabs(xob[j])));
      const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
      asb[b] = s;
      const float inv = 1.0f / s;
      int8_t* ae = asi8 + (size_t)b * 16;
      for (int j = 0; j < 8; ++j) {
        int qe = (int)std::lround(xeb[j] * inv), qo = (int)std::lround(xob[j] * inv);
        ae[j] = (int8_t)std::max(-127, std::min(127, qe));
        ae[8 + j] = (int8_t)std::max(-127, std::min(127, qo));
      }
    }
  }

  // gate_up output row `row` (in [0, 2I)) dotted with activation over K = H. ``e`` is
  // the layer-local expert row (0..num_experts); the layer bases (already resolved
  // once per task/pass by the caller via tbl_at) pick the layer's own tensors.
  // bf16 uses the interleaved bf16 row; nvfp4 uses the pre-split fp32 even/odd halves
  // (or, with AVX-VNNI, the per-16-block int8 activations).
  inline float gemm1_dot(const bf16_t* gate_up_l, const uint8_t* gu_packed_l,
                         const uint8_t* gu_scale_l, const uint16_t* gu_global_l, int e, int row,
                         const bf16_t* x, const float* xe, const float* xo, const int8_t* xi8,
                         const float* xas, const Q8KBlock* xq8k) {
    if (fmt == WF_BF16) {
      const bf16_t* w = gate_up_l + ((size_t)e * (2 * I) + row) * H;
      return dot(w, x, H);
    }
    if (fmt == WF_Q4_0) {
      const uint8_t* w =
          gu_packed_l + ((size_t)e * (2 * I) + row) * (size_t)q4_gu_row_bytes;
      if (use_q4k_q5_1) return q4kdot(w, xq8k, H);
      return q4dot(w, xi8, xas, H);  // W4A8: int8 activations (Q8_0), scale in xas
    }
    const size_t r = (size_t)e * (2 * I) + row;
    if (use_vnni)
      return nvi8dot(gu_packed_l + r * (size_t)(H / 2), gu_scale_l + r * (size_t)(H / 16),
                     fp16_to_f32(gu_global_l[r]), xi8, H, e4m3_lut, xas);
    return nvdot(gu_packed_l + r * (size_t)(H / 2), gu_scale_l + r * (size_t)(H / 16),
                 fp16_to_f32(gu_global_l[r]), xe, xo, H, e2m1_lut, e4m3_lut);
  }

  // down output row `row` (in [0, H)) dotted with the intermediate over K = I. Same
  // layer-local-base convention as gemm1_dot.
  inline float gemm2_dot(const bf16_t* down_l, const uint8_t* dn_packed_l,
                         const uint8_t* dn_scale_l, const uint16_t* dn_global_l, int e, int row,
                         const bf16_t* g, const float* ge, const float* go, const int8_t* gi8,
                         const float* gas, const Q8_1Block* gq8_1) {
    if (fmt == WF_BF16) {
      const bf16_t* w = down_l + ((size_t)e * H + row) * I;
      return dot(w, g, I);
    }
    if (fmt == WF_Q4_0) {
      const uint8_t* w = dn_packed_l + ((size_t)e * H + row) * (size_t)q4_dn_row_bytes;
      if (use_q4k_q5_1) return q5_1dot(w, gq8_1, I);
      return q4dot(w, gi8, gas, I);  // W4A8: int8 activations (Q8_0), scale in gas
    }
    const size_t r = (size_t)e * H + row;
    if (use_vnni)
      return nvi8dot(dn_packed_l + r * (size_t)(I / 2), dn_scale_l + r * (size_t)(I / 16),
                     fp16_to_f32(dn_global_l[r]), gi8, I, e4m3_lut, gas);
    return nvdot(dn_packed_l + r * (size_t)(I / 2), dn_scale_l + r * (size_t)(I / 16),
                 fp16_to_f32(dn_global_l[r]), ge, go, I, e2m1_lut, e4m3_lut);
  }

  void pin_self(int tid) {
#if CPU_MOE_HAS_AFFINITY
    if (core_ids.empty()) return;
    const int cpu = core_ids[tid % static_cast<int>(core_ids.size())];
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)tid;
#endif
  }

  ~CpuMoeExecutor() {
    coord_stop.store(true);
    if (coord_thread.joinable()) coord_thread.join();
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      stop = true;
    }
    task_cv.notify_all();
    for (auto& th : workers)
      if (th.joinable()) th.join();
    for (MoeTask* t : owned_tasks) delete t;
  }

  uintptr_t create_task(int layer_id, int num_tokens, uintptr_t x_ptr,
                        uintptr_t ids_ptr, uintptr_t w_ptr, uintptr_t y_ptr) {
    MoeTask* t = new MoeTask{this,
                             layer_id,
                             num_tokens,
                             reinterpret_cast<const bf16_t*>(x_ptr),
                             reinterpret_cast<const int32_t*>(ids_ptr),
                             reinterpret_cast<const float*>(w_ptr),
                             reinterpret_cast<bf16_t*>(y_ptr)};
    owned_tasks.push_back(t);
    return reinterpret_cast<uintptr_t>(t);
  }

  const char* isa_name() const { return isa; }

  void barrier(int& local_sense) {
    local_sense ^= 1;
    if (bar_count.fetch_add(1) + 1 == num_threads) {
      bar_count.store(0);
      bar_sense.store(local_sense);
    } else {
      while (bar_sense.load() != local_sense) {
#if CPU_MOE_X86
        _mm_pause();
#endif
      }
    }
  }

  void do_pass1(const MoeTask* t, int64_t p) {
    if (fmt == WF_MXFP4) {
      do_pass1_mxfp4(t, p);
      return;
    }
    if (fmt == WF_DSFP4) {
      do_pass1_dsfp4(t, p);
      return;
    }
    const int64_t ib = p % n_iblk;
    const int64_t tk = p / n_iblk;
    const int k = static_cast<int>(tk % top_k);
    const int tok = static_cast<int>(tk / top_k);
    const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
    if (e < 0 || e >= num_experts) return;
    const float w_in = apply_on_input ? t->w[static_cast<size_t>(tok) * top_k + k] : 1.0f;
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const bf16_t* gate_up_l = reinterpret_cast<const bf16_t*>(tbl_at(gate_up_tbl, t->layer_id));
    const uint8_t* gu_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(gate_up_tbl, t->layer_id));
    const uint8_t* gu_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(gu_scale_tbl, t->layer_id));
    const uint16_t* gu_global_l =
        reinterpret_cast<const uint16_t*>(tbl_at(gu_global_tbl, t->layer_id));
    const bf16_t* x_row = t->x + (size_t)tok * H;
    const float* xe = needs_di ? xe_scratch.data() + (size_t)tok * (H / 2) : nullptr;
    const float* xo = needs_di ? xo_scratch.data() + (size_t)tok * (H / 2) : nullptr;
    const int8_t* xi8 =
        (use_vnni || use_q4a8) ? xi8_scratch.data() + (size_t)tok * H : nullptr;
    const float* xas = use_vnni ? xas_scratch.data() + (size_t)tok * (H / 16)
                     : use_q4a8 ? xas_scratch.data() + (size_t)tok * (H / 32)
                                  : nullptr;
    const Q8KBlock* xq8k = use_q4k_q5_1
                               ? xq8k_scratch.data() + (size_t)tok * (H / 256)
                               : nullptr;
    bf16_t* g_row = g_scratch.data() + ((size_t)tok * top_k + k) * I;
    const int i0 = static_cast<int>(ib) * IBLK;
    const int i1 = std::min(I, i0 + IBLK);
    const bool clamped = act == ACT_SWIGLUOAI || act == ACT_SWIGLU_CLAMP;
    const float up_bias = act == ACT_SWIGLUOAI ? 1.0f : 0.0f;
    const float lim = swiglu_limit, alpha = swiglu_alpha;
    for (int i = i0; i < i1; ++i) {
      // gate = row i, up = row I+i
      float gate =
          gemm1_dot(gate_up_l, gu_packed_l, gu_scale_l, gu_global_l, e, i, x_row, xe, xo, xi8,
                    xas, xq8k) * w_in;
      float up = gemm1_dot(gate_up_l, gu_packed_l, gu_scale_l, gu_global_l, e, I + i, x_row,
                           xe, xo, xi8, xas, xq8k) * w_in;
      if (clamped) {
        // clamp(gate, max=lim) * sigmoid(alpha * gate) * (clamp(up, +-lim) + up_bias)
        // -- swigluoai carries the +1 up bias (gpt-oss/MiniMax); swiglu_clamp
        // (GLM-5.3) does not. lim == +inf: no clamp.
        if (gate > lim) gate = lim;
        if (up > lim) up = lim;
        else if (up < -lim) up = -lim;
        const float glu = gate / (1.0f + std::exp(-gate * alpha));
        g_row[i] = f32_to_bf16(glu * (up + up_bias));
      } else {
        g_row[i] = f32_to_bf16(act_apply(act, gate) * up);
      }
    }
    if (use_q4k_q5_1) {
      // Quantize the worker-owned 32-value tile here, avoiding a second queue
      // and barrier before the Q5_1 down GEMV.
      quant_q8_1(g_row + i0, i1 - i0,
                 gq8_1_scratch.data() +
                     ((size_t)tok * top_k + k) * (I / 32) + ib);
    }
  }

  void do_pass2(const MoeTask* t, int64_t p) {
    if (fmt == WF_MXFP4) {
      do_pass2_mxfp4(t, p);
      return;
    }
    if (fmt == WF_DSFP4) {
      do_pass2_dsfp4(t, p);
      return;
    }
    const int64_t hb = p % n_hblk;
    const int tok = static_cast<int>(p / n_hblk);
    const int h0 = static_cast<int>(hb) * HBLK;
    const int h1 = std::min(H, h0 + HBLK);
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const bf16_t* down_l = reinterpret_cast<const bf16_t*>(tbl_at(down_tbl, t->layer_id));
    const uint8_t* dn_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(down_tbl, t->layer_id));
    const uint8_t* dn_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(dn_scale_tbl, t->layer_id));
    const uint16_t* dn_global_l =
        reinterpret_cast<const uint16_t*>(tbl_at(dn_global_tbl, t->layer_id));
    bf16_t* y_row = t->y + (size_t)tok * H;
    float acc[HBLK];
    for (int c = 0; c < h1 - h0; ++c) acc[c] = 0.0f;
    for (int k = 0; k < top_k; ++k) {
      const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
      if (e < 0 || e >= num_experts) continue;
      const float w_out = apply_on_input ? 1.0f : t->w[static_cast<size_t>(tok) * top_k + k];
      const size_t gr = (size_t)tok * top_k + k;
      const bf16_t* g_row = g_scratch.data() + gr * I;
      const float* ge = needs_di ? ge_scratch.data() + gr * (I / 2) : nullptr;
      const float* go = needs_di ? go_scratch.data() + gr * (I / 2) : nullptr;
      const int8_t* gi8 = (use_vnni || use_q4a8) ? gi8_scratch.data() + gr * I : nullptr;
      const float* gas = use_vnni ? gas_scratch.data() + gr * (I / 16)
                       : use_q4a8 ? gas_scratch.data() + gr * (I / 32)
                                    : nullptr;
      const Q8_1Block* gq8_1 = use_q4k_q5_1
                                  ? gq8_1_scratch.data() + gr * (I / 32)
                                  : nullptr;
      for (int h = h0; h < h1; ++h) {
        acc[h - h0] +=
            gemm2_dot(down_l, dn_packed_l, dn_scale_l, dn_global_l, e, h, g_row, ge, go, gi8,
                      gas, gq8_1) * w_out;
      }
    }
    for (int c = 0; c < h1 - h0; ++c) {
      y_row[h0 + c] = f32_to_bf16(acc[c]);
    }
  }

  // Batched Q4_K/Q5_1 pass 1. One work item owns an expert group and an I tile;
  // walking rows first keeps the group's repeated expert rows hot while each route
  // still receives the same per-route intermediate and Q8_1 scratch as the normal path.
  void do_pass1_grouped(const MoeTask* t, int64_t p) {
    const int64_t ib = p % n_iblk;
    const int group = static_cast<int>(p / n_iblk);
    const int e = grouped_experts[group];
    const int i0 = static_cast<int>(ib) * IBLK;
    const int i1 = std::min(I, i0 + IBLK);
    const int64_t r0 = grouped_offsets[group], r1 = grouped_offsets[group + 1];
    const uint8_t* gu_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(gate_up_tbl, t->layer_id));
    const bool clamped = act == ACT_SWIGLUOAI || act == ACT_SWIGLU_CLAMP;
    const float up_bias = act == ACT_SWIGLUOAI ? 1.0f : 0.0f;
    const float lim = swiglu_limit, alpha = swiglu_alpha;
    for (int i = i0; i < i1; ++i) {
      for (int64_t ri = r0; ri < r1;) {
        const int width = static_cast<int>(std::min<int64_t>(4, r1 - ri));
        const Q8KBlock* xq8k[4];
        float gates[4], ups[4];
        for (int r = 0; r < width; ++r) {
          const int64_t route = grouped_routes[ri + r];
          const int tok = static_cast<int>(route / top_k);
          xq8k[r] = xq8k_scratch.data() + (size_t)tok * (H / 256);
        }
        const uint8_t* gate_w = gu_packed_l +
            ((size_t)e * (2 * I) + i) * (size_t)q4_gu_row_bytes;
        const uint8_t* up_w = gu_packed_l +
            ((size_t)e * (2 * I) + I + i) * (size_t)q4_gu_row_bytes;
        if (width == 1) {
          gates[0] = q4kdot(gate_w, xq8k[0], H);
          ups[0] = q4kdot(up_w, xq8k[0], H);
        } else {
          q4kdot_multi(gate_w, xq8k, width, H, gates);
          q4kdot_multi(up_w, xq8k, width, H, ups);
        }
        for (int r = 0; r < width; ++r) {
          const int64_t route = grouped_routes[ri + r];
          const float w_in = apply_on_input ? t->w[route] : 1.0f;
          bf16_t* g_row = g_scratch.data() + (size_t)route * I;
          float gate = gates[r] * w_in;
          float up = ups[r] * w_in;
          if (clamped) {
            if (gate > lim) gate = lim;
            if (up > lim) up = lim;
            else if (up < -lim) up = -lim;
            const float glu = gate / (1.0f + std::exp(-gate * alpha));
            g_row[i] = f32_to_bf16(glu * (up + up_bias));
          } else {
            g_row[i] = f32_to_bf16(act_apply(act, gate) * up);
          }
        }
        ri += width;
      }
    }
    for (int64_t ri = r0; ri < r1; ++ri) {
      const int64_t route = grouped_routes[ri];
      quant_q8_1(g_scratch.data() + (size_t)route * I + i0, i1 - i0,
                 gq8_1_scratch.data() + (size_t)route * (I / 32) + ib);
    }
  }

  // Grouped pass 2 assigns each H tile to one worker. It computes all grouped
  // routes into disjoint route_out slices, then reduces those slices in the original
  // token/k order before writing y. Thus no output races or second barrier are needed.
  void do_pass2_grouped(const MoeTask* t, int64_t p) {
    const int64_t hb = p % n_hblk;
    const int h0 = static_cast<int>(hb) * HBLK;
    const int h1 = std::min(H, h0 + HBLK);
    const uint8_t* dn_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(down_tbl, t->layer_id));
    for (size_t gi = 0; gi < grouped_experts.size(); ++gi) {
      const int e = grouped_experts[gi];
      for (int h = h0; h < h1; ++h) {
        const int64_t r0 = grouped_offsets[gi], r1 = grouped_offsets[gi + 1];
        for (int64_t ri = r0; ri < r1;) {
          const int width = static_cast<int>(std::min<int64_t>(4, r1 - ri));
          const Q8_1Block* gq8_1[4];
          float values[4];
          for (int r = 0; r < width; ++r) {
            const int64_t route = grouped_routes[ri + r];
            gq8_1[r] = gq8_1_scratch.data() + (size_t)route * (I / 32);
          }
          const uint8_t* down_w = dn_packed_l +
              ((size_t)e * H + h) * (size_t)q4_dn_row_bytes;
          if (width == 1) values[0] = q5_1dot(down_w, gq8_1[0], I);
          else q5_1dot_multi(down_w, gq8_1, width, I, values);
          for (int r = 0; r < width; ++r) {
            const int64_t route = grouped_routes[ri + r];
            const float w_out = apply_on_input ? 1.0f : t->w[route];
            route_out.data()[(size_t)route * H + h] = values[r] * w_out;
          }
          ri += width;
        }
      }
    }
    for (int tok = 0; tok < t->num_tokens; ++tok) {
      bf16_t* y_row = t->y + (size_t)tok * H;
      for (int h = h0; h < h1; ++h) {
        float acc = 0.0f;
        for (int k = 0; k < top_k; ++k) {
          const int64_t route = (size_t)tok * top_k + k;
          const int e = t->ids[route];
          if (e >= 0 && e < num_experts) acc += route_out.data()[(size_t)route * H + h];
        }
        y_row[h] = f32_to_bf16(acc);
      }
    }
  }

  // ----------------------------- mxfp4 (gpt-oss) -----------------------------
  // Transposed split-K layout (N innermost), so the GEMV streams K and accumulates
  // a contiguous block of N output columns -> cache-line-efficient, no repack, no
  // extra host memory. Pass 1 fuses gate_up + clamped-swiglu(+bias); pass 2 fuses
  // down(+bias) * router-weight, summed over the token's routes.
  //
  // Dequant: w = E2M1[code] * 2^(e8m0_scale - 127); two codes per byte (low nibble
  // first), one e8m0 scale per 32 contiguous K. Matches kernel/triton/mxfp4_moe.py.

  void do_pass1_mxfp4(const MoeTask* t, int64_t p) {
    const int64_t ib = p % n_iblk;
    const int64_t tk = p / n_iblk;
    const int k = static_cast<int>(tk % top_k);
    const int tok = static_cast<int>(tk / top_k);
    const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
    if (e < 0 || e >= num_experts) return;
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const uint8_t* gu_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(gate_up_tbl, t->layer_id));
    const uint8_t* gu_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(gu_scale_tbl, t->layer_id));
    const bf16_t* gu_bias_l = reinterpret_cast<const bf16_t*>(tbl_at(gu_bias_tbl, t->layer_id));
    const int N2 = 2 * I;            // gate_up output width (gate/up interleaved)
    const int Hh = H / 2;            // packed-K rows
    const int i0 = static_cast<int>(ib) * IBLK;
    const int i1 = std::min(I, i0 + IBLK);
    const int nunit = i1 - i0;       // intermediate units owned by this tile
    const int col0 = 2 * i0;         // first gate_up column
    const int ncol = 2 * nunit;      // gate_up columns owned by this tile
    const bf16_t* x_row = t->x + (size_t)tok * H;
    const uint8_t* blk_e = gu_packed_l + (size_t)e * Hh * N2;
    const uint8_t* scl_e = gu_scale_l + (size_t)e * (size_t)(H / 32) * N2;
    float gu[2 * IBLK];
    mxgemv(gu, blk_e + col0, scl_e + col0, x_row, Hh, N2, ncol, e2m1_lut, e8m0_lut);
    const bf16_t* bias_e = gu_bias_l + (size_t)e * N2 + col0;
    bf16_t* g_row = g_scratch.data() + ((size_t)tok * top_k + k) * I;
    const float lim = swiglu_limit, alpha = swiglu_alpha;
    for (int j = 0; j < nunit; ++j) {
      float gate = gu[2 * j] + bf16_to_f32(bias_e[2 * j]);
      float up = gu[2 * j + 1] + bf16_to_f32(bias_e[2 * j + 1]);
      if (gate > lim) gate = lim;
      if (up > lim) up = lim;
      else if (up < -lim) up = -lim;
      const float glu = gate / (1.0f + std::exp(-gate * alpha));  // gate * sigmoid(alpha*gate)
      g_row[i0 + j] = f32_to_bf16(glu * (up + 1.0f));
    }
  }

  void do_pass2_mxfp4(const MoeTask* t, int64_t p) {
    const int64_t hb = p % n_hblk;
    const int tok = static_cast<int>(p / n_hblk);
    const int Ih = I / 2;            // packed-K rows
    const int h0 = static_cast<int>(hb) * HBLK;
    const int h1 = std::min(H, h0 + HBLK);
    const int nh = h1 - h0;
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const uint8_t* dn_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(down_tbl, t->layer_id));
    const uint8_t* dn_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(dn_scale_tbl, t->layer_id));
    const bf16_t* dn_bias_l = reinterpret_cast<const bf16_t*>(tbl_at(dn_bias_tbl, t->layer_id));
    float acc[HBLK];
    for (int c = 0; c < nh; ++c) acc[c] = 0.0f;
    for (int k = 0; k < top_k; ++k) {
      const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
      if (e < 0 || e >= num_experts) continue;
      const float wt = t->w[static_cast<size_t>(tok) * top_k + k];
      const bf16_t* g_row = g_scratch.data() + ((size_t)tok * top_k + k) * I;
      const uint8_t* blk_e = dn_packed_l + (size_t)e * Ih * H;
      const uint8_t* scl_e = dn_scale_l + (size_t)e * (size_t)(I / 32) * H;
      float part[HBLK];
      mxgemv(part, blk_e + h0, scl_e + h0, g_row, Ih, H, nh, e2m1_lut, e8m0_lut);
      const bf16_t* bias_e = dn_bias_l + (size_t)e * H + h0;
      for (int c = 0; c < nh; ++c) acc[c] += (part[c] + bf16_to_f32(bias_e[c])) * wt;
    }
    bf16_t* y_row = t->y + (size_t)tok * H;
    for (int c = 0; c < nh; ++c) y_row[h0 + c] = f32_to_bf16(acc[c]);
  }

  // ----------------------------- ds_fp4 (DSV4) -------------------------------
  // Row-major e2m1 + e8m0/32 (no global); silu-swiglu with clamp; FP8-roundtripped
  // activations (x once in submit -> xq_scratch; the intermediate g in a dedicated
  // round-trip phase between the two passes). Router weight applies on the down output.

  void do_pass1_dsfp4(const MoeTask* t, int64_t p) {
    const int64_t ib = p % n_iblk;
    const int64_t tk = p / n_iblk;
    const int k = static_cast<int>(tk % top_k);
    const int tok = static_cast<int>(tk / top_k);
    const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
    if (e < 0 || e >= num_experts) return;
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const uint8_t* gu_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(gate_up_tbl, t->layer_id));
    const uint8_t* gu_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(gu_scale_tbl, t->layer_id));
    const int N2 = 2 * I, Hh = H / 2, Hs = H / 32;
    // fp8-roundtripped input, pre-deinterleaved to fp32 even/odd halves.
    const float* xe = xe_scratch.data() + (size_t)tok * (H / 2);
    const float* xo = xo_scratch.data() + (size_t)tok * (H / 2);
    const uint8_t* gp = gu_packed_l + (size_t)e * N2 * Hh;
    const uint8_t* gs = gu_scale_l + (size_t)e * N2 * Hs;
    bf16_t* g_row = g_scratch.data() + ((size_t)tok * top_k + k) * I;
    const int i0 = static_cast<int>(ib) * IBLK;
    const int i1 = std::min(I, i0 + IBLK);
    const float lim = swiglu_limit;
    for (int i = i0; i < i1; ++i) {
      // gate_up is stored bf16 by the reference GEMV before swiglu; round to match.
      float gate = bf16_to_f32(f32_to_bf16(
          dsdot(gp + (size_t)i * Hh, gs + (size_t)i * Hs, xe, xo, H, e2m1_lut, e8m0_lut)));
      float up = bf16_to_f32(f32_to_bf16(dsdot(
          gp + (size_t)(I + i) * Hh, gs + (size_t)(I + i) * Hs, xe, xo, H, e2m1_lut, e8m0_lut)));
      if (lim > 0.0f) {
        if (gate > lim) gate = lim;
        if (up > lim) up = lim;
        else if (up < -lim) up = -lim;
      }
      const float glu = gate / (1.0f + std::exp(-gate));  // silu(gate)
      g_row[i] = f32_to_bf16(glu * up);
    }
  }

  // Prepare one intermediate row (token,route) for the down GEMV: ds_fp4 first FP8
  // round-trips it (DSV4 act_quant), then both formats deinterleave to fp32 even/odd
  // (reused across every down output row).
  void prep_g_row(int64_t r) {
    bf16_t* g = g_scratch.data() + (size_t)r * I;
    if (use_q4k_q5_1) {
      quant_q8_1(g, I, gq8_1_scratch.data() + (size_t)r * (I / 32));
      return;
    }
    if (use_q4a8) {  // q4_0 W4A8: Q8_0-quantize the intermediate row for the down GEMV.
      quant_q8_0(g, I, gi8_scratch.data() + (size_t)r * I,
                 gas_scratch.data() + (size_t)r * (I / 32));
      return;
    }
    if (fmt == WF_DSFP4) fp8_roundtrip_bf16(g, g, I);
    float* ge = ge_scratch.data() + (size_t)r * (I / 2);
    float* go = go_scratch.data() + (size_t)r * (I / 2);
    deinterleave_bf16_f32(g, ge, go, I);
    if (use_vnni)
      quant_i8_pg16(ge, go, I, gi8_scratch.data() + (size_t)r * I,
                    gas_scratch.data() + (size_t)r * (I / 16));
  }

  void do_pass2_dsfp4(const MoeTask* t, int64_t p) {
    const int64_t hb = p % n_hblk;
    const int tok = static_cast<int>(p / n_hblk);
    const int Ih = I / 2, Is = I / 32;
    const int h0 = static_cast<int>(hb) * HBLK;
    const int h1 = std::min(H, h0 + HBLK);
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const uint8_t* dn_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(down_tbl, t->layer_id));
    const uint8_t* dn_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(dn_scale_tbl, t->layer_id));
    bf16_t* y_row = t->y + (size_t)tok * H;
    for (int h = h0; h < h1; ++h) {
      float acc = 0.0f;
      for (int k = 0; k < top_k; ++k) {
        const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
        if (e < 0 || e >= num_experts) continue;
        const float wt = t->w[static_cast<size_t>(tok) * top_k + k];
        const float* ge = ge_scratch.data() + ((size_t)tok * top_k + k) * (I / 2);
        const float* go = go_scratch.data() + ((size_t)tok * top_k + k) * (I / 2);
        const uint8_t* dp = dn_packed_l + (size_t)e * (size_t)H * Ih + (size_t)h * Ih;
        const uint8_t* ds = dn_scale_l + (size_t)e * (size_t)H * Is + (size_t)h * Is;
        // The reference rounds each route's weighted down output to bf16 before the
        // fp32 sum over routes (down [T, top_k, H] bf16 -> .sum(dim=1)).
        acc += bf16_to_f32(f32_to_bf16(dsdot(dp, ds, ge, go, I, e2m1_lut, e8m0_lut) * wt));
      }
      y_row[h] = f32_to_bf16(acc);
    }
  }

  void run_task_body(const MoeTask* t) {
    int local_sense = 0;
    for (;;) {
      int64_t p = p1_next.fetch_add(1, std::memory_order_relaxed);
      if (p >= p1_total) break;
      if (grouped_q4k_q5_1) do_pass1_grouped(t, p);
      else do_pass1(t, p);
    }
    barrier(local_sense);
    // Row-major fp4 and q4_0 W4A8 prepare intermediate rows before the down GEMV.
    // The Q4_K/Q5_1 path quantizes each Q8_1 tile in do_pass1, so it needs no
    // second work queue or barrier here.
    if (needs_di || use_q4a8) {
      for (;;) {
        int64_t r = prt_next.fetch_add(1, std::memory_order_relaxed);
        if (r >= prt_total) break;
        prep_g_row(r);
      }
      barrier(local_sense);
    }
    for (;;) {
      int64_t p = p2_next.fetch_add(1, std::memory_order_relaxed);
      if (p >= p2_total) break;
      if (grouped_q4k_q5_1) do_pass2_grouped(t, p);
      else do_pass2(t, p);
    }
  }

  void worker_loop(int tid) {
    pin_self(tid);
    uint64_t my_gen = 0;
    for (;;) {
      MoeTask* t;
      {
        std::unique_lock<std::mutex> lk(task_mtx);
        task_cv.wait(lk, [&] { return stop || cur_gen != my_gen; });
        if (stop) return;
        my_gen = cur_gen;
        t = cur_task;
      }
      run_task_body(t);
      if (done_count.fetch_add(1) + 1 == num_threads) {
        completed.store(my_gen, std::memory_order_release);
        {
          std::lock_guard<std::mutex> lk(sync_mtx);
        }
        sync_cv.notify_all();
      }
    }
  }

  void submit(MoeTask* t) {
    n_iblk = (I + IBLK - 1) / IBLK;
    n_hblk = (H + HBLK - 1) / HBLK;
    // Grow the per-token intermediate scratch if a larger batch shows up than the
    // construction-time hint (CUDA-graph capture warms the largest bs first, so
    // this happens at most once, before any capture, while the pool is idle).
    const size_t need = static_cast<size_t>(t->num_tokens) * top_k * I;
    if (need > g_scratch.size()) g_scratch.resize(need);
    grouped_q4k_q5_1 = false;
    grouped_experts.clear();
    grouped_offsets.clear();
    grouped_routes.clear();
    if (use_q4k_q5_1 && t->num_tokens > 1) {
      std::vector<int> group_for_expert(static_cast<size_t>(num_experts), -1);
      std::vector<int64_t> group_counts;
      const size_t routes = static_cast<size_t>(t->num_tokens) * top_k;
      for (size_t route = 0; route < routes; ++route) {
        const int e = t->ids[route];
        if (e < 0 || e >= num_experts) continue;
        int group = group_for_expert[e];
        if (group < 0) {
          group = static_cast<int>(grouped_experts.size());
          group_for_expert[e] = group;
          grouped_experts.push_back(e);
          group_counts.push_back(0);
        }
        ++group_counts[group];
      }
      grouped_offsets.resize(grouped_experts.size() + 1);
      for (size_t group = 0; group < grouped_experts.size(); ++group)
        grouped_offsets[group + 1] = grouped_offsets[group] + group_counts[group];
      grouped_routes.resize(static_cast<size_t>(grouped_offsets.back()));
      std::vector<int64_t> group_cursor(grouped_offsets.begin(), grouped_offsets.end() - 1);
      for (size_t route = 0; route < routes; ++route) {
        const int e = t->ids[route];
        if (e < 0 || e >= num_experts) continue;
        const int group = group_for_expert[e];
        grouped_routes[static_cast<size_t>(group_cursor[group]++)] =
            static_cast<int64_t>(route);
      }
      bool duplicate = false;
      for (size_t group = 0; group < grouped_experts.size(); ++group) {
        if (grouped_offsets[group + 1] - grouped_offsets[group] > 1) {
          duplicate = true;
          break;
        }
      }
      if (duplicate) {
        grouped_q4k_q5_1 = true;
        route_out.resize(routes * static_cast<size_t>(H));
      } else {
        grouped_experts.clear();
        grouped_offsets.clear();
        grouped_routes.clear();
      }
    }
    p1_total = grouped_q4k_q5_1
                   ? static_cast<int64_t>(grouped_experts.size()) * n_iblk
                   : static_cast<int64_t>(t->num_tokens) * top_k * n_iblk;
    p2_total = grouped_q4k_q5_1 ? static_cast<int64_t>(n_hblk)
                                : static_cast<int64_t>(t->num_tokens) * n_hblk;
    prt_total = (needs_di || use_q4a8)
                    ? static_cast<int64_t>(t->num_tokens) * top_k
                    : 0;
    p1_next.store(0, std::memory_order_relaxed);
    p2_next.store(0, std::memory_order_relaxed);
    prt_next.store(0, std::memory_order_relaxed);
    done_count.store(0, std::memory_order_relaxed);
    bar_count.store(0, std::memory_order_relaxed);
    bar_sense.store(0, std::memory_order_relaxed);
    // ds_fp4: FP8 round-trip the per-token input once, up front (single-threaded;
    // tiny for decode, and done before the workers are woken below).
    if (needs_di) {
      const size_t xn = static_cast<size_t>(t->num_tokens) * H;
      if (xn / 2 > xe_scratch.size()) {
        xe_scratch.resize(xn / 2);
        xo_scratch.resize(xn / 2);
        ge_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * (I / 2));
        go_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * (I / 2));
        if (use_vnni) {
          xi8_scratch.resize(xn);
          xas_scratch.resize(xn / 16);
          gi8_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * I);
          gas_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * (I / 16));
        }
      }
      const bool ds = (fmt == WF_DSFP4) && !input_prequant;
      if (ds && xn > xq_scratch.size()) xq_scratch.resize(xn);
      for (int tok = 0; tok < t->num_tokens; ++tok) {
        const bf16_t* src = t->x + (size_t)tok * H;
        if (ds) {  // DSV4 FP8-round-trips the input before the gate_up GEMV
          bf16_t* xq = xq_scratch.data() + (size_t)tok * H;
          fp8_roundtrip_bf16(src, xq, H);
          src = xq;
        }
        float* xe = xe_scratch.data() + (size_t)tok * (H / 2);
        float* xo = xo_scratch.data() + (size_t)tok * (H / 2);
        deinterleave_bf16_f32(src, xe, xo, H);
        if (use_vnni)
          quant_i8_pg16(xe, xo, H, xi8_scratch.data() + (size_t)tok * H,
                        xas_scratch.data() + (size_t)tok * (H / 16));
      }
    }
    // q4_0 W4A8: Q8_0-quantize the per-token input once (single-threaded, tiny for decode).
    if (use_q4a8) {
      const size_t xn = static_cast<size_t>(t->num_tokens) * H;
      if (xn > xi8_scratch.size()) {
        xi8_scratch.resize(xn);
        xas_scratch.resize(xn / 32);
        gi8_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * I);
        gas_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * (I / 32));
      }
      for (int tok = 0; tok < t->num_tokens; ++tok)
        quant_q8_0(t->x + (size_t)tok * H, H, xi8_scratch.data() + (size_t)tok * H,
                   xas_scratch.data() + (size_t)tok * (H / 32));
    }
    if (use_q4k_q5_1) {
      const size_t nblocks = static_cast<size_t>(t->num_tokens) * (H / 256);
      if (nblocks > xq8k_scratch.size())
        xq8k_scratch.resize(nblocks);
      for (int tok = 0; tok < t->num_tokens; ++tok)
        quant_q8_k(t->x + (size_t)tok * H, H,
                   xq8k_scratch.data() + (size_t)tok * (H / 256));
      const size_t routes = static_cast<size_t>(t->num_tokens) * top_k;
      const size_t gblocks = routes * (I / 32);
      if (gblocks > gq8_1_scratch.size()) gq8_1_scratch.resize(gblocks);
    }
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      cur_task = t;
      ++cur_gen;
      submitted.store(cur_gen, std::memory_order_release);
    }
    task_cv.notify_all();
  }

  void sync() {
    const uint64_t target = submitted.load(std::memory_order_acquire);
    std::unique_lock<std::mutex> lk(sync_mtx);
    sync_cv.wait(lk, [&] { return completed.load(std::memory_order_acquire) >= target; });
  }

  void submit_with_cuda_stream(uintptr_t stream, uintptr_t task) {
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &CpuMoeExecutor::submit_cb,
                       reinterpret_cast<void*>(task));
  }

  void sync_with_cuda_stream(uintptr_t stream, uintptr_t task) {
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &CpuMoeExecutor::sync_cb,
                       reinterpret_cast<void*>(task));
  }

  // Register a (layer, batch-size) slot's task so the coordinator can dispatch it on a
  // flag bump.
  void register_flag_task(int slot, uintptr_t task) {
    std::lock_guard<std::mutex> lk(flag_task_mtx);
    if (static_cast<int>(flag_task.size()) <= slot) flag_task.resize(slot + 1, nullptr);
    flag_task[slot] = reinterpret_cast<MoeTask*>(task);
  }

  int64_t flag_served_count(int slot) const {
    return (slot >= 0 && slot < static_cast<int>(flag_served.size())) ? flag_served[slot] : 0;
  }

  // Start the busy-poll coordinator over the mapped-pinned flag arrays. ``pin_core`` >= 0
  // pins the coordinator to that logical CPU (the worker auto-sizing reserves it), so its
  // polling never migrates onto / contends with a GEMV worker's core.
  void start_flag_coordinator(uintptr_t ready_ptr, uintptr_t done_ptr, int num_slots,
                              int pin_core) {
    ready_flags = reinterpret_cast<volatile int64_t*>(ready_ptr);
    done_flags = reinterpret_cast<volatile int64_t*>(done_ptr);
    coord_num_slots = num_slots;
    {
      std::lock_guard<std::mutex> lk(flag_task_mtx);
      if (static_cast<int>(flag_task.size()) < num_slots) flag_task.resize(num_slots, nullptr);
    }
    flag_served.assign(num_slots, 0);
    coord_stop.store(false);
    coord_thread = std::thread([this, pin_core] {
#if CPU_MOE_HAS_AFFINITY
      if (pin_core >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(pin_core, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      }
#endif
      coordinator_loop();
    });
  }

  void coordinator_loop() {
    // Idle backoff (edge-device friendliness): spin hot only while decode traffic is
    // flowing, with a TIME-based hot window (an iteration count varies 2-6x with slot
    // count and pause cost, so some configs dozed every token). 50 ms since the last
    // flag comfortably covers intra- and inter-token gaps -- including host-func-only
    // stretches (prefill bursts, --moe-cpu-layers subsets) -- so steady decode never
    // sleeps and keeps the ~us-level wakeup. Past it, i.e. the engine is actually idle,
    // escalate to timed sleeps capped at 2 ms: a dozing coordinator costs <0.1% of a
    // core instead of 100%, and the only price is a <=2 ms discovery delay on the FIRST
    // MoE layer after an idle period (irrelevant next to prefill). The clock is sampled
    // every 1024 empty polls (~0.1-1 ms) to keep the hot loop cheap.
    using coord_clock = std::chrono::steady_clock;
    constexpr auto kHotWindow = std::chrono::milliseconds(50);
    constexpr int64_t kSleepCapUs = 2000;
    auto last_active = coord_clock::now();
    unsigned empty_polls = 0;  // unsigned: the hot-phase ++ must not overflow into UB
    int64_t sleep_us = 100;
    bool dozing = false;
    while (!coord_stop.load(std::memory_order_relaxed)) {
      bool any = false;
      for (int L = 0; L < coord_num_slots; ++L) {
        // Binary handshake (memop-compatible: the GPU-side WAIT compares against an
        // immediate baked at graph capture, so the protocol resets per step instead of
        // counting). Acquire: everything the GPU made visible before setting ready --
        // the D2H input copies -- is visible to the worker pool after this read.
        if (flag_load_acquire(&ready_flags[L]) != 0) {
          flag_store_release(&ready_flags[L], 0);  // consume this step's doorbell
          MoeTask* t;
          {
            std::lock_guard<std::mutex> lk(flag_task_mtx);
            t = (L < static_cast<int>(flag_task.size())) ? flag_task[L] : nullptr;
          }
          if (t != nullptr) {
            submit(t);
            sync();
          }
          // Release: the workers' y stores are visible before the GPU sees done.
          flag_store_release(&done_flags[L], 1);
          if (L < static_cast<int>(flag_served.size())) ++flag_served[L];
          any = true;
        }
      }
      if (any) {
        last_active = coord_clock::now();
        empty_polls = 0;
        sleep_us = 100;
        dozing = false;
        continue;
      }
      // Hot phase: pause-spin, consulting the clock only every 1024 empty polls (the
      // amortization must gate the CLOCK, not the sleep -- gating the sleep decision
      // let 1023/1024 idle iterations hot-scan all slots, ~20% of a core at 992 slots).
      // Doze phase: sleep on EVERY iteration until traffic returns; one wake-up scan
      // (~0.5 us) per 2 ms sleep is ~0.03% duty.
      if (!dozing) {
        if ((++empty_polls & 1023u) != 0 ||
            coord_clock::now() - last_active < kHotWindow) {
#if CPU_MOE_X86
          _mm_pause();
#endif
          continue;
        }
        dozing = true;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
      sleep_us = std::min<int64_t>(sleep_us * 2, kSleepCapUs);
    }
    // Teardown: release any in-flight (or future) spin-wait immediately so a replay
    // caught mid-shutdown exits its sync kernel now instead of owning the watchdog
    // stall. Runs before the destructor's join() returns, while the flag arrays are
    // still alive on the Python side.
    for (int L = 0; L < coord_num_slots; ++L) {
      flag_store_release(&done_flags[L], INT64_MAX);
    }
  }

  // Eager (non-graph) path: run one task to completion on the pool.
  void run_task(uintptr_t task) {
    MoeTask* t = reinterpret_cast<MoeTask*>(task);
    submit(t);
    sync();
  }

  static void CUDART_CB submit_cb(void* ud) {
    MoeTask* t = reinterpret_cast<MoeTask*>(ud);
    t->exec->submit(t);
  }
  static void CUDART_CB sync_cb(void* ud) {
    MoeTask* t = reinterpret_cast<MoeTask*>(ud);
    t->exec->sync();
  }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("_quantize_q8_1_for_test",
        [](torch::Tensor input, bool avx2) {
          TORCH_CHECK(input.device().is_cpu(), "Q8_1 test input must be on CPU");
          TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
                      "Q8_1 test input must have bfloat16 dtype");
          TORCH_CHECK(input.dim() == 1 && input.numel() % 32 == 0,
                      "Q8_1 test input must be a 1D multiple of 32");
          input = input.contiguous();
          auto output = torch::empty({input.numel() / 32, 36},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
          auto* src = reinterpret_cast<const bf16_t*>(input.data_ptr());
          auto* dst = reinterpret_cast<Q8_1Block*>(output.data_ptr());
          if (avx2) {
#if CPU_MOE_X86
            TORCH_CHECK(__builtin_cpu_supports("avx2"), "AVX2 is not available");
            quant_q8_1_avx2(src, static_cast<int>(input.numel()), dst);
#else
            TORCH_CHECK(false, "AVX2 is not available on this architecture");
#endif
          } else {
            quant_q8_1_scalar(src, static_cast<int>(input.numel()), dst);
          }
          return output;
        },
        py::arg("input"), py::arg("avx2"));
  m.def("supports_gguf_q4k_q5_1", [] { return true; });
  py::class_<CpuMoeExecutor>(m, "CpuMoeExecutor")
      .def(py::init<int, int, int, int, int, int, int, int, int, int, uintptr_t, uintptr_t,
                    uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t,
                    double, double, std::vector<int>, int, int>(),
           py::arg("num_threads"), py::arg("num_layers"), py::arg("num_experts"),
           py::arg("top_k"), py::arg("hidden_size"), py::arg("inter_size"),
           py::arg("max_tokens"), py::arg("activation_id"),
           py::arg("apply_router_weight_on_input"), py::arg("weight_format"),
           py::arg("gate_up_ptr"), py::arg("down_ptr"), py::arg("gate_up_scale_ptr"),
           py::arg("gate_up_global_ptr"), py::arg("down_scale_ptr"),
           py::arg("down_global_ptr"), py::arg("gate_up_bias_ptr"),
           py::arg("down_bias_ptr"), py::arg("swiglu_alpha"), py::arg("swiglu_limit"),
           py::arg("core_ids"), py::arg("ggml_gate_up_type") = 2,
           py::arg("ggml_down_type") = 2)
      .def("create_task", &CpuMoeExecutor::create_task, py::arg("layer_id"),
           py::arg("num_tokens"), py::arg("x_ptr"), py::arg("ids_ptr"), py::arg("w_ptr"),
           py::arg("y_ptr"))
      .def("submit_with_cuda_stream", &CpuMoeExecutor::submit_with_cuda_stream,
           py::arg("stream"), py::arg("task"), py::call_guard<py::gil_scoped_release>())
      .def("sync_with_cuda_stream", &CpuMoeExecutor::sync_with_cuda_stream,
           py::arg("stream"), py::arg("task"), py::call_guard<py::gil_scoped_release>())
      .def("run_task", &CpuMoeExecutor::run_task, py::arg("task"),
           py::call_guard<py::gil_scoped_release>())
      .def("register_flag_task", &CpuMoeExecutor::register_flag_task,
           py::arg("slot"), py::arg("task"))
      .def("flag_served_count", &CpuMoeExecutor::flag_served_count, py::arg("slot"))
      .def("start_flag_coordinator", &CpuMoeExecutor::start_flag_coordinator,
           py::arg("ready_ptr"), py::arg("done_ptr"), py::arg("num_slots"),
           py::arg("pin_core"))
      .def("set_input_prequant",
           [](CpuMoeExecutor& e, bool v) { e.input_prequant = v; },
           py::arg("value"))
      .def("q8_1_quant_name", &CpuMoeExecutor::q8_1_quant_name)
      .def("isa_name", &CpuMoeExecutor::isa_name);
  m.def("memops_probe", &cumemops_probe, py::arg("stream"), py::arg("scratch_addr"));
  m.def("memop_submit", &cumemop_submit, py::arg("stream"), py::arg("done_addr"),
        py::arg("ready_addr"), py::arg("slot"));
  m.def("memop_sync", &cumemop_sync, py::arg("stream"), py::arg("done_addr"),
        py::arg("slot"));
  // ABI capability marker: the highest ActKind this build implements in the
  // GENERIC epilogue. CpuMoeExecutor.__init__ probes it before requesting an act
  // id the epilogue must handle -- a prebuilt .so from before ACT_SWIGLUOAI
  // accepts id 3 without error and silently computes the wrong activation
  // (act_apply falls through to gelu_tanh); the probe turns a stale extension
  // into a loud rebuild instruction instead of wrong model outputs.
  m.def("max_generic_act_id", []() { return static_cast<int>(ACT_SWIGLU_CLAMP); });
}
