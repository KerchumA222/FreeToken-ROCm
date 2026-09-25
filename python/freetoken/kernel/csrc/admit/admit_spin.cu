// Disk-tier expert admission inside a CUDA/HIP graph without a host node per layer.
//
// Captured decode graphs used to admit each MoE layer's GPU-cache misses through a host
// function node: D2H the miss list, run the host tier's ensure, H2D the host slots back.
// A host node costs ~51 us on an RX 6800 even with nothing to do, and a graph cannot skip
// one, so every layer paid it (~3.2 ms of a Qwen3.8-Flash-Next token).
//
// Here one small kernel reads the miss count on the GPU and returns at once when it is
// zero. Otherwise it posts the miss list into coherent pinned memory and spin-waits for
// the reply of a host poller thread, which runs the (Python) host-tier ensure. The spin
// is bounded by wall-clock time: on timeout the kernel flags an error and returns, the
// forward computes garbage for that step, and the engine raises. It can never hang.
//
// Request buffer (int32): [0] seq, [1] layer, [2] n, [3] error flag (GPU -> host),
// [4..) miss ids. Reply buffer: [0] seq, [1] status (0 ok), [4..) host slots.

#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <chrono>
#include <thread>

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#define FT_STREAM() at::hip::getCurrentHIPStream().stream()
#define FT_SLEEP() __builtin_amdgcn_s_sleep(2)
#define FT_WALLCLOCK() __builtin_amdgcn_s_memrealtime()
static constexpr long long kTicksPerSecond = 100000000LL;  // s_memrealtime: 100 MHz
#else
#include <ATen/cuda/CUDAContext.h>
#define FT_STREAM() at::cuda::getCurrentCUDAStream().stream()
#define FT_SLEEP() __nanosleep(100)
#define FT_WALLCLOCK() globaltimer()
static __device__ __forceinline__ long long globaltimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
static constexpr long long kTicksPerSecond = 1000000000LL;  // ns
#endif

constexpr int kHeader = 4;

__global__ void admit_spin_kernel(const int64_t* __restrict__ num_indices, int32_t* __restrict__ src,
                                  int* req, int* resp, int* counter, int layer, long long timeout_ticks) {
  __shared__ int s_ok;
  const int64_t n = num_indices[0];
  if (n <= 0) {
    return;  // no GPU-cache miss in this layer: nothing for the host to do
  }
  volatile int* vreq = req;
  volatile int* vresp = resp;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    vreq[kHeader + i] = src[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int seq = ++(*counter);
    vreq[1] = layer;
    vreq[2] = static_cast<int>(n);
    __threadfence_system();
    vreq[0] = seq;
    __threadfence_system();
    const long long start = FT_WALLCLOCK();
    bool done = false;
    while (!(done = (vresp[0] == seq)) && FT_WALLCLOCK() - start < timeout_ticks) {
      FT_SLEEP();
    }
    __threadfence_system();
    s_ok = done && vresp[1] == 0;
    if (!s_ok) {
      vreq[3] = 1;  // the engine checks this after the step and raises
      __threadfence_system();
    }
  }
  __syncthreads();
  if (s_ok) {
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
      src[i] = vresp[kHeader + i];
    }
  }
}

void launch_admit(torch::Tensor num_indices, torch::Tensor src_indices, int64_t req, int64_t resp,
                  torch::Tensor counter, int64_t layer, double timeout_s) {
  TORCH_CHECK(num_indices.scalar_type() == torch::kInt64 && src_indices.scalar_type() == torch::kInt32);
  const long long ticks = static_cast<long long>(timeout_s * kTicksPerSecond);
#ifdef USE_ROCM
  hipLaunchKernelGGL(admit_spin_kernel, dim3(1), dim3(256), 0, FT_STREAM(),
                     num_indices.data_ptr<int64_t>(), src_indices.data_ptr<int32_t>(),
                     reinterpret_cast<int*>(req), reinterpret_cast<int*>(resp),
                     counter.data_ptr<int>(), static_cast<int>(layer), ticks);
#else
  admit_spin_kernel<<<1, 256, 0, FT_STREAM()>>>(
      num_indices.data_ptr<int64_t>(), src_indices.data_ptr<int32_t>(),
      reinterpret_cast<int*>(req), reinterpret_cast<int*>(resp), counter.data_ptr<int>(),
      static_cast<int>(layer), ticks);
#endif
}

// ---- host poller -------------------------------------------------------------------

namespace {
std::atomic<bool> g_stop{false};
std::thread g_thread;
pybind11::object* g_callback = nullptr;
}  // namespace

// ``callback(layer, n) -> int`` runs under the GIL for every request: it reads the miss
// ids from the request buffer, fills the host slots into the reply buffer and returns a
// status (0 ok). Spins while requests keep coming; after ~20 ms idle it naps 200 us
// between polls, so an idle server does not hold a core.
void start_poller(int64_t req_host, int64_t resp_host, pybind11::object callback) {
  TORCH_CHECK(!g_thread.joinable(), "admission poller already running");
  g_stop = false;
  g_callback = new pybind11::object(callback);
  g_thread = std::thread([req_host, resp_host]() {
    volatile int* req = reinterpret_cast<volatile int*>(req_host);
    volatile int* resp = reinterpret_cast<volatile int*>(resp_host);
    int last = req[0];
    auto idle_since = std::chrono::steady_clock::now();
    while (!g_stop.load(std::memory_order_relaxed)) {
      const int seq = req[0];
      if (seq == last) {
        if (std::chrono::steady_clock::now() - idle_since > std::chrono::milliseconds(20)) {
          std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        continue;
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      int status = 1;
      {
        pybind11::gil_scoped_acquire gil;
        try {
          status = (*g_callback)(req[1], req[2]).cast<int>();
        } catch (...) {
          status = 1;
        }
      }
      resp[1] = status;
      std::atomic_thread_fence(std::memory_order_seq_cst);
      resp[0] = seq;
      last = seq;
      idle_since = std::chrono::steady_clock::now();
    }
  });
}

void stop_poller() {
  g_stop = true;
  if (g_thread.joinable()) {
    pybind11::gil_scoped_release nogil;
    g_thread.join();
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_admit", &launch_admit, "");
  m.def("start_poller", &start_poller, "");
  m.def("stop_poller", &stop_poller, "");
}
