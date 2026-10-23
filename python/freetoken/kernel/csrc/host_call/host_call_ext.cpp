// Host call from inside a captured CUDA/HIP graph: the stream hands a task word to a host thread that runs a Python callback, and resumes when it returns.
//
// The GPU side is GpuSync (cpu_moe/gpu_sync.h): submit(stream, task) rings the doorbell and wait(stream) holds the stream until the thread has called
// cpu_done(). Both are stream memory operations, so they record as graph nodes where the runtime supports it (probe() checks) and the stream runs
// no kernel while it waits. One call is in flight at a time, as every user captures a single stream.
//
// `pending_ptr` (optional) is the host address of an int64 the stream wrote before submit: when it reads 0 the call needs no host work, and the thread
// answers without taking the GIL. The callback reports its own failures; one that raises anyway is dropped here. The stream is released whatever
// happens, so a failed callback can never hang a replay.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>

#include <pybind11/pybind11.h>

#include "../cpu_moe/gpu_sync.h"

namespace py = pybind11;
using freetoken::cpu_moe::GpuSync;

namespace {

class HostCall {
 public:
  HostCall(uintptr_t pending_ptr, py::object callback)
      : pending_(reinterpret_cast<const volatile int64_t*>(pending_ptr)), callback_(std::move(callback)), sync_(std::make_unique<GpuSync>()) {}

  ~HostCall() { stop(); }

  void start() {
    if (thread_.joinable()) return;
    stop_.store(false);
    thread_ = std::thread([this] { loop(); });
  }

  // Joins the thread and releases a stream still waiting on it. Call without a callback in flight on another Python thread that needs this one.
  void stop() {
    stop_.store(true);
    if (thread_.joinable()) {
      py::gil_scoped_release nogil;
      thread_.join();
    }
    sync_->cpu_done();
  }

  void submit(uintptr_t stream, uint64_t task) { sync_->gpu_submit(stream, task); }
  void wait(uintptr_t stream) { sync_->gpu_wait(stream); }
  // Release a stream the thread will never answer (it died, or the owner is giving up).
  void release() { sync_->cpu_done(); }
  uint64_t served() const { return served_.load(); }
  bool running() const { return thread_.joinable() && !stop_.load(); }

 private:
  void loop() {
    // Spin while calls keep coming, then back off to timed sleeps so an idle server does not hold a core.
    using clock = std::chrono::steady_clock;
    constexpr auto kHotWindow = std::chrono::milliseconds(50);
    constexpr int64_t kSleepCapUs = 2000;
    auto last_active = clock::now();
    unsigned empty_polls = 0;
    int64_t sleep_us = 100;
    bool dozing = false;
    while (!stop_.load(std::memory_order_relaxed)) {
      const uint64_t task = sync_->cpu_poll();
      if (task != 0) {
        if (pending_ == nullptr || freetoken::cpu_moe::flag_load_acquire(pending_) != 0) {
          py::gil_scoped_acquire gil;
          try {
            callback_(task);
          } catch (...) {
          }
        }
        sync_->cpu_done();
        served_.fetch_add(1, std::memory_order_relaxed);
        last_active = clock::now();
        empty_polls = 0;
        sleep_us = 100;
        dozing = false;
        continue;
      }
      if (!dozing) {
        if ((++empty_polls & 1023u) != 0 || clock::now() - last_active < kHotWindow) continue;
        dozing = true;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
      sleep_us = std::min<int64_t>(sleep_us * 2, kSleepCapUs);
    }
  }

  const volatile int64_t* pending_;
  py::object callback_;
  std::unique_ptr<GpuSync> sync_;
  std::thread thread_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> served_{0};
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<HostCall>(m, "HostCall")
      .def(py::init<uintptr_t, py::object>(), py::arg("pending_ptr"), py::arg("callback"))
      .def("start", &HostCall::start)
      .def("stop", &HostCall::stop)
      .def("submit", &HostCall::submit, py::arg("stream"), py::arg("task"))
      .def("wait", &HostCall::wait, py::arg("stream"))
      .def("release", &HostCall::release)
      .def("served", &HostCall::served)
      .def("running", &HostCall::running);
  m.def("probe", &GpuSync::probe, py::arg("stream"));
}
