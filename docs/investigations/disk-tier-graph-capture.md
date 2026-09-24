# Handoff: CUDA graph capture for the MoE disk tier

**Status:** graph-safe expert admission is built and committed. A host callback error
associated with the graph-on request hang was identified: the callback mutated inference
tensors outside inference mode. It now enters `torch.inference_mode()`. This remains opt-in
(`FREETOKEN_DISK_TIER_GRAPH=1`). On the controlled RX 6800 workload below, graph-on
measured 4.92 tok/s against 4.38 tok/s eager (+12.5%); this is a single-prompt workload, not
a general 1.5x improvement. Hardware measurements below were made on the RX 6800
(gfx1030) box, Qwen3.8-Flash-Next Q4 v3.

## Why this work exists

The disk tier used to force decode eager -- `_admit_to_host_tier` read `num_indices`
back to the host each MoE layer, which capture forbids, so `engine.py` turned
`cuda_graph_max_bs` to 0 whenever a tier was attached. That is a far bigger cost than
the fetch it protects.

Measured at bs=1, 4.44 tok/s (225 ms/token):

| resource | demand | capacity | utilization |
|---|---|---|---|
| disk | 1.34 GB/s sustained | 6.0-6.9 GB/s (QD8-12) | ~20% |
| GPU | 27-30% busy | -- | 30% |
| PCIe 4.0 x16 | ~5.6 GB/s | ~25 GB/s | 22% |
| FP32 compute | ~18 GFLOP/s | ~16 TFLOP/s | 0.11% |

Neither the disk nor the GPU is the bottleneck. Subtracting the two (~65 ms of GPU work
and ~50 ms of disk at plateau rate) leaves ~110 ms a token unaccounted for -- a residual,
not a direct measurement, but it is consistent with the profile: Triton launch machinery
(`compute_cache_key`, `get_arg_specialization`, `driver.__call__`) was ~17.5% of *active*
main-thread time, and the main thread is only non-idle ~25% of the wall.
A captured graph replays 48 layers of trivial work in **0.192 ms**.

Per-token byte budget, for reference: one routed expert is 3.07 MB (gate Q4_K 921,600 B
+ up Q4_K 921,600 B + down Q5_1 1,228,800 B), top-10 of 512 over 48 layers = **1.47 GB
of expert weights per token**, of which ~290 MB actually reaches the device; the rest is
absorbed by the VRAM slot cache (3363 slots) and the pinned host pool. Arithmetic
intensity is 3.2 FLOP/byte against a 31.6 balance point, so this is memory-bound even
with everything resident -- compute is never the constraint at bs=1.

## What is committed

```
c5d51ca fix(ple): require the wait-sync probe to prove memops capture
73fe793 feat(moe): admit disk-tier experts from a graph host node
ea5323a feat(kernel): resolve stream memops from hip as well as cuda
9fa9fc0 perf(moe): admit disk-tier experts in one device round trip
eea84cf fix(sampling): skip the cooperative launch on hip
```

`9fa9fc0` is the only one that moves the needle today: admission went from three device
stalls per MoE layer (144 pipeline drains a token) to one. **4.44 -> 4.66 tok/s**, both
plateaus over seven runs. Note the profiler badly undersells this -- the syncs were ~2%
of main-thread *self* time; their real cost is the idle they create.

`eea84cf` is load-bearing for running anything at all on gfx1030 (see Traps).

## The handshake

`offload_cache.py`: `_admit_to_host_tier` (line ~1117) branches on
`torch.cuda.is_current_stream_capturing()`. The captured arm, `_admit_capture`, emits
per MoE layer:

```
D2H  num_indices, src_indices -> pinned
HOST node: host_tier.ensure(layer, misses) -> pinned slot vector
H2D  pinned slots -> src_indices
```

The stream does not advance past the host node until the callback returns, which is the
same barrier the old sync provided. Supporting pieces: `moe/graph_host.py` (resolves
`hipLaunchHostFunc` / `cudaLaunchHostFunc` / `cuLaunchHostFunc`),
`prepare_graph_admission` called from `graph.py:154` before capture opens, and
`raise_admission_error` called by the scheduler after `copy_done.synchronize()` -- a host
callback cannot unwind into the driver, so failures are parked on the cache and re-raised
on the engine thread only after replay completion.

Cost: **23.6 us/layer, ~1.1 ms/token** (2.2 ms with a contended GIL), against the ~110
ms/token of eager launch overhead it is meant to displace.

`tests/moe/test_graph_admission.py` covers node timing, replay, zero-miss layers and
error parking with a stand-in tier. A CPU regression now calls the real callback body
from a foreign thread with inference tensors and verifies that it writes the mapped slot
IDs. Callback errors are checked by the scheduler only after the forward's existing
`copy_done.synchronize()` barrier and before token or cache commits; that check adds no
GPU synchronization. A separate CPU regression validates this ordering and that an
error prevents final-batch token output.

## Plateau throughput measurement

Measured 2026-09-14 on `ajkerchum@192.168.1.61`: Radeon RX 6800 (gfx1030, 16 GiB
VRAM), ROCm 7.1 / PyTorch 2.11.0+rocm7.1, Qwen3.8-Flash-Next-Q4_K-v3.gguf. Both
services used the same working tree and model, `--moe-backend offload`,
`--moe-cache-auto --moe-host-cache-size 2048`, `--ple-backend disk`, `--memory-ratio 0.9`,
`--max-seq-len-override 8192`, and `--max-running-requests 1`. Graph-on used
`FREETOKEN_DISK_TIER_GRAPH=1 --cuda-graph-max-bs 1`; eager used the variable unset and
`--cuda-graph-max-bs 0`. `AMD_SERIALIZE_KERNEL`, `FT_GGUF_BACKEND`, and
`FREETOKEN_ROCM_DECODE_TRACE` were unset in both runs.

The request was the same greedy streamed completion in both modes: prompt
`The capital of France is a city with museums, historic landmarks, and a river running through it. Explain the main attractions in one sentence.`, `max_tokens=48`,
`temperature=0`, `ignore_eos=true`, with `stream_options.include_usage=true`. Each mode had
four warmups and seven measured requests. The first warmup was cold (graph 1.24 tok/s;
eager 2.51 tok/s); warmups 2-4 stabilized at 4.91-4.93 tok/s graph-on and 4.38-4.40 tok/s
eager. Throughput follows the decode convention of counting tokens after the first:
`(completion_tokens - 1) / (last_text_time - first_text_time)`. The SSE reader uses
`iter_lines(chunk_size=1)` to reduce buffering; prefill is excluded. Every run returned 47
completion tokens. All 22 output strings were byte-for-byte equal; their SHA-256 was
`7a7afd5ffd208cb6524d2a0a4a0f505761fa76e3c24dbd72d4501192bec7adea`.

| mode | seven measured samples (tok/s) | median | range |
|---|---|---:|---:|
| graph-on | 4.9350, 4.7538, 4.9232, 4.8713, 4.9205, 4.8678, 4.9211 | 4.9205 | 4.7538-4.9350 |
| eager | 4.3632, 4.4029, 4.3588, 4.3942, 4.3754, 4.4493, 4.3587 | 4.3754 | 4.3587-4.4493 |

That is a 12.5% median increase for graph-on in this A/B (4.92 vs 4.38 tok/s). This
compares graph admission with eager admission on the same working tree, not against `main`
or the historical 4.44 tok/s workload. It is a single serial request, one prompt, and one
token length; it does not establish multi-request behavior or a general speedup. An
earlier timing draft counted tokens through `[DONE]` and stopped the interval at `[DONE]`;
those uncorrected proxy artifacts are retained with `proxy-superseded` filenames and are
not part of this result. The corrected request, per-call timings, text, token counts, and
hashes are retained in
[`disk-tier-graph-throughput-graph.json`](disk-tier-graph-throughput-graph.json) and
[`disk-tier-graph-throughput-eager.json`](disk-tier-graph-throughput-eager.json); the
SSE timing runner is [`benchmark_disk_tier_graph.py`](benchmark_disk_tier_graph.py).

Reproduce the two service modes from `~/ft-mtp` (model path and options match the run):

```bash
cd ~/ft-mtp
nohup env -u FREETOKEN_ROCM_DECODE_TRACE FREETOKEN_DISK_TIER_GRAPH=1 ~/.venvs/ft/bin/python -m freetoken.cli serve --model ~/models/Qwen3.8-Flash-Next-Q4/Qwen3.8-Flash-Next-Q4_K-v3.gguf --host 127.0.0.1 --port 8199 --moe-backend offload --max-running-requests 1 --moe-cache-auto --moe-host-cache-size 2048 --ple-backend disk --memory-ratio 0.9 --max-seq-len-override 8192 --cuda-graph-max-bs 1 >/tmp/disk-tier-graph-throughput.log 2>&1 </dev/null &
nohup env -u FREETOKEN_DISK_TIER_GRAPH -u FREETOKEN_ROCM_DECODE_TRACE ~/.venvs/ft/bin/python -m freetoken.cli serve --model ~/models/Qwen3.8-Flash-Next-Q4/Qwen3.8-Flash-Next-Q4_K-v3.gguf --host 127.0.0.1 --port 8199 --moe-backend offload --max-running-requests 1 --moe-cache-auto --moe-host-cache-size 2048 --ple-backend disk --memory-ratio 0.9 --max-seq-len-override 8192 --cuda-graph-max-bs 0 >/tmp/disk-tier-eager-throughput.log 2>&1 </dev/null &
```

Start only one server at a time, redirecting its output to a log, and wait for
`API server is ready to serve` before sending requests. Run the request loop with the
service ready on `127.0.0.1:8199`:

```bash
scp docs/investigations/benchmark_disk_tier_graph.py ajkerchum@192.168.1.61:/tmp/benchmark_disk_tier_graph.py
source ~/.venvs/ft/bin/activate
python /tmp/benchmark_disk_tier_graph.py --mode graph --warmups 4 --runs 7 --output /tmp/disk-tier-graph-throughput.json
python /tmp/benchmark_disk_tier_graph.py --mode eager --warmups 4 --runs 7 --output /tmp/disk-tier-eager-throughput.json
```

Both services were stopped after their runs, and ports 8199/8200 and their workers were
confirmed clear.

### Do not try to do this with a flag handshake

Stream memops (`hipStreamWriteValue64` / `hipStreamWaitValue64`) **are not capturable on
HIP**. They execute eagerly at capture time and leave no node. This probe settles it:

```python
with torch.cuda.graph(g):
    x.add_(1.0)
    _ple_store.memop_write(torch.cuda.current_stream().cuda_stream, flag.data_ptr(), 5)
print(int(flag[0]))          # 5  -> ran eagerly, was not captured
flag.zero_(); g.replay(); torch.cuda.synchronize()
print(int(flag[0]))          # 0  -> no node was recorded
```

A first pass at this was built on flags and had to be thrown away. It is easy to fool
yourself here: capturing a WAIT and then timing a replay *looks* like it works, because
the eager wait parks the capture stream and the replay's synchronize inherits that
delay. `c5d51ca` hardens `DiskRowTable._probe_wait_sync` against exactly this -- a zero
return only means the driver accepted the call, so the probe now captures a write and
checks it did not land until replay. ROCm correctly reads `launch-gating` again.

## Diagnosed callback thread context

The first report looked like a prefill hang, but later request-phase instrumentation showed
the first prefill had completed and the stall was in the first captured decode. A serialized
run's Python stack stopped at Triton sampling, which identified a wait site rather than the
kernel or callback responsible. Instrumentation confirmed that this decode took the graph
path. A temporary `torch.cuda.synchronize()` after replay let the callbacks run and surfaced
the concrete failure:

```
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed
```

The callback runs on a HIP driver thread, outside the engine thread's inference-mode
context, while its pinned `n`, `src`, and `out` buffers were created in inference mode.
Writing `out` therefore raised; callback exceptions are parked because they cannot unwind
through the driver. `_admit_callback` now wraps its tensor reads and output write in
`torch.inference_mode()`. This fix does not add a replay synchronization.

On the RX 6800 with graph admission enabled, default HIP GGUF backend, and
`AMD_SERIALIZE_KERNEL` unset, the 5-token prompt `The capital of France is` returned
` Paris. Paris` (max_tokens=4). A longer greedy comparison used this exact request for both
graph-on and eager service runs:

```
prompt: The capital of France is a city with museums, historic landmarks, and a river running through it. Explain the main attractions in one sentence.
max_tokens=16, temperature=0, ignore_eos=true
```

Both returned 15 completion tokens and the same text:
`\n\n<think>\nThe user asks me to explain the main attractions of Paris (`. The graph-on
request used `FREETOKEN_DISK_TIER_GRAPH=1` and `--cuda-graph-max-bs 1`; the eager control
used `--cuda-graph-max-bs 0`. This verifies correct output for the tested request, not a
throughput improvement or the absence of other graph issues.

The focused local CPU suite reports **24 passed, 6 skipped**. On the RX 6800,
`tests/moe/test_graph_admission.py` plus the new foreign-thread inference-buffer regression
report **5 passed**. Remote tests and both serving comparisons ran on
`ajkerchum@192.168.1.61` in `~/ft-mtp`; the final graph and eager service logs are
`/tmp/disk-tier-graph-correctness16.log` and `/tmp/disk-tier-eager-correctness16.log`, with
the eager response saved in `/tmp/disk-tier-eager-correctness16.response`. Both test
services were stopped after collecting results. The warmed throughput comparison is
recorded above; broader prompts and batch sizes remain untested.

Local CPU verification (CPU-only `.venv`; graph replay tests skip without a GPU):
`rtk uv run --no-sync python -m pytest tests/scheduler/test_disk_tier_admission_error.py tests/scheduler/test_abort_inflight_prefill.py::test_abort_inflight_final_chunk_marks_then_drains tests/scheduler/test_abort_inflight_prefill.py::test_abort_inflight_intermediate_chunk_marks_then_drains tests/scheduler/test_abort_inflight_prefill.py::test_abort_starved_decode_req_frees_immediately tests/moe/test_host_tier.py tests/moe/test_graph_admission.py tests/models/qwen4_exp/test_ple_disk.py -q`
reported **24 passed, 6 skipped**.

## Reproducing

The serving env on the VM is `~/.venvs/ft` (python 3.11, torch 2.11.0+rocm7.1,
pytorch-triton-rocm 3.5.1, flashlib 0.3.0). The checkout is `~/ft-mtp`, kept in sync by
rsync from the workstation -- **it is not a git remote and holds no commits of its own.**

Rebuild after a C++ change:

```bash
VIRTUAL_ENV=$HOME/.venvs/ft uv pip install -e . --no-deps --no-build-isolation
```

Serve (`/tmp/serve.sh`):

```bash
python -m freetoken.cli serve \
  --model ~/models/Qwen3.8-Flash-Next-Q4/Qwen3.8-Flash-Next-Q4_K-v3.gguf \
  --host 127.0.0.1 --port 8199 \
  --moe-backend offload --max-running-requests 1 \
  --moe-cache-auto --moe-host-cache-size 2048 \
  --ple-backend disk --memory-ratio 0.9 \
  --max-seq-len-override 8192 --cuda-graph-max-bs 1
```

`FREETOKEN_ROCM_DECODE_TRACE=1` gives a one-shot per-stage trace of the first real decode.

## Traps

**Benchmarking here is treacherous. Two effects will fabricate a result.**

* *Cold Triton JIT.* The first request after a restart compiles every kernel. An early
  "baseline" of 0.83 tok/s in this work was entirely that. Always warm first.
* *Page cache.* Throughput climbs across roughly four runs (2.87 -> 3.68 -> 4.70) as the
  GGUF's hot regions land in cache, then plateaus to within +-0.03. **Report the plateau
  over 7-9 runs, never a single shot.** Baseline and candidate must both be plateaued;
  the +5% above is 4.42-4.47 vs 4.63-4.70 measured that way.

**Orphaned workers.** `pkill -f "freetoken.cli serve"` kills the frontend but leaves the
`multiprocessing.spawn` workers holding port 8200, and the next launch dies with
`EADDRINUSE` on the *distributed rendezvous* port, not 8199. Kill
`multiprocessing.spawn` and `multiprocessing.resource_tracker` too, and wait for both
ports to clear.

**Triton version.** Installing `flashlib` pulls `triton==3.8.0`, which shadows the ROCm
build and breaks the sampling kernel launch path. Install `pytorch-triton-rocm` from the
rocm7.1 index afterwards.

**Cooperative launch is unusable on gfx1030** (fixed in `eea84cf`, but know the shape).
`_fused_plan` asks for `2 x 30 SMs = 60` CTAs/row. 60 is rejected with
`hipErrorCooperativeLaunchTooLarge`, surfaced as a bare `SystemError` that no string
match catches -- and the rejection poisons the HIP context, so catching it does not
help; the retry inherits a dead stream. 30 is *accepted* and then **spins forever** at
the grid barrier. HIP therefore takes `force_single` from the start, which costs 0.81
ms/call over a 248k vocab.

**Pre-existing test failures**, unrelated to any of this -- confirm against a stashed
tree before blaming a change: `tests/moe` has 5 (four `test_cpu_moe*cuda_graph_replay`,
one `test_adjust_config_converts_moe_cache_rate_to_cache_size`). A clean run is
153 passed / 5 failed / 9 skipped.

## Ruled out, do not redo

* **A bigger pinned host pool.** Residency looks like the obvious lever and it is not:
  pinned memory is unreclaimable and evicts the page cache, which was serving more reads
  than the pool does.

  | slots | residency | device reads | throughput |
  |---|---|---|---|
  | 1024 | 4.2% | 20.2 GB | 4.51 |
  | **2048** | **8.3%** | **16.8 GB** | **4.66** |
  | 4096 | 16.7% | 20.4 GB | ~4.0, erratic |

  2048 is already optimal on a 30 GB box. This is a config ceiling, not a RAM shortage.

* **More read parallelism.** `host_tier.ensure` submits one job per missing
  (expert, bank) pair to `ThreadPoolExecutor(max_workers=8)`; each job reads that bank's
  extents. The device knee is QD8-12 (QD1 1.66-2.41 GB/s, QD8 4.87-6.85 GB/s,
  QD16+ flat or worse), so 8 workers is at the knee. There is no 3x sitting in the read
  path -- an early claim in this work that there was came from reading `read_expert`
  without its caller.

## Leftovers

The follow-on Qwen3.8 IQ2 work, including the retained symmetric Q2 down-projection
format, quality probes, and the current 15.8833 tok/s long-prompt result, is recorded in
[qwen38-iq2-s-results.md](qwen38-iq2-s-results.md). The retained VM checkpoint is
`~/models/Qwen3.8-Flash-Next-IQ2_S-Q2SYM/`; it occupies 61.3 GiB and leaves about
46 GiB free. Do not create another full derivative without removing or moving this one.

* The n-gram PLE table (`per_layer_token_embd`, Q8_0) is 54.40 GB -- 41% of the
  checkpoint -- and must stay on disk; it cannot be pinned on this box. It costs nothing:
  16 heads x one 170-byte row = **2.7 KB per token**, ~0.02% of the expert traffic, read
  through io_uring + O_DIRECT. All the pressure is the routed experts; do not go
  optimizing the PLE path.
* Still open from the MTP work and untouched here: greedy speculative output diverges
  from plain decoding on ~1.6% of tokens on both model families. The test that would
  settle whether it is argmax flipping on near-ties is logging the top-1/top-2 logit gap
  at a divergence point.
* `benchmarks/bench_decode_moe.py` does not pass `--moe-host-cache-size`, so it cannot
  exercise the disk tier as-is.
