# Handoff: CUDA graph capture for the MoE disk tier

**Status:** the graph-safe expert admission is built, tested and committed, but it is
opt-in (`FREETOKEN_DISK_TIER_GRAPH=1`) because turning capture on wedges the *prefill*
path. That hang is the only thing between this and a ~1.5x decode win. Everything below
was measured on the RX 6800 (gfx1030) box, Qwen3.8-Flash-Next Q4 v3.

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
`raise_admission_error` called from `forward_batch` -- a host callback cannot unwind into
the driver, so failures are parked on the cache and re-raised on the engine thread.

Cost: **23.6 us/layer, ~1.1 ms/token** (2.2 ms with a contended GIL), against the ~110
ms/token of eager launch overhead it is meant to displace.

Covered by `tests/moe/test_graph_admission.py` (4 tests): the node does not fire during
capture, fires once per replay, zero-miss layers skip the tier, and a failing host
surfaces on the engine thread rather than hanging the stream or serving stale experts.

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

## The blocker

With `FREETOKEN_DISK_TIER_GRAPH=1`, capture succeeds ("disk tier: expert admission on
graph host nodes", graphs captured in ~5 s) and then the **first real request hangs in
prefill**, GPU pegged at 99%, same stack for 60 s+:

```
causal_conv1d_varlen (causal_conv1d_triton.py:462)
_conv_prefill (models/qwen4_exp/gdn.py:101)
forward_batch (engine/engine.py:1107)
```

Two things matter about that line. It is in **prefill, which is never captured**. And it
is not a kernel -- `causal_conv1d_triton.py:462` is `max_seq_len = int(seq_lens.max().item())`,
a device->host sync. So the correct reading is *some earlier GPU work never completes*,
and this sync is merely where the scheduler first waits on it. Do not go looking for a
bug in the conv kernel.

Isolated cleanly: **the same tree with `--cuda-graph-max-bs 0` generates correct output.**
So this is capture, not the handshake -- and since the disk tier has always force-disabled
capture, this configuration has never run before. Treat it as a pre-existing fault the
blanket disable was masking, not as a regression.

Suggested next steps, cheapest first:

1. `AMD_SERIALIZE_KERNEL=3` to name the kernel that never retires.
2. Determine whether a host node is still outstanding when prefill stalls. The plausible
   deadlock is the callback: it runs on a driver thread holding the GIL and calls
   `host_tier.ensure`, which fans disk reads across a `ThreadPoolExecutor` whose workers
   need the GIL to run Python. `concurrent.futures` should release the GIL while waiting,
   but this is unverified and is the first thing to rule out. A quick test: swap the
   callback body for a no-op that only fills `out` with the identity mapping. If the hang
   disappears, it is the callback, not capture.
3. If the hang persists with a no-op callback, capture alone is at fault; bisect by
   capturing a graph with no host nodes at all on this model.

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

* **More read parallelism.** `host_tier.ensure` already fans misses across
  `ThreadPoolExecutor(max_workers=8)`; only `read_expert`'s extents are serial within one
  expert-bank. The device knee is QD8-12 (QD1 1.66-2.41 GB/s, QD8 4.87-6.85 GB/s,
  QD16+ flat or worse), so 8 workers is at the knee. There is no 3x sitting in the read
  path -- an early claim in this work that there was came from reading `read_expert`
  without its caller.

## Leftovers

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
