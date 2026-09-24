# ROCm CPU-MoE graph replay investigation

## Finding

Issue [#350](https://github.com/FlashML-org/FreeToken/issues/350) reported silent output corruption from CPU-MoE CUDA-graph replay on ROCm. The default flag handshake was selected because HIP resolved `hipStreamWriteValue64` and `hipStreamWaitValue64`, and the eager memops probe succeeded. That probe only established that the stream operations worked outside graph capture. On this ROCm build, a write accepted during capture executed eagerly instead of becoming a replay node. A later replay therefore had no reliable ready/done dependency with the CPU coordinator. The C++ memop wrappers log runtime errors rather than returning them, so a missing wait node cannot be detected through their return value.

The additional capture probe originally added for writes could reject this HIP behavior, but testing only writes still leaves the paired wait unverified. The fix therefore does not use flag sync on HIP. When `FREETOKEN_CPU_MOE_FLAG_SYNC` is enabled, HIP selects the `cudaLaunchHostFunc` submit/sync path, which passed the changing-input replay tests on the RX6800. CUDA retains the capture probe. That probe now creates its graph and scratch tensors on the executor's device and captures on that device's current stream. PyTorch 2.11.0+rocm7.1 reports the supported signature as `torch.cuda.graph(cuda_graph, pool=None, stream=None, capture_error_mode='global')`; the stream and device context select the target device.

The policy is covered by `tests/moe/test_cpu_moe_flag_sync.py`. Local CPU verification:

```text
rtk proxy uv run --no-sync python -m pytest \
  tests/moe/test_cpu_moe_flag_sync.py \
  tests/moe/test_hybrid_fetch.py \
  tests/moe/test_cpu_moe.py \
  tests/moe/test_cpu_moe_q4_0.py -q
4 passed, 27 skipped
```

The skips are GPU-only cases on the local CPU-only environment. On the RX6800 VM, after deploying `python/freetoken/moe/cpu_executor.py`, the existing graph-replay regressions passed with the default setting (flag sync enabled in the environment, but safely disabled by the ROCm platform gate):

```text
cd ~/ft-mtp
~/.venvs/ft/bin/python -m pytest \
  tests/moe/test_cpu_moe.py::test_cpu_moe_decode_cuda_graph_replay \
  tests/moe/test_cpu_moe.py::test_cpu_moe_decode_cuda_graph_replay_mxfp4 \
  tests/moe/test_cpu_moe.py::test_cpu_moe_decode_cuda_graph_replay_dsfp4 \
  tests/moe/test_cpu_moe_q4_0.py::test_cpu_moe_decode_q4_0_cuda_graph_replay -q
4 passed in 5.58s
```

Before this correction, all four tests failed with numerical mismatches in flag mode; running the same command with `FREETOKEN_CPU_MOE_FLAG_SYNC=0` passed all four. With the fix, the default mode passes and logs that HIP uses the host-function handshake. This confirms the fallback path on the tested RX6800 and PyTorch 2.11.0+rocm7.1 runtime.

## Qwen3.8-Flash-Next mixed GGUF kernels

The target file `/home/ajkerchum/models/Qwen3.8-Flash-Next-Q4/Qwen3.8-Flash-Next-Q4_K-v3.gguf` resolves its routed expert banks to Q4_K (GGML type 12) for gate/up and Q5_1 (type 7) for down. Its live config is `H=2560, I=640, E=512, L=48`; the packed banks are `gate_up [512, 1280, 1440]` and `down [512, 2560, 480]`. The CPU executor's `q4_0` kernel assumes 18 bytes per 32 values for both bank rows. Q4_K happens to have the same row width at these dimensions (10 blocks of 256 values at 144 bytes each, versus 80 Q4_0 blocks at 18 bytes), so the shape-only check passes while the Q4_0 decoder would silently misinterpret its block layout. Q5_1 has 24 bytes per 32 values, so its 640-value down rows occupy 480 bytes rather than the 360 bytes expected by the Q4_0 check. Initialization therefore fails on the down bank:

```text
AssertionError: (torch.Size([512, 2560, 480]), 640)
```

The failing check expected `(I // 32) * 18 == 360` bytes at `I=640`; the Q5_1 row has 480 bytes. The model loader preserves bank types, so treating the model's broad `q4_0` CPU format tag as support for its mixed Q4_K/Q5_1 banks was not valid.

The CPU executor now carries the gate/up and down GGML type ids from the model quantization method into the native extension. It accepts the existing Q4_0/Q4_0 pair and the target checkpoint's Q4_K/Q5_1 pair, rejects other pairs before dispatch, and checks each bank against its own packed row width. The new path quantizes the input to Q8_K for gate/up and the activated intermediate to Q8_1 for down, then uses the corresponding llama.cpp dot-product equations. Runtime ISA selection keeps scalar fallbacks and selects AVX2 implementations on the RX6800 VM's Zen 2 host.

The synthetic numerical test forces both scalar and AVX2 paths and compares them with gguf-py's dequantizer plus FreeToken's dense GPU MoE reference. The graph test changes activations, routing ids, and routing weights between replays. A slow checkpoint test reads only experts 17 and 291 from layer 0 through `GgufExpertStore`, so it validates the actual file layout while allocating about 6 MiB rather than the full expert bank. On the RX6800 VM:

```text
~/.venvs/ft/bin/python -m pytest \
  tests/moe/test_cpu_moe_q4_0.py \
  tests/moe/test_cpu_moe_gguf_types.py -q
12 passed in 26.51s
```

The real-checkpoint test alone passed in 14.43s. A 64-expert, layer-0 microbenchmark rotated through a 187.5 MiB working set and timed only the native top-2 CPU task after eight warmups. Alternating the forced ISA order produced:

| ISA | Median per layer | Effective packed-weight bandwidth |
| --- | ---: | ---: |
| scalar | 0.804-0.853 ms | 7.21-7.64 GB/s |
| AVX2 | 0.313-0.318 ms | 19.35-19.63 GB/s |

AVX2 is 2.53-2.73x faster in this bounded test. This is a kernel result, not an end-to-end token-rate claim. A complete resident CPU/hybrid bank for this checkpoint is 75.5 GB (70.3 GiB), while the VM has 30 GiB of RAM, so the full target cannot start with the current resident-bank CPU executor. Using these kernels with the bounded disk tier requires a cache-aware CPU dispatch path that resolves routed experts from host slots instead of assuming every layer's full bank is resident.

The down projection now processes one routed expert across an output tile before moving to the next expert. This keeps each expert's packed rows contiguous and hoists the route metadata out of the inner output loop. A top-10 version of the same 64-expert benchmark alternated the old and new extension three times each. The old medians were 1.268, 1.335, and 1.337 ms; the route-major medians were 1.304, 1.251, and 1.231 ms. The mean of those medians improved from 1.313 to 1.262 ms (3.9%), raising effective packed-weight bandwidth from 23.4 to 24.4 GB/s. The focused CPU MoE suite passed 32 tests after the change.

Q8_1 preparation for the mixed path is also fused into pass one. Each intermediate tile is exactly one 32-value Q8_1 block, so its existing worker can quantize the block immediately after writing it. The pass-one barrier still orders all Q8_1 writes before the down projection, while the separate row-preparation queue and second barrier are skipped. Two fixed-input, 200-sample alternating runs measured route-major baselines of 1.308 and 1.240 ms and fused results of 1.223 and 1.231 ms. The mean of the medians improved from 1.274 to 1.227 ms (3.7%). The same 32-test CPU MoE suite passed after fusion.

Several follow-up kernel experiments were measured and reverted. A fuller llama.cpp-style Q4_K AVX2 reduction was neutral or slower. Pairing gate and up dots measured 1.252 ms against a 1.250 ms baseline. Static scheduling usually improved about 1.7% but produced a 24% straggler regression, so it was not safe to retain. Adding a 512-byte Q4_K software prefetch produced candidate medians of 1.249 and 1.318 ms against alternating retained medians of 1.243 and 1.253 ms. Increasing the down tile from 32 to 64 rows fell to 1.464 ms and 20.98 GB/s because one route's tile nearly filled Zen 2's 32 KiB L1. Finally, a two-accumulator Q5_1 loop unroll passed the scalar/AVX2 numerical tests but measured 1.252 and 1.387 ms against retained medians of 1.243, 1.253, and 1.316 ms. These results leave the retained kernel near its memory-bandwidth ceiling without speculative scheduling or prefetch changes.

Expert reuse changes that conclusion for duplicate routes. During submit, the mixed path groups valid final expert or host-slot IDs only when a multi-token batch contains a duplicate. Grouped pass one walks each expert's output rows and computes up to four routes with one packed Q4_K decode; grouped pass two does the same for Q5_1 down rows, stores each route's weighted result in a disjoint H-tile slice, and reduces those slices in the original token/top-k order. The width-one case calls the existing single-route dot, and an all-unique batch keeps the existing route-major path, so reuse metadata adds no work to the normal fallback. Invalid and inactive IDs remain outside the groups and contribute zero as before. The graph replay test changes both IDs and activations, exercising fresh grouping and route reduction on every replay.

The controlled Qwen3.8 layer-0 matrix used the same 64-expert working set, H=2560/I=640, top-k=10, eight pinned Zen 2 workers, and fixed 8-warmup/200-sample runs. Stage-1 grouping improved the shared bs=4 case from about 4.04 ms to 3.32-3.37 ms. The first multi-dot version reached 2.65-2.68 ms. After restoring the width-one fallback, a separate 60-sample confirmation run measured 2.385 ms for the final width-aware version. That confirmation was about 29% below the stage-1 median and 41% below the original shared baseline, but it is a confirmation rather than a synchronized A/B. Its logical packed-weight rate was 51.52 GB/s, but only 12.88 GB/s of unique packed weights were fetched because one expert served four routes. The bs=2 shared case was 1.690 ms, 36.35 logical GB/s, and 18.18 unique GB/s. These unique rates are below the earlier roughly 25 GB/s physical bandwidth boundary, so compute-side ideas that were neutral for one route may have value when a decoded row feeds several activation vectors.

The final matrix retained the width-aware Q4_K and Q5_1 multi-dot kernels for duplicate groups, with the existing single-route dot used for singleton groups and all-unique batches. A separate width-four, two-accumulator Q5_1 multi-route branch was rejected: controlled bs=4 shared measured 2.487 ms versus a 2.396 ms retained baseline (3.8% slower), and pairs measured 3.337 versus 3.304 ms (1.0% slower), while unique routing was within noise. This is separate from the earlier single-route Q5_1 experiment described above. The benchmark is reproducible with `docs/investigations/benchmark_cpu_moe_reuse.py`; its report includes both logical route bandwidth and unique physical bandwidth so those two cases are not conflated.

A later Q4_K multi-route experiment decoded the low and high nibbles from each packed 32-byte weight vector together, intending to remove the duplicate packed-weight load. It passed the numerical guard but was rejected after two counterbalanced 120-sample runs: bs=2 shared regressed from 1.703 to 4.572 ms, bs=4 pairs from 3.300 to 9.014 ms, and bs=4 shared from 2.381 to 5.149 ms. The added activation work and register pressure cost more than the saved load on Zen 2, so the original width-aware loop remains in use.

The Q8_1 activation preparation now has an AVX2 implementation with scalar fallback and a narrow `FREETOKEN_CPU_MOE_Q8_1={scalar,avx2}` override for controlled tests. It widens BF16 values, finds each 32-value block maximum, rounds away from zero to preserve the existing `std::lround` behavior, and packs the GGML Q8_1 bytes. The direct 896-block helper benchmark measured 0.1246 ms scalar versus 0.0189 ms AVX2 over 200 samples, and the scalar/AVX2 bytes matched exactly, including signed half ties and zero blocks.

The real Qwen3.8-Flash-Next-Q4_K-v3 checkpoint was then tested with H=2560, I=640, top-k=10, 8 workers, 64 resident experts, and 120 measured samples per case. Each run used eight warmups and rotated routes; two counterbalanced process runs used scalar then AVX2 and AVX2 then scalar. The benchmark printed the selected quantizer for every executor, confirming the override was sampled at construction. Results below are medians in milliseconds; paired means across the two orders are shown where useful:

| batch/pattern | scalar run A | AVX2 run A | scalar run B | AVX2 run B | paired AVX2 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| bs=1 unique | 1.287 | 1.250 | 1.234 | 1.283 | -0.5% |
| bs=2 unique | 2.424 | 2.410 | 2.360 | 2.496 | -2.5% |
| bs=2 one-duplicate | 2.374 | 2.392 | 2.486 | 2.468 | 0.0% |
| bs=2 shared | 1.814 | 1.692 | 1.730 | 1.680 | +4.9% |
| bs=4 unique | 4.846 | 4.848 | 4.707 | 4.870 | -1.7% |
| bs=4 one-duplicate | 4.818 | 4.864 | 4.857 | 4.842 | -0.3% |
| bs=4 pairs | 3.282 | 3.320 | 3.374 | 3.307 | +1.8% |
| bs=4 shared | 2.348 | 2.406 | 2.442 | 2.369 | +0.3% |

The end-to-end effect is therefore small on this bandwidth-bound real workload, with a repeatable signal only for bs=2 shared. The AVX2 quantizer remains retained because it is isolated, byte-compatible, has no consistent route regression in the counterbalanced runs, and removes most of the measured Q8_1 preparation cost. The reproducible command is:

```text
FREETOKEN_CPU_MOE_Q8_1=scalar|avx2 \
  python docs/investigations/benchmark_cpu_moe_reuse.py \
  --model /home/ajkerchum/models/Qwen3.8-Flash-Next-Q4/Qwen3.8-Flash-Next-Q4_K-v3.gguf \
  --samples 120 --warmups 8 --experts 64
```

## Cache-policy experiment

The current global cache uses timestamp LRU. Hybrid capped miss selection uses per-expert recency to choose which misses cross PCIe; global slot eviction remains LRU. A 380-token trace with 2,048 slots measured a 60.1% LRU hit rate, versus 63.7% for a static oracle, 44.6% for naive LFU, and 57.7% for a decayed-frequency policy. A larger 3,363-slot simulation predicted 71.8% for LRU and 79.0% for the static oracle. A practical simulation that learned for 128 tokens, then fixed 75% of the slots as a protected frequency set, produced a charged 75.3% overall hit rate and 77.7% after warmup.

Those trace results suggested trying frequency-ranked hybrid misses, but the first live result did not establish a win. On 191-token requests, recency had a 35.21 s median, compared with 35.67 s for the second recency control and 38.40 s for a short recency run; frequency measured 39.43 s. Disk-tier variance was substantial and the outputs diverged, so this was not a clean performance A/B. The runtime now exposes frequency-ranked capped misses behind `--moe-hybrid-fetch-policy frequency`; the default remains timestamp LRU for global slot eviction and recency-ranked capped misses.

The Qwen3.6 coding-session trace gives a more relevant workload. It used eight turns of one repository-oriented coding flow and produced 888 decode steps. The routing distribution had entropy 6.60, the top eight experts accounted for 24.9% of routed occurrences, and 193.5 of 256 experts were distinct on average. The hot-32 set had 45.6% Jaccard overlap between halves, and the experts learned during the first 128 steps accounted for 47.8% of later routed occurrences. In an offline protected-tier simulation, LRU reached 95.69% hit rate, an oracle protected set reached 98.40%, and a frequency protected set reached 96.80% overall and 96.45% after the warmup. Those figures describe a hypothetical protected resident tier; they do not describe the implemented hybrid policy, which only uses the frozen frequency ranking to choose which capped misses cross PCIe while the global GPU slot pool still evicts by LRU.

The implemented policy was also exercised on that workload. Live recency took 90.740 s and frequency took 89.957 s, effectively neutral at this sample size. Later repeated medians were 10.667 s for recency and 10.944 s for frequency. The workload therefore shows why frequency is worth testing during real coding sessions, while still leaving the default on recency until a longer controlled run establishes a stable advantage.

## Bounded host-slot CPU and hybrid decode

The CPU executor can now use the disk tier's bounded host pool instead of requiring the full 70.3 GiB expert bank to reside in RAM. A stream host callback admits the raw routed expert IDs, rewrites the pinned route buffer to host-pool slot IDs, and then submits the existing native CPU task. The native executor uses the pool capacity as its row bound while retaining the model's 512-expert count for route validation. The callback is retained for graph lifetime, preserves negative inactive routes, deduplicates repeated expert IDs, and parks admission failures for the scheduler to raise after replay.

Hybrid decode has an additional ordering constraint because its GPU-selected experts and CPU overflow experts share the bounded host pool. `ensure_experts_hybrid` first admits GPU misses and rewrites their copy source indices. Bounded hybrid then queues `copy_missing()` before the CPU admission callback, so the callback cannot evict a slot while PCIe is still reading it. CPU GEMV still overlaps the later GPU expert GEMM. The original submit-before-copy order remains in use for a resident full bank, where eviction is impossible.

The real-checkpoint graph test uses a two-slot pool, changes routes among experts 17, 42, 291, and 300, and forces cross-layer eviction while comparing with dequantized GPU reference output. The focused RX6800 suite reported 45 passes:

```text
~/.venvs/ft/bin/python -m pytest \
  tests/moe/test_hybrid_fetch.py \
  tests/moe/test_host_tier.py \
  tests/moe/test_cpu_moe_q4_0.py \
  tests/moe/test_cpu_moe_gguf_types.py -q
45 passed in 27.81s
```

An end-to-end smoke run used the Qwen3.8 checkpoint, 2,048 host slots, the automatic 3,363-slot GPU cache, `--moe-strategy hybrid --moe-hybrid-max-fetch 1`, one running request, and a 512-token sequence limit. Eager and graph modes both returned `PASS` for the same greedy request. After warmup, eager took 21.94s for 39 completion tokens. Two graph replays took 13.38s for 39 completion tokens and 13.75s for 41 completion tokens, or 2.91 and 2.98 completion tokens/s when dividing the full HTTP duration. These are smoke samples with variable reasoning-token counts rather than a plateau A/B; they establish working bounded hybrid replay and show that graph launch removal is material for this path.

A short graph-mode fetch-cap sweep did not justify moving away from the conservative default. Fetch cap 2 returned `PASS` in 13.72s for 41 completion tokens (2.99 tokens/s). Fetch cap 4 returned `PASS` in 14.60s for 39 tokens (2.67 tokens/s). Different caps can perturb greedy generation: some cap-4 and cap-5 runs ended after the reasoning field without visible content, while a separate cap-4 arithmetic prompt correctly returned `4`. No admission error occurred, and an audit found no route-index or host-slot ordering defect. Exact-text comparison across CPU/GPU splits is therefore not a stable numerical check; a per-layer merged-output comparison is the next useful diagnostic before changing the automatic split.

Automatic bandwidth selection also cannot currently use this checkpoint's measured CPU result. The GGUF loader reports the broad `q4_0` CPU format tag, while the bandwidth profile has no `q4_0` mapping and `benchbw` cannot construct those banks. It therefore falls back to a cap of one. A generic `pcie / cpu` fallback would also be wrong for this bounded path because its PCIe copy and CPU admission are serialized; only CPU GEMV and the later GPU GEMM overlap. Selecting a larger cap needs direct end-to-end or per-layer measurements rather than the resident-bank overlap formula.

An existing local Qwen3.6-35B-A3B Q4_0exp GGUF was compatible: `_expert_types` resolved all three expert tensors to GGML type 2 (Q4_0). The resolved runtime CPU executor reported `fmt=q4_0 H=2048 I=512 experts=256 layers=40 top_k=8`.

## Bounded end-to-end comparison

The RX6800 had no active benchmark or server before the run. Each mode used the same checkpoint and settings, with graph batch size 1, memory ratio 0.85, cache rate 0.05, and one request. Hybrid used explicit `--moe-hybrid-max-fetch 1` because `ft bench bw --model qwen3.6-moe` does not measure q4_0, and `--formats q4_0` is rejected by that command's supported-format list. These results validate output only; one request per mode is too small and differently warmed to establish a performance conclusion.

Offload server:

```text
~/.venvs/ft/bin/ft serve \
  --model /home/ajkerchum/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-Q4_0exp.gguf \
  --moe-strategy offload --moe-cache-rate 0.05 --memory-ratio 0.85 \
  --max-running-requests 1 --max-seq-len-override 512 \
  --host 127.0.0.1 --port 8201
```

Hybrid server, after stopping offload:

```text
~/.venvs/ft/bin/ft serve \
  --model /home/ajkerchum/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-Q4_0exp.gguf \
  --moe-strategy hybrid --moe-hybrid-max-fetch 1 \
  --moe-cache-rate 0.05 --memory-ratio 0.85 \
  --max-running-requests 1 --max-seq-len-override 512 \
  --host 127.0.0.1 --port 8201
```

Both received this same request with `temperature=0` and `max_tokens=64`:

```json
{"model":"Qwen3.6-35B-A3B-Q4_0exp.gguf","messages":[{"role":"user","content":"Say exactly PASS and nothing else."}],"temperature":0,"max_tokens":64,"stream":false}
```

Both returned `PASS` with identical reasoning text and 52 completion tokens. The one-run server logs reported 1.59 tok/s for offload and 2.96 tok/s for hybrid; HTTP durations were 4.09s and 4.35s. These are diagnostic samples only. After testing, both serving processes were stopped; ports 8199–8202 were clear and the RX6800 returned to 0% use with 25 MB allocated.
