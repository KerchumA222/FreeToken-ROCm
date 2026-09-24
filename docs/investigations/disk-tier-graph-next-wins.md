# Follow-up performance opportunities for disk-tier graph decode

**Status:** no single-stream code change was justified. A controlled two-stream run found an aggregate-throughput gain from using a bs=2 graph, with a clear per-request latency cost. A direct host-tier profile now shows that cold admission is dominated by concurrent disk reads; route conversion and cache bookkeeping are small, so no Python optimization was justified.

## Validated: serve two different requests together

Measured 2026-09-14 on the RX 6800 (gfx1030), ROCm 7.1 / PyTorch 2.11.0+rocm7.1, with Qwen3.8-Flash-Next-Q4_K-v3.gguf. The graph-on service used the same disk-tier settings as the bs=1 measurement: offload MoE, 2048 host slots, disk PLE, memory ratio 0.9, max sequence length 8192. Only `--max-running-requests 2 --cuda-graph-max-bs 2` differed. `FREETOKEN_DISK_TIER_GRAPH=1` was enabled and decode tracing was unset.

The two greedy prompts were different to avoid artificial shared routing locality. Each had one warmup and three serial samples. After one concurrent warmup, five measured pairs were sent together. Every response had 47 completion tokens. For each prompt, all serial and concurrent output strings matched byte-for-byte. The hashes were `7a7afd5ffd208cb6524d2a0a4a0f505761fa76e3c24dbd72d4501192bec7adea` (Paris prompt) and `83c125deae3aee9736fba3181579c7ad5403a6b54c73631761be93e264d95077` (soup prompt).

| Measurement | Serial, one request at a time | Two requests together |
|---|---:|---:|
| Paris prompt median | 4.985 tok/s | 3.246 tok/s |
| Soup prompt median | 5.502 tok/s | 3.246 tok/s |
| Aggregate for both streams | 5.230 tok/s | 6.492 tok/s median |
| Aggregate samples | -- | 6.537, 6.492, 6.462, 6.496, 6.331 tok/s |
| First-text to last-text per request | 9.23 s / 8.36 s | about 14.17 s each |

The aggregate rises 24.1% against the serial aggregate computed from the two serial medians. Each request takes longer: about 53.6% for the Paris prompt and 69.5% for the soup prompt. This is a capacity option for concurrent traffic, not a single-stream latency improvement. The measured aggregate interval spans from the first stream's first text to the last stream's last text and counts tokens after the first for each stream.

Reproduce with the normal serve command from `disk-tier-graph-capture.md`, changing the graph settings to `--max-running-requests 2 --cuda-graph-max-bs 2`, then run:

```bash
python docs/investigations/benchmark_disk_tier_graph_batch.py \
  --warmups 1 --serial-runs 3 --concurrent-runs 5 \
  --output /tmp/disk-tier-graph-bs2-results.json
```

The exact per-response timings, output strings, token counts, hashes, and five aggregate samples are retained in [`disk-tier-graph-bs2-results.json`](disk-tier-graph-bs2-results.json). The runner is [`benchmark_disk_tier_graph_batch.py`](benchmark_disk_tier_graph_batch.py).

## Ranked bs=1 leads

1. **Leave host-tier admission unchanged after profiling.** The direct profiler ([`benchmark_host_tier_admission.py`](benchmark_host_tier_admission.py)) ran on 2026-09-16 against Qwen3.8-Flash-Next-Q4_K-v3 on the RX 6800 (ROCm 7.1, PyTorch 2.11.0+rocm7.1), using 48 layers, 512 experts, top-k 10, 2048 host slots, and 8 workers. Three runs after the page cache was warm produced cold `ensure` sums of 189.04, 192.01, and 188.32 ms for one 48-layer token (median 189.04 ms; 3.92 ms per layer). The cold path read 960 expert-bank jobs, or 1,474,560,000 bytes (1.37 GiB), and the concurrent read wall time was 187.81-191.51 ms. Replaying the same routes made all layers hits: 0.283-0.452 ms total, with a 0.006 ms median layer. The route conversion helper alone took 2.06-2.14 ms per token (0.039 ms median layer), roughly 1% of the cold admission wall time. Disk I/O is therefore the clear dominant component; reducing list/dict churn cannot produce a material gain, and no runtime code change was made.

Reproduce the direct profile with the checkpoint's two bank types explicitly selected:

```bash
python docs/investigations/benchmark_host_tier_admission.py \
  --model /home/ajkerchum/models/Qwen3.8-Flash-Next-Q4/Qwen3.8-Flash-Next-Q4_K-v3.gguf \
  --experts 512 --layers 48 --top-k 10 --capacity 2048 --workers 8 \
  --gate-up-type 12 --down-type 7
```

The profiler fails before admission if any layer would be requantized under those bank types.

2. **Keep bs=2 as a throughput setting where request latency permits.** The controlled result above is already a concrete system-level win for two distinct concurrent requests. It comes from improved batching and utilization while the per-stream rate falls, so operators should choose it based on aggregate capacity rather than expecting faster individual responses.

3. **Leave greedy argmax alone for now.** A warmed RX 6800 microcheck of `torch.argmax` on a contiguous `[1, 248320]` fp16 row measured 18.8 microseconds per call including host launch and final synchronization (one 100-call timed sample). That is about 0.01% of the roughly 203 ms/token bs=1 decode interval. Capturing or replacing this op has little room to improve end-to-end rate; the much slower non-greedy sampling path is not exercised by this greedy workload.

The n-gram PLE table still moves only about 2.7 KB per token, so it remains a low-priority target. Host-pool capacity and read-worker-count changes were already measured and ruled out in `disk-tier-graph-capture.md`.

## Environment cleanup and limits

The batch-2 and profile services were launched as owned process groups and stopped after their runs; ports 8199 and 8200 were verified clear. The warmed bs=1 sanity sample with the temporary profiler setup measured 4.947 tok/s, but it is a single sample and is not a comparison. The host-tier profile used a direct microbenchmark and did not start a service. The only project change for this investigation is the benchmark script; no runtime code change was made.
