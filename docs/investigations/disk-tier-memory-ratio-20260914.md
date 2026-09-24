# Disk-tier graph memory-ratio comparison

**Measured:** 2026-09-14 on `ajkerchum@192.168.1.61`, Radeon RX 6800 (gfx1030),
ROCm 7.1 / PyTorch 2.11.0+rocm7.1, Qwen3.8-Flash-Next-Q4_K-v3.gguf. The result favors
`--memory-ratio 0.93` for this workload: it had the highest measured median. The
difference from 0.90 is small (1.7%) and the sample ranges overlap, so this is a setting
to validate in a randomized repeat, not evidence to change a default. At 0.95, the service
ran without OOM but retained only 0.41 GiB free after graph capture and measured slower.

## Method

Each arm started a fresh serial service with graph-on disk admission:
`FREETOKEN_DISK_TIER_GRAPH=1`, `--cuda-graph-max-bs 1`, offload MoE,
`--moe-cache-auto --moe-host-cache-size 2048`, disk PLE, max sequence length 8192, and
max running requests 1. Only `--memory-ratio` changed. `AMD_SERIALIZE_KERNEL`,
`FT_GGUF_BACKEND`, and `FREETOKEN_ROCM_DECODE_TRACE` were unset; the default HIP GGUF
backend was used.

The same greedy streamed request was run serially in every arm:

```
The capital of France is a city with museums, historic landmarks, and a river running through it. Explain the main attractions in one sentence.
```

Request options were `max_tokens=48`, `temperature=0`, `ignore_eos=true`, and streamed
usage reporting.
The runner performed three warmups followed by six measured requests per arm. Decode
throughput is `(completion_tokens - 1) / (last_text_time - first_text_time)`, with SSE
lines read at `chunk_size=1`; prefill is excluded. All requests returned 47 completion
tokens and the same output string. Its SHA-256 in every arm was
`7a7afd5ffd208cb6524d2a0a4a0f505761fa76e3c24dbd72d4501192bec7adea`.

| `--memory-ratio` | Auto GPU expert slots | Free VRAM before graph | Free VRAM after graph | Three warmups (tok/s) | Six measured samples (tok/s) | Median (tok/s) | Range (tok/s) |
|---:|---:|---:|---:|---|---|---:|---:|
| 0.90 | 3363 | 1.57 GiB | 1.21 GiB | 1.2922, 4.9116, 4.9232 | 4.9069, 4.9587, 4.8826, 4.9624, 4.8958, 4.9442 | 4.9255 | 4.8826–4.9624 |
| 0.93 | 3530 | 1.09 GiB | 0.72 GiB | 1.2676, 5.0312, 5.0268 | 5.0158, 4.9507, 4.9999, 4.9604, 5.0470, 5.0232 | 5.0079 | 4.9507–5.0470 |
| 0.95 | 3642 | 0.77 GiB | 0.41 GiB | 4.2134, 4.6082, 4.3456 | 4.5926, 4.6473, 4.5973, 4.6807, 4.5919, 4.6461 | 4.6217 | 4.5919–4.6807 |

Each service reported 15.95 GiB free before model loading. The disk host tier remained
2048 slots (5.86 GiB) in all arms. None of the runs OOMed. The free-VRAM margin at 0.95
is narrow; it is not a safe basis for increasing the ratio further.

## Interpretation

The 0.93 median is 1.7% above 0.90, while the measured ranges overlap. The 0.95 median is
6.2% below 0.90 and 7.7% below 0.93. Additional GPU expert slots therefore did not
improve this workload monotonically. The single highest observed sample was at 0.93
(5.0470 tok/s), and output identity was preserved across all 27 requests.

The arms were sequential, not randomized. A targeted GPU pytest ran after the 0.90 arm and
before 0.93; the disk page cache and GPU thermal/clock state were not reset between runs.
The 0.90 and 0.93 arms each had a slow first warmup (about 1.3 tok/s), while the 0.95 arm
started at 4.21 tok/s after earlier runs had warmed the compilation and disk paths. These
effects limit what can be inferred from the 1.7% edge. For this machine and workload, use
0.93 as the candidate setting when testing the best observed median, then repeat in
randomized order before treating the difference as stable. Keep the configured default
unchanged.

## Raw data

The runner saved every warmup and measured request, including timing, output text,
completion-token count, finish reason, and text hash:

- [0.90 raw samples](disk-tier-memory-ratio-090.json)
- [0.93 raw samples](disk-tier-memory-ratio-093.json)
- [0.95 raw samples](disk-tier-memory-ratio-095.json)

All arms used `benchmark_disk_tier_graph.py` with `--mode graph --warmups 3 --runs 6`.
The service was stopped between arms; ports 8199 and 8200 were clear after cleanup.
