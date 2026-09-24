# Decode and MTP speculative decoding vs llama.cpp (RX 6800)

**Status:** FreeToken decode is at parity with llama.cpp on this model and GPU, and MTP
speculative decoding is within 10% of it at every draft depth.

Measured 2026-09-24 on the RX 6800 (gfx1030, 16 GiB), ROCm 7.1 / PyTorch 2.11.0+rocm7.1.
Both engines served the same file, `Qwen3.6-35B-A3B-MTP-Q2K-Q3K.gguf` (13.65 GiB: Q2_K
routed gate/up, Q3_K routed down, Q8_0 everything else, inline MTP head), fully resident
in VRAM, one request at a time. The client ([`bench_spec_decode.py`](bench_spec_decode.py)) sends 8 greedy prompts x 256 tokens,
one warmup pass and two measured passes, and times decode as
`(completion_tokens - 1) / (last_text - first_text)` for both servers identically.

| Mode | llama.cpp | FreeToken | FreeToken / llama.cpp |
|---|---:|---:|---:|
| plain decode | 71.4 | 73.6 | 103% |
| MTP, 1 draft | 92.0 | 86.0 | 93.5% |
| MTP, 2 drafts | -- | 90.9 | -- |
| MTP, 3 drafts | 98.8 | 90.1 | 91.2% |

FreeToken acceptance: 0.877 / 1.471 / 1.930 drafts accepted per round at depth 1 / 2 / 3.
llama.cpp: 0.88 at depth 1 (mean length 1.88), mean length ~2.5-2.6 at depth 3. Greedy
speculative output matches plain decode up to near-tie argmax flips (see
[`mtp_divergence.py`](mtp_divergence.py)).

Serving commands:

```bash
# FreeToken
ft serve --model $M --max-running-requests 1 --max-seq-len-override 4096 \
  --cuda-graph-max-bs 1 --moe-strategy offload --moe-cache-auto --memory-ratio 0.95 \
  [--speculative-draft-tokens K]
# llama.cpp (the fork's default VBR KV cache cannot fit the MTP context; f16 KV, no auto-fit)
llama-server -m $M -ngl 99 -fit off -ctk f16 -ctv f16 -c 4096 -np 1 -fa on \
  [--spec-type draft-mtp --spec-draft-n-max K]
```

## Where the time went, and what changed

Start: plain 36.3 tok/s, MTP 27.3 tok/s (and MTP silently off after five requests).

| Change | Plain | MTP |
|---|---:|---:|
| GDN in_proj/out_proj served packed (were fp16 rocBLAS, 65% of decode GPU time) | 44.9 | |
| Router + shared-expert gate as one small-M Triton GEMV (rocBLAS Tensile tiles) | 61.5 -> 73.2 | |
| Same-type packed slots of a fused linear merged into one GEMV | 69.2 | |
| Verify as a decode-shaped step: attention rows as decode queries, recurrent GDN with per-row state for rollback, partial accept instead of reject-and-restage | | 27.5 |
| Verify captured in CUDA graphs (and the three bugs below) | | 74.6 |
| Greedy sample + draft head inside the verify graph; draft-head logits for the accepted row only | | 77.0 |
| MMVQ reads the weights once for several vectors (was once per vector) | | 84.1 |
| MTP head dense projections served packed; host trims | | 86.3 |
| Drafts of any depth (the head's chain runs inside the verify graph) | | 90.9 |

Bugs that were hiding the real numbers, all fixed:

- `is_greedy` treated temperature 0 with a model-default top_p as sampled: no drafts.
- Rejections and partial accepts leaked KV pages; `page_size > 1` freed only half of them.
- MTP checkpoints crashed at load with speculation off.
- The rollback-snapshot slot was never reclaimed from the prefix tree: speculation turned
  off after five requests.
- The detokenizer streamed a request's first token twice when one reply carried several.
- The draft head was fed the rejected draft instead of the correction.
- Graph capture recorded the warmup forward's hidden states for the draft head.

## Remaining gap

A depth-1 verify round costs ~1.6x a decode step here against ~1.46x for llama.cpp. The
per-round GPU work is ~20 ms, leaving ~1 ms of host work in the non-overlapped
speculative loop. The draft head's chained steps are single-request only; batched chains
need per-request KV ranges in the step's attention metadata.

## Qwen3.8-Flash-Next IQ2_S-Q2SYM on the disk tier

The same changes, on the disk-bound 61.3 GiB checkpoint with the recorded service settings
(`--moe-backend offload --moe-cache-auto --moe-host-cache-size 2048 --ple-backend disk
--memory-ratio 0.96`, `FREETOKEN_DISK_TIER_GRAPH=1`, graph bs 1), runner
[`benchmark_disk_tier_graph.py`](benchmark_disk_tier_graph.py). The control is the
pre-session tree (64efe1e) run the same day, since page-cache state moves these numbers
by several percent between days (the 2026-09-18 figures above were 14.10 / 15.88).

| Benchmark | 64efe1e (same day) | Current | Change |
|---|---:|---:|---:|
| 48-token repeated prompt, LRU | 18.56 | 19.64 | +5.8% |
| 512-token hash-table prompt, LRU | 13.16 | 14.30 | +8.7% |
| 512-token prompt, frequency 0.95 | 14.45 | 15.36 | +6.3% |

GPU expert slots: 8230 (control) vs 8202. Before the merged-slot upload fix the current
tree lost 170 slots to allocator holes and the long prompts gained only 0.6-1.7%.

MTP still does not pay here. With a symmetric-Q2_0 sidecar built for this checkpoint
(`mtp-Qwen3.8-Flash-Next-IQ2XSQ2SYM.gguf`, same patched llama-q4e command with the
sidecar as input), depth 1 measured 14.29 (short) / 11.16 (long) at 0.817 drafts
accepted per round: a verify moves two tokens' routed experts, which is the scarce
resource on this configuration. qwen4_exp also carries per-request PLE slot state that the
per-row GDN rollback does not snapshot, so it runs the older snapshot-and-restage verify
(eager, one draft).
