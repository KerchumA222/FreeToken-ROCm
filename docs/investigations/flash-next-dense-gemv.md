# Flash-Next decode: two rocBLAS GEMVs were half the GPU time

**Result:** plain decode on the RX 6800 disk-tier service went from 19.6 to 34.0 tok/s on
the short prompt, and from 14-16 to 26.4 tok/s on the long one. Measured 2026-09-24,
`Qwen3.8-Flash-Next-IQ2_S-Q2SYM`, the service in
[disk-tier-expert-prefetch.md](disk-tier-expert-prefetch.md).

## Finding it

Host-callback timing (`FT_TIER_STATS` now logs time inside the admission host nodes)
showed that disk and file-cache reads were only 13-20% of a token: 6.5 ms of 52 ms short,
14 of 70 long. An eager profile
(`FT_PROFILE_DECODE=200,16,<path> FT_PROFILE_SHAPES=1`) found the rest:

| op | calls / token | us / call | ms / token |
|---|---:|---:|---:|
| `aten::mm [1,6144] x [6144,2560]` (GDN `ssm_out`) | 36 | 536 | 19.3 |
| `aten::mm [1,2560] x [2560,512]` (MoE router) | 48 | 83 | 4.0 |

54% of GPU time was these two fp16 rocBLAS GEMMs, run as large Tensile tiles at M=1.

- **`ssm_out` was served dense.** The GGUF stores GDN value heads in ggml's tiled order,
  so the reader permutes `ssm_out`'s input columns. IQ2_XS blocks are 256 wide and a head
  is 128, so the packed bytes could not be permuted, and the weight was dequantized to
  fp16. It now stays packed in the file's column order and the module gathers its
  6144-wide input into that order (`linear_attn.out_in_perm`, one `index_select`).
- **The router went through `nn.Linear`.** `Qwen4ExpMoE` now uses the Triton small-M
  GEMV, like the Qwen3.5 MoE router.

## Numerics

Against the previous plain path, over the divergence harness's prompts:

| change | median | p99 | max |
|---|---:|---:|---:|
| router only | 0.094 | 1.17 | 1.81 |
| router + packed `ssm_out` | 0.19 | 3.31 | 4.86 |

These are drifts in the top-1 logit. The packed path quantizes `ssm_out`'s input to
q8_1, as llama.cpp does for every IQ2_XS matmul, on top of a 2-bit weight. Output is
coherent. `FT_GGUF_DENSE_MODULES=linear_attn.out_proj` restores the dense weight.

## Consequence for MTP

Compute is ~1.7x faster, so a verify round's extra expert reads now weigh more:

| tok/s | short | long |
|---|---:|---:|
| plain | 34.0 | 26.4 |
| MTP, adaptive depth up to 3 | 41.2 | 22.3 |
| MTP depth 1 | 39.1 | 22.4 |

A depth-1 round spends ~31 ms on disk, against ~6 ms for a plain token.

The selector (`speculative/depth.py`) now also considers depth 0, a plain step. On the
long run it still settles on depth 2 (22.4 tok/s): inside the MTP configuration a plain
step costs 52.5 ms, not a plain server's 38 ms. Three things add to it:

- the draft head still runs, to keep a draft ready;
- verify rounds churn the expert cache (13 ms of disk against ~6);
- the head and the verify graphs take ~130 GPU cache slots.

On this model, disk-tier MTP pays on the short, repetitive run (41.1 against 34.0) and
not on the long run (22.4 against 26.4).

A plain step whose next round is also plain now skips the draft: the head only refreshes
its attention state (`Qwen4ExpMTPHead.refresh_kv`: KV and QSA index keys, no MoE, mixer
or lm_head), so a later draft still attends to every position. Long run 22.4 -> 24.0
tok/s (short 41.0). A depth-0 step inside the MTP configuration now costs 47.9 ms,
against 52.5 before and 38 in a plain server. What remains is mostly cache churn from the
verify rounds in between (10.8 ms of disk against ~6).

## Second pass: small GEMVs, IQ2_XS dot product, launch floor

Measured with `FT_PROFILE_DECODE` (`FT_PROFILE_SORT=count` for a table by call count) and
a per-shape MMVQ microbenchmark (random packed weights, M=1, inside a CUDA graph).

- **Every dense M<=4 linear on HIP now uses the Triton small GEMV**
  (`kernel/triton/small_gemv.dense_linear`), not rocBLAS. This covers the QSA indexer's
  BF16 `index_qk_proj`, at 85 us a call under rocBLAS.
- **IQ2_XS dot product (`vec_dot_iq2_xs_q8_1`): dp4a with a carry-free sign negate.**
  ROCm has no packed-byte compare or subtract, so the signs come from a 16-entry byte-mask
  table and `(g ^ s) + (s & 0x01010101)`. That is exact because IQ2_XS grid bytes are
  never 0. Outputs are bit-identical to the vendored scalar version:

  | shape (M=1) | before | after |
  |---|---:|---:|
  | GDN in, 6240 x 2560 | 32.2 us | 20.8 us |
  | ssm_out, 2560 x 6144 | 29.9 us | 19.2 us |
  | attention q+k, 12800 x 2560 | 49.9 us | 37.9 us |
  | shared expert gate/up, 1280 x 2560 | 11.1 us | 9.2 us |
  | hyper-connection down, 320 x 10240 | 12.3 us | 10.4 us |

  Plain decode, short prompt: 35.3 -> 36.7 tok/s.
- Rows per MMVQ workgroup (`GGML_CUDA_MMV_Y` 1/2/4/8) made no difference.
- **Launch floor.** A trivial kernel costs 2.8 us inside a HIP graph on the RX 6800, and a
  Flash-Next decode token runs ~2,800 kernels. Of those, 531 are `quantize_q8_1`, one per
  packed matmul, and ~900 are the hyper-connection mix/combine chain (twice a layer).
- **Split-K MMVQ for thin matrices (4 waves per row) was tried and reverted.** Hyper-
  connection down went from 10.4 to 9.5 us, and `ssm_out` got slower (19.2 -> 23.6 us).
  Thin GEMVs sit at a two-kernel floor (quantize + GEMV), not at their arithmetic.
- **Disk-tier admission host nodes: ~66 us each, 48 a token (~3.2 ms, ~12%).** Measured in
  a HIP graph (`kernel + D2H/H2D copies` 11.9 us, `+ empty host node` 62.5 us per layer).
  A graph cannot skip the node on a layer without GPU misses (~31 of 48 on the short
  prompt). Stream-memop waits are not captured on this HIP (the PLE disk backend probes
  them and falls back to launch-gating). A device-side skip would therefore need a
  GPU-side spin on a pinned flag, answered by a host poller thread.

## Third pass: q8_1 activations from their producers

A packed GEMV quantizes its fp16 input to q8_1 (one launch per matmul, 531 a token).
Now:

- the C++ op splits into `ggml_quantize_q8_1` and `ggml_mul_mat_vec_q8`;
- a fused GGUF module with several packed runs (GDN in_proj, attention qkv) quantizes
  once;
- the hyper-connection rmsnorm, silu and gate-mix kernels and the shared expert's
  `silu_and_mul` write q8_1 blocks next to their fp16 output (`kernel/triton/q8_1.py`,
  bit-identical to the CUDA quantizer). The blocks ride on the output tensor
  (`layers/q8_act.py`) to the GEMV.

Plain decode is identical to before (zero logit drift over 1,024 tokens against
`FT_Q8_FUSE=0`). Short prompt: 36.7 -> 37.9 tok/s.

Same session (the long run's absolute numbers move with page-cache state between
sessions):

| tok/s | short | long |
|---|---:|---:|
| plain | 37.8 | 23.4 |
| MTP adaptive, depth up to 3 | 47.9 | 26.0 |

MTP now wins on both prompts. Admitting draft-only experts as least recently used
(`FT_SPEC_COLD_DRAFT`) was tried and removed: short 42.3, long 25.9. Accepted drafts
reuse those experts.

## Fourth pass: admission without host nodes

Captured decode graphs admitted each MoE layer's GPU-cache misses through a host function
node: ~66 us a layer (51 us of it the node itself), 48 layers a token, including the ~31
without a miss. Now (`kernel/admit_spin.py`, `csrc/admit/admit_spin.cu`, default on ROCm,
`FT_ADMIT_SPIN=0` to disable):

- a one-block kernel reads the miss count and returns at once when it is zero;
- otherwise it posts the miss list into coherent pinned memory (`hipHostMallocCoherent`)
  and spin-waits for a C++ poller thread, which runs the host tier's `ensure` under the
  GIL and writes the host slots back;
- the spin is capped at 5 s wall clock (`s_memrealtime`). On timeout the kernel flags an
  error, the step computes garbage and the engine raises, so it cannot hang the GPU.

A standalone harness measured a 5.3 us handshake (13.9 us with the Python callback),
against a 66 us host node. It also exercised the timeout path (poller stopped: the
kernel returns after the cap). Output is bit-identical to the host-node path (zero drift
over 1,024 tokens).

| tok/s (same session) | short | long |
|---|---:|---:|
| plain, host nodes | 37.8 | 23.4 |
| plain, spin admission | 40.9 | 24.6 |
| MTP adaptive, host nodes | 47.9 | 26.0 |
| MTP adaptive, spin admission | 50.0 | 25.4 |

## Prefill

A 2k-token prefill forward (eager profile, `FT_PROFILE_PHASE=prefill`) was ~19 s of GPU
time:

| part | GPU s | share |
|---|---:|---:|
| dense IQ-type linears through MMVQ, 6 rows a launch (no MMQ kernel for IQ quants) | 6.7 | 35% |
| routed experts through the per-(token, expert) `moe_vec_q` kernel | 5.1 | 27% |
| expert staging copies host -> GPU | 4.5 | 23% |
| QSA sparse attention | 1.5 | 8% |
| GDN chunked recurrence | 0.9 | 5% |

Fixes:

- **Chunk size.** `--moe-cache-auto` never budgeted for the prefill chunk: on the
  disk-tier service a ~2k-token prompt OOMed (8192-token chunk, ~0.2 GB headroom). The
  engine now probes activation growth with dummy prefills and caps the chunk. With a disk
  tier it also reserves VRAM for prefill (default 2 GB, at most 1/8 of the card). Every
  chunk re-streams the experts it routes to, so the chunk size is the prefill speed.
- **Dense IQ linears** with more than 64 rows dequantize once and run one GEMM.
- **Routed experts** with at least 16 rows per routed expert run as grouped GEMMs: groups
  of up to 8 experts of similar load are dequantized (IQ2_XS by the CUDA dequantizer,
  Q2_0/Q2_0_SYM by a new Triton one) and batched with `torch.bmm`, and results accumulate
  into an fp32 [T, H] buffer.

| prefill tok/s | ~630 tokens | ~2.5k | ~7.5k |
|---|---:|---:|---:|
| before (384-token chunks at 0.96) | ~43 | OOM | OOM |
| chunk fit + 2 GB reserve | ~50 | 107 | 134 |
| + dequant GEMMs + grouped experts | 58 | 149 | 283 |

Decode is unchanged (40.7 tok/s). Prefill numerics move from q8_1-activation GEMVs to fp16
GEMMs. Against the previous path, first-generated-token logits drift: median 0, p99 3.8.

Then:

- The disk-tier prefill fill gathers experts the GPU slot cache already holds device to
  device. Disk reads per 8k chunk fell from 4-9 GiB to 1.3 GiB.
- The grouped gate_up GEMM runs as `W @ x^T`. On gfx1030, rocBLAS runs `bmm(x, W^T)` over
  the dequantized row-major weight at ~4 TF/s and `bmm(W, x^T)` at ~9-12 TF/s; down, at
  K = 640, is already fast as `x @ W^T`.

| prefill tok/s | ~630 tokens | ~2.5k | ~7.5k |
|---|---:|---:|---:|
| + GPU-resident experts + flipped gate_up | 57-60 | 166-168 | 305 |

Where a prefill spends its time now:

- **~630 tokens:** ~11 s wall, 4.9 s of GPU. The chunk routes to ~95% of every layer's
  experts, so it moves ~25 GB (disk / page cache -> pinned -> PCIe); GPU-side staging
  copies alone are 2.0 s. Grouping experts at 4 rows each instead of 16 did not help
  (56.5 vs 59.7 tok/s).
- **~7.5k tokens:** ~20 s of GPU (eager). Grouped expert GEMMs are 32% (~115 GFLOP/s
  before the flip), and QSA sparse attention is 25% (the per-row split-K decode kernel
  runs for every prefill row, ~100 GFLOP/s).

## QSA prefill attention as batched GEMMs

The per-row QSA kernel (`_qsa_sparse_paged_gqa_splitk_kernel`) runs one program per
(query row, KV head), each gathering that row's ~2k selected tokens. On a 7.5k-token
prefill it was 25% of GPU time (404 ms a layer). The selections of neighbouring rows
nearly coincide. Dumping one prefill's selections (`FT_QSA_DUMP=<path>`), with a row
selecting ~55 of its visible 64-token blocks:

| rows per tile | union of blocks | K/V loads / | extra work |
|---:|---:|---:|---:|
| 4 | 57.7 | 3.8x | +5% |
| 8 | 58.5 | 7.5x | +6% |
| 32 | 59.0 | 29.9x | +7% |

- **A Triton tiled kernel** (5 rows x 12 heads per program, exact per-row token masks) was
  correct and no faster (421 vs 404 ms). It scaled with 1/tile rows because on gfx1030
  (no matrix units) Triton's `tl.dot` runs ~0.9 TFLOP/s: compute-bound, not load-bound.
- **As rocBLAS batched GEMMs** (`qsa_gemm_attention`), with tiles of 32 rows, 8 tiles a
  batch:
  - gather the union's pages once per KV head, straight into GEMM layout;
  - one Triton row kernel does scale, the per-row token mask and softmax in fp32;
  - PV, then `index_copy_` back.
  - Result: 119 ms a layer (3.4x), max abs error 0.002 vs the per-row kernel (fp16
    scores). Permuting a gathered [page, token, head, dim] block had run at ~26 GB/s and
    cost more than the GEMMs; per-head gathers fixed that.

Prefill batches of 256+ rows use it (`FT_QSA_GEMM=0` disables). Prefill tok/s:
~630 tokens 61, ~2.5k 164, ~7.5k 369 (from 305).

## Prefill: layout work

- **MoE transposes in Triton** (`kernel/triton/moe_prefill.py`). The flipped gate_up GEMM
  needs its gathered rows transposed and its result transposed back. torch's strided
  copies ran at ~26 GB/s (21% of a 7.5k-token prefill). A gather-transpose (6x faster
  than torch's) and a silu_and_mul that reads the transposed GEMM output replace them.
  7.5k 369 -> 383 tok/s, 2.5k 164 -> 179.
- **Dense GEMMs by row count** (`layers/gguf._dense_gemm`). On gfx1030 rocBLAS runs
  `x @ W^T` at 4-6 TF/s for 64-1024 rows, but `W @ x^T` (x^T as a view) at 15-25 TF/s at
  every size, and `x @ W^T` at ~25 TF/s from 2k rows up. Below 2k rows the dequantized
  path runs `W @ x^T` and a Triton transpose.
- **Q4_K and the other MMQ types** dequantize once and use that path from 256 rows: MMQ
  ran ~12 TF/s. 2.5k tokens 179 -> 190 tok/s.

| prefill tok/s | ~630 | ~2.5k | ~7.5k |
|---|---:|---:|---:|
| start of the day (0.96 memory ratio) | ~43 | OOM | OOM |
| now | 61 | 190 | 385 |

## Prefill: overlapping the disk reads

The disk-tier log now reports read and chunk time per prefill chunk. Reads were not
overlapped: the look-ahead layer's experts were read on the main thread before the
current layer's expert GEMMs were queued, so the GPU idled behind the host.

| prompt | chunk | reading |
|---|---:|---:|
| ~630 tokens | 10 s | 7.5 s |
| ~2.5k | 13 s | 7.5 s |
| ~7.5k | 19 s | 7.8 s |

The look-ahead fill now splits into:

- a plan on the main thread (host-tier bookkeeping; `claim_free` claims pool slots);
- a read on a worker thread (`fill_claims` and the misses into the pinned stage), after
  the previous copy out of that stage has retired;
- the H2D/D2D copies, queued when the layer is waited on.

Output is byte-identical to the synchronous path (`FT_PREFILL_ASYNC=0`). Short and
mid-length chunks now take as long as their reads.

| prefill tok/s | ~630 | ~2.5k | ~7.5k |
|---|---:|---:|---:|
| synchronous reads | 61 | 190 | 385 |
| async look-ahead reads | 71 | 256 | 485 |

These benchmarks run with an empty GPU expert cache (the startup probe resets it, and
`max_tokens=1` requests never decode), so every chunk reads ~29.5 GB. With a warm cache,
the GPU-resident experts (~1/3) are gathered device to device.

## Prefill: GDN kernel tiles for RDNA2

The fla chunked GDN kernels carry NVIDIA-tuned tiles. On gfx1030 (no matrix units) the
same kernels run 1.3-3.3x faster with smaller tiles and more warps. At a 7.5k-token
prefill of Flash-Next's 48 value heads, per layer:

| kernel | NVIDIA tile | RDNA2 tile | time |
|---|---|---|---:|
| `chunk_fwd_o` | BK 128, BV 64, 4 warps | BK 32, BV 64, 8 warps | 17.4 -> 7.8 ms |
| `chunk_gated_delta_rule_fwd_h` | BV 32, 4 warps | BV 16, 8 warps | 15.8 -> 4.9 ms |
| `recompute_w_u` | 4 warps, 3 stages | 2 warps, 1 stage | 8.6 -> 6.8 ms |

Outputs are identical: the divergence harness trace is byte-identical. They apply on
gfx103x only (`fla.utils.is_rdna2`, from Triton's driver target, so CUDA is not
initialized at import). The `SGLANG_GDN_CHUNK_H_*` env knobs still override
`delta_h`. Prefill at ~7.5k tokens 485 -> 545 tok/s; shorter prompts are read-bound.

## Where a long prefill stands (7.5k tokens, one chunk)

~13-14 s wall with ~9 s of overlapped reads; 12.3 s of GPU time (eager profile):

| part | GPU s | note |
|---|---:|---|
| grouped expert GEMMs | 3.4 | ~35 TFLOP (48 layers x 0.74), ~10 TF/s |
| expert staging copies | 3.1 | on the copy stream, mostly overlapped |
| dense GEMMs | 2.2 | ~37 TFLOP, ~17 TF/s |
| GDN | 0.8 | |
| gathers / index_add / sort | ~1.0 | |

The GEMMs are near what rocBLAS reaches on this card (15-27 TF/s); 512 experts at ~150
tokens each keep the expert GEMMs small. Up to ~2.5k tokens a chunk is bound by its
~29.5 GB of expert reads (~3.5 GB/s from page cache + virtual disk). More read threads
(16, 32) did not help.

Grouped down projection as `W @ a` (kept, `FT_MOE_DOWN_FLIP`, default below 256 padded rows):
1.4x faster in isolation at ~150 rows per expert and 7.5% faster per MoE layer in a 7.5k-token
microbench (99 vs 107 ms), but no end-to-end change on this VM (524 vs 542 tok/s, 13.2-14.3 s
chunks either way): shaving GPU time off the expert GEMMs does not shorten a chunk that waits
on staging and reads. Kept for hosts with faster storage or more page cache. Greedy output on
7.5k/2.8k-token chats differs from the old path no earlier than the old path differs from
itself run to run (atomic fp32 accumulation in both); `tests/moe/test_grouped_down_flip.py`
checks both layouts against a per-token reference.

## Prefill tier fills as DMA, and a warm GPU cache (2026-09-25)

The disk-tier prefill fill moved staged and pooled expert rows into the GPU buffer with the
zero-copy index kernel. In isolation it and DMA both reach ~25-28 GB/s on this link, but
during prefill the kernel's ~3 s per 7.5k chunk runs on compute units beside the GEMMs.
The fill now issues `copy_` DMA per run of consecutive rows (`FT_TIER_DMA`, default on;
more than 128 runs per bank falls back to the kernel). tok/s at ~630 / ~2.5k / ~7.5k:

| | 630 | 2.5k | 7.5k |
|---|---:|---:|---:|
| cold GPU cache, kernel | 70.4 | 259 | 523 |
| cold GPU cache, DMA | 68.7 | 266 | 566 |
| warm GPU cache, kernel | 86.3 | 301 | 585 |
| warm GPU cache, DMA | 83.7 | 304 | 582 |

"Warm" is `bench_prefill.py --warm-tokens 512`: a decode first fills the GPU expert cache
(7374 slots, ~30% of experts), and prefill then gathers ~26% of a long chunk's rows device to
device. Every earlier number here was cold, because `max_tokens=1` requests never populate
the cache; warm is what a second turn of a conversation sees and is 10-20% faster. The page
cache cannot be prewarmed usefully on this VM: 32.2 GiB of experts against 30 GB of RAM.

## Queue each layer's copies from the reader thread (2026-09-25)

A timeline of a warm 7.5k-token chunk (`FT_PROFILE_TRACE=1` exports the torch profiler
trace next to the table) showed the compute stream idle 5.7 of 16.5 s (stack profiling
inflates the chunk). The largest share, 1.6 s in 68 gaps of 10 ms or more, sat behind the
next layer's H2D copies: the look-ahead read finished on its worker thread, but its copies
were only queued from `wait_prefill_layer` when the layer's GEMMs needed them, so ~20-30 ms
of PCIe per layer ran with nothing to overlap. The reader thread now queues the copies
itself as soon as its read lands (`FT_FILL_IN_READER`, default on; its work runs under
`torch.inference_mode()`, which is thread-local).

Two smaller fixes on the way: the fill no longer builds its index tensors with blocking
copies when DMA does the copying, and the GPU-cache gather's indices are pinned. Neither
moved the numbers alone. Queuing from `release_prefill_layer` instead helped only +3%: by
then the next layer's read is rarely done, and blocking on it holds back the next layer.

Warm GPU cache, tok/s at ~630 / ~2.5k / ~7.5k: 88-92 / 306-312 / 596-613 before, 89 / 328 /
695 after (chunks ~12 -> 10.2-11.0 s). Greedy chat output on 7.5k / 2.8k-token prompts stays
coherent and diverges from earlier runs no sooner than runs of the old path diverge from
each other. The prefill log now also reports how long the main thread waited on reads.
