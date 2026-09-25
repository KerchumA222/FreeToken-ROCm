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
