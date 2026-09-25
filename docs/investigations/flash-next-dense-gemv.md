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
