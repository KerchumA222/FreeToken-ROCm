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
