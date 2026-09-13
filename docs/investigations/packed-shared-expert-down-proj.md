# Resolved: GGUF MMQ reads past the end of a row on a partial K tile

**Status:** root-caused, fixed, verified. The `FT_GGUF_DENSE_MODULES` mitigation is no
longer needed. Originally filed as "packed `mlp.shared_expert.down_proj` corrupts
memory"; that framing named the wrong tensor -- see "Why the mitigation worked".

## The bug

`mul_mat_q` (`mmq.cuh`) and `moe_q` (`moe.cuh`) walk a row in steps of
`blocks_per_warp = WARP_SIZE_GGUF / QI<type>` blocks, calling `load_tiles_*` with the
row base already advanced to `ib0`. The loaders then index
`bx0 + i * blocks_per_row + kbx` with `kbx` spanning a **full** tile, whatever is
actually left in the row. When `blocks_per_row % blocks_per_warp != 0`, the final tile
reads past the end of the row.

The guarded inner compute loop never consumes those values, which is why every
numerical comparison passes -- the read itself is the defect. It faults only when the
over-read crosses into an unmapped page, so whether a given model dies depends on
allocation layout, not on the weights.

Only the 32-element block types are affected (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0). For a K-quant
`kbx` is always 0, which is why those loaders take `ib0` unused.

Worked example, Q5_0 with `in_features=640`: 20 blocks per row, `QI5_0 = 4` so the step
is 8 blocks, giving offsets 0, 8, 16. At 16 only 4 blocks remain and 8 are loaded -- 4
blocks, 88 bytes, past the row.

## The fix

Thread `ib0` into the loader signature and clamp the block index to what is left:

```c
const int kbx_eff  = min(kbx,  blocks_per_row - ib0 - 1);
const int kbxd_eff = min(kbxd, blocks_per_row - ib0 - 1);
```

Lanes past the end re-read the last valid block instead of leaving the row; the compute
loop already ignores them. This mirrors the `i = min(i, i_max)` row clamp the same
loaders already use. Touches `ggml-common.h` (the `load_tiles_cuda_t` typedef),
`mmq.cuh` and `moe.cuh` (the two call sites), and `vecdotq.cuh` (the loaders).

## Verification

`tests/kernels/test_gguf_mmq.py` reserves HIP virtual address space, maps exactly the
tensor's pages, and leaves the next page unmapped, so an over-read faults deterministically.

| case | without fix | with fix |
|---|---|---|
| Q5_0, in=640 (`[6-440]`) | **fault** | pass |
| Q8_0, in=640 (`[8-680]`) | pass | pass |

Q8_0 at 640 passing either way is expected and is a useful control: `QI8_0 = 8` gives a
4-block step, which tiles 20 blocks exactly.

**The JIT does not rebuild on a `.cuh` edit.** `~/.cache/torch_extensions/py311_cpu/
freetoken_gguf_kernels/` kept a stale `.so` across source changes, so an A/B run without
clearing it compares a binary against itself. Delete that directory between arms; a
clean rebuild is ~90 s.

End to end on the RX 6800, Qwen3.8-Flash-Next Q4 (v3), **no** `FT_GGUF_DENSE_MODULES`:
correct answers with natural stops, 0 illegal accesses, 3.24 tok/s -- matching what the
mitigation used to deliver.

## Why the mitigation worked, and why the name was wrong

`ffn_down_shexp` is Q8_0 at `in_features=640`: 20 blocks, 4-block step, remainder 0. It
**never over-read**. The tensors that do, in the same checkpoint:

| tensor | type | in | blocks | step | remainder |
|---|---|---:|---:|---:|---:|
| `output_hc_up` / `hc_*_up` | Q5_0 | 320 | 10 | 8 | **2** |
| `ffn_down_exps` (MoE bank) | Q5_1 | 640 | 20 | 8 | **4** |
| `ffn_down_shexp` | Q8_0 | 640 | 20 | 4 | 0 |
| `ssm_out` | Q5_0 | 6144 | 192 | 8 | 0 |

Forcing `down_proj` dense removed one allocation from the pool and shifted everything
after it, so the *other* over-reads landed on mapped pages. It masked the fault rather
than avoiding it -- which is also why the v2/v3 recipes appeared to fail for unrelated
reasons and why "the same type and shape works in the Q8_0 checkpoint" looked like a
contradiction. It was never about that module.

A superseded revision of this document concluded that the v3 failure was numerical
instability in MMQ's Q8_1 activation quantization, on the strength of per-layer captures
showing the packed Q8_0 down projection locally correct (0.33%-1.64% relative error) with
NaN appearing before layer 44. Those measurements were sound; the inference was not. The
kernel fix alone makes v3 correct with the module packed, which activation-quantization
error could not explain.

## Leftovers

* `tests/kernels/test_gguf_mmq.py` must `torch.cuda.synchronize()` before unmapping --
  tearing down memory a queued kernel still references faults the context, and every
  later test then fails for that reason rather than its own.
* Unrelated and still open: `tests/kernels/test_mxfp8_linear.py` fails on gfx1030 with
  triton `OutOfResources` (LDS limit), and it poisons the HIP context, so a full
  `pytest tests/kernels` run cascades. Run with `-x`, or skip that file, to see real
  results.
* Unrelated: one reasoning-model prompt returned coherent `reasoning_content` with empty
  `content` at `finish_reason: stop`. Parser/template edge case, not this bug.
