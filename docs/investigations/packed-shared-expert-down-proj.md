# Open bug: packed `mlp.shared_expert.down_proj` corrupts memory (qwen4_exp, Q4 GGUF)

**Status:** unresolved, mitigated. **Branch:** `merge/upstream-qwen4-exp` (also on
`feat/mtp-speculative`). **Mitigation commit:** `9792dbd`.

Serving `model.layers.*.mlp.shared_expert.down_proj` as packed GGUF blocks corrupts GPU
memory on a Q4_K_M requant of Qwen3.8-Flash-Next. Forcing that one module dense is
correct and stable. The root cause is **not** found, and the obvious explanations are
each contradicted by a measurement below.

## Symptom

Two faces of the same corruption, depending on where it lands:

1. **Illegal access.** Layers 0-8 of the first prefill complete; layer 9 dies in
   `mlp.shared_expert` with `hipErrorIllegalAddress`. Deterministic: always layer 9.
2. **Silent garbage.** With a slightly different quant recipe (v3, below), it does not
   fault -- generation runs to `max_tokens` with empty content every time. Same cause,
   corruption landing in mapped memory.

Failure 2 is the dangerous one: it looks like a quality problem, not a crash.

## Reproduce

Host `192.168.1.61`, RX 6800 (gfx1030), torch 2.10.0+rocm7.0, hip 7.0.51831.

```bash
cd ~/ft2/FreeToken-ROCm
export LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so
export HSA_OVERRIDE_GFX_VERSION=10.3.0
# omit FT_GGUF_DENSE_MODULES to reproduce; set it to make the bug go away
.venv/bin/ft serve \
  --model-path ~/models/Qwen3.8-Flash-Next-Q4/Qwen3.8-Flash-Next-Q4_K-v3.gguf \
  --port 1923 --moe-strategy offload --moe-host-cache-size 3200 \
  --moe-cache-size 1024 --dtype auto --max-seq-len 4096 --memory-ratio 0.92
```

```bash
curl -s http://127.0.0.1:1923/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","prompt":"Explain how photosynthesis works in plants, step by step.",
       "temperature":0,"max_tokens":40}'
```

* **Without** `FT_GGUF_DENSE_MODULES` -> empty content at 499 tokens (v3), or
  `hipErrorIllegalAddress` at layer 9 (v2 recipe).
* **With** `FT_GGUF_DENSE_MODULES=mlp.shared_expert.down_proj` -> correct output,
  natural stop, stable over many requests. 3.42 tok/s.

A 5-token prompt does **not** reproduce it; a 13-token one does. `_MMVQ_SAFE = 6`
(`python/freetoken/layers/gguf.py`) switches MMVQ -> MMQ at 7 tokens, so that is a
plausible split -- but see "MMQ is not it" below.

## The contradiction that kills the easy theories

The failing module in the v3 file is **Q8_0, shape 640 -> 2560**. That is the *same
type and the same shape* it has in the Q8_0 checkpoint, where serving it packed is
correct and has been for hundreds of requests (2.21 tok/s, right answers). So the
trigger is **not** the module, its type, or its shape on their own. Something about the
surrounding Q4 context matters.

## Ruled out (each measured, not argued)

| Hypothesis | Evidence against |
|---|---|
| The MoE kernel | `ggml_moe_a8_vec` matches a dequantized reference on the real Q4 banks for Q4_K and Q5_1, tokens 1/4/13/64/256/512, top_k=10, rel err <= 0.027 |
| The dense kernel | `fused_mul_mat_gguf` matches for **every** (type, in_features) pair in the checkpoint -- Q4_K, Q5_0, Q6_K, Q8_0 -- at m = 1, 4, 13, 64, incl. the failing layer's own tensors |
| The composition | `cat(2x Q4_K gate_up) -> silu_and_mul -> Q5_0 down` with layer 9's real tensors is finite and correct at m = 1, 13, 64 |
| MMQ vs MMVQ | the dense sweep above covers m=13 and m=64, which are the MMQ path, for every type present |
| The routed-union prefill staging (`96ea3d8`) | reproduces with `_ROUTED_STAGE_MAX_SHARE = 0.0` (whole-layer staging), on both short and long prompts |
| A race | reproduces under `HIP_LAUNCH_BLOCKING=1` + `AMD_SERIALIZE_KERNEL=3` |
| KV sizing | Q4 auto-allocated 233,088 KV tokens vs Q8's 70,528; capping with `--num-tokens 70528` still reproduces |
| Host pool geometry | same layer 9 at `--moe-host-cache-size` 1600 and 3200 |
| `ffn_down_shexp` being Q5_0 | v3 pins it to Q8_0 everywhere (verified: 48/48 Q8_0) and it still fails |
| Layer 9 being special in the data | layers 6 and 7 also have Q5_0 `ffn_down_shexp` (in v2) and *run* -- they presumably corrupt silently first |

## Where the fault is attributed

Under `AMD_SERIALIZE_KERNEL=3` the fault attributes to
`freetoken/kernel/triton/moe_shared_gate.py:65` (`shared_gate_sigmoid`) via
`models/qwen4_exp/moe.py`. That kernel is shape-asserted and fully masked, and its
shapes do not depend on the quantization -- it is very likely the first kernel to
*touch* an already-corrupted allocation, not the cause.

Sub-op tracing inside the MoE (sync after each step) puts the boundary precisely:
`[moe] gate` completes, then `self.shared_expert.forward(hidden_states)` faults.

## Code

* `python/freetoken/models/qwen4_exp/gguf.py` -- `gguf_module_types()` decides what is
  served packed; `FT_GGUF_DENSE_MODULES` (the mitigation) is the `_force_dense` check
  in `one()`, ~line 307.
* `python/freetoken/layers/quantization/linear/gguf.py` -- `MmvqGgufLinearKernel.apply`
  (line 44) and `GgufLinearMethod.create_weights`; the latter allocates
  `torch.empty(out, row_bytes(in_features, t), dtype=uint8)` per slot.
* `python/freetoken/layers/gguf.py` -- `fused_mul_mat_gguf` (line 119), the
  MMVQ/MMQ/dequant dispatch and `_MMVQ_SAFE`.
* `python/freetoken/kernel/csrc/gguf/gguf_kernel.cu` -- the vendored kernels.

## Leads worth trying

1. **Allocation adjacency.** The standalone kernel tests pass because a freshly
   allocated tensor has slack after it; in-model the weight may not. Place a packed
   weight so it ends exactly at the end of a mapped region and re-run the dense sweep --
   if an over-read is real, it should fault there and the over-read size is measurable.
2. **`create_weights` padding probe.** Over-allocate every packed slot by a page in
   `GgufLinearMethod.create_weights` and see whether the corruption disappears. This is
   a *diagnostic*, not a fix -- if it helps, it localises the bug to an over-read and
   gives its bound.
3. **Bisect the context.** The same module+type+shape works in the Q8_0 file. Requantize
   one module class at a time from Q8_0 toward Q4 until it breaks; that isolates which
   *neighbouring* tensor's type flips the behaviour.
4. **Check `row_bytes` alignment assumptions.** in=640 gives 440 B (Q5_0), 680 B (Q8_0),
   480 B (Q5_1) per row -- none 16-byte aligned. Compare against in=2560 (Q4_K, 1440 B)
   which is. The vendored kernels may assume an alignment the allocator happens to
   satisfy for some shapes.
5. **`compute-sanitizer`/`rocgdb`** on a single prefill would settle it directly; not
   attempted.

## Impact if unfixed

The mitigation costs the packed path for one module: ~157 MB of fp16 in VRAM across 48
layers, and that module's GEMM runs dense. With it set, Q4 is correct and 1.55x faster
than Q8_0 (3.42 vs 2.21 tok/s, 22.4 GB vs 41.4 GB read per 39-token completion). The
risk of leaving it un-root-caused is failure mode 2: a different checkpoint or recipe
could corrupt quietly rather than crash.
