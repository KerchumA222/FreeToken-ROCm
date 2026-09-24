# Qwen3.8-Flash-Next IQ2_S on RX 6800

Measured 2026-09-17 on the RX 6800 (gfx1030), ROCm 7.1 / PyTorch
2.11.0+rocm7.1, with one running request and a batch-1 CUDA graph.

## Quantization

The source was the six-shard Q8_0 GGUF. The conversion used the model-specific
importance matrix and kept the source split:

```bash
/home/ajkerchum/src/llama-q4e/build/bin/llama-quantize \
  --allow-requantize --keep-split \
  --imatrix /home/ajkerchum/models/Qwen3.8-Flash-Next-GGUF/imatrix_unsloth.gguf_file \
  --tensor-type output_hc_down.weight=q4_k \
  /home/ajkerchum/models/Qwen3.8-Flash-Next-GGUF/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
  /home/ajkerchum/models/Qwen3.8-Flash-Next-IQ2_S/Qwen3.8-Flash-Next-IQ2_S.gguf \
  IQ2_S 8
```

Quantization took 9,322,280 ms (2h35m). The six output shards contain 1,224
tensors and occupy 77,187,764,800 bytes (71.9 GiB, 73,601.47 MiB reported by
the quantizer) at 3.49 bpw. The previous Q4 v3 file occupies 124.2 GiB, so the
new checkpoint is 42.1% smaller.

The FreeToken GGUF reader validated every shard range and resolved the model as
Qwen4Exp with 48 layers, hidden size 2,560, 512 routed experts, and top-k 10.
All emitted tensor types have FreeToken kernels:

| Type | Tensor count |
|---|---:|
| IQ2_XS | 552 |
| IQ4_NL | 194 |
| IQ3_S | 15 |
| Q4_K | 49 |
| Q5_K | 1 |
| BF16 | 24 |
| F16 | 1 |
| F32 | 388 |

Every layer's routed gate and up banks are IQ2_XS. Every routed down bank is
IQ4_NL because its width of 640 is not divisible by the IQ2 family's 256-value
block. The routed-expert bank is 42.77 GiB.

## Symmetric Q2 down projections

The 640-wide routed down projections do fit the llama-q4e fork's 64-value Q2_0
block. Its original codebook was rejected: the quantizer chose
`scale = max(abs(weight))` for levels `{-1, 0, 1, 2}`, so the fourth level was
effectively unused. Requantizing every routed down tensor with it raised the
short median to 18.5982 tok/s and the long median to 12.1905 tok/s, but all six
bounded quality probes terminated after only 1-8 completion tokens.

The retained experiment changes that 18-byte block to the symmetric four-level
codebook `{-3, -1, 1, 3}`. Four least-squares scale refinements per block reduce
reconstruction MSE by 56.2% on 40,960 sampled down-projection blocks versus the
initial `max(abs(weight)) / 3` scale. A metadata marker distinguishes these files
from ordinary type-42 Q2_0 files:

```text
freetoken.q2_0.codebook = symmetric_odd
```

FreeToken maps marked tensors to its internal `Q2_0_SYM` type while continuing
to decode unmarked Q2_0 with the fork's original semantics. The conversion used
the patched llama-q4e quantizer recorded in
[llama-q4e-q2sym.patch](llama-q4e-q2sym.patch):

```bash
/home/ajkerchum/src/llama-q4e/build/bin/llama-quantize \
  --allow-requantize --keep-split \
  --imatrix /home/ajkerchum/models/Qwen3.8-Flash-Next-GGUF/imatrix_unsloth.gguf_file \
  --tensor-type ffn_down_exps.weight=q2_0 \
  --override-kv freetoken.q2_0.codebook=str:symmetric_odd \
  /home/ajkerchum/models/Qwen3.8-Flash-Next-IQ2_S/Qwen3.8-Flash-Next-IQ2_S-00001-of-00006.gguf \
  /home/ajkerchum/models/Qwen3.8-Flash-Next-IQ2_S-Q2SYM/Qwen3.8-Flash-Next-IQ2_S-Q2SYM \
  COPY 8
```

The selective `COPY` conversion took 205.9 seconds. It changed exactly the 48
`ffn_down_exps.weight` tensors. Their packed bytes fell from 22,649,241,600 to
11,324,620,800; total model size fell from 73,601.47 MiB to 62,801.47 MiB
(61.3 GiB). The disk-tier expert bank is 32.23 GiB, and auto sizing at ratio
0.96 increased GPU residency from 6,201 to 8,230 slots (+32.7%).

The symmetric block has a dedicated gfx1030 `dp4a` MoE-vector kernel. CPU
reference dequantization and the RX 6800 kernel agree. This experiment currently
supports routed GPU offload decode; dense Q2 operations and CPU/hybrid Q2 expert
execution are outside its scope.

For the favorable repeated 48-token prompt, the median is 18.3456 tok/s, 1.8%
above the IQ2_S control. The 511-token hash-table prompt is the useful result:
LRU reaches 14.1045 tok/s, 40.0% above the 10.0767 tok/s IQ2_S LRU control. With
the 128-call frequency policy and a 0.95 protected fraction, three learned samples
span 15.6423-15.9109 tok/s with a 15.8833 tok/s median. That is 12.6% above
Q2_0_SYM LRU and 57.6% above the original long-prompt control.

The deterministic quality suite derives the correct syllogism result and correctly
diagnoses the mutable-default bug. The 767-token coding probes still exhaust their
completion budget during coherent reasoning, matching the pre-Q2 behavior. With a
1,536-token budget, the interval-merging probe stops normally after 1,248 tokens and
returns executable code; four functional cases pass, including the required touching
interval and no-input-mutation checks. Strict answer-only prompts can still stop after
reasoning without visible content. This is the previously observed template/finalization
issue rather than the immediate-EOS failure of the rejected codebook.

For a long, coherent single-user coding session, run the retained checkpoint with the
measured 0.95 frequency policy:

```bash
cd ~/ft-mtp
FREETOKEN_DISK_TIER_GRAPH=1 ~/.venvs/ft/bin/python -m freetoken.cli serve \
  --model ~/models/Qwen3.8-Flash-Next-IQ2_S-Q2SYM/Qwen3.8-Flash-Next-IQ2_S-Q2SYM-00001-of-00006.gguf \
  --host 127.0.0.1 --port 8199 --moe-backend offload \
  --max-running-requests 1 --moe-cache-auto --moe-host-cache-size 2048 \
  --moe-cache-policy frequency --moe-cache-frequency-warmup 128 \
  --moe-cache-frequency-protect-fraction 0.95 \
  --ple-backend disk --memory-ratio 0.96 --max-seq-len-override 8192 \
  --cuda-graph-max-bs 1
```

Use the default LRU policy when requests frequently switch between unrelated topics;
the frequency policy is measured to help after its per-layer learning window on a
stable workload.

Artifacts:

- [Q2_0_SYM short LRU](disk-tier-graph-iq2-s-q2sym-short.json)
- [Q2_0_SYM long LRU](disk-tier-graph-iq2-s-q2sym-long.json)
- [Q2_0_SYM frequency learning](disk-tier-graph-iq2-s-q2sym-frequency095-long.json)
- [Q2_0_SYM low-reasoning quality](qwen38-iq2-s-q2sym-quality-low.json)
- [Q2_0_SYM 1,536-token coding probe](qwen38-iq2-s-q2sym-merge1536.json)
- [Rejected original Q2_0 short](disk-tier-graph-iq2-s-q2down-full-short-rejected.json)
- [Rejected original Q2_0 long](disk-tier-graph-iq2-s-q2down-full-long-rejected.json)
- [Rejected original Q2_0 quality](qwen38-iq2-s-q2down-full-quality-rejected.json)

## Throughput

The control and candidates used the same greedy 48-token completion and the
same measurement definition: `(completion_tokens - 1) / (last text time - first
text time)`. Each reported median has four warmups and seven measured requests.
The service command was:

```bash
FREETOKEN_DISK_TIER_GRAPH=1 ~/.venvs/ft/bin/python -m freetoken.cli serve \
  --model /home/ajkerchum/models/Qwen3.8-Flash-Next-IQ2_S/Qwen3.8-Flash-Next-IQ2_S-00001-of-00006.gguf \
  --host 127.0.0.1 --port 8199 --moe-backend offload \
  --max-running-requests 1 --moe-cache-auto --moe-host-cache-size 2048 \
  --ple-backend disk --memory-ratio 0.95 --max-seq-len-override 8192 \
  --cuda-graph-max-bs 1
```

| Checkpoint / setting | GPU slots | Free after graph | Median tok/s | Relative to current Q4 |
|---|---:|---:|---:|---:|
| Q4 v3, ratio 0.90 | 3,363 | 1.21 GiB | 4.7052 | 1.00x |
| IQ2_S, ratio 0.90 | 5,651 | 1.20 GiB | 11.4316 | 2.43x |
| IQ2_S, ratio 0.93 | 5,926 | 0.72 GiB | 11.9015 | 2.53x |
| IQ2_S, explicit 6,000 slots | 6,000 | 0.44 GiB | 12.2351 | 2.60x |
| IQ2_S, explicit 6,060 slots | 6,060 | 0.42 GiB | 12.2568 | 2.60x |
| IQ2_S, explicit 6,090 slots | 6,090 | about 0.4 GiB | 12.7405 | 2.71x |
| IQ2_S, explicit 6,100 slots | 6,100 | about 0.4 GiB | 17.6646 | 3.75x |
| IQ2_S, ratio 0.95 | 6,109 | 0.40 GiB | 17.8700 | 3.80x |
| IQ2_S, ratio 0.96 | 6,201 | 0.24 GiB | 18.0164 | 3.83x |
| IQ2_S, ratio 0.97 | 6,292 | 0.08 GiB | 18.0359 | 3.83x |

The ten-slot transition between 6,090 and 6,100 is a cache-capacity cliff. The
extra residency removes recurring expert transfers; it is much larger than the
arithmetic differences between the kernel candidates below. Ratio 0.96 also
completed a 1,023-token request without an allocation failure. Ratio 0.97 buys
only 0.1% and leaves too little headroom to recommend.

The short repeated prompt is a favorable cache workload. A warmed 511-token
completion for a different technical prompt sustained 9.8838 tok/s at ratio
0.96. Long, diverse generations still churn through a working set larger than
the resident cache, so 18 tok/s should not be treated as an agentic-session
average.

Increasing the host pool from 2,048 to 3,360 IQ2 slots used 5.85 GiB instead of
3.56 GiB and measured 17.6796 tok/s. It did not improve steady decode or the
first two time-to-first-token samples, so the smaller host pool remains the
measured choice.

The opt-in frequency policy was tested with a 128-call learning window. Protecting
25% of slots was too small to repay its launch overhead: the 511-token hash-table
prompt measured 9.9057 tok/s against a fresh LRU control at 10.0767 tok/s. Raising
the protected fraction changed the result. A 75% share reached 10.3548 tok/s, 90%
reached 11.0642 tok/s, and the maximum safe 95% setting reached 11.1281 tok/s over
three additional steady samples. The best setting is 10.4% faster than LRU on the
stable learned workload. The nominal 95% request is safety-capped to 5,664 protected
slots and leaves 537 ordinary LRU slots.

The static set does not provide that gain after every topic change. After learning
the hash-table prompt, a related B-tree prompt measured 10.4107 tok/s versus an LRU
control at 10.3956 tok/s, effectively neutral. A more severe earlier transition from
the short benchmark into the technical prompt made the 25% policy 7.3% slower than
LRU. LRU therefore remains the general default. For a long, coherent coding session,
`frequency` with a 0.90 fraction is the more conservative measured experiment; 0.95
is the fastest same-topic setting.

A zero-protection frequency control measured 17.5307 tok/s. This isolates the
three extra per-layer observe/rank/mark launches at a 2.7% cost; the protected
residency recovered part of that overhead but not enough to beat LRU. Two fused
adaptive alternatives were also rejected. Temporary second-access protection
measured 12.0076 tok/s with a 32-token TTL; a one-token TTL recovered to 17.6957
tok/s on the short prompt and 9.9332 tok/s on the long prompt, still 1.8% and
1.4% below LRU. A bounded 25% protected segment with its own LRU oscillated from
12.12 to 17.88 tok/s and had a 15.9921 tok/s median. The adaptive runtime code
was removed after measurement.

An experimental near-static variant protected 6,144 of 6,201 slots and left only
57 for LRU. It measured 11.1330 tok/s with wider variance, no material improvement
over the safe 95% setting. The override was removed; rare experts still benefit from
a real dynamic pool.

Combining frequency observation, one-shot ranking, and sentinel refresh into one
post-LRU Triton kernel reduced four admission-related launches to two, but measured
11.0258 tok/s against the retained 11.1281 tok/s. The larger steady-state kernel cost
more than the removed launches on RDNA2, so the separate kernels were restored.

Raw measurements:

- [Q4 control](disk-tier-graph-q4-control-20260917.json)
- [IQ2 ratio 0.90](disk-tier-graph-iq2-s-ratio090.json)
- [IQ2 ratio 0.93](disk-tier-graph-iq2-s-ratio093.json)
- [IQ2 ratio 0.95](disk-tier-graph-iq2-s-ratio095.json)
- [IQ2 ratio 0.96](disk-tier-graph-iq2-s-ratio096.json)
- [IQ2 ratio 0.97](disk-tier-graph-iq2-s-ratio097.json)
- [IQ2 6,000 slots](disk-tier-graph-iq2-s-cache6000.json)
- [IQ2 6,060 slots](disk-tier-graph-iq2-s-cache6060.json)
- [IQ2 6,090 slots](disk-tier-graph-iq2-s-cache6090.json)
- [IQ2 6,100 slots](disk-tier-graph-iq2-s-cache6100.json)
- [IQ2 3,360 host slots](disk-tier-graph-iq2-s-host3360.json)
- [IQ2 LRU long-prompt control](disk-tier-graph-iq2-s-lru-long-control.json)
- [IQ2 frequency 25%, clean long-prompt learning](disk-tier-graph-iq2-s-frequency025-coldlong.json)
- [IQ2 frequency 25%, stale short-prompt learning](disk-tier-graph-iq2-s-frequency025-after-short-long.json)
- [IQ2 frequency 25%, short repeated prompt](disk-tier-graph-iq2-s-frequency025.json)
- [IQ2 frequency 0%, launch-overhead control](disk-tier-graph-iq2-s-frequency000.json)
- [IQ2 frequency 75%, clean long-prompt learning](disk-tier-graph-iq2-s-frequency075-coldlong.json)
- [IQ2 frequency 90%, clean long-prompt learning](disk-tier-graph-iq2-s-frequency090-coldlong.json)
- [IQ2 frequency 95%, clean long-prompt learning](disk-tier-graph-iq2-s-frequency095-coldlong.json)
- [IQ2 frequency 95%, additional steady samples](disk-tier-graph-iq2-s-frequency095-steady.json)
- [IQ2 frequency 95%, related-topic switch](disk-tier-graph-iq2-s-frequency095-related-switch.json)
- [IQ2 LRU related-topic control](disk-tier-graph-iq2-s-lru-btree.json)
- [Rejected near-static frequency cache](disk-tier-graph-iq2-s-frequency-nearstatic-rejected.json)
- [Rejected fused frequency bookkeeping](disk-tier-graph-iq2-s-frequency095-fused-rejected.json)
- [Rejected reuse TTL 32](disk-tier-graph-iq2-s-reuse-ttl32-rejected.json)
- [Rejected reuse TTL 1, short prompt](disk-tier-graph-iq2-s-reuse-ttl1-rejected.json)
- [Rejected reuse TTL 1, long prompt](disk-tier-graph-iq2-s-reuse-ttl1-long-rejected.json)
- [Rejected bounded reuse segment](disk-tier-graph-iq2-s-reuse-segment025-rejected.json)

## Quality probe

A deterministic bounded probe did not show a reasoning loop. Exact arithmetic
returned `391` for 17 times 23, and a syllogism returned the correct `NO`. The
mutable-default, interval-merging, and cache-concurrency traces identified the
correct bugs and algorithms, but requests capped at 384, 512, and 1,024 output
tokens ended while the model was still emitting `reasoning_content`. Repeated
full lines were absent or limited to code structure. The empty final answers in
those cases came from exhausting the completion budget, not the 8K context
window.

This probe is enough to reject catastrophic IQ2 corruption, but not enough to
claim Q4-equivalent coding quality. Coding clients need a larger output budget
or a reasoning policy that reserves room for the final answer. An exact-format
probe also stopped early with empty content, which may be a chat-template
transition issue and needs separate investigation.

## Rejected MTP decoding

The model's MTP sidecar was requantized to match the target's IQ2_XS gate/up and
IQ4_NL down banks. MTP also required graph replay to expose the captured hidden
state to the draft head; that graph correctness fix is retained. With one draft
token, residency fell to 6,089 slots and the same short benchmark measured a
median of 11.1068 tok/s versus 18.0164 tok/s without MTP, a 38.4% regression.
The sidecar therefore does not pay on this RX 6800 configuration.

- [Rejected IQ2 MTP benchmark](disk-tier-graph-iq2-s-mtp-rejected.json)

## Rejected kernel and launch candidates

Three gfx1030 candidates passed correctness but did not improve the end-to-end
plateau:

| Candidate | Candidate tok/s | Control tok/s | Decision |
|---|---:|---:|---|
| IQ2_XS scalar products -> packed `dp4a` | 17.6553 | 17.8675 | reverted |
| IQ4_NL scalar lookup -> AMD byte permute | 17.6376 | 17.8675 | reverted |
| two output rows per IQ expert block | 17.9858 | 18.0164 | reverted |

The first two ports reduce source-level instruction counts, but RDNA2 did not
benefit in this workload. The launch change was neutral. The benchmark artifacts
are retained for the rejected candidates; none of their runtime code remains.
