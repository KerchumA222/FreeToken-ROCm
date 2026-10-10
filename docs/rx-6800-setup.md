# RX 6800 setup and decode tuning

This guide records a working FreeToken setup for a 16 GiB Radeon RX 6800
(`gfx1030`), focused on decode throughput (TG) for Qwen3.8 Flash-Next MoE
GGUF checkpoints. It is a practical starting point, not a guarantee that every
ROCm installation or checkpoint will reach the numbers below.

## Tested software and checkout

The Linux RX 6800 runs recorded in this repository used Python 3.11, ROCm 7.1
(one later profiling pass used ROCm 7.2), and PyTorch 2.11.0 built for ROCm
7.1. The card was reported as `gfx1030`. The RX 6800 setup is Linux; the
Windows ROCm setup documented elsewhere in this fork targets newer GPUs and
is not a substitute for these Linux steps.

Get the fork branch that contains the KV-cache and routing-bias features:

```bash
git clone --branch feat/kv-cache-q8_0 --single-branch \
  https://github.com/KerchumA222/FreeToken-ROCm.git
cd FreeToken-ROCm
uv venv --python 3.11
source .venv/bin/activate
```

Install a ROCm-compatible PyTorch 2.11 build and the ROCm runtime/toolchain
matching the host first. Follow AMD's instructions for the host's Linux
distribution ([ROCm for Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html),
[PyTorch on ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html));
the available wheel and supported-GPU matrix depends on the ROCm release. Do
not let a generic dependency install replace ROCm PyTorch with a CUDA or CPU
build. This checkout's `accel` extra includes CUDA-specific dependencies, so
do not install `.[accel]` into the AMD environment. Install the ROCm-compatible
runtime dependencies for this fork, then install the checkout with
`uv pip install -e . --no-deps --no-build-isolation`. Verify the runtime
before serving:

```bash
python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
rocminfo | grep -m 1 gfx1030
```

The Python check should show a ROCm (`+rocm`) PyTorch build, a non-empty HIP
version, an available device, and the RX 6800. `rocminfo` should list
`gfx1030`. If those checks fail, fix the ROCm/PyTorch environment before
debugging FreeToken kernels.

## Checkpoint choice

Two Flash-Next GGUFs have RX 6800 measurements in this repo:

- The existing three-shard IQ3_XXS checkpoint is the more conservative
  starting point if matching the previous model behavior matters.
- The [DASLab GSQ-RCO Coder GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
  IQ1_M variant has half as many experts per layer and is smaller overall. It
  can improve short-prompt TG and leave more expert capacity available, but it
  also changes the quantization and model behavior. The limited comparisons in
  [the evaluation](investigations/daslab-pruned-flash-next.md) do not establish
  broad coding-quality parity.

The serving command accepts a local GGUF path. For a sharded checkpoint, pass
the first shard (the filename containing `00001-of-...`). Keep all shards
together in the same directory.

## Recommended first-run profile

TurboQuant saves VRAM for longer KV histories, while a modest resident-expert
bias can avoid some offload misses. Start with 128k capacity and generation
bias 0.8; leave prefill bias at zero. This is the best balanced starting point
we have repeatedly used, not a claim that it preserves every model's quality.

```bash
MODEL="/path/to/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"

FT_ROUTE_BIAS=0.8 \
FT_ROUTE_BIAS_PREFILL=0 \
ft serve --model "$MODEL" \
  --moe-strategy offload \
  --moe-cache-auto \
  --kv-reserve-tokens 131072 \
  --max-seq-len-override 131072 \
  --memory-ratio 0.90 \
  --max-running-requests 1 \
  --cuda-graph-max-bs 1 \
  --max-prefill-length 8192 \
  --cache-type radix \
  --kv-cache-dtype turboquant_4bit_nc
```

Replace `MODEL` with the actual first-shard path. `--moe-cache-auto` divides
available VRAM between the expert cache and the requested KV floor. The
resolved cache sizes are printed at startup; check the log for both allocated
KV tokens and GPU expert slots. Keep the server at one active request when
measuring single-request TG.

For an initial performance reference, a warmed IQ3_XXS run with TurboQuant,
decode bias 0.8, prefill bias 0, and a 32,762-token prompt measured 19.37 tok/s
for TG on the RX 6800. That was one timed sample, so use it as an order-of-
magnitude check rather than a promised result. The DASLab IQ1_M checkpoint at
the same requested 128k capacity measured 16.18 tok/s after a 32,762-token
prompt in its recorded comparison; its short-prompt mean was higher. The
checkpoint formats and expert sets differ, so these are not a controlled
head-to-head speed ranking.

## Tuning for more decode speed

Change one setting at a time and compare the same checkpoint, prompt, context
length, output budget, and warmup procedure.

1. **Try a little more generation bias.** `FT_ROUTE_BIAS` is an additive
   router-logit margin in favor of GPU-resident experts, not a percentage. It
   changes which experts are selected; selected experts still receive weights
   from their original router logits. Try `0.8`, then `1`, and only then `2`
   if decode speed matters more than conservative routing. In the DASLab
   IQ1_M 256k sweep, bias 1 averaged 18.28 tok/s and bias 2 averaged 23.10
   tok/s after 32k input. The bias-2 quality check was only a small fixed set
   of elementary probes, not a broad coding evaluation. Higher bias can
   change answers; validate it on the actual agent workload before adopting
   it.

   ```bash
   FT_ROUTE_BIAS=1 FT_ROUTE_BIAS_PREFILL=0 ft serve ...
   ```

2. **Keep prefill bias separate from generation bias.** Start with
   `FT_ROUTE_BIAS_PREFILL=0`. It can alter prompt processing and has not shown
   a reliable TG improvement. A small prefill margin may be worth an A/B if
   first-token latency is the problem, but assess its quality separately.

3. **Trade context capacity against expert residency.** TurboQuant 4-bit
   (`turboquant_4bit_nc`) uses substantially less KV memory than 16-bit KV.
   Still, doubling the reservation from 128k to 256k reduced the automatically
   sized GPU expert cache from 2,061 to 1,404 slots in one RX 6800 setup and
   reduced TG. Increase both `--kv-reserve-tokens` and
   `--max-seq-len-override` only when the workload needs the larger window,
   and stay within the checkpoint's advertised context length. If you need
   less context, reducing both gives the auto-sized expert cache more room.

4. **Leave lookahead prefetch off initially.** `FT_PREFETCH` is opt-in. The
   tested bias-aware GPU and CPU prefetch variants reduced some cache misses
   but did not beat prefetch-off TG consistently; predictor work competes for
   an RX 6800 GPU that is already busy during decode.

5. **Treat the TurboQuant QSA prefill GEMM as experimental.** The normal QSA
   GEMM dispatch is enabled by default. The additional
   `FT_QSA_TURBOQUANT_GEMM=1` path had a short-prefill win in some measurements
   but no repeatable improvement around 32k. Leave it unset for a baseline;
   enable it only for a measured workload-specific A/B. It does not improve
   decode TG.

6. **Keep the serving workload simple while tuning.** Use one running request
   and graph batch size 1 for single-user decode tests. The default 8192-token
   prefill chunk is a sensible starting point; smaller chunks may reduce peak
   prefill memory at the cost of more chunk overhead. `--memory-ratio 0.90`
   leaves some VRAM headroom; raise it only if the resolved cache sizes are
   too small and the workload remains stable.

## Measure and validate

TG means decode tokens per second after prompt processing; it is different
from prefill throughput and time to first token. For a useful A/B:

- Use the same model files and exact prompt, with a fixed output-token count.
- Warm each server in the same way, then collect multiple timed trials.
- Avoid accidentally timing a prefix-cache hit when comparing prefill. Use a
  fresh server or a prompt that has not already been cached.
- Record the startup-resolved KV capacity and GPU expert slots, the two bias
  values, and the exact context length. Use `ft ctl stats` and the server log
  to confirm the server is healthy and using the intended capacities.
- Run the real coding-agent task at each bias setting. TG alone does not
  measure whether routing changes preserve useful answers or tool use.

For the available serving flags and their defaults, see the [CLI reference](cli.md).
