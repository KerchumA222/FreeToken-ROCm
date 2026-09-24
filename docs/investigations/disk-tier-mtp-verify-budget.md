# MTP on the disk tier: per-row rollback and a verify expert budget (Qwen3.8-Flash-Next)

**Status:** per-row rollback and QSA verify graphs are on. MTP runs +30-48% over plain
decode on the short prompt and +35% on the long one (depth 1). The verify budget (`FT_VERIFY_BUDGET`, off by default) loses at every budget
tried. Measured 2026-09-24 on
the RX 6800, `Qwen3.8-Flash-Next-IQ2_S-Q2SYM`, with the same disk-tier service as
[disk-tier-expert-prefetch.md](disk-tier-expert-prefetch.md), symmetric
Q2_0 sidecar `mtp-Qwen3.8-Flash-Next-IQ2XSQ2SYM.gguf`.

## Per-row rollback for PLE slot states

The per-row GDN rollback (`LinearStatePool.alloc_spec`/`restore_spec`) now also records
the per-step values of slot states (`ple_conv`, `ple_ngram_ctx`). Models opt in with
`supports_spec_slot_states`. Before this, qwen4_exp used the snapshot/re-run path.

| path | short tok/s | long tok/s |
|---|---:|---:|
| plain decode (no MTP) | 19.6 | 14.1 |
| MTP, old snapshot path | 14.29 | 11.16 |
| MTP, per-row rollback | 18.20 | 14.48 |

Acceptance is ~0.77 a round.

## Verify budget

`FT_TIER_STATS` also histograms draft-only expert misses per verify round: routed experts
that the draft row needs, the accepted row does not, and are not GPU-resident. Unbudgeted,
18% of rounds need none and 60% need 16 or more. Acceptance is ~0.77 in every bucket, so
the miss count does not predict rejection.

`FT_VERIFY_BUDGET=<n>` counts those misses layer by layer through the verify forward. Once
a round passes `n`, the draft is marked dead: its remaining non-resident experts are
swapped for row 0's (no extra reads), and the draft is rejected at commit.

| budget | short tok/s | long tok/s | accepted / round | dead rounds |
|---|---:|---:|---:|---:|
| 0 | 16.83 | 8.46 | 0.08 | 91% |
| 4 | 16.79 | 8.53 | 0.10 | 90% |
| 8 | 16.61 | 8.94 | 0.15 | 84% |
| 16 | 16.82 | 10.02 | 0.29 | 63% |
| 32 | 16.62 | 11.80 | 0.51 | 34% |
| none | 18.20 | 14.48 | 0.77 | 0% |

Throughput rises steadily with the budget; no budget beats unbudgeted MTP or plain decode.

## Why it loses

A verify round costs about the same whether or not the draft's experts are read:

- At budget 0, ~1.08 tokens a round at 8.46 tok/s is ~128 ms a round.
- Unbudgeted, 1.77 tokens a round at 14.48 tok/s is ~122 ms a round.
- Plain decode is ~71 ms a token.

So the draft's disk reads are not what makes verify expensive. Both sides stall on every
layer (row 0's own misses), and per-layer read time is dominated by round-trip latency, not
by the number of experts. Killing the draft throws away acceptance while keeping the
~50 ms verify overhead.

The overhead was QSA running verify eagerly: it had no `prepare_for_capture_rows`, so
verify rows took the ragged extend path outside a CUDA graph while plain decode replayed a
graph.

## Verify graphs for QSA

QSA now captures a uniform verify (`prepare_for_capture_rows`/`prepare_for_replay_rows`).
The captured forward runs the same ragged extend as the eager one, with its addressing
restaged onto static buffers per replay. Two PLE pieces had to become capture-safe:

- a uniform verify builds its `PLEMetadata` from the FLA metadata (no host staging, no
  fresh mask) and caches its conv indices on the device;
- the disk PLE backend stages a verify graph's tokens into the graph's pinned buffer,
  ordered after the previous graph like decode.

| | short tok/s | long tok/s | accepted / round |
|---|---:|---:|---:|
| plain decode | 19.6 | 14.1 | -- |
| depth 1, eager verify | 18.20 | 14.48 | 0.77 |
| **depth 1, graph verify** | **25.57** | **19.04** | 0.77 |
| depth 2, graph verify | 27.96 | 16.90 | 1.20 |
| depth 3, graph verify | 29.06 | 17.71 | 1.43 |

The graph rows were measured without `FT_TIER_STATS`. The verify-miss counting it enables
costs ~4-9% a round, so it now runs only with `FT_TIER_STATS` or `FT_VERIFY_BUDGET`. The long
prompt is 3 runs and noisy (depth 2: 16.8-18.4). Depth 1 wins there because each extra
verify row adds its own non-resident experts to every stalled layer. On the short prompt
more depth keeps paying.

Acceptance at depth 1 is unchanged by capture. The divergence harness gives the same drift
against plain decode as eager verify, so capture changes nothing numerically. That drift
(median 0.125, p99 1.93, max 3.3) comes from QSA's extend path selecting blocks slightly
differently from its decode path. It predates this work: `FT_SPEC_ROLLBACK=snapshot`
shows the same drift.

## Depth > 1 on Flash-Next

The draft chain used to feed the head its own narrow `[T, hidden]` output, where the
Flash-Next head takes the wide `[T, hc*hidden]` residual. It also built Triton metadata for
each step, but the MTP layer is QSA. Now the head's `forward_chain` returns the block's
wide pre-mixer output for the next step, and each backend builds that step's metadata
(`chain_step_metadata`). The drift at depth 2 against plain decode matches depth 1: median
0.125, p99 1.78.

## Why deeper drafts lose on the long run, and an adaptive depth

The scheduler times every verify round per depth (`FT_SPEC_ADAPT=0` logs without adapting;
`FT_SPEC_DEPTH=<k>` forces one). Fixed depths, moving averages at the end of each run:

| | tokens / round | ms / round | disk ms | tok/s |
|---|---:|---:|---:|---:|
| short, depth 2 | 2.52 | 81 | 0 | 31 |
| short, depth 3 | 2.84 | 95 | 0 | 30 |
| long, depth 2 | 1.71 | 101 | 13 | 17 |
| long, depth 3 | 2.14 | 142 | 35 | 15 |

- **Acceptance drops on long prose.** At depth 2 the long run accepts 0.7 drafts a round
  against 1.5 on the short run. The short benchmark repeats one prompt.
- **Each extra verify row costs ~15-25 ms of disk reads on the long run**, and nothing on
  the short run, whose experts stay cached.
- **Each chained head step costs ~3.5 ms.** Measured by forcing depth 1 under
  `--speculative-draft-tokens 3` (22.9 tok/s) and `2` (24.1), against a true depth-1 run
  (25.4).

`speculative/depth.py` picks the depth per round:

- Tokens for depth `d` are estimated from every round at depth `k >= d`, as
  `min(accepted, d) + 1`, because shallower drafts are prefixes of the same chain.
- Seconds are measured per depth, skipping the first 6 rounds after a switch, which pay
  cold reads.
- A probe every 128 rounds keeps the stalest depth current.
- The verify graphs are captured per (rows, chain length), 9 for depth 3, so a
  shallower round does not run the full chain.

| `--speculative-draft-tokens 3` | fixed depth 1 | fixed depth 2 | fixed depth 3 | adaptive |
|---|---:|---:|---:|---:|
| Flash-Next short | 25.4 | 28.0 | 29.1 | 28.0 |
| Flash-Next long | 17.8-19.0 | 16.9-19.5 | 16.2-17.7 | 17.7 |
| Qwen3.6-35B-A3B in VRAM | -- | 90.8 | 90.8 | 89.7 |

The long runs are noisy (5 runs; each depth spans 1-3 tok/s across sessions). Adaptive sits
within a few percent of the best fixed depth on every workload, and avoids the worst one:
depth 3 on the long run, or depth 1 on the short run. It is on by default when
`--speculative-draft-tokens` is above 1.

Adaptive loses its few percent to probes, and to settling that is still too short on the
disk tier. A depth-3 probe on the short run measured 192 ms with 74 ms of disk, against a
steady state of 95 ms with 0 ms.
