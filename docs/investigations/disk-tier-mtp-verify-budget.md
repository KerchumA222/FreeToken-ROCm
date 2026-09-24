# MTP on the disk tier: per-row rollback and a verify expert budget (Qwen3.8-Flash-Next)

**Status:** per-row rollback for qwen4_exp is on (a win). The verify budget
(`FT_VERIFY_BUDGET`, off by default) loses at every budget tried. Measured 2026-09-24 on
the RX 6800, `Qwen3.8-Flash-Next-IQ2_S-Q2SYM`, with the same disk-tier service as
[disk-tier-expert-prefetch.md](disk-tier-expert-prefetch.md), MTP depth 1, symmetric
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

That overhead most likely comes from QSA running verify eagerly. It has no
`prepare_for_capture_rows`, so verify rows take the extend path outside a CUDA graph,
while plain decode replays a graph. Not yet measured directly. QSA's extend path is also
behind the small logit drift between MTP and plain decode on this model. The drift
predates this work; `FT_SPEC_ROLLBACK=snapshot` shows the same drift: median 0.11,
p99 1.44, max 2.5.

## Next

QSA rows-as-decode support (`prepare_for_capture_rows`) would graph-capture verify and
remove the extend/decode drift. With ~0.77 acceptance and a verify round near plain-decode
cost, MTP could reach up to ~1.7x plain on the disk tier. That ceiling is optimistic: it
assumes the draft's extra reads cost nothing.
