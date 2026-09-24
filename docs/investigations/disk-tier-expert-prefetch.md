# Lookahead expert prefetch for the disk tier (Qwen3.8-Flash-Next)

**Status:** implemented behind `FT_PREFETCH` (off by default); not a win yet. The
prediction is good, but predicting costs more than the stalls it removes. Measured
2026-09-24 on the RX 6800, `Qwen3.8-Flash-Next-IQ2_S-Q2SYM`, the recorded disk-tier service
(`--moe-cache-auto --moe-host-cache-size 2048 --ple-backend disk --memory-ratio 0.96`,
graph bs 1, LRU).

## Where decode time goes

`FT_TIER_STATS=4800` logs the host tier every 4800 ensures (~100 tokens at 48 MoE
layers). Without prefetch every layer of every token stalls on disk, and the pinned host
pool almost never hits (the page cache is what caches):

```
host tier: 4800 ensures, 4800 stalled (100.0%), 8703 expert misses of 8720, 4.277 s in disk reads
```

~42 ms of blocking reads per ~70 ms token (14.1 tok/s on the 512-token prompt), one read
round trip per layer.

## How predictable is routing?

`FT_ROUTE_TRACE=<path>` (eager only) records every decode step's router input and routed
ids per layer; [`route_prediction.py`](route_prediction.py) applies layer L+d's router to
layer L's input. 1,962 decode steps; "new" = experts the previous token did not use at
that layer (64% of routed, the likely misses):

| lookahead d | predictions K | recall (all) | recall (new) |
|---:|---:|---:|---:|
| 1 | 10 | 67.8% | 63.7% |
| 1 | 20 | 85.5% | 83.5% |
| 1 | 30 | 90.5% | 89.2% |
| 2 | 20 | 77.8% | 74.9% |
| 2 | 30 | 84.4% | 82.3% |
| 3 | 30 | 80.9% | 77.9% |

The previous token's set at the same layer recalls 35.9%.

## Prefetch as implemented

Each MoE block, on single-row decode steps, applies layer L+d's router to its own input
(small GEMV + softmax + top-K), drops experts already in the GPU slot cache (and, with a
third field, those under a router probability), and stages the rest. Layer L's admission
host node starts those reads on the host tier's thread pool before its own blocking
`ensure`; layer L+d's `ensure` then waits only on reads still in flight. Guesses a layer
did not use return to the LRU cold end as their reads finish; in-flight guesses are capped
at a quarter of the pool.

| `FT_PREFETCH` | short tok/s | long tok/s | layers stalled | prefetched / used per ~100 tok |
|---|---:|---:|---:|---:|
| off | 19.61 | 14.01 | 100% | -- |
| `1,10` | 17.27 | 13.68 | 57% | 13,000 / 4,400 |
| `1,20` | 17.31 | 13.05 | 27% | 40,000 / 6,200 |
| `2,20` | 17.26 | 12.29 | 40% | 42,000 / 5,300 |
| `2,30` | 16.59 | 10.40 | 33% | 69,000 / 5,700 |
| `3,30` | 16.60 | 10.04 | 40% | 69,000 / 5,200 |
| `1,10,0.05` | 17.92 | 13.44 | 99% | ~150 / ~100 |
| `1,20,0.08` | 17.88 | 13.41 | 100% | ~40 / ~25 |
| `2,10,0.05` | 17.81 | 13.43 | 99% | ~165 / ~75 |

Two costs sink it:

1. **Predicting costs ~5 ms a token.** With a probability floor almost nothing is fetched
   (the router's softmax over 512 experts is too flat for 5-8% to select anything), yet
   the short prompt still drops 19.6 -> 17.9 tok/s: ~6 extra kernels per layer and the
   Python in 48 host callbacks a token.
2. **Unselective prefetch wastes the disk.** `1,20` removes 73% of stalls and halves
   blocking read time, but 85% of its reads are never used; they compete with the real
   reads and churn the page cache.

## What would make it pay

- Fuse the lookahead into the current layer's router: one small GEMV over the
  concatenated `[E_L + E_{L+d}, H]` weight, and one kernel that does top-K, the GPU-residency
  filter and compaction to a short list (~3 non-resident experts a layer).
- Keep the host callback Python-free when that list is empty, and cheap when it is not.
- Rank by predicted probability *and* recency outside the GPU cache, not raw top-K.
