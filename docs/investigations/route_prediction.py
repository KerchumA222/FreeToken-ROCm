"""How well can a decode step's expert routing be predicted ahead of time?

Input: an FT_ROUTE_TRACE file (qwen4_exp MoE: per decode step and layer, the router input
and the routed top-k ids) and the checkpoint the routers come from.

  python route_prediction.py /tmp/route-flash.pt MODEL.gguf

For lookahead distance d, layer L+d's router is applied to layer L's input (available
d layers early) and its top-K compared with the experts L+d actually used. Recall is
reported over all routed experts and over the "new" ones -- those the previous token did
not use at that layer, the likely cache misses that a prefetch has to find.
"""

from __future__ import annotations

import argparse
import sys

import torch


def load_routers(model_path: str, num_layers: int) -> list[torch.Tensor]:
    from freetoken.models.gguf.dequant import dequant_any
    from freetoken.models.gguf.reader import iter_gguf_tensors

    want = {f"blk.{i}.ffn_gate_inp.weight": i for i in range(num_layers)}
    out: list[torch.Tensor | None] = [None] * num_layers
    for t in iter_gguf_tensors(model_path):
        i = want.get(t.name)
        if i is not None:
            out[i] = dequant_any(t, torch.float32).reshape(t.shape)
    assert all(w is not None for w in out), "missing router weights"
    return out  # each [E, H]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("model")
    ap.add_argument("--max-d", type=int, default=4)
    args = ap.parse_args()

    rows = torch.load(args.trace)
    num_layers = max(r[0] for r in rows) + 1
    # Group into decode steps: a step is layers 0..L-1 in order.
    steps, cur = [], []
    for layer, x, ids in rows:
        if layer == 0 and cur:
            if len(cur) == num_layers:
                steps.append(cur)
            cur = []
        cur.append((x.float()[0], ids[0]))
    if len(cur) == num_layers:
        steps.append(cur)
    k = steps[0][0][1].numel()
    print(f"{len(steps)} decode steps, {num_layers} layers, top-{k}", file=sys.stderr)

    W = load_routers(args.model, num_layers)
    X = torch.stack([torch.stack([s[l][0] for s in steps]) for l in range(num_layers)])  # [L, T, H]
    ids = torch.stack([torch.stack([s[l][1] for s in steps]) for l in range(num_layers)])  # [L, T, k]
    E = W[0].shape[0]

    def as_set(t):  # [T, k] -> [T, E] bool
        m = torch.zeros(t.shape[0], E, dtype=torch.bool)
        m.scatter_(1, t.long(), True)
        return m

    actual = [as_set(ids[l]) for l in range(num_layers)]
    prev = [torch.cat([torch.zeros(1, E, dtype=torch.bool), a[:-1]]) for a in actual]
    new = [a & ~p for a, p in zip(actual, prev)]
    n_new = sum(int(n[1:].sum()) for n in new)
    n_all = sum(int(a[1:].sum()) for a in actual)
    print(f"new experts per step (not used by the previous token at that layer): "
          f"{n_new / n_all:.1%} of routed")

    print("\nd  K    recall(all)  recall(new)  [layer L+d router on layer L input]")
    for d in range(0, args.max_d + 1):
        for K in (k, 2 * k, 3 * k):
            hit_all = hit_new = tot_all = tot_new = 0
            for L in range(num_layers - d):
                logits = X[L] @ W[L + d].T                      # [T, E]
                pred = as_set(logits.topk(K, dim=-1).indices)
                a, n = actual[L + d][1:], new[L + d][1:]
                p = pred[1:]
                hit_all += int((p & a).sum()); tot_all += int(a.sum())
                hit_new += int((p & n).sum()); tot_new += int(n.sum())
            print(f"{d}  {K:<4} {hit_all / tot_all:10.1%}  {hit_new / max(tot_new, 1):10.1%}")
    # Temporal baseline: the previous token's experts at the same layer.
    print("\nprevious token's set at the same layer: recall(all) "
          f"{sum(int((p[1:] & a[1:]).sum()) for p, a in zip(prev, actual)) / n_all:.1%}")


if __name__ == "__main__":
    main()
