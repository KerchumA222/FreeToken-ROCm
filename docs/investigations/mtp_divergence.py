"""Locate where greedy MTP speculation first commits a different token than plain decode,
and measure the target's logit margin there.

  # 1. serve with FT_SPEC_TRACE=/tmp/plain.jsonl (no speculation), then:
  python mtp_divergence.py run --port 8199 --out /tmp/plain.out.json
  # 2. restart with FT_SPEC_TRACE=/tmp/spec.jsonl --speculative-draft-tokens 1, then:
  python mtp_divergence.py run --port 8199 --out /tmp/spec.out.json
  # 3.
  python mtp_divergence.py analyze /tmp/plain.jsonl /tmp/spec.jsonl

The trace records every sampled row's top-5 logits and the sequence index the row
predicts. Per request, the argmax of the last forward predicting an index is the token
committed there (greedy, match acceptance). Before the first
divergence both runs condition on identical tokens, so the logit differences there are
pure kernel numerics -- the noise floor a near-tie flip has to fall under.
"""

from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
from collections import defaultdict

PROMPTS = [
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a haiku sequence of four stanzas about autumn in a mountain village.",
    "What were the main causes of the French Revolution? Answer in two paragraphs.",
    "Derive the formula for the sum of the first n squares, step by step.",
    "Write a bash script that finds the ten largest files under a directory.",
    "Describe the water cycle to a ten-year-old.",
    "Compare TCP and UDP, and give one use case where each is the better choice.",
    "Translate into French and then explain the grammar: 'I would have gone if I had known.'",
]


def run(args) -> None:
    out = []
    for p in PROMPTS:
        body = json.dumps({
            "model": "m", "messages": [{"role": "user", "content": p}],
            "temperature": 0, "max_tokens": args.max_tokens, "ignore_eos": True,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/chat/completions", body,
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            resp = json.load(r)
        msg = resp["choices"][0]["message"]
        out.append({"prompt": p, "content": msg.get("content"),
                    "reasoning": msg.get("reasoning_content"), "usage": resp.get("usage")})
        print(f"{resp.get('usage', {}).get('completion_tokens')} tokens: {p[:50]}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


def load(path):
    """-> {prompt_key: (committed {idx: tok}, last record per predicted idx, prompt_len)}"""
    recs = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        recs[r["uid"]].append(r)
    runs = {}
    for uid, rs in recs.items():
        first = rs[0]
        if first["kind"] != "prefill":
            continue
        pred = {}
        for r in rs:
            pred[r["idx"]] = r
        prompt_len = max(r["fed_from"] + len(r["fed"]) for r in rs if r["kind"] == "prefill")
        # Greedy and match-accepted: every committed token is the target's argmax at the
        # last forward that predicted its index (a later forward never re-predicts an
        # index once it is committed). Host input_ids lag under overlap scheduling, so the
        # fed lists are not a reliable record of commits.
        committed = {i: r["ids"][0] for i, r in pred.items()}
        runs[tuple(first["fed"][-16:])] = (committed, pred, prompt_len)
    return runs


def analyze(args) -> None:
    plain, spec = load(args.plain), load(args.spec)
    drift, plain_gaps = [], []
    rows = []
    for key, (pc, pp, plen) in plain.items():
        if key not in spec:
            print("unmatched request in spec trace; skipping")
            continue
        sc, sp, _ = spec[key]
        end = min(max(pc), max(sc))
        d = next((i for i in range(plen, end + 1) if pc.get(i) != sc.get(i)), None)
        for i in range(plen, (d if d is not None else end) + 1):
            if i in pp and i in sp:
                a, b = pp[i], sp[i]
                plain_gaps.append(a["vals"][0] - a["vals"][1])
                if a["ids"][0] == b["ids"][0]:
                    drift.append(abs(a["vals"][0] - b["vals"][0]))
        if d is None:
            rows.append((plen, None, end - plen + 1))
            continue
        a, b = pp[d], sp[d]
        b_of_a = dict(zip(b["ids"], b["vals"])).get(a["ids"][0])
        rows.append((plen, d, dict(
            plain_top=list(zip(a["ids"][:3], [round(v, 4) for v in a["vals"][:3]])),
            spec_top=list(zip(b["ids"][:3], [round(v, 4) for v in b["vals"][:3]])),
            plain_gap=round(a["vals"][0] - a["vals"][1], 4),
            spec_gap=round(b["vals"][0] - b["vals"][1], 4),
            spec_kind=b["kind"], spec_logit_of_plain_top=b_of_a,
            plain_committed=pc.get(d), spec_committed=sc.get(d),
        )))

    n_tok = 0
    for plen, d, info in rows:
        if d is None:
            print(f"prompt_len={plen}: identical over {info} generated tokens")
            n_tok += info
            continue
        n_tok += d - plen
        print(f"prompt_len={plen}: first divergence at generated token {d - plen}")
        for k, v in info.items():
            print(f"    {k}: {v}")
    if drift:
        q = statistics.quantiles(drift, n=100)
        print(f"\npre-divergence top-1 logit drift, spec vs plain (same context, n={len(drift)}):"
              f" median {statistics.median(drift):.4f}  p90 {q[89]:.4f}  p99 {q[98]:.4f}"
              f"  max {max(drift):.4f}")
    if plain_gaps:
        g = sorted(plain_gaps)
        print(f"plain top1-top2 gap over those rows: median {statistics.median(g):.4f};"
              f" fraction below 0.05: {sum(x < 0.05 for x in g) / len(g):.3%},"
              f" below 0.25: {sum(x < 0.25 for x in g) / len(g):.3%}")
    print(f"generated tokens compared before first divergence: {n_tok}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--port", type=int, default=8199)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("plain")
    a.add_argument("spec")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else analyze(args)


if __name__ == "__main__":
    main()
