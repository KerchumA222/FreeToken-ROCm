"""Decode throughput of an OpenAI-compatible server (FreeToken or llama-server).

Greedy, fixed-length streamed completions over a fixed prompt set. Decode rate is
``(completion_tokens - 1) / (last_text_time - first_text_time)``, so prefill is excluded
and both servers are timed identically from the client side. Warmup passes absorb JIT and
page-cache effects before the measured passes.

  python bench_spec_decode.py --port 8199 --label ft-spec1 --out /tmp/ft-spec1.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

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


def one(port: int, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": "m", "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1.0, "max_tokens": max_tokens, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    usage = None
    text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    text.append(piece)
    n = usage["completion_tokens"] if usage else None
    return {
        "completion_tokens": n,
        "ttft_s": first - t0,
        "decode_tok_s": (n - 1) / (last - first) if n and last > first else None,
        "text": "".join(text),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup-passes", type=int, default=1)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for _ in range(args.warmup_passes):
        for p in PROMPTS:
            one(args.port, p, args.max_tokens)
    runs = []
    for i in range(args.passes):
        for j, p in enumerate(PROMPTS):
            r = one(args.port, p, args.max_tokens)
            r.update(pass_=i, prompt_idx=j)
            runs.append(r)
    rates = [r["decode_tok_s"] for r in runs if r["decode_tok_s"]]
    per_prompt = [statistics.median(r["decode_tok_s"] for r in runs if r["prompt_idx"] == j)
                  for j in range(len(PROMPTS))]
    summary = {
        "label": args.label,
        "median_tok_s": statistics.median(rates),
        "mean_tok_s": statistics.fmean(rates),
        "min_tok_s": min(rates), "max_tok_s": max(rates),
        "per_prompt_median": [round(x, 2) for x in per_prompt],
        "tokens": sorted({r["completion_tokens"] for r in runs}),
    }
    print(json.dumps(summary))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "runs": runs}, f, indent=1)


if __name__ == "__main__":
    main()
