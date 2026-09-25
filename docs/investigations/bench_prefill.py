"""Prefill throughput against a running FreeToken (or llama.cpp) server.

Sends prompts of several lengths with ``max_tokens=1`` and reports the wall time and
prefill tokens/second (prompt tokens from the server's usage block). Each prompt starts
with a unique nonce so no run reuses another's prefix cache.

  python bench_prefill.py --lengths 512 2048 6000 --runs 3 --out prefill.json

``--warm-tokens N`` decodes N tokens first, so a disk-tier server's GPU expert cache holds
what a conversation leaves behind (prefill reads cached experts device to device) rather
than starting empty, which ``max_tokens=1`` requests never change.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.request
import uuid

_TEXT = (pathlib.Path(__file__).resolve().parents[2] / "README.md").read_text(errors="ignore")


def prompt_of(approx_tokens: int) -> str:
    body = _TEXT
    while len(body) < approx_tokens * 4:
        body += "\n" + _TEXT
    return f"[{uuid.uuid4().hex}]\n" + body[: approx_tokens * 4]


def complete(url: str, model: str, prompt: str, max_tokens: int = 1) -> tuple[float, int]:
    req = urllib.request.Request(
        url,
        data=json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                         "temperature": 0, "ignore_eos": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        body = json.loads(r.read())
    return time.perf_counter() - t0, int(body["usage"]["prompt_tokens"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8199/v1/completions")
    ap.add_argument("--model", default="default")
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 6000])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warm-tokens", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    complete(args.url, args.model, prompt_of(64), max(1, args.warm_tokens))  # warm the path
    results = []
    for n in args.lengths:
        runs = []
        for _ in range(args.runs):
            dt, toks = complete(args.url, args.model, prompt_of(n))
            runs.append({"seconds": dt, "prompt_tokens": toks, "tok_s": toks / dt})
        med = sorted(r["tok_s"] for r in runs)[len(runs) // 2]
        print(f"prompt ~{runs[0]['prompt_tokens']:6d} tokens: median {med:8.1f} tok/s "
              f"({min(r['seconds'] for r in runs):.2f}-{max(r['seconds'] for r in runs):.2f} s)")
        results.append({"length": n, "runs": runs, "median_tok_s": med})
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
