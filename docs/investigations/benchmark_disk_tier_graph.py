#!/usr/bin/env python3
"""Measure streamed decode throughput for the disk-tier graph handoff."""

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone

import requests


PROMPT = (
    "The capital of France is a city with museums, historic landmarks, and a river "
    "running through it. Explain the main attractions in one sentence."
)


def run_one(session, endpoint, request_body):
    started = time.perf_counter()
    first_text = None
    last_text = None
    final_at = None
    text_parts = []
    completion_tokens = None
    status = None
    with session.post(endpoint, json=request_body, stream=True, timeout=600) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                final_at = time.perf_counter()
                break
            event = json.loads(data)
            if event.get("usage"):
                completion_tokens = event["usage"].get("completion_tokens")
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                status = choice["finish_reason"]
            piece = choice.get("text") or ""
            if piece:
                now = time.perf_counter()
                if first_text is None:
                    first_text = now
                last_text = now
                text_parts.append(piece)
    if first_text is None or last_text is None or final_at is None or completion_tokens is None:
        raise RuntimeError(
            f"incomplete stream: first_text={first_text is not None}, "
            f"last_text={last_text is not None}, done={final_at is not None}, "
            f"usage={completion_tokens}"
        )
    text = "".join(text_parts)
    duration = last_text - first_text
    if duration <= 0 or completion_tokens <= 1:
        raise RuntimeError(f"invalid decode interval/count: {duration=}, {completion_tokens=}")
    return {
        "completion_tokens": completion_tokens,
        "decode_seconds_first_text_to_last_text": duration,
        "decode_tokens_after_first_per_second": (completion_tokens - 1) / duration,
        "first_text_seconds_from_request": first_text - started,
        "done_seconds_from_request": final_at - started,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "finish_reason": status,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("graph", "eager"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--url", default="http://127.0.0.1:8199/v1/completions")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-Q4_K-v3.gguf")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()

    body = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    records = []
    with requests.Session() as session:
        for index in range(args.warmups + args.runs):
            result = run_one(session, args.url, body)
            result.update({"index": index, "phase": "warmup" if index < args.warmups else "measured"})
            records.append(result)
            print(
                f"{args.mode} {result['phase']} {index + 1}/{args.warmups + args.runs}: "
                f"{result['completion_tokens']} tokens, "
                f"{result['decode_tokens_after_first_per_second']:.4f} tok/s, "
                f"{result['decode_seconds_first_text_to_last_text']:.3f}s",
                flush=True,
            )

    measured = [
        r["decode_tokens_after_first_per_second"]
        for r in records
        if r["phase"] == "measured"
    ]
    output = {
        "mode": args.mode,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.url,
        "request": body,
        "warmup_count": args.warmups,
        "measured_count": args.runs,
        "throughput_method": "(completion_tokens - 1) / (last_text_time - first_text_time); iter_lines(chunk_size=1)",
        "measured_median_tok_s": statistics.median(measured),
        "measured_min_tok_s": min(measured),
        "measured_max_tok_s": max(measured),
        "records": records,
    }
    with open(args.output, "w") as file:
        json.dump(output, file, indent=2)
        file.write("\n")
    print(
        f"{args.mode} measured median={output['measured_median_tok_s']:.4f} "
        f"min={output['measured_min_tok_s']:.4f} max={output['measured_max_tok_s']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
