#!/usr/bin/env python3
"""Compare serial and concurrent greedy streams on the disk-tier graph path."""

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone

import requests

PROMPTS = (
    "The capital of France is a city with museums, historic landmarks, and a river "
    "running through it. Explain the main attractions in one sentence.",
    "A good tomato soup balances sweetness and acidity; explain a simple way to "
    "improve its flavor in one sentence.",
)


def run_one(session, endpoint, prompt, max_tokens, *, pair_started=None):
    body = {
        "model": "Qwen3.8-Flash-Next-Q4_K-v3.gguf",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    pair_started = started if pair_started is None else pair_started
    first_text = last_text = None
    parts = []
    completion_tokens = None
    with session.post(endpoint, json=body, stream=True, timeout=600) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                completion_tokens = event["usage"].get("completion_tokens")
            choices = event.get("choices") or []
            if not choices:
                continue
            piece = choices[0].get("text") or ""
            if piece:
                now = time.perf_counter()
                if first_text is None:
                    first_text = now
                last_text = now
                parts.append(piece)
    if first_text is None or last_text is None or completion_tokens is None:
        raise RuntimeError("incomplete streamed response")
    text = "".join(parts)
    duration = last_text - first_text
    return {
        "completion_tokens": completion_tokens,
        "decode_seconds_first_to_last_text": duration,
        "decode_tokens_after_first_per_second": (completion_tokens - 1) / duration,
        "first_text_seconds_from_request": first_text - started,
        "first_text_seconds_from_pair": first_text - pair_started,
        "last_text_seconds_from_pair": last_text - pair_started,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8199/v1/completions")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--serial-runs", type=int, default=3)
    parser.add_argument("--concurrent-runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    serial = [[], []]
    concurrent_pairs = []
    with requests.Session() as session:
        for prompt_index, prompt in enumerate(PROMPTS):
            for _ in range(args.warmups):
                run_one(session, args.url, prompt, args.max_tokens)
            for _ in range(args.serial_runs):
                serial[prompt_index].append(
                    run_one(session, args.url, prompt, args.max_tokens)
                )
        for _ in range(args.warmups):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(
                    lambda prompt: run_one(session, args.url, prompt, args.max_tokens),
                    PROMPTS,
                ))
        for _ in range(args.concurrent_runs):
            pair_started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                pair = list(pool.map(
                    lambda prompt: run_one(
                        session, args.url, prompt, args.max_tokens, pair_started=pair_started
                    ),
                    PROMPTS,
                ))
            span = max(item["last_text_seconds_from_pair"] for item in pair) - min(
                item["first_text_seconds_from_pair"] for item in pair
            )
            concurrent_pairs.append({
                "requests": pair,
                "decode_span_seconds": span,
                "aggregate_decode_tokens_after_first_per_second":
                    sum(item["completion_tokens"] - 1 for item in pair) / span,
            })

    output = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.url,
        "max_tokens": args.max_tokens,
        "throughput_method": "(completion_tokens-1)/(last_text-first_text); iter_lines(chunk_size=1)",
        "prompts": list(PROMPTS),
        "warmup_count_per_prompt": args.warmups,
        "serial_runs_per_prompt": args.serial_runs,
        "concurrent_pair_runs": args.concurrent_runs,
        "serial": serial,
        "concurrent_pairs": concurrent_pairs,
    }
    with open(args.output, "w") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")

    serial_rates = [statistics.median(item["decode_tokens_after_first_per_second"] for item in values)
                    for values in serial]
    serial_pair_aggregate = 2 / sum(1 / value for value in serial_rates)
    concurrent_rates = [
        pair["aggregate_decode_tokens_after_first_per_second"]
        for pair in concurrent_pairs
    ]
    concurrent_stream_rates = [
        statistics.median(pair["requests"][index]["decode_tokens_after_first_per_second"]
                          for pair in concurrent_pairs)
        for index in range(2)
    ]
    for index in range(2):
        hashes = {
            item["text_sha256"] for item in serial[index]
        } | {
            pair["requests"][index]["text_sha256"] for pair in concurrent_pairs
        }
        if len(hashes) != 1:
            raise RuntimeError(f"greedy output changed for prompt {index}: {sorted(hashes)}")
    print(f"serial per-prompt medians: {serial_rates}")
    print(f"serial aggregate: {serial_pair_aggregate:.4f} tok/s")
    print(f"concurrent per-stream medians: {concurrent_stream_rates}")
    print(f"concurrent aggregate samples: {concurrent_rates}")
    print(f"concurrent aggregate median: {statistics.median(concurrent_rates):.4f} tok/s")


if __name__ == "__main__":
    main()
