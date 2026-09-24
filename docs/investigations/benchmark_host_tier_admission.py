#!/usr/bin/env python3
"""Profile bounded host-tier admission without starting an inference server.

The benchmark wraps the already imported ``GgufExpertStore.read_expert`` method,
so each worker read is timed from the same path used by ``HostExpertCache.ensure``.
It reports route conversion, cold misses, repeated hits, read bytes, and aggregate
read wall time.  The route trace is synthetic but has the model's layer/expert/top-k
shape; pass a fixed seed when comparing changes.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

import torch

from freetoken.moe.cpu_executor import _remap_host_ids
from freetoken.moe.disk_store import GgufExpertStore
from freetoken.moe.host_tier import HostExpertCache


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "median_ms": statistics.median(values),
        "p90_ms": ordered[max(0, int(len(values) * 0.9) - 1)],
        "sum_ms": sum(values),
    }


class _ReadProbe:
    """Forward a store while recording every concurrent expert read."""

    def __init__(self, store: GgufExpertStore):
        self.store = store
        self.samples: list[tuple[int, int, int]] = []

    def read_expert(self, bank: str, layer: int, expert: int, dst) -> int:
        started = time.perf_counter_ns()
        result = self.store.read_expert(bank, layer, expert, dst)
        finished = time.perf_counter_ns()
        self.samples.append((started, finished, int(dst.nbytes)))
        return result

    def __getattr__(self, name):
        return getattr(self.store, name)


class _FakeTier:
    def ensure(self, _layer: int, experts: list[int]) -> list[int]:
        return list(range(len(experts)))


def _run_token(tier: HostExpertCache, routes: list[list[int]]) -> list[float]:
    elapsed = []
    for layer, experts in enumerate(routes):
        started = time.perf_counter_ns()
        tier.ensure(layer, experts)
        elapsed.append((time.perf_counter_ns() - started) / 1e6)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--gate-up-type", type=int, default=12, help="GGML Q4_K is 12"
    )
    parser.add_argument(
        "--down-type", type=int, default=7, help="GGML Q5_1 is 7"
    )
    parser.add_argument("--seed", type=int, default=9173)
    args = parser.parse_args()

    random.seed(args.seed)
    routes = [
        [random.sample(range(args.experts), args.top_k) for _ in range(args.layers)]
        for _ in range(2)
    ]

    with GgufExpertStore(
        args.model,
        args.experts,
        {"gate_up": args.gate_up_type, "down": args.down_type},
        args.layers,
    ) as store:
        requantized = store.requantized_layers()
        if requantized:
            raise SystemExit(
                "checkpoint has requantized layers; use matching bank types: "
                f"{requantized}"
            )
        probe = _ReadProbe(store)
        tier = HostExpertCache(
            probe, args.experts, args.capacity, pin=True, workers=args.workers
        )
        cold = _run_token(tier, routes[0])
        cold_read_samples = list(probe.samples)
        hit = _run_token(tier, routes[0])
        second = _run_token(tier, routes[1])

        conversion = []
        fake = _FakeTier()
        for experts in routes[0]:
            ids = torch.tensor(experts, dtype=torch.int32)
            started = time.perf_counter_ns()
            _remap_host_ids(ids, fake, 0, args.experts)
            conversion.append((time.perf_counter_ns() - started) / 1e6)

        read_ms = [
            (finished - started) / 1e6
            for started, finished, _ in cold_read_samples
        ]
        read_bytes = sum(nbytes for _, _, nbytes in cold_read_samples)
        read_wall_ms = (
            (max(finished for _, finished, _ in cold_read_samples)
             - min(started for started, _, _ in cold_read_samples))
            / 1e6
            if cold_read_samples
            else 0.0
        )
        output = {
            "model": args.model,
            "experts": args.experts,
            "layers": args.layers,
            "top_k": args.top_k,
            "capacity": args.capacity,
            "workers": args.workers,
            "bank_types": {
                "gate_up": args.gate_up_type,
                "down": args.down_type,
            },
            "route_conversion_ms": _summary(conversion),
            "cold_ensure_ms": _summary(cold),
            "hit_ensure_ms": _summary(hit),
            "second_token_ensure_ms": _summary(second),
            "cold_read_calls": len(cold_read_samples),
            "cold_read_bytes": read_bytes,
            "cold_read_call_ms": _summary(read_ms),
            "cold_read_wall_ms": read_wall_ms,
            "bytes_per_expert": {
                name: store.expert_bytes(name) for name in store.banks
            },
            "stats": tier.stats.as_dict(),
        }
        tier.close()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
