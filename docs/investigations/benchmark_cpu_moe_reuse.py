"""Microbenchmark duplicate-expert reuse in the mixed Q4_K/Q5_1 CPU kernel."""

from __future__ import annotations

import argparse
import os
import runpy
import statistics
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--experts", type=int, default=64)
    return parser.parse_args()


def route_ids(batch_size: int, pattern: str, base: int, top_k: int, experts: int) -> torch.Tensor:
    rows = []
    for token in range(batch_size):
        if pattern in ("unique", "one-duplicate"):
            offset = token * top_k
        elif pattern == "pairs":
            offset = (token // 2) * top_k
        elif pattern == "shared":
            offset = 0
        else:
            raise ValueError(pattern)
        rows.append([(base + offset + route) % experts for route in range(top_k)])
    if pattern == "one-duplicate":
        rows[1][0] = rows[0][0]
    return torch.tensor(rows, dtype=torch.int32)


def main() -> None:
    args = parse_args()
    # Keep the mixed-format dots on AVX2 while independently A/B-testing Q8_1.
    os.environ["FREETOKEN_CPU_MOE_ISA"] = "avx2"
    os.environ.setdefault("FREETOKEN_CPU_MOE_Q8_1", "avx2")
    torch.manual_seed(1234)
    helpers = runpy.run_path("tests/moe/test_cpu_moe_q4_0.py")
    expert_ids = tuple(range(args.experts))
    cache = helpers["_load_real_q4_k_q5_1_cache"](
        os.path.expanduser(args.model), layer=0, experts=expert_ids
    )
    expert_bytes = sum(
        tensor.numel()
        for tensor in (cache.bank_sources["gate_up"][0], cache.bank_sources["down"][0])
    ) // args.experts

    from freetoken.moe.cpu_executor import CpuMoeExecutor

    top_k = 10
    executor = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=8,
        max_tokens=4,
        device=torch.device("cuda"),
        fmt="q4_0",
        ggml_types=(12, 7),
    )
    cases = [
        (1, "unique"),
        (2, "unique"),
        (2, "one-duplicate"),
        (2, "shared"),
        (4, "unique"),
        (4, "one-duplicate"),
        (4, "pairs"),
        (4, "shared"),
    ]
    for batch_size, pattern in cases:
        io = executor._io_for(batch_size)
        io["x"].copy_(torch.randn(batch_size, 2560, dtype=torch.bfloat16) * 0.1)
        io["w"].copy_(torch.rand(batch_size, top_k, dtype=torch.float32))
        task = executor._task_for(0, batch_size)
        samples = []
        unique_count = len(set(route_ids(batch_size, pattern, 0, top_k, args.experts).reshape(-1).tolist()))
        for iteration in range(args.warmups + args.samples):
            base = (iteration * 17) % args.experts
            io["ids"].copy_(route_ids(batch_size, pattern, base, top_k, args.experts))
            start = time.perf_counter()
            executor._ext.run_task(task)
            elapsed = time.perf_counter() - start
            if iteration >= args.warmups:
                samples.append(elapsed)
        median = statistics.median(samples)
        p95 = sorted(samples)[int(len(samples) * 0.95) - 1]
        logical_bytes = batch_size * top_k * expert_bytes
        unique_bytes = unique_count * expert_bytes
        print(
            f"bs={batch_size} pattern={pattern} unique={unique_count}/{batch_size * top_k} "
            f"quant={executor._ext.q8_1_quant_name()} samples={len(samples)} "
            f"median_ms={median * 1000:.3f} p95_ms={p95 * 1000:.3f} "
            f"logical_gbs={logical_bytes / median / 1e9:.3f} "
            f"unique_gbs={unique_bytes / median / 1e9:.3f}"
        )


if __name__ == "__main__":
    main()
