#!/usr/bin/env python3
"""Copy trunk expert importance data to a shape-compatible MTP layer."""

import argparse
import struct

import numpy as np
from gguf import GGUFReader


EXPERT_TENSORS = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--source-layer", type=int, default=47)
    parser.add_argument("--target-layer", type=int, default=48)
    args = parser.parse_args()

    reader = GGUFReader(args.source)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    entries: list[tuple[str, np.ndarray]] = []
    for suffix in EXPERT_TENSORS:
        source = f"blk.{args.source_layer}.{suffix}"
        sums = np.asarray(tensors[f"{source}.in_sum2"].data, dtype=np.float32)
        counts = np.asarray(tensors[f"{source}.counts"].data, dtype=np.float32)
        rows = sums.reshape(counts.size, -1)
        normalized = rows / np.maximum(counts.reshape(-1, 1), 1.0)
        entries.append((f"blk.{args.target_layer}.{suffix}", normalized.reshape(-1)))

    with open(args.output, "wb") as file:
        file.write(struct.pack("<i", len(entries)))
        for name, values in entries:
            encoded = name.encode()
            file.write(struct.pack("<i", len(encoded)))
            file.write(encoded)
            file.write(struct.pack("<ii", 1, values.size))
            file.write(values.astype("<f4", copy=False).tobytes())
        file.write(struct.pack("<i", 1))
        dataset = b"copied trunk expert importance for the shape-compatible MTP layer"
        file.write(struct.pack("<i", len(dataset)))
        file.write(dataset)


if __name__ == "__main__":
    main()
