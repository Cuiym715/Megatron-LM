#!/usr/bin/env python3
"""Build a reproducible variable-length Megatron dataset for DSPP tests."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megatron.data.indexed_dataset import MMapIndexedDatasetBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--training-lengths",
        default="96,64,24,16",
        help="Comma-separated LM target counts; one label token is appended per document.",
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()

    lengths = [int(value) for value in args.training_lengths.split(",")]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("training lengths must be positive")
    if args.repeats <= 0 or args.vocab_size < 3:
        raise ValueError("repeats must be positive and vocab size must be at least 3")

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = [prefix.with_suffix(suffix) for suffix in (".bin", ".idx", ".json")]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))

    builder = MMapIndexedDatasetBuilder(str(prefix.with_suffix(".bin")), dtype=np.int32)
    raw_lengths = []
    for _ in range(args.repeats):
        for training_length in lengths:
            raw_length = training_length + 1
            tokens = torch.arange(1, raw_length + 1, dtype=torch.int64)
            tokens = tokens.remainder(args.vocab_size - 1).add_(1)
            builder.add_item(tokens)
            builder.end_document()
            raw_lengths.append(raw_length)
    builder.finalize(str(prefix.with_suffix(".idx")))

    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "kind": "synthetic-dspp-indexed-dataset",
                "training_lengths_pattern": lengths,
                "raw_lengths_pattern": [length + 1 for length in lengths],
                "repeats": args.repeats,
                "documents": len(raw_lengths),
                "vocab_size": args.vocab_size,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(raw_lengths)} documents to {prefix}")


if __name__ == "__main__":
    main()
