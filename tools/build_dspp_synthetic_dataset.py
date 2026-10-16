#!/usr/bin/env python3
"""Build a reproducible variable-length Megatron dataset for DSPP tests."""

import argparse
import hashlib
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
    parser.add_argument(
        "--raw-lengths-file",
        type=Path,
        help=(
            "Newline-delimited raw document lengths. When provided, each document "
            "produces raw_length-1 LM targets and --training-lengths is ignored."
        ),
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--token-pattern",
        choices=("sequential", "infinipipe"),
        default="sequential",
    )
    args = parser.parse_args()

    source_sha256 = None
    if args.raw_lengths_file is not None:
        source_bytes = args.raw_lengths_file.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        raw_lengths_pattern = [
            int(line.strip())
            for line in source_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
        if not raw_lengths_pattern or any(length < 2 for length in raw_lengths_pattern):
            raise ValueError("raw document lengths must all be at least 2")
        lengths = [length - 1 for length in raw_lengths_pattern]
    else:
        lengths = [int(value) for value in args.training_lengths.split(",")]
        raw_lengths_pattern = [length + 1 for length in lengths]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("training lengths must be positive")
    if args.repeats <= 0 or args.vocab_size < 3:
        raise ValueError("repeats must be positive and vocab size must be at least 3")

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = [prefix.with_suffix(suffix) for suffix in (".bin", ".idx", ".json")]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        metadata_path = prefix.with_suffix(".json")
        if len(existing) == len(paths) and args.raw_lengths_file is not None:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected = {
                "raw_lengths_sha256": source_sha256,
                "repeats": args.repeats,
                "documents": len(lengths) * args.repeats,
                "vocab_size": args.vocab_size,
                "token_pattern": args.token_pattern,
            }
            actual = {key: metadata.get(key) for key in expected}
            if actual == expected:
                print(
                    f"verified existing {len(raw_lengths_pattern) * args.repeats} "
                    f"document dataset at {prefix}"
                )
                return
            raise ValueError(
                f"existing dataset at {prefix} does not match the shared input: "
                f"expected {expected}, found {actual}"
            )
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))

    builder = MMapIndexedDatasetBuilder(str(prefix.with_suffix(".bin")), dtype=np.int32)
    raw_lengths = []
    document_index = 0
    for _ in range(args.repeats):
        for training_length in lengths:
            raw_length = training_length + 1
            if args.token_pattern == "infinipipe":
                usable_vocab = max(args.vocab_size - 1024, 1)
                base = 17 + document_index * 9973
                tokens = torch.arange(raw_length, dtype=torch.int64)
                tokens = tokens.add_(base).remainder_(usable_vocab).add_(1024)
            else:
                tokens = torch.arange(1, raw_length + 1, dtype=torch.int64)
                tokens = tokens.remainder(args.vocab_size - 1).add_(1)
            builder.add_item(tokens)
            builder.end_document()
            raw_lengths.append(raw_length)
            document_index += 1
    builder.finalize(str(prefix.with_suffix(".idx")))

    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "kind": "synthetic-dspp-indexed-dataset",
                "training_lengths_pattern": lengths,
                "raw_lengths_pattern": raw_lengths_pattern,
                "raw_lengths_file": (
                    str(args.raw_lengths_file.resolve())
                    if args.raw_lengths_file is not None
                    else None
                ),
                "raw_lengths_sha256": source_sha256,
                "repeats": args.repeats,
                "documents": len(raw_lengths),
                "vocab_size": args.vocab_size,
                "token_pattern": args.token_pattern,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(raw_lengths)} documents to {prefix}")


if __name__ == "__main__":
    main()
