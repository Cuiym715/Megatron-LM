#!/usr/bin/env python3
"""Evaluate bounded per-iteration DSPP chunk selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megatron.core.datasets.dspp_chunk_selector import (
    load_dspp_length_trace,
    load_profiled_chunk_bank,
    select_per_iteration_chunks,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length-trace", required=True)
    candidate_source = parser.add_mutually_exclusive_group(required=True)
    candidate_source.add_argument("--chunks", help="comma-separated capacities")
    candidate_source.add_argument(
        "--chunk-bank", help="chunk_menu.json containing the complete measured profile bank"
    )
    parser.add_argument("--pipeline-size", type=int, required=True)
    parser.add_argument("--ordering-mode", default="warmup-short-steady-long")
    parser.add_argument("--packing-strategy", default="baseline")
    parser.add_argument("--sequence-group-size", type=int, default=1)
    parser.add_argument("--groups-per-iteration", type=int, default=1)
    parser.add_argument("--raw-lengths-include-label", action="store_true")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--execution-profile")
    parser.add_argument("--ordering-profile")
    parser.add_argument("--p2p-profile")
    parser.add_argument("--hidden-size", type=int, default=1)
    parser.add_argument("--dtype-bytes", type=int, default=1)
    parser.add_argument("--communication-cost-mode", choices=("ignore", "profile"), default="ignore")
    parser.add_argument("--allow-low-profile-coverage", action="store_true")
    parser.add_argument("--min-profiled-compute-fraction", type=float, default=0.90)
    parser.add_argument("--max-low-confidence-compute-fraction", type=float, default=0.25)
    parser.add_argument("--long-sequence-threshold", type=int, default=8192)
    parser.add_argument(
        "--selection-strategy",
        choices=("branch-and-bound", "exhaustive"),
        default="branch-and-bound",
    )
    parser.add_argument("--efficiency-top-k", type=int, default=4)
    parser.add_argument("--efficiency-slack", type=float, default=0.05)
    parser.add_argument("--current-chunk-size", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    trace = load_dspp_length_trace(
        args.length_trace,
        sequence_group_size=args.sequence_group_size,
        groups_per_iteration=args.groups_per_iteration,
        raw_lengths_include_label=args.raw_lengths_include_label,
        max_iterations=args.max_iterations,
    )
    candidate_chunks = (
        tuple(entry.capacity for entry in load_profiled_chunk_bank(args.chunk_bank))
        if args.chunk_bank
        else tuple(int(item) for item in args.chunks.split(",") if item.strip())
    )
    decisions = select_per_iteration_chunks(
        trace,
        candidate_chunks=candidate_chunks,
        pipeline_size=args.pipeline_size,
        ordering_mode=args.ordering_mode,
        packing_strategy=args.packing_strategy,
        execution_profile_path=args.execution_profile,
        ordering_profile_path=args.ordering_profile,
        p2p_profile_path=args.p2p_profile,
        hidden_size=args.hidden_size,
        dtype_bytes=args.dtype_bytes,
        communication_cost_mode=args.communication_cost_mode,
        require_profile_coverage=not args.allow_low_profile_coverage,
        min_profiled_compute_fraction=args.min_profiled_compute_fraction,
        max_low_confidence_compute_fraction=args.max_low_confidence_compute_fraction,
        long_sequence_threshold=args.long_sequence_threshold,
        selection_strategy=args.selection_strategy,
        profile_bank_path=args.chunk_bank,
        efficiency_top_k=args.efficiency_top_k,
        efficiency_slack=args.efficiency_slack,
        current_chunk_size=args.current_chunk_size,
    )
    payload = {
        "candidate_chunks": list(candidate_chunks),
        "chunk_bank": args.chunk_bank,
        "selection_strategy": args.selection_strategy,
        "pipeline_size": args.pipeline_size,
        "ordering_mode": args.ordering_mode,
        "packing_strategy": args.packing_strategy,
        "execution_profile": args.execution_profile,
        "communication_cost_mode": args.communication_cost_mode,
        "iterations": [decision.to_dict() for decision in decisions],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    counts = {}
    for decision in decisions:
        counts[decision.selected_chunk_size] = counts.get(decision.selected_chunk_size, 0) + 1
    mean_decision_ms = sum(item.decision_ms for item in decisions) / len(decisions)
    print(f"iterations={len(decisions)} selected_counts={counts} mean_decision_ms={mean_decision_ms:.3f}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
