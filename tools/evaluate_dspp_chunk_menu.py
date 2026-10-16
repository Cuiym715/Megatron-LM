#!/usr/bin/env python3
"""Evaluate fixed chunk candidates and per-iteration oracle regret."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megatron.core.datasets.dspp_chunk_selector import (
    build_dspp_chunk_menu,
    load_dspp_length_trace,
    select_global_fixed_chunk,
)
from megatron.core.datasets.dspp_chunk_tuner import estimate_dspp_chunk_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length-trace", required=True)
    parser.add_argument("--chunks", help="comma-separated calibrated candidate chunk sizes")
    parser.add_argument("--min-chunk-size", type=int)
    parser.add_argument("--max-chunk-size", type=int)
    parser.add_argument("--chunk-alignment", type=int, default=256)
    parser.add_argument("--saturation-chunk-size", type=int)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--pipeline-size", type=int, required=True)
    parser.add_argument(
        "--ordering-mode", default="warmup-short-steady-long",
        choices=("input", "warmup-short-steady-long", "warmup-short-steady-critical"),
    )
    parser.add_argument(
        "--packing-strategy", default="baseline",
        choices=("baseline", "attention-critical"),
    )
    parser.add_argument("--sequence-group-size", type=int, default=1)
    parser.add_argument("--groups-per-iteration", type=int, default=1)
    parser.add_argument("--raw-lengths-include-label", action="store_true")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--execution-profile")
    parser.add_argument("--p2p-profile")
    parser.add_argument("--hidden-size", type=int, default=1)
    parser.add_argument("--dtype-bytes", type=int, default=1)
    parser.add_argument("--communication-cost-mode", choices=("ignore", "profile"), default="ignore")
    parser.add_argument("--allow-low-profile-coverage", action="store_true")
    parser.add_argument("--min-profiled-compute-fraction", type=float, default=0.90)
    parser.add_argument("--max-low-confidence-compute-fraction", type=float, default=0.25)
    parser.add_argument("--activation-bytes-per-live-token", type=float)
    parser.add_argument("--activation-budget-bytes", type=int)
    parser.add_argument("--memory-safety-factor", type=float, default=1.20)
    parser.add_argument("--communication-bytes-per-chunk-token", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trace = load_dspp_length_trace(
        args.length_trace,
        sequence_group_size=args.sequence_group_size,
        groups_per_iteration=args.groups_per_iteration,
        raw_lengths_include_label=args.raw_lengths_include_label,
        max_iterations=args.max_iterations,
    )
    if args.chunks:
        candidate_chunks = [int(item) for item in args.chunks.split(",")]
    else:
        if args.min_chunk_size is None or args.max_chunk_size is None:
            parser.error("provide --chunks or both --min-chunk-size/--max-chunk-size")
        candidate_chunks = build_dspp_chunk_menu(
            trace,
            min_chunk_size=args.min_chunk_size,
            max_chunk_size=args.max_chunk_size,
            alignment=args.chunk_alignment,
            saturation_chunk_size=args.saturation_chunk_size,
            max_candidates=args.max_candidates,
        )
    feasibility_fn = None
    if args.activation_bytes_per_live_token is not None:
        if args.activation_budget_bytes is None:
            parser.error("--activation-budget-bytes is required with activation calibration")

        def feasibility_fn(chunk, groups):
            layout = estimate_dspp_chunk_layout(
                groups,
                chunk_size=chunk,
                pipeline_size=args.pipeline_size,
                ordering_mode=args.ordering_mode,
                packing_strategy=args.packing_strategy,
            )
            activation = (
                layout.peak_live_tokens
                * args.activation_bytes_per_live_token
                * args.memory_safety_factor
            )
            communication = chunk * args.communication_bytes_per_chunk_token
            return activation + communication <= args.activation_budget_bytes

    result = select_global_fixed_chunk(
        trace,
        candidate_chunks=candidate_chunks,
        pipeline_size=args.pipeline_size,
        ordering_mode=args.ordering_mode,
        packing_strategy=args.packing_strategy,
        execution_profile_path=args.execution_profile,
        p2p_profile_path=args.p2p_profile,
        hidden_size=args.hidden_size,
        dtype_bytes=args.dtype_bytes,
        communication_cost_mode=args.communication_cost_mode,
        require_profile_coverage=not args.allow_low_profile_coverage,
        min_profiled_compute_fraction=args.min_profiled_compute_fraction,
        max_low_confidence_compute_fraction=args.max_low_confidence_compute_fraction,
        feasibility_fn=feasibility_fn,
    )
    payload = {
        "selected_chunk_size": result.selected_chunk_size,
        "fixed_median_ms": result.fixed_median_ms,
        "oracle_median_ms": result.oracle_median_ms,
        "regret": result.regret,
        "oracle_choices": list(result.oracle_choices),
        "communication_cost_mode": args.communication_cost_mode,
        "trace_iterations": len(trace),
        "candidate_chunks": list(candidate_chunks),
        "ordering_mode": args.ordering_mode,
        "packing_strategy": args.packing_strategy,
        "memory_filter": {
            "activation_bytes_per_live_token": args.activation_bytes_per_live_token,
            "activation_budget_bytes": args.activation_budget_bytes,
            "memory_safety_factor": args.memory_safety_factor,
            "communication_bytes_per_chunk_token": args.communication_bytes_per_chunk_token,
        },
        "scores": [score.__dict__ for score in result.scores],
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(
        f"selected fixed chunk={result.selected_chunk_size}; "
        f"iterations={len(trace)}; predicted fixed/oracle regret={result.regret:.4%}"
    )


if __name__ == "__main__":
    main()
