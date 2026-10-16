#!/usr/bin/env python3
"""Offline PP-free DSPP compute shape profiler.

The model-specific adapter is intentionally external.  It receives
``adapter(shape=DsppComputeShape, device=...)`` and returns a callable that
executes one real F/B/W task, optionally with ``work_units`` metadata.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megatron.core.datasets.dspp_compute_profile import (
    DsppCapacityProbeResult,
    build_compute_shape_plan,
    discover_capacity_range,
    load_adapter,
    profile_with_adapter,
    save_compute_profile,
    suggest_refinement_capacities,
)


def _csv(value, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _run_capacity_probe(
    capacity: int,
    *,
    adapter: str,
    adapter_kwargs_json: str,
    operations: tuple[str, ...],
    history_ratio: int,
    device: str,
    timeout_seconds: int,
) -> DsppCapacityProbeResult:
    results = []
    probe_script = Path(__file__).with_name("probe_dspp_compute_capacity.py")
    for operation in operations:
        command = [
            sys.executable,
            str(probe_script),
            "--adapter", adapter,
            "--adapter-kwargs-json", adapter_kwargs_json,
            "--capacity", str(capacity),
            "--history-ratio", str(history_ratio),
            "--operation", operation,
            "--device", device,
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return DsppCapacityProbeResult(
                capacity=capacity,
                feasible=False,
                reason=f"{operation}: capacity probe timed out",
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if completed.returncode or not lines:
            reason = (completed.stderr or completed.stdout or "probe process failed").splitlines()
            return DsppCapacityProbeResult(
                capacity=capacity,
                feasible=False,
                reason=f"{operation}: {reason[-1][:500] if reason else 'probe process failed'}",
            )
        try:
            result = DsppCapacityProbeResult.from_dict(json.loads(lines[-1]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return DsppCapacityProbeResult(
                capacity=capacity,
                feasible=False,
                reason=f"{operation}: invalid probe output: {error}",
            )
        results.append(result)
        if not result.feasible:
            return DsppCapacityProbeResult(
                capacity=capacity,
                feasible=False,
                peak_allocated_bytes=result.peak_allocated_bytes,
                peak_reserved_bytes=result.peak_reserved_bytes,
                total_memory_bytes=result.total_memory_bytes,
                reason=f"{operation}: {result.reason or 'not feasible'}",
            )
    return DsppCapacityProbeResult(
        capacity=capacity,
        feasible=True,
        peak_allocated_bytes=max(result.peak_allocated_bytes for result in results),
        peak_reserved_bytes=max(result.peak_reserved_bytes for result in results),
        total_memory_bytes=min(
            result.total_memory_bytes for result in results if result.total_memory_bytes > 0
        ) if any(result.total_memory_bytes > 0 for result in results) else 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="module:function")
    parser.add_argument(
        "--adapter-kwargs-json",
        default="{}",
        help="JSON object forwarded to the model-specific adapter",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-capacity", type=int)
    parser.add_argument("--max-capacity", type=int)
    parser.add_argument(
        "--capacities",
        help="Optional comma-separated capacities to profile exactly within min/max bounds",
    )
    parser.add_argument(
        "--auto-capacity-range",
        action="store_true",
        help="Discover legal min and memory-feasible max without prior chunk bounds",
    )
    parser.add_argument("--alignment", type=int, default=256)
    parser.add_argument("--legal-start-capacity", type=int, default=1)
    parser.add_argument("--search-alignment", type=int)
    parser.add_argument("--capacity-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-probe-attempts", type=int, default=64)
    parser.add_argument("--probe-history-ratio", type=int, default=4)
    parser.add_argument("--probe-operations", default="B,W")
    parser.add_argument("--probe-timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--profile-max-capacity",
        type=int,
        help="Optional practical profiling ceiling; discovery still records the full feasible max",
    )
    parser.add_argument("--stage-count", type=int, default=1)
    parser.add_argument("--max-history-tokens", type=int)
    parser.add_argument("--operations", default="F,B,W")
    parser.add_argument("--history-ratios", default="0,1,2,4")
    parser.add_argument("--tail-ratios", default="0.25,0.5,0.75")
    parser.add_argument("--item-counts", default="2,4")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-shapes", type=int, default=8192)
    parser.add_argument("--refinement-rounds", type=int, default=2)
    parser.add_argument("--refinement-points-per-round", type=int, default=6)
    parser.add_argument("--efficiency-threshold", type=float, default=0.90)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()
    adapter_kwargs = json.loads(args.adapter_kwargs_json)
    if not isinstance(adapter_kwargs, dict):
        raise SystemExit("--adapter-kwargs-json must decode to an object")
    adapter = partial(load_adapter(args.adapter), **adapter_kwargs)
    supplied_bounds = args.min_capacity is not None or args.max_capacity is not None
    if supplied_bounds and (args.min_capacity is None or args.max_capacity is None):
        raise SystemExit("--min-capacity and --max-capacity must be supplied together")
    auto_range = args.auto_capacity_range or not supplied_bounds
    if auto_range and supplied_bounds:
        raise SystemExit("use either auto capacity discovery or explicit min/max bounds")
    capacity_discovery = None
    if auto_range:
        search_alignment = args.search_alignment or args.alignment
        probe = partial(
            _run_capacity_probe,
            adapter=args.adapter,
            adapter_kwargs_json=args.adapter_kwargs_json,
            operations=_csv(args.probe_operations, str),
            history_ratio=args.probe_history_ratio,
            device=args.device,
            timeout_seconds=args.probe_timeout_seconds,
        )
        capacity_discovery = discover_capacity_range(
            probe,
            legal_start_capacity=args.legal_start_capacity,
            search_alignment=search_alignment,
            memory_utilization=args.capacity_memory_utilization,
            max_probe_attempts=args.max_probe_attempts,
        )
        args.min_capacity = capacity_discovery.min_capacity
        args.max_capacity = capacity_discovery.max_capacity
        print(
            "capacity discovery: "
            f"legal_min={args.min_capacity}, feasible_max={args.max_capacity}, "
            f"probes={len(capacity_discovery.probes)}",
            flush=True,
        )
    profile_min_capacity = args.min_capacity
    profile_max_capacity = args.max_capacity
    if args.profile_max_capacity is not None:
        if args.profile_max_capacity < profile_min_capacity:
            raise SystemExit("--profile-max-capacity must be >= discovered/supplied min capacity")
        profile_max_capacity = min(profile_max_capacity, args.profile_max_capacity)
    metadata = {
        "adapter": args.adapter,
        "adapter_kwargs": adapter_kwargs,
        "device": args.device,
        "capacity_bounds": [args.min_capacity, args.max_capacity],
        "profile_capacity_bounds": [profile_min_capacity, profile_max_capacity],
        "capacity_discovery": (
            capacity_discovery.to_dict() if capacity_discovery is not None else None
        ),
        "alignment": args.alignment,
        "stage_count": args.stage_count,
        "operations": list(_csv(args.operations, str)),
        "history_ratios": list(_csv(args.history_ratios, int)),
        "tail_ratios": list(_csv(args.tail_ratios, float)),
        "item_counts": list(_csv(args.item_counts, int)),
        "max_history_tokens": args.max_history_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cuda_graph_requested": not args.no_cuda_graph,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if args.device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(args.device).index
        index = torch.cuda.current_device() if index is None else index
        metadata["cuda_device_name"] = torch.cuda.get_device_name(index)
        metadata["cuda_capability"] = list(torch.cuda.get_device_capability(index))
    shapes = build_compute_shape_plan(
        min_capacity=profile_min_capacity,
        max_capacity=profile_max_capacity,
        alignment=args.alignment,
        stage_count=args.stage_count,
        operations=_csv(args.operations, str),
        history_ratios=_csv(args.history_ratios, int),
        tail_ratios=_csv(args.tail_ratios, float),
        item_counts=_csv(args.item_counts, int),
        max_history_tokens=args.max_history_tokens,
        max_shapes=args.max_shapes,
        capacities=(
            _csv(args.capacities, int) if args.capacities is not None else None
        ),
    )
    records = list(profile_with_adapter(
        shapes,
        adapter=adapter,
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
        use_cuda_graph=not args.no_cuda_graph,
    ))
    save_compute_profile(args.output, records, metadata=metadata)
    for round_index in range(max(0, args.refinement_rounds)):
        capacities = suggest_refinement_capacities(
            records,
            min_capacity=profile_min_capacity,
            max_capacity=profile_max_capacity,
            alignment=args.alignment,
            efficiency_threshold=args.efficiency_threshold,
            max_new_capacities=args.refinement_points_per_round,
        )
        if not capacities:
            break
        refined_shapes = build_compute_shape_plan(
            min_capacity=profile_min_capacity,
            max_capacity=profile_max_capacity,
            alignment=args.alignment,
            stage_count=args.stage_count,
            operations=_csv(args.operations, str),
            history_ratios=_csv(args.history_ratios, int),
            tail_ratios=_csv(args.tail_ratios, float),
            item_counts=_csv(args.item_counts, int),
            max_history_tokens=args.max_history_tokens,
            max_shapes=args.max_shapes,
            capacities=capacities,
        )
        records.extend(profile_with_adapter(
            refined_shapes,
            adapter=adapter,
            device=args.device,
            warmup=args.warmup,
            repeats=args.repeats,
            use_cuda_graph=not args.no_cuda_graph,
        ))
        save_compute_profile(args.output, records, metadata=metadata)
        print(
            f"refinement round {round_index + 1}: profiled capacities={list(capacities)}",
            flush=True,
        )
    print(f"profiled {len(records)} compute shapes -> {args.output}")


if __name__ == "__main__":
    main()
