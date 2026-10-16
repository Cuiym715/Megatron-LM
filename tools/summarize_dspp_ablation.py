#!/usr/bin/env python3
"""Aggregate DSPP ablation metrics with one cold-start iteration per run removed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped = {}
    runs = []
    for status_path in sorted(args.result_root.glob("*/*/exit_status.txt")):
        result_dir = status_path.parent
        suite = result_dir.parent.name
        config, seed_text = result_dir.name.rsplit("_seed", 1)
        status = int(status_path.read_text().strip())
        metrics_path = result_dir / "metrics.json"
        record = {
            "suite": suite,
            "config": config,
            "seed": int(seed_text),
            "status": status,
            "metrics": str(metrics_path),
        }
        if status == 0 and metrics_path.exists():
            payload = json.loads(metrics_path.read_text())
            iterations = payload.get("iterations", [])
            steady = iterations[1:]
            record["completed_iterations"] = len(iterations)
            record["steady_iterations"] = len(steady)
            record["chunks"] = [row.get("chunk_size") for row in iterations]
            record["steady_iteration_ms"] = [row["iteration_ms"] for row in steady]
            record["steady_valid_tokens"] = [row["valid_tokens"] for row in steady]
            record["steady_physical_microbatches"] = [
                row["physical_microbatches"] for row in steady
            ]
            record["peak_allocated_bytes_rank0"] = payload["summary"].get(
                "peak_allocated_bytes_rank0"
            )
            grouped.setdefault((suite, config), []).append(record)
        runs.append(record)

    summaries = []
    for (suite, config), records in sorted(grouped.items()):
        times = [value for row in records for value in row["steady_iteration_ms"]]
        tokens = [value for row in records for value in row["steady_valid_tokens"]]
        total_ms = sum(times)
        physical = [
            value for row in records for value in row["steady_physical_microbatches"]
        ]
        summaries.append(
            {
                "suite": suite,
                "config": config,
                "successful_runs": len(records),
                "steady_iterations": len(times),
                "mean_iteration_ms": sum(times) / len(times),
                "median_iteration_ms": median(times),
                "p95_iteration_ms": percentile(times, 0.95),
                "effective_tokens_per_second": sum(tokens) / (total_ms / 1000.0),
                "mean_physical_microbatches": sum(physical) / len(physical),
                "peak_allocated_bytes_rank0": max(
                    row["peak_allocated_bytes_rank0"] or 0 for row in records
                ),
            }
        )

    output = {"cold_start_iterations_excluded_per_run": 1, "runs": runs, "summary": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
