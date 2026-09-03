"""Small, rank-zero DSPP iteration metric accumulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_dspp_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    durations = [float(record["iteration_ms"]) for record in records]
    seconds = sum(durations) / 1000.0
    valid_tokens = sum(int(record["valid_tokens"]) for record in records)
    capacity = sum(int(record["physical_token_capacity"]) for record in records)
    padding = sum(int(record["schedule_padding_slots"]) for record in records)
    slots = sum(int(record["schedule_slots"]) for record in records)
    return {
        "iterations": len(records),
        "effective_tokens_per_second": valid_tokens / seconds if seconds else 0.0,
        "iteration_median_ms": _percentile(durations, 0.5),
        "iteration_p95_ms": _percentile(durations, 0.95),
        "packing_utilization": valid_tokens / capacity if capacity else 0.0,
        "schedule_padding_ratio": padding / slots if slots else 0.0,
        "peak_allocated_bytes_rank0": max(
            (int(record.get("peak_allocated_bytes_rank0", 0)) for record in records),
            default=0,
        ),
        "peak_reserved_bytes_rank0": max(
            (int(record.get("peak_reserved_bytes_rank0", 0)) for record in records),
            default=0,
        ),
    }


def append_dspp_metric(args: Any, iteration_ms: float, *, peak_allocated: int,
                       peak_reserved: int) -> None:
    current = dict(getattr(args, "_dspp_last_iteration_metrics", {}))
    if not current:
        return
    current.update(
        {
            "iteration": int(getattr(args, "curr_iteration", -1)) + 1,
            "iteration_ms": float(iteration_ms),
            "peak_allocated_bytes_rank0": int(peak_allocated),
            "peak_reserved_bytes_rank0": int(peak_reserved),
        }
    )
    history = getattr(args, "_dspp_metric_history", None)
    if history is None:
        history = []
        args._dspp_metric_history = history
    history.append(current)


def write_dspp_metrics(args: Any) -> None:
    path = getattr(args, "dspp_metrics_path", None)
    history = getattr(args, "_dspp_metric_history", [])
    if not path or not history:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(
            {"summary": summarize_dspp_metrics(history), "iterations": history},
            stream,
            indent=2,
        )
