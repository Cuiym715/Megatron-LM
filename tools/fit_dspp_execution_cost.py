#!/usr/bin/env python3
"""Fit a DSPP execution model from timeline JSON files.

Usage:
  python tools/fit_dspp_execution_cost.py \
      --input src/results/dspp_chrome/timeline/iteration_000002_rank_0.json \
      --output src/results/dspp_execution_cost_model.json

The input may be repeated, a timeline directory, or a merged JSON containing
an ``events``/``records`` list.  No GPU is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from collections import defaultdict
from typing import Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megatron.core.datasets.dspp_execution_cost import (
    DsppExecutionCostModel,
    meta_from_timeline_record,
)


def _paths(values: Iterable[str]):
    for value in values:
        path = Path(value)
        if path.is_dir():
            yield from sorted(path.glob("*.json"))
        else:
            yield path


def _records(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if "events" in payload:
        return payload["events"]
    if "records" in payload:
        return payload["records"]
    return []


def _prediction_report(model, records):
    groups = defaultdict(list)
    source_counts = defaultdict(int)
    for record in records:
        kind = str(record.get("kind", ""))[:1]
        if kind not in "FBW" or "stage" not in record or "duration_ms" not in record:
            continue
        try:
            meta = meta_from_timeline_record(record)
        except (KeyError, TypeError, ValueError):
            continue
        prediction = model.predict_detailed(int(record["stage"]), kind, meta)
        actual = float(record["duration_ms"])
        absolute = abs(actual - prediction.duration_ms)
        relative = absolute / max(actual, 1e-6)
        groups[(int(record["stage"]), kind)].append((absolute, relative))
        source_counts[prediction.source] += 1
    rows = []
    for (stage, op), errors in sorted(groups.items()):
        absolute = [item[0] for item in errors]
        relative = [item[1] for item in errors]
        rows.append({
            "stage": stage,
            "op": op,
            "samples": len(errors),
            "absolute_error_median_ms": _percentile(absolute, 0.5),
            "absolute_error_p95_ms": _percentile(absolute, 0.95),
            "relative_error_median": _percentile(relative, 0.5),
            "relative_error_p95": _percentile(relative, 0.95),
        })
    total = sum(source_counts.values())
    return {
        "records": total,
        "source_counts": dict(sorted(source_counts.items())),
        "profile_covered_fraction": (
            sum(
                value for source, value in source_counts.items()
                if source in {"exact-profile", "interpolated-profile", "fitted-profile"}
            ) / max(1, total)
        ),
        "groups": rows,
    }


def _percentile(values, percentile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--validation-input", action="append", default=[])
    parser.add_argument(
        "--include-validation-in-output",
        action="store_true",
        help="Evaluate on held-out records first, then refit the saved model on all records",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--base-profile",
        help="Optional PP-free compute profile. When supplied, timeline records "
        "fit a stage/op affine correction while preserving its shape-dependent curve.",
    )
    parser.add_argument("--report", help="Optional calibration coverage JSON")
    args = parser.parse_args()
    records = []
    for path in _paths(args.input):
        records.extend(_records(path))
    if not records:
        raise SystemExit("no timeline events found")
    validation_records = []
    for path in _paths(args.validation_input):
        validation_records.extend(_records(path))
    base_model = (
        DsppExecutionCostModel.from_json(args.base_profile)
        if args.base_profile else None
    )
    evaluation_model = (
        base_model.with_stage_op_calibration(records)
        if base_model is not None else DsppExecutionCostModel.from_records(records)
    )
    output_records = (
        records + validation_records
        if args.include_validation_in_output else records
    )
    model = (
        base_model.with_stage_op_calibration(output_records)
        if base_model is not None else DsppExecutionCostModel.from_records(output_records)
    )
    model.save(args.output)
    if args.report:
        groups = defaultdict(list)
        for record in records:
            kind = str(record.get("kind", ""))[:1]
            if kind in "FBW" and "stage" in record and "duration_ms" in record:
                groups[(int(record["stage"]), kind)].append(record)

        def _fit_stats(stage, op):
            calibration = model.stage_op_calibrations.get((stage, op))
            fitted = model.fitted.get((stage, op))
            if calibration is not None:
                return {
                    "residual_median_ms": calibration.residual_median_ms,
                    "residual_p95_ms": calibration.residual_p95_ms,
                    "duration_median_ms": 0.0,
                    "scale": calibration.scale,
                    "overhead_ms": calibration.overhead_ms,
                }
            return {
                "residual_median_ms": fitted.residual_median_ms if fitted else 0.0,
                "residual_p95_ms": fitted.residual_p95_ms if fitted else 0.0,
                "duration_median_ms": fitted.duration_median_ms if fitted else 0.0,
                "scale": 1.0,
                "overhead_ms": 0.0,
            }

        report = {
            "fit_records": len(records),
            "output_records": len(output_records),
            "sequence_chunks": sorted({
                meta_from_timeline_record(row).chunk_size
                for row in output_records if row.get("items")
            }),
            "groups": [],
        }
        for (stage, op), rows in sorted(groups.items()):
            stats = _fit_stats(stage, op)
            stats.update({
                "stage": stage,
                "op": op,
                "samples": len(rows),
                "unique_signatures": len({row.get("signature") for row in rows}),
                "sequence_chunks": sorted({
                    meta_from_timeline_record(row).chunk_size for row in rows
                }),
            })
            report["groups"].append(stats)
        if validation_records:
            report["held_out_validation"] = _prediction_report(
                evaluation_model, validation_records
            )
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output} from {len(output_records)} task records")


if __name__ == "__main__":
    main()
