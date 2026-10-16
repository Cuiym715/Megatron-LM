#!/usr/bin/env python3
"""Merge compatible compute profiles and plot the measured chunk frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megatron.core.datasets.dspp_compute_profile import (
    generate_candidate_menu,
    load_compute_profile,
    save_compute_profile,
)


FAMILIES = ("full", "tail", "middle", "pack")


def _record_key(record):
    shape = record.shape
    return (
        shape.stage,
        shape.operation,
        shape.capacity,
        shape.query_lengths,
        shape.history_lengths,
    )


def _pareto_capacities(scores):
    vectors = {
        score.capacity: tuple(
            getattr(score, f"{family}_efficiency") for family in FAMILIES
        )
        for score in scores
    }
    return tuple(sorted(
        capacity
        for capacity, vector in vectors.items()
        if not any(
            other_capacity != capacity
            and all(a >= b for a, b in zip(other, vector, strict=True))
            and any(a > b for a, b in zip(other, vector, strict=True))
            for other_capacity, other in vectors.items()
        )
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--combined-profile")
    parser.add_argument("--efficiency-threshold", type=float, default=0.90)
    args = parser.parse_args()

    # Later files override an identical shape from an earlier file. This lets
    # a targeted warmup/recheck replace a cold-start-contaminated point.
    records = {}
    metadata = {}
    for profile_path in args.profiles:
        payload = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        if not metadata:
            metadata = dict(payload.get("metadata", {}))
        for record in load_compute_profile(profile_path):
            records[_record_key(record)] = record
    merged = tuple(records[key] for key in sorted(records))
    capacities = sorted({record.shape.capacity for record in merged})
    metadata.update({
        "profile_capacity_bounds": [capacities[0], capacities[-1]],
        "merged_from": [str(Path(path)) for path in args.profiles],
    })
    if args.combined_profile:
        save_compute_profile(args.combined_profile, merged, metadata=metadata)

    menu = generate_candidate_menu(
        merged,
        efficiency_threshold=args.efficiency_threshold,
        max_candidates=max(1, len(capacities)),
    )
    pareto = _pareto_capacities(menu.scores)
    plateau = tuple(
        score.capacity
        for score in menu.scores
        if score.overall_efficiency >= args.efficiency_threshold
    )
    summary = {
        "profiles": args.profiles,
        "capacities": capacities,
        "profile_records": len(merged),
        "efficiency_threshold": args.efficiency_threshold,
        "pareto_capacities": list(pareto),
        "robust_plateau_capacities": list(plateau),
        "scores": [
            {
                "capacity": score.capacity,
                **{
                    family: getattr(score, f"{family}_efficiency")
                    for family in FAMILIES
                },
                "robust": score.overall_efficiency,
                "pareto": score.capacity in pareto,
                "robust_plateau": score.capacity in plateau,
            }
            for score in menu.scores
        ],
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt

    score_by_capacity = {score.capacity: score for score in menu.scores}
    colors = {
        "full": "#2f6fdf",
        "tail": "#14a05a",
        "middle": "#e75b0a",
        "pack": "#8b3fd6",
    }
    figure, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    figure.suptitle("DSPP measured compute efficiency across chunk capacities", fontsize=18)

    for axis, zoom in ((axes[0], False), (axes[1], True)):
        shown = [capacity for capacity in capacities if not zoom or capacity >= 2048]
        for family in FAMILIES:
            axis.plot(
                shown,
                [getattr(score_by_capacity[c], f"{family}_efficiency") for c in shown],
                marker="o",
                linewidth=2,
                label=family,
                color=colors[family],
            )
        axis.axhline(
            args.efficiency_threshold,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f"robust threshold={args.efficiency_threshold:.2f}" if not zoom else None,
        )
        axis.set_ylim(0.0 if not zoom else 0.65, 1.04)
        axis.set_xlabel("chunk capacity")
        axis.set_ylabel("normalized measured throughput")
        axis.grid(alpha=0.25)
        axis.set_title("Full measured range" if not zoom else "Expanded 2K–8K view")
        if not zoom:
            axis.set_xscale("log", base=2)
            axis.legend(loc="lower right")

    robust = [score_by_capacity[c].overall_efficiency for c in capacities]
    axes[2].plot(capacities, robust, color="#374151", marker="o", linewidth=2)
    axes[2].axhline(args.efficiency_threshold, color="gray", linestyle="--", linewidth=1.5)
    dominated = [c for c in capacities if c not in pareto]
    axes[2].scatter(
        dominated,
        [score_by_capacity[c].overall_efficiency for c in dominated],
        color="#9ca3af",
        edgecolor="#4b5563",
        s=70,
        zorder=3,
        label="dominated in 4-family space",
    )
    axes[2].scatter(
        pareto,
        [score_by_capacity[c].overall_efficiency for c in pareto],
        color="#dc2626",
        edgecolor="#7f1d1d",
        s=95,
        zorder=4,
        label="4-family Pareto point",
    )
    label_offsets = {
        3072: (-8, 10),
        3840: (0, 13),
        4096: (9, -13),
        4608: (0, -13),
        5120: (0, 10),
        5632: (0, -13),
    }
    for capacity in capacities:
        if capacity >= 2048:
            x_offset, y_offset = label_offsets.get(capacity, (0, 8))
            axes[2].annotate(
                str(capacity),
                (capacity, score_by_capacity[capacity].overall_efficiency),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    axes[2].set_xlabel("chunk capacity")
    axes[2].set_ylabel("robust efficiency = min(four families)")
    axes[2].set_ylim(0.0, 1.04)
    axes[2].set_title("Robust score and four-family Pareto membership")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="lower right")

    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(f"capacities={capacities}")
    print(f"pareto={list(pareto)} robust_plateau={list(plateau)}")
    print(f"wrote {output} and {summary_path}")


if __name__ == "__main__":
    main()
