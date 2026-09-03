"""Opt-in CUDA-event timeline and compact SVG output for DSPP.

The normal training path only constructs a disabled recorder.  Synchronizing,
per-task CUDA events, filesystem output, and the profiling barrier are all
restricted to the explicitly selected iteration.
"""

from __future__ import annotations

import contextlib
import html
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from megatron.core.datasets.dspp_batch_plan import DsppMicrobatchMeta
from megatron.core.datasets.dspp_ordering import task_signature


_COLORS = {
    "F0": "#6C8EBF",
    "F1": "#82B366",
    "B0": "#D79B00",
    "B1": "#B85450",
    "W0": "#9673A6",
    "W1": "#D6B656",
}


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_timeline(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_stage: Dict[int, List[Tuple[float, float]]] = {}
    durations: Dict[str, List[float]] = {}
    durations_by_stage: Dict[int, Dict[str, List[float]]] = {}
    for record in records:
        stage = int(record["stage"])
        begin, end = float(record["start_ms"]), float(record["end_ms"])
        by_stage.setdefault(stage, []).append((begin, end))
        durations.setdefault(str(record["kind"]), []).append(end - begin)
        durations_by_stage.setdefault(stage, {}).setdefault(
            str(record["kind"]), []
        ).append(end - begin)

    stage_summary = {}
    for stage, intervals in sorted(by_stage.items()):
        intervals.sort()
        merged: List[List[float]] = []
        for begin, end in intervals:
            if merged and begin <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([begin, end])
        first = min(begin for begin, _ in intervals)
        last = max(end for _, end in intervals)
        busy = sum(end - begin for begin, end in merged)
        span = last - first
        stage_summary[str(stage)] = {
            "first_compute_ms": first,
            "last_compute_ms": last,
            "busy_ms": busy,
            "bubble_ms": max(0.0, span - busy),
            "compute_utilization": busy / span if span else 0.0,
        }
    all_starts = [float(record["start_ms"]) for record in records]
    all_ends = [float(record["end_ms"]) for record in records]
    return {
        "measurement": (
            "CUDA-event compute-stream wall-time envelope. P2P API calls are outside "
            "task ranges; concurrent communication may still contend for GPU resources."
        ),
        "critical_span_ms": max(all_ends) - min(all_starts) if records else 0.0,
        "stages": stage_summary,
        "task_duration_ms": {
            kind: {
                "median": _percentile(values, 0.5),
                "p95": _percentile(values, 0.95),
            }
            for kind, values in sorted(durations.items())
        },
        "task_duration_ms_by_stage": {
            str(stage): {
                kind: {
                    "median": _percentile(values, 0.5),
                    "p95": _percentile(values, 0.95),
                }
                for kind, values in sorted(per_kind.items())
            }
            for stage, per_kind in sorted(durations_by_stage.items())
        },
    }


def build_profile_costs(records: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Return max-stage measured F+B+W time for each task signature."""

    per_task_stage: Dict[Tuple[int, int, int], Tuple[str, float]] = {}
    for record in records:
        key = (
            int(record["logical_microbatch"]),
            int(record["physical_microbatch"]),
            int(record["stage"]),
        )
        signature = str(record["signature"])
        old_signature, old_cost = per_task_stage.get(key, (signature, 0.0))
        per_task_stage[key] = (old_signature, old_cost + float(record["duration_ms"]))
    per_task: Dict[Tuple[int, int], Tuple[str, float]] = {}
    for (logical_id, physical_id, _stage), (signature, cost) in per_task_stage.items():
        key = (logical_id, physical_id)
        old_signature, old_cost = per_task.get(key, (signature, 0.0))
        per_task[key] = (old_signature, max(old_cost, cost))
    by_signature: Dict[str, List[float]] = {}
    for signature, cost in per_task.values():
        by_signature.setdefault(signature, []).append(cost)
    return {
        signature: sum(costs) / len(costs)
        for signature, costs in sorted(by_signature.items())
    }


def render_timeline_svg(records: Sequence[Mapping[str, Any]], title: str) -> str:
    stages = sorted({int(record["stage"]) for record in records})
    width, left, right, row_height = 2400, 130, 25, 48
    height = 78 + row_height * max(1, len(stages)) + 48
    max_end = max((float(record["end_ms"]) for record in records), default=1.0)
    scale = (width - left - right) / max(max_end, 1e-9)
    rows = {stage: 74 + index * row_height for index, stage in enumerate(stages)}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="20" y="28" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
    ]
    legend_x = left
    for kind in ("F0", "F1", "B0", "B1", "W0", "W1"):
        parts.append(
            f'<rect x="{legend_x}" y="42" width="16" height="12" '
            f'fill="{_COLORS[kind]}" stroke="#444" stroke-width="0.35"/>'
            f'<text x="{legend_x + 21}" y="53" font-family="monospace" font-size="11">{kind}</text>'
        )
        legend_x += 72
    parts.append(
        f'<text x="{legend_x + 8}" y="53" font-family="sans-serif" font-size="11">'
        'One compute lane per stage; W shorter than 3 px is a visibility marker.</text>'
    )
    for stage in stages:
        y = rows[stage]
        parts.append(
            f'<text x="18" y="{y + 16}" font-family="monospace" font-size="14">stage {stage}</text>'
        )
        parts.append(
            f'<line x1="{left}" y1="{y + 24}" x2="{width-right}" y2="{y + 24}" stroke="#dddddd"/>'
        )
    for record in records:
        stage = int(record["stage"])
        begin, end = float(record["start_ms"]), float(record["end_ms"])
        x = left + begin * scale
        actual_width = (end - begin) * scale
        kind = str(record["kind"])
        rect_width = max(3.0 if kind.startswith("W") else 1.0, actual_width)
        microbatch_id = record.get("microbatch_id")
        if microbatch_id is None:
            microbatch_id = f'{record["logical_microbatch"]}:{record["physical_microbatch"]}'
        short_label = f'm{microbatch_id} {kind}'
        label = (
            f'{short_label} task={record["logical_microbatch"]}:'
            f'{record["physical_microbatch"]}'
        )
        y = rows[stage]
        parts.append(
            f'<rect x="{x:.2f}" y="{y}" width="{rect_width:.2f}" height="22" '
            f'fill="{_COLORS.get(kind, "#999999")}" stroke="#444" stroke-width="0.35">'
            f'<title>{html.escape(label)} {begin:.3f}-{end:.3f} ms</title></rect>'
        )
        if rect_width >= 20:
            parts.append(
                f'<text x="{x + 2:.2f}" y="{y + 15}" font-family="monospace" font-size="10" '
                f'fill="white">{html.escape(short_label)}</text>'
            )
    parts.append(
        f'<text x="{left}" y="{height-15}" font-family="sans-serif" font-size="12">0 ms</text>'
    )
    parts.append(
        f'<text x="{width-right-80}" y="{height-15}" font-family="sans-serif" font-size="12">{max_end:.2f} ms</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


class DsppTimelineRecorder:
    def __init__(self, args: Any, *, stage: int, ordering: Mapping[str, Any],
                 microbatch_report: Optional[str] = None):
        selected = int(getattr(args, "dspp_timeline_iteration", 0) or 0)
        current = int(getattr(args, "curr_iteration", -1)) + 1
        output_dir = getattr(args, "dspp_timeline_dir", None)
        self.enabled = bool(output_dir and selected == current)
        torch_profile_selected = int(
            getattr(args, "dspp_torch_profiler_iteration", 0) or 0
        )
        torch_profile_dir = getattr(args, "dspp_torch_profiler_dir", None)
        self.torch_profile_enabled = bool(
            torch_profile_dir and torch_profile_selected == current
        )
        self.recording = self.enabled or self.torch_profile_enabled
        self.stage = stage
        self.iteration = current
        self.output_dir = Path(output_dir) if output_dir else None
        self.ordering = dict(ordering)
        self.microbatch_ids = {
            tuple(task): index
            for index, task in enumerate(self.ordering.get("entrance_order", ()))
        }
        self.microbatch_report = microbatch_report
        self.records: List[Dict[str, Any]] = []
        self.anchor = None
        self.anchor_cpu_ns = None
        if self.enabled:
            dist.barrier()
            self.anchor = torch.cuda.Event(enable_timing=True)
            self.anchor.record()
            self.anchor_cpu_ns = time.perf_counter_ns()

    @contextlib.contextmanager
    def task(
        self,
        kind: str,
        task_id: Tuple[int, int],
        meta: DsppMicrobatchMeta,
        *,
        chunk: int,
    ):
        if not self.enabled and not self.torch_profile_enabled:
            yield
            return
        label = f"DSPP/{kind}/s{self.stage}/mb{task_id[0]}:{task_id[1]}"
        begin = end = None
        with contextlib.ExitStack() as stack:
            if self.torch_profile_enabled:
                stack.enter_context(torch.profiler.record_function(label))
            if self.enabled:
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.nvtx.range_push(label)
                begin.record()
            try:
                yield
            finally:
                if self.enabled:
                    end.record()
                    torch.cuda.nvtx.range_pop()
        if self.enabled:
            self.records.append(
                {
                    "kind": kind,
                    "stage": self.stage,
                    "chunk": chunk,
                    "logical_microbatch": task_id[0],
                    "physical_microbatch": task_id[1],
                    "microbatch_id": self.microbatch_ids.get(task_id),
                    "signature": task_signature(meta),
                    "valid_tokens": meta.valid_token_count,
                    "items": [
                        {
                            "sequence_id": item.sequence_id,
                            "segment_id": item.segment_id,
                            "segment_count": item.segment_count,
                            "token_offset": item.token_offset,
                            "token_length": item.token_length,
                            "sequence_length": item.sequence_length,
                        }
                        for item in meta.items
                    ],
                    "_begin": begin,
                    "_end": end,
                }
            )

    @contextlib.contextmanager
    def communication(self, label: str):
        if not self.enabled and not self.torch_profile_enabled:
            yield
            return
        full_label = f"DSPP/P2P/{label}"
        with contextlib.ExitStack() as stack:
            if self.torch_profile_enabled:
                stack.enter_context(torch.profiler.record_function(full_label))
            if self.enabled:
                torch.cuda.nvtx.range_push(full_label)
            try:
                yield
            finally:
                if self.enabled:
                    torch.cuda.nvtx.range_pop()

    def finish(self) -> Optional[Path]:
        if not self.enabled:
            return None
        torch.cuda.synchronize()
        serializable = []
        for record in self.records:
            begin, end = record.pop("_begin"), record.pop("_end")
            record["start_ms"] = self.anchor.elapsed_time(begin)
            record["end_ms"] = self.anchor.elapsed_time(end)
            record["duration_ms"] = begin.elapsed_time(end)
            serializable.append(record)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rank = dist.get_rank()
        rank_path = self.output_dir / f"iteration_{self.iteration:06d}_rank_{rank}.json"
        with rank_path.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "iteration": self.iteration,
                    "rank": rank,
                    "stage": self.stage,
                    "anchor_cpu_ns": self.anchor_cpu_ns,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "ordering": self.ordering,
                    "measurement": "CUDA-event compute-stream wall-time envelope; P2P calls are outside task ranges",
                    "events": serializable,
                },
                stream,
                indent=2,
            )
        dist.barrier()
        if rank == 0:
            paths = sorted(self.output_dir.glob(f"iteration_{self.iteration:06d}_rank_*.json"))
            payloads = []
            memory_by_stage = {}
            for path in paths:
                with path.open("r", encoding="utf-8") as stream:
                    payload = json.load(stream)
                payloads.append(payload)
                memory_by_stage[str(payload["stage"])] = {
                    "peak_allocated_bytes": payload["peak_allocated_bytes"],
                    "peak_reserved_bytes": payload["peak_reserved_bytes"],
                }
            reference_cpu_ns = min(payload["anchor_cpu_ns"] for payload in payloads)
            all_records = []
            for payload in payloads:
                offset_ms = (payload["anchor_cpu_ns"] - reference_cpu_ns) / 1e6
                for record in payload["events"]:
                    record["start_ms"] += offset_ms
                    record["end_ms"] += offset_ms
                    all_records.append(record)
            all_records.sort(key=lambda item: (item["start_ms"], item["stage"]))
            stem = self.output_dir / f"iteration_{self.iteration:06d}"
            with stem.with_suffix(".svg").open("w", encoding="utf-8") as stream:
                stream.write(render_timeline_svg(all_records, f"DSPP iteration {self.iteration}"))
            with Path(f"{stem}_summary.json").open("w", encoding="utf-8") as stream:
                summary = summarize_timeline(all_records)
                summary["memory_by_stage"] = memory_by_stage
                json.dump(summary, stream, indent=2)
            with Path(f"{stem}_costs.json").open("w", encoding="utf-8") as stream:
                json.dump(
                    {"source": "CUDA events", "costs": build_profile_costs(all_records)},
                    stream,
                    indent=2,
                )
            if self.microbatch_report is not None:
                with Path(f"{stem}_microbatches.md").open("w", encoding="utf-8") as stream:
                    stream.write(self.microbatch_report)
        dist.barrier()
        return rank_path
