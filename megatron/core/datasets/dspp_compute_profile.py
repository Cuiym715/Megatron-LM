"""Offline, PP-free DSPP compute profiling and candidate generation.

The profiler operates on *physical compute-task shapes*, not on complete
training iterations.  A task is described by its fixed capacity, per-item
query lengths and history lengths.  This lets candidate generation account for
the fact that a capacity ``c`` induces full segments, tails, middle-of-chain
segments and residual packs with different kernel shapes.

The module intentionally has no dependency on a particular model.  The CLI
accepts a small adapter callback that constructs one real F/B/W compute task
for a requested shape, so the same procedure can be reused with different
models, GPUs, dtypes and attention implementations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import defaultdict
import importlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DsppComputeShape:
    stage: int
    operation: str
    capacity: int
    query_lengths: Tuple[int, ...]
    history_lengths: Tuple[int, ...]
    work_units: Optional[float] = None

    @property
    def item_count(self) -> int:
        return len(self.query_lengths)

    @property
    def valid_tokens(self) -> int:
        return sum(self.query_lengths)

    @property
    def padding_tokens(self) -> int:
        return max(0, self.capacity - self.valid_tokens)

    @property
    def key_lengths(self) -> Tuple[int, ...]:
        return tuple(h + q for q, h in zip(self.query_lengths, self.history_lengths))

    @property
    def token_work(self) -> float:
        return float(self.valid_tokens)

    @property
    def attention_work(self) -> float:
        return float(sum(q * (h + q) for q, h in zip(self.query_lengths, self.history_lengths)))

    @property
    def category(self) -> str:
        if self.item_count > 1:
            return "pack"
        # A tail is any physical item that does not fill its capacity.  This
        # includes an independent short sequence (history=0), not only a
        # continuation segment.  Treating short independent items as ``full``
        # would make the menu blind to the residual-packing boundary.
        if self.query_lengths[0] < self.capacity:
            return "tail"
        if self.history_lengths[0] > 0:
            return "middle"
        return "full"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["query_lengths"] = list(self.query_lengths)
        payload["history_lengths"] = list(self.history_lengths)
        payload["category"] = self.category
        payload["token_work"] = self.token_work
        payload["attention_work"] = self.attention_work
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DsppComputeShape":
        shape = cls(
            stage=int(payload.get("stage", 0)),
            operation=str(payload.get("operation", payload.get("op", "F"))),
            capacity=int(payload["capacity"]),
            query_lengths=tuple(int(value) for value in payload["query_lengths"]),
            history_lengths=tuple(int(value) for value in payload["history_lengths"]),
            work_units=(
                float(payload["work_units"])
                if payload.get("work_units") is not None
                else None
            ),
        )
        shape.validate()
        return shape

    def validate(self) -> None:
        if self.operation not in {"F", "B", "W"}:
            raise ValueError(f"unsupported compute operation: {self.operation}")
        if self.capacity <= 0 or not self.query_lengths:
            raise ValueError("compute shape must have positive capacity and items")
        if len(self.query_lengths) != len(self.history_lengths):
            raise ValueError("query/history item counts must match")
        if any(q <= 0 or h < 0 for q, h in zip(self.query_lengths, self.history_lengths)):
            raise ValueError("query lengths must be positive and history non-negative")
        if self.valid_tokens > self.capacity:
            raise ValueError("physical shape exceeds its capacity")


@dataclass(frozen=True)
class DsppComputeProfileRecord:
    shape: DsppComputeShape
    median_ms: float
    p95_ms: float
    repeats: int
    source: str = "cuda-graph"

    def to_dict(self) -> Dict[str, Any]:
        payload = self.shape.to_dict()
        payload.update(
            median_ms=float(self.median_ms),
            p95_ms=float(self.p95_ms),
            repeats=int(self.repeats),
            source=self.source,
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DsppComputeProfileRecord":
        return cls(
            shape=DsppComputeShape.from_dict(payload),
            median_ms=float(payload.get("median_ms", payload.get("duration_ms", 0.0))),
            p95_ms=float(payload.get("p95_ms", payload.get("median_ms", 0.0))),
            repeats=int(payload.get("repeats", 1)),
            source=str(payload.get("source", "profile")),
        )


@dataclass(frozen=True)
class DsppCandidateScore:
    capacity: int
    full_efficiency: float
    tail_efficiency: float
    middle_efficiency: float
    pack_efficiency: float
    overall_efficiency: float
    coverage: float
    shape_count: int
    retained: bool
    pareto: bool = False
    robust_plateau: bool = False
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DsppCandidateMenu:
    candidates: Tuple[int, ...]
    scores: Tuple[DsppCandidateScore, ...]
    plateau_efficiency: float
    threshold: float
    profile_records: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 2,
            "candidates": list(self.candidates),
            # ``candidates`` remains the compact compute-efficient subset for
            # backward compatibility. Runtime memory filtering and bounded
            # planning use the complete measured bank instead.
            "profile_bank": [score.to_dict() for score in self.scores],
            "pareto_capacities": [
                score.capacity for score in self.scores if score.pareto
            ],
            "robust_plateau_capacities": [
                score.capacity for score in self.scores if score.robust_plateau
            ],
            "scores": [score.to_dict() for score in self.scores],
            "plateau_efficiency": self.plateau_efficiency,
            "threshold": self.threshold,
            "profile_records": self.profile_records,
        }


@dataclass(frozen=True)
class DsppCapacityProbeResult:
    capacity: int
    feasible: bool
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    total_memory_bytes: int = 0
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DsppCapacityProbeResult":
        return cls(
            capacity=int(payload["capacity"]),
            feasible=bool(payload["feasible"]),
            peak_allocated_bytes=int(payload.get("peak_allocated_bytes", 0)),
            peak_reserved_bytes=int(payload.get("peak_reserved_bytes", 0)),
            total_memory_bytes=int(payload.get("total_memory_bytes", 0)),
            reason=payload.get("reason"),
        )


@dataclass(frozen=True)
class DsppCapacityDiscovery:
    min_capacity: int
    max_capacity: int
    search_alignment: int
    memory_utilization: float
    probes: Tuple[DsppCapacityProbeResult, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_capacity": self.min_capacity,
            "max_capacity": self.max_capacity,
            "search_alignment": self.search_alignment,
            "memory_utilization": self.memory_utilization,
            "probes": [probe.to_dict() for probe in self.probes],
        }


def discover_capacity_range(
    probe: Callable[[int], DsppCapacityProbeResult],
    *,
    legal_start_capacity: int = 1,
    search_alignment: int = 64,
    memory_utilization: float = 0.90,
    max_probe_attempts: int = 64,
) -> DsppCapacityDiscovery:
    """Discover the legal and memory-feasible capacity interval.

    The probe runs in an isolated process in the CLI.  This pure function owns
    only the search policy so it can be tested without CUDA.  No historical
    chunk value or training input participates in the search.
    """

    if legal_start_capacity <= 0 or search_alignment <= 0:
        raise ValueError("capacity discovery bounds and alignment must be positive")
    if not 0 < memory_utilization <= 1:
        raise ValueError("memory_utilization must be in (0, 1]")
    if max_probe_attempts <= 0:
        raise ValueError("max_probe_attempts must be positive")

    probes: list[DsppCapacityProbeResult] = []
    cache: Dict[int, bool] = {}

    def accepted(capacity: int) -> bool:
        if capacity in cache:
            return cache[capacity]
        if len(probes) >= max_probe_attempts:
            raise RuntimeError(
                "capacity discovery exhausted max_probe_attempts before finding a bound"
            )
        result = probe(capacity)
        if result.capacity != capacity:
            raise ValueError("capacity probe returned a result for a different capacity")
        probes.append(result)
        within_budget = (
            result.total_memory_bytes <= 0
            or result.peak_reserved_bytes
            <= memory_utilization * result.total_memory_bytes
        )
        cache[capacity] = result.feasible and within_budget
        return cache[capacity]

    # Find the first executable point.  If tiny shapes are unsupported by a
    # kernel, refine the transition at integer granularity: legal minimum is a
    # model/software property and must not be conflated with menu alignment.
    previous_failure = 0
    candidate = legal_start_capacity
    while not accepted(candidate):
        previous_failure = candidate
        candidate *= 2
    first_success = candidate
    if previous_failure:
        low = previous_failure + 1
        high = first_success
        while low < high:
            middle = (low + high) // 2
            if accepted(middle):
                high = middle
            else:
                low = middle + 1
        min_capacity = low
    else:
        min_capacity = first_success

    # Search the upper bound on an aligned grid.  The legal minimum itself may
    # be unaligned and remains part of the recorded interval, while menu points
    # use search_alignment for a bounded portable search resolution.
    aligned_min = max(
        search_alignment,
        ((min_capacity + search_alignment - 1) // search_alignment) * search_alignment,
    )
    if aligned_min == min_capacity or accepted(aligned_min):
        last_success = aligned_min
    else:
        # The feasible interval can be narrower than one search step.
        return DsppCapacityDiscovery(
            min_capacity=min_capacity,
            max_capacity=min_capacity,
            search_alignment=search_alignment,
            memory_utilization=memory_utilization,
            probes=tuple(probes),
        )

    first_failure = last_success * 2
    while accepted(first_failure):
        last_success = first_failure
        first_failure *= 2

    low_step = last_success // search_alignment
    high_step = first_failure // search_alignment
    while high_step - low_step > 1:
        middle_step = (low_step + high_step) // 2
        middle = middle_step * search_alignment
        if accepted(middle):
            low_step = middle_step
        else:
            high_step = middle_step
    return DsppCapacityDiscovery(
        min_capacity=min_capacity,
        max_capacity=low_step * search_alignment,
        search_alignment=search_alignment,
        memory_utilization=memory_utilization,
        probes=tuple(probes),
    )


def _aligned(value: int, alignment: int) -> int:
    return max(1, int(round(value / alignment) * alignment))


def build_compute_shape_plan(
    *,
    min_capacity: int,
    max_capacity: int,
    alignment: int = 256,
    stage_count: int = 1,
    operations: Sequence[str] = ("F", "B", "W"),
    history_ratios: Sequence[int] = (0, 1, 2, 4),
    tail_ratios: Sequence[float] = (0.25, 0.5, 0.75),
    item_counts: Sequence[int] = (2, 4),
    max_history_tokens: Optional[int] = None,
    max_shapes: int = 8192,
    capacities: Optional[Sequence[int]] = None,
) -> Tuple[DsppComputeShape, ...]:
    """Build a bounded, data-independent profile plan.

    Capacities start on a geometric grid.  The plan includes the shape family
    induced by every capacity, so a candidate is not judged solely by its full
    ``q=capacity`` task.
    """

    if min_capacity <= 0 or max_capacity < min_capacity or alignment <= 0:
        raise ValueError("invalid compute profile capacity bounds")
    if capacities is None:
        sampled = []
        # Keep the exact legal minimum in discovery metadata, but start the
        # performance grid at the first complete alignment point whenever the
        # feasible interval is wider than one alignment step.  Tiny legal
        # shapes are useful for kernel legality, not for a practical menu.
        value = max(
            alignment,
            ((min_capacity + alignment - 1) // alignment) * alignment,
        )
        if value > max_capacity:
            value = min_capacity
        while value < max_capacity:
            sampled.append(value)
            next_value = _aligned(max(value + alignment, value * 2), alignment)
            if next_value <= value:
                break
            value = next_value
        sampled.append(_aligned(max_capacity, alignment))
        capacities = sampled
    capacities = sorted(set(
        min(max_capacity, max(min_capacity, _aligned(int(c), alignment)))
        for c in capacities
    ))

    shapes = []
    for capacity in capacities:
        histories = tuple(
            min(int(ratio * capacity), max_history_tokens)
            if max_history_tokens is not None
            else int(ratio * capacity)
            for ratio in history_ratios
        )
        for stage in range(stage_count):
            for operation in operations:
                if operation == "W":
                    # Deferred weight-gradient GEMMs operate on the fixed
                    # dense capacity. Query/history/packing metadata does not
                    # change their matrix shapes, so profile W once rather
                    # than duplicating an identical task for every family.
                    shapes.append(
                        DsppComputeShape(
                            stage=stage,
                            operation=operation,
                            capacity=capacity,
                            query_lengths=(capacity,),
                            history_lengths=(0,),
                        )
                    )
                    continue
                for history in histories:
                    shapes.append(
                        DsppComputeShape(
                            stage=stage,
                            operation=operation,
                            capacity=capacity,
                            query_lengths=(capacity,),
                            history_lengths=(history,),
                        )
                    )
                for ratio in tail_ratios:
                    query = max(1, min(capacity - 1, int(round(capacity * ratio))))
                    for history in histories:
                        shapes.append(
                            DsppComputeShape(
                                stage=stage,
                                operation=operation,
                                capacity=capacity,
                                query_lengths=(query,),
                                history_lengths=(history,),
                            )
                        )
                for item_count in item_counts:
                    if item_count <= 1 or item_count > capacity:
                        continue
                    full_item = max(1, capacity // item_count)
                    half_item = max(1, capacity // (2 * item_count))
                    for query in (full_item, half_item):
                        shapes.append(
                            DsppComputeShape(
                                stage=stage,
                                operation=operation,
                                capacity=capacity,
                                query_lengths=(query,) * item_count,
                                history_lengths=(0,) * item_count,
                            )
                        )
                    # A real residual pack may begin with one continuation
                    # item and then place independent short sequences after
                    # it.  Profile that boundary explicitly as a separate
                    # physical family.
                    continuation_q = max(1, capacity // 2)
                    remaining = max(1, capacity - continuation_q)
                    short_q = max(1, remaining // (item_count - 1))
                    while continuation_q + short_q * (item_count - 1) > capacity:
                        short_q -= 1
                    if short_q > 0:
                        shapes.append(
                            DsppComputeShape(
                                stage=stage,
                                operation=operation,
                                capacity=capacity,
                                query_lengths=(continuation_q,) + (short_q,) * (item_count - 1),
                                history_lengths=(capacity,) + (0,) * (item_count - 1),
                            )
                        )
    # Clipped history bounds and integer ratios may produce the same physical
    # shape. Measure it once.
    shapes = list(dict.fromkeys(shapes))
    # Never silently drop a shape family: doing so can make a capacity appear
    # fast only because its tail/pack/middle case was not measured.  Callers
    # that need a smaller run must reduce the capacity grid or the family
    # ratios explicitly.
    if len(shapes) > max_shapes:
        raise ValueError(
            f"compute profile plan has {len(shapes)} shapes, exceeding max_shapes={max_shapes}; "
            "reduce the capacity grid/family ratios or increase max_shapes"
        )
    return tuple(shapes)


def load_compute_profile(path: str | Path) -> Tuple[DsppComputeProfileRecord, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("records", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("compute profile must contain a list of records")
    return tuple(DsppComputeProfileRecord.from_dict(row) for row in rows)


def save_compute_profile(
    path: str | Path,
    records: Iterable[DsppComputeProfileRecord],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "version": 1,
                "metadata": dict(metadata or {}),
                "records": [record.to_dict() for record in records],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _work_units(record: DsppComputeProfileRecord) -> float:
    shape = record.shape
    work = shape.work_units
    if work is None:
        # Operation-local fallback.  It is never compared across F/B/W or
        # across unrelated shape families; it only removes the unavoidable
        # scaling with q/k when an adapter does not provide exact work units.
        work = shape.attention_work if shape.operation in {"F", "B"} else shape.token_work
    return max(float(work), 1e-9)


def _family_key(record: DsppComputeProfileRecord) -> Tuple[Any, ...]:
    shape = record.shape
    q = max(shape.valid_tokens, 1)
    history_ratio = round(sum(shape.history_lengths) / q, 3)
    utilization = round(shape.valid_tokens / max(shape.capacity, 1), 3)
    return (
        shape.stage,
        shape.operation,
        shape.category,
        shape.item_count,
        history_ratio,
        utilization,
    )


def generate_candidate_menu(
    records: Sequence[DsppComputeProfileRecord],
    *,
    efficiency_threshold: float = 0.90,
    min_coverage: float = 0.75,
    lower_quantile: float = 0.20,
    max_candidates: int = 8,
    require_work_units: bool = True,
) -> DsppCandidateMenu:
    """Retain capacities whose induced shape family is on the compute plateau."""

    if not records:
        raise ValueError("cannot generate candidates from an empty profile")
    if not 0 < efficiency_threshold <= 1 or not 0 <= lower_quantile < 1:
        raise ValueError("invalid efficiency threshold or quantile")
    if require_work_units and any(record.shape.work_units is None for record in records):
        raise ValueError(
            "candidate generation requires adapter-provided work_units for every shape; "
            "raw latency cannot be compared across different q/k sizes"
        )
    by_capacity: Dict[int, list[DsppComputeProfileRecord]] = {}
    for record in records:
        by_capacity.setdefault(record.shape.capacity, []).append(record)

    # Normalize throughput only against the same physical family.  A full
    # F task, a middle continuation, and a 4-item residual pack do not have a
    # common FLOP unit or kernel efficiency, so comparing their raw
    # work/duration values would bias the menu toward whichever family happens
    # to have the largest analytical proxy.
    family_rows: Dict[Tuple[Any, ...], list[Tuple[int, DsppComputeProfileRecord]]] = defaultdict(list)
    for capacity, rows in by_capacity.items():
        for row in rows:
            family_rows[_family_key(row)].append((capacity, row))
    normalized: Dict[int, Dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rows in family_rows.values():
        plateau = max(
            _work_units(row) / max(row.median_ms, 1e-9)
            for _capacity, row in rows
        )
        for capacity, row in rows:
            ratio = (_work_units(row) / max(row.median_ms, 1e-9)) / max(plateau, 1e-9)
            normalized[capacity][row.shape.category].append(min(1.0, max(0.0, ratio)))

    raw_scores = []
    for capacity, rows in sorted(by_capacity.items()):
        by_category: Dict[str, list[float]] = {
            name: normalized[capacity].get(name, [])
            for name in ("full", "tail", "middle", "pack")
        }
        category_scores = {
            category: _percentile(values, lower_quantile) if values else 0.0
            for category, values in by_category.items()
        }
        present = sum(bool(values) for values in by_category.values())
        coverage = present / 4.0
        overall = min(category_scores.values()) if present == 4 else 0.0
        raw_scores.append((capacity, category_scores, overall, coverage, sum(len(v) for v in by_category.values())))

    # Scores are already ratios to the best measured family throughput.  The
    # plateau is therefore a dimensionless reference, not a FLOP-weighted
    # cross-category sum.
    plateau = 1.0
    category_names = ("full", "tail", "middle", "pack")
    score_vectors = {
        capacity: tuple(category_scores[name] for name in category_names)
        for capacity, category_scores, _overall, _coverage, _count in raw_scores
    }
    pareto = set()
    for capacity, vector in score_vectors.items():
        dominated = any(
            other != capacity
            and all(a >= b for a, b in zip(other_vector, vector))
            and any(a > b for a, b in zip(other_vector, vector))
            for other, other_vector in score_vectors.items()
        )
        if not dominated:
            pareto.add(capacity)

    scores = []
    for capacity, category_scores, overall, coverage, shape_count in raw_scores:
        robust = overall >= efficiency_threshold
        specialist = (
            capacity in pareto
            and max(category_scores.values()) >= efficiency_threshold
        )
        retained = coverage >= min_coverage and (robust or specialist)
        reason = None
        if coverage < min_coverage:
            reason = "insufficient-shape-family-coverage"
        elif not robust and not specialist:
            reason = (
                "dominated-compute-shape-tradeoff"
                if capacity not in pareto
                else "below-compute-efficiency-plateau"
            )
        scores.append(
            DsppCandidateScore(
                capacity=capacity,
                full_efficiency=category_scores["full"],
                tail_efficiency=category_scores["tail"],
                middle_efficiency=category_scores["middle"],
                pack_efficiency=category_scores["pack"],
                overall_efficiency=overall,
                coverage=coverage,
                shape_count=shape_count,
                retained=retained,
                pareto=capacity in pareto,
                robust_plateau=robust,
                rejection_reason=reason,
            )
        )
    retained_scores = [score for score in scores if score.retained]
    if not retained_scores:
        feasible = [score for score in scores if score.coverage >= min_coverage]
        if feasible:
            # A profile should not produce an unusable empty menu merely
            # because every measured point is below a strict plateau
            # threshold.  Keep the most robust measured point and expose the
            # fallback in its rejection reason.
            best = max(feasible, key=lambda score: (score.overall_efficiency, -score.capacity))
            retained_scores = [best]
            scores = [
                score if score.capacity != best.capacity else DsppCandidateScore(
                    **{**score.to_dict(), "retained": True,
                       "rejection_reason": "fallback-best-profile"}
                )
                for score in scores
            ]
    if len(retained_scores) > max_candidates:
        # Preserve the best measured representative for every physical
        # category before filling by robust score. This maintains menu
        # diversity when the future input mix is not known offline.
        keep = set()
        for field in (
            "full_efficiency", "tail_efficiency",
            "middle_efficiency", "pack_efficiency",
        ):
            leader = max(retained_scores, key=lambda score: getattr(score, field))
            keep.add(leader.capacity)
        keep.add(min(retained_scores, key=lambda score: score.capacity).capacity)
        for score in sorted(retained_scores, key=lambda row: row.overall_efficiency, reverse=True):
            if len(keep) >= max_candidates:
                break
            keep.add(score.capacity)
        retained_scores = [score for score in retained_scores if score.capacity in keep]
        scores = [
            score if score.capacity in keep else DsppCandidateScore(
                **{**score.to_dict(), "retained": False, "rejection_reason": "menu-size-limit"}
            )
            for score in scores
        ]
    return DsppCandidateMenu(
        candidates=tuple(sorted(score.capacity for score in retained_scores)),
        scores=tuple(scores),
        plateau_efficiency=float(plateau),
        threshold=float(efficiency_threshold),
        profile_records=len(records),
    )


def suggest_refinement_capacities(
    records: Sequence[DsppComputeProfileRecord],
    *,
    min_capacity: int,
    max_capacity: int,
    alignment: int,
    efficiency_threshold: float = 0.90,
    max_new_capacities: int = 6,
) -> Tuple[int, ...]:
    """Choose the next profile points from the measured efficiency curve.

    This is deliberately profile-driven.  It adds points around measured
    plateau boundaries/local maxima and bisects the widest unresolved
    intervals.  No training result or preselected chunk value participates.
    """

    if max_new_capacities <= 0:
        return ()
    measured = sorted({record.shape.capacity for record in records})
    if len(measured) < 2:
        return ()
    menu = generate_candidate_menu(
        records,
        efficiency_threshold=efficiency_threshold,
        min_coverage=0.0,
        require_work_units=True,
        max_candidates=max(len(measured), 1),
    )
    scores = {score.capacity: score.overall_efficiency for score in menu.scores}
    proposed: Dict[int, float] = {}

    def offer(value: int, priority: float) -> None:
        value = _aligned(value, alignment)
        if value < min_capacity or value > max_capacity or value in scores:
            return
        proposed[value] = max(priority, proposed.get(value, float("-inf")))

    # Probe one alignment step around efficient measured points.  This catches
    # tile/kernel-selection boundaries that a geometric grid alone can skip.
    for index, capacity in enumerate(measured):
        score = scores.get(capacity, 0.0)
        left = scores.get(measured[index - 1], score) if index else score
        right = scores.get(measured[index + 1], score) if index + 1 < len(measured) else score
        local_peak = score >= left and score >= right
        if score >= efficiency_threshold or local_peak:
            priority = 20.0 + score
            offer(capacity - alignment, priority)
            offer(capacity + alignment, priority)

    # Bisect intervals where the profile crosses the plateau threshold, has a
    # steep efficiency change, or is still wide in log-space.
    for left, right in zip(measured, measured[1:]):
        if right - left <= alignment:
            continue
        left_score = scores.get(left, 0.0)
        right_score = scores.get(right, 0.0)
        crosses = (left_score >= efficiency_threshold) != (right_score >= efficiency_threshold)
        near_plateau = max(left_score, right_score) >= efficiency_threshold
        slope = abs(right_score - left_score)
        width_steps = (right - left) / alignment
        priority = (100.0 if crosses else 0.0) + (10.0 if near_plateau else 0.0)
        priority += 5.0 * slope + math.log2(max(width_steps, 1.0))
        offer((left + right) // 2, priority)

    ranked = sorted(proposed.items(), key=lambda item: (-item[1], item[0]))
    return tuple(sorted(value for value, _priority in ranked[:max_new_capacities]))


def load_adapter(spec: str) -> Callable[..., Any]:
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"compute profile adapter is not callable: {spec}")
    return function


def profile_with_adapter(
    shapes: Sequence[DsppComputeShape],
    *,
    adapter: Callable[..., Any],
    device: str = "cuda",
    warmup: int = 3,
    repeats: int = 10,
    use_cuda_graph: bool = True,
) -> Tuple[DsppComputeProfileRecord, ...]:
    """Measure adapter-created tasks without PP or communication.

    The adapter must return either a callable or ``{"run": callable,
    "work_units": float}``.  The callable owns all model-specific tensor,
    KV-cache and autograd setup.  It must not allocate shape-dependent tensors
    inside the timed call if CUDA Graph replay is requested.
    """

    import gc
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA compute profiling requires a CUDA device")
    if repeats <= 0:
        raise ValueError("compute profiling repeats must be positive")
    results = []
    for shape in shapes:
        built = adapter(shape=shape, device=device)
        if callable(built):
            run = built
            work_units = shape.work_units
        elif isinstance(built, Mapping) and callable(built.get("run")):
            run = built["run"]
            work_units = built.get("work_units", shape.work_units)
            task_use_cuda_graph = use_cuda_graph and bool(
                built.get("use_cuda_graph", True)
            )
        else:
            raise TypeError("adapter must return callable or {'run': callable}")
        if callable(built):
            task_use_cuda_graph = use_cuda_graph
        if work_units is not None and shape.work_units != float(work_units):
            shape = DsppComputeShape(
                stage=shape.stage,
                operation=shape.operation,
                capacity=shape.capacity,
                query_lengths=shape.query_lengths,
                history_lengths=shape.history_lengths,
                work_units=float(work_units),
            )
        for _ in range(max(0, warmup)):
            run()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            graph = None
            source = "cuda-events"
            if task_use_cuda_graph:
                try:
                    graph = torch.cuda.CUDAGraph()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(graph):
                        run()
                    torch.cuda.synchronize()
                    source = "cuda-graph"
                except (RuntimeError, torch.AcceleratorError):
                    # Some custom kernels cannot be captured after an
                    # environment-specific allocator/context transition.
                    # Keep the shape measurable with CUDA events rather than
                    # discarding the whole offline calibration.
                    graph = None
                    torch.cuda.synchronize()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            for start, end in zip(starts, ends):
                start.record()
                graph.replay() if graph is not None else run()
                end.record()
            torch.cuda.synchronize()
            durations = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        else:
            import time

            durations = []
            for _ in range(repeats):
                start = time.perf_counter()
                run()
                durations.append((time.perf_counter() - start) * 1000.0)
            source = "host-timer"
        results.append(
            DsppComputeProfileRecord(
                shape=shape,
                median_ms=float(median(durations)),
                p95_ms=float(_percentile(durations, 0.95)),
                repeats=len(durations),
                source=source,
            )
        )
        # Release adapter-owned weights, KV buffers and CUDA graphs after each
        # shape. Otherwise a long offline scan accumulates one model replica
        # per shape and eventually measures allocator pressure instead of
        # kernel latency.
        del starts, ends, graph, run, built
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return tuple(results)
