"""Global fixed-chunk selection and fixed-vs-oracle regret analysis."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from statistics import median
import json
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple

from .dspp_dag import (
    DsppDagPlan,
    build_dspp_iteration_batch_plans,
    estimate_dspp_compute_lower_bound,
    plan_dspp_iteration,
)
from .dspp_execution_cost import load_execution_cost_model
from .dspp_chunk_tuner import _structural_chunk_candidates


LengthTrace = Sequence[Sequence[Sequence[int]]]


@dataclass(frozen=True)
class DsppProfileBankEntry:
    capacity: int
    full_efficiency: float = 0.0
    tail_efficiency: float = 0.0
    middle_efficiency: float = 0.0
    pack_efficiency: float = 0.0
    overall_efficiency: float = 0.0
    pareto: bool = False
    robust_plateau: bool = False


@lru_cache(maxsize=16)
def load_profiled_chunk_bank(path: str | Path) -> Tuple[DsppProfileBankEntry, ...]:
    """Load every measured capacity, including inefficient OOM fallbacks."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = ()
    if isinstance(payload, Mapping):
        rows = payload.get("profile_bank") or payload.get("scores") or ()
    entries = []
    for row in rows:
        capacity = int(row.get("capacity", 0))
        if capacity <= 0:
            continue
        entries.append(DsppProfileBankEntry(
            capacity=capacity,
            full_efficiency=float(row.get("full_efficiency", row.get("full", 0.0))),
            tail_efficiency=float(row.get("tail_efficiency", row.get("tail", 0.0))),
            middle_efficiency=float(row.get("middle_efficiency", row.get("middle", 0.0))),
            pack_efficiency=float(row.get("pack_efficiency", row.get("pack", 0.0))),
            overall_efficiency=float(row.get("overall_efficiency", row.get("robust", 0.0))),
            pareto=bool(row.get("pareto", False)),
            robust_plateau=bool(row.get("robust_plateau", row.get("retained", False))),
        ))
    if not entries and isinstance(payload, Mapping):
        entries = [
            DsppProfileBankEntry(capacity=int(value), robust_plateau=True)
            for value in payload.get("candidates", ()) if int(value) > 0
        ]
    bank = tuple(sorted({entry.capacity: entry for entry in entries}.values(), key=lambda x: x.capacity))
    if not bank:
        raise ValueError("DSPP chunk profile bank contains no positive capacities")
    return bank


@lru_cache(maxsize=16)
def load_profiled_chunk_menu(path: str | Path) -> Tuple[int, ...]:
    """Load the bounded offline compute menu used by runtime selection."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("candidates") if isinstance(payload, Mapping) else None
    menu = tuple(sorted({int(value) for value in (values or ()) if int(value) > 0}))
    if not menu:
        raise ValueError("DSPP chunk menu contains no positive candidates")
    return menu


def _percentile(values, percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("inf")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _microbatch_family(meta, chunk_size: int) -> str:
    """Map a physical microbatch to the offline efficiency families."""
    if len(meta.items) > 1:
        return "pack"
    item = meta.items[0]
    if item.token_length < chunk_size:
        return "tail"
    if item.token_offset > 0:
        return "middle"
    return "full"


def _profile_efficiency(
    bank: Sequence[DsppProfileBankEntry], chunk_size: int, family: str
) -> Optional[float]:
    """Return exact or local linear efficiency for one profile family."""
    if not bank:
        return None
    attribute = f"{family}_efficiency"
    points = [
        (entry.capacity, float(getattr(entry, attribute, 0.0)))
        for entry in bank
        if float(getattr(entry, attribute, 0.0)) > 0.0
    ]
    if not points:
        return None
    points.sort()
    for capacity, value in points:
        if capacity == chunk_size:
            return value
    if chunk_size < points[0][0] or chunk_size > points[-1][0]:
        return None
    for (left_c, left_v), (right_c, right_v) in zip(points, points[1:]):
        if left_c <= chunk_size <= right_c:
            if right_c == left_c:
                return left_v
            ratio = (chunk_size - left_c) / (right_c - left_c)
            return left_v + ratio * (right_v - left_v)
    return None


def _cheap_efficiency_score(
    batch_plans, chunk_size: int, bank, execution_model
) -> Tuple[float, float]:
    """Estimate time and weighted efficiency without constructing a DAG."""
    total_work = 0.0
    weighted_efficiency = 0.0
    work_by_family = {family: 0.0 for family in ("full", "middle", "tail", "pack")}
    for plan in batch_plans:
        for meta in plan.microbatches:
            family = _microbatch_family(meta, chunk_size)
            # F/B attention work and deferred W token work are both counted;
            # this is only a ranking score, while the final DAG uses measured
            # operation-specific durations.
            attention_work = sum(
                float(item.token_length)
                * float(item.token_offset + item.token_length)
                for item in meta.items
            )
            work = 2.0 * attention_work + float(meta.valid_token_count)
            work_by_family[family] += max(work, 1.0)
    for family, work in work_by_family.items():
        if work <= 0.0:
            continue
        efficiency = _profile_efficiency(bank, chunk_size, family) or 1.0
        total_work += work / max(efficiency, 1.0e-6)
        weighted_efficiency += work * efficiency
    useful_work = sum(work_by_family.values())
    normalized_efficiency = weighted_efficiency / max(useful_work, 1.0e-6)
    # The efficiency score decides which candidates enter the DAG.  This
    # separate millisecond estimate is only reported/tie-broken and comes
    # from operation-specific profile records, not from a FLOPs-only proxy.
    profiled_time = 0.0
    for plan in batch_plans:
        for meta in plan.microbatches:
            fallback = sum(float(item.estimated_flops) for item in meta.items) / 1.0e6
            for op in "FBW":
                profiled_time += max(
                    0.0,
                    execution_model.predict_detailed(
                        0, op, meta, fallback=fallback
                    ).duration_ms,
                )
    return float(profiled_time or total_work), float(normalized_efficiency)


def _screen_efficiency_candidates(
    feasible_chunks: Sequence[int],
    batch_plans_by_chunk: Mapping[int, Sequence],
    *,
    profile_bank: Sequence[DsppProfileBankEntry],
    execution_model,
    top_k: Optional[int],
    slack: float,
    current_chunk_size: Optional[int],
) -> Tuple[set[int], Dict[int, Tuple[float, float, int]]]:
    """Build top-k plus structural candidates from an iteration score."""
    scored = {}
    capacities = sorted(feasible_chunks)
    for chunk in feasible_chunks:
        scored[chunk] = _cheap_efficiency_score(
            batch_plans_by_chunk[chunk], chunk, profile_bank, execution_model
        )
    ordered = sorted(scored, key=lambda c: (scored[c][0], -scored[c][1], c))
    if not ordered:
        return set(), {}
    best_time = scored[ordered[0]][0]
    keep = set(ordered if top_k is None else ordered[: max(1, int(top_k))])
    if slack > 0.0:
        keep.update(
            chunk for chunk in ordered
            if scored[chunk][0] <= best_time * (1.0 + slack)
        )
    # Local minima of estimated compute time represent efficiency peaks. Endpoints
    # are handled separately so a boundary peak is never lost.
    for chunk in ordered:
        position = capacities.index(chunk)
        if position > 0 and position + 1 < len(capacities):
            left, right = capacities[position - 1], capacities[position + 1]
            if scored[chunk][0] <= scored[left][0] and scored[chunk][0] <= scored[right][0]:
                keep.add(chunk)
    # Largest directly measured feasible point is a profile anchor, distinct
    # from an interpolated memory boundary.
    measured = {entry.capacity for entry in profile_bank}
    measured_feasible = [chunk for chunk in feasible_chunks if chunk in measured]
    if measured_feasible:
        keep.add(max(measured_feasible))
    if current_chunk_size in scored:
        keep.add(current_chunk_size)
    ranks = {chunk: rank for rank, chunk in enumerate(ordered, 1)}
    return keep, {
        chunk: (scored[chunk][0], scored[chunk][1], ranks[chunk])
        for chunk in scored
    }


@dataclass(frozen=True)
class DsppChunkCandidateScore:
    chunk_size: int
    iterations: int
    median_makespan_ms: float
    p95_makespan_ms: float
    mean_makespan_ms: float
    feasible: bool
    profiled_compute_fraction: float = 0.0
    extrapolated_compute_fraction: float = 1.0
    low_confidence_compute_fraction: float = 1.0
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class DsppFixedChunkSelection:
    selected_chunk_size: int
    scores: Tuple[DsppChunkCandidateScore, ...]
    oracle_median_ms: float
    fixed_median_ms: float
    regret: float
    oracle_choices: Tuple[int, ...]


@dataclass(frozen=True)
class DsppPerIterationCandidate:
    """One candidate evaluation for one real metadata iteration."""

    chunk_size: int
    predicted_makespan_ms: float
    predicted_p95_ms: float
    feasible: bool
    planning_ms: float
    compute_lower_bound_ms: float = float("inf")
    fully_planned: bool = True
    profiled_compute_fraction: float = 0.0
    extrapolated_compute_fraction: float = 1.0
    low_confidence_compute_fraction: float = 1.0
    rejection_reason: Optional[str] = None
    cheap_score_ms: float = float("inf")
    efficiency_score: float = 0.0
    screen_rank: int = 0
    selected_by_screen: bool = False


@dataclass(frozen=True)
class DsppPerIterationDecision:
    """Runtime chunk decision and its auditable candidate costs."""

    iteration_index: int
    selected_chunk_size: int
    decision_ms: float
    sequence_count: int
    total_tokens: int
    max_sequence_length: int
    mean_sequence_length: float
    long_sequence_fraction: float
    candidates: Tuple[DsppPerIterationCandidate, ...]
    selection_strategy: str = "exhaustive"
    efficiency_screen_ms: float = 0.0
    screened_candidate_count: int = 0

    def to_dict(self) -> dict:
        return {
            "iteration_index": self.iteration_index,
            "selected_chunk_size": self.selected_chunk_size,
            "decision_ms": self.decision_ms,
            "sequence_count": self.sequence_count,
            "total_tokens": self.total_tokens,
            "max_sequence_length": self.max_sequence_length,
            "mean_sequence_length": self.mean_sequence_length,
            "long_sequence_fraction": self.long_sequence_fraction,
            "selection_strategy": self.selection_strategy,
            "efficiency_screen_ms": self.efficiency_screen_ms,
            "screened_candidate_count": self.screened_candidate_count,
            "profile_bank_size": len(self.candidates),
            "fully_planned_candidates": sum(
                candidate.fully_planned for candidate in self.candidates
            ),
            "pruned_candidates": sum(
                candidate.feasible and not candidate.fully_planned
                for candidate in self.candidates
            ),
            "candidates": [candidate.__dict__ for candidate in self.candidates],
        }


def select_per_iteration_chunks(
    length_trace: LengthTrace,
    *,
    candidate_chunks: Sequence[int],
    pipeline_size: int,
    ordering_mode: str = "warmup-short-steady-long",
    packing_strategy: str = "baseline",
    execution_profile_path: Optional[str] = None,
    profile_bank_path: Optional[str] = None,
    ordering_profile_path: Optional[str] = None,
    p2p_profile_path: Optional[str] = None,
    hidden_size: int = 1,
    dtype_bytes: int = 1,
    p2p_percentile: float = 0.5,
    communication_cost_mode: str = "ignore",
    p2p_bandwidth_bytes_per_ms: float = 25.0e6,
    p2p_latency_ms: float = 0.02,
    require_profile_coverage: bool = False,
    min_profiled_compute_fraction: float = 0.90,
    max_low_confidence_compute_fraction: float = 0.25,
    long_sequence_threshold: int = 8192,
    feasibility_fn: Optional[Callable[[int, Sequence[Sequence[int]]], bool]] = None,
    selection_strategy: str = "branch-and-bound",
    efficiency_top_k: Optional[int] = 4,
    efficiency_slack: float = 0.05,
    current_chunk_size: Optional[int] = None,
) -> Tuple[DsppPerIterationDecision, ...]:
    """Select a chunk per iteration using efficiency screening then a DAG.

    The cheap stage accounts for the current iteration's physical microbatch
    mix.  Full weighted-DAG planning is restricted to its top-k plus the
    measured memory-boundary anchor, local score minima, and current chunk.
    ``selection_strategy`` only controls exact planning inside that screened
    set; it does not generate the screened set.
    """
    menu = tuple(sorted({int(chunk) for chunk in candidate_chunks if int(chunk) > 0}))
    if not menu or not length_trace:
        raise ValueError("chunk menu and length trace must be non-empty")
    if selection_strategy not in {"branch-and-bound", "exhaustive"}:
        raise ValueError("selection_strategy must be 'branch-and-bound' or 'exhaustive'")
    if efficiency_top_k is not None and efficiency_top_k <= 0:
        raise ValueError("efficiency_top_k must be positive or None")
    if efficiency_slack < 0.0:
        raise ValueError("efficiency_slack must be non-negative")
    profile_bank = (
        load_profiled_chunk_bank(profile_bank_path) if profile_bank_path else ()
    )
    execution_model = load_execution_cost_model(execution_profile_path)
    decisions = []
    for iteration_index, groups in enumerate(length_trace):
        lengths = [int(length) for group in groups for length in group]
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError("each iteration must contain positive sequence lengths")
        decision_start = time.perf_counter()
        states = {}
        batch_plans_by_chunk = {}
        feasible_chunks = []
        for chunk in menu:
            candidate_start = time.perf_counter()
            if feasibility_fn is not None and not feasibility_fn(chunk, groups):
                states[chunk] = {"candidate": DsppPerIterationCandidate(
                    chunk_size=chunk, predicted_makespan_ms=float("inf"),
                    predicted_p95_ms=float("inf"), feasible=False,
                    planning_ms=(time.perf_counter() - candidate_start) * 1000.0,
                    fully_planned=False, rejection_reason="memory-infeasible"),
                    "bound": None}
                continue
            try:
                    batch_plans = build_dspp_iteration_batch_plans(
                        groups, chunk_size=chunk, packing_strategy=packing_strategy
                    )
                    batch_plans_by_chunk[chunk] = batch_plans
                    cheap_score, efficiency_score = _cheap_efficiency_score(
                        batch_plans, chunk, profile_bank, execution_model
                    )
            except (ValueError, RuntimeError) as error:
                states[chunk] = {"candidate": DsppPerIterationCandidate(
                    chunk_size=chunk, predicted_makespan_ms=float("inf"),
                    predicted_p95_ms=float("inf"), feasible=False,
                    planning_ms=(time.perf_counter() - candidate_start) * 1000.0,
                    fully_planned=False,
                    rejection_reason=f"packing-failed: {str(error).splitlines()[0][:160]}"),
                    "bound": None}
                continue
            feasible_chunks.append(chunk)
            states[chunk] = {"candidate": DsppPerIterationCandidate(
                chunk_size=chunk, predicted_makespan_ms=float("inf"),
                predicted_p95_ms=float("inf"), feasible=True,
                planning_ms=(time.perf_counter() - candidate_start) * 1000.0,
                fully_planned=False, cheap_score_ms=cheap_score,
                efficiency_score=efficiency_score), "bound": None}

        screen_start = time.perf_counter()
        screened, score_info = _screen_efficiency_candidates(
            feasible_chunks, batch_plans_by_chunk, profile_bank=profile_bank,
            execution_model=execution_model,
            top_k=efficiency_top_k, slack=efficiency_slack,
            current_chunk_size=current_chunk_size,
        )
        screen_ms = (time.perf_counter() - screen_start) * 1000.0
        ordered = sorted(
            feasible_chunks, key=lambda c: (score_info[c][0], c)
        )
        ranks = {chunk: rank for rank, chunk in enumerate(ordered, 1)}
        for chunk in feasible_chunks:
            candidate = states[chunk]["candidate"]
            cheap, efficiency, _rank = score_info[chunk]
            states[chunk]["candidate"] = DsppPerIterationCandidate(
                **{**candidate.__dict__, "cheap_score_ms": cheap,
                   "efficiency_score": efficiency, "screen_rank": ranks[chunk],
                   "selected_by_screen": chunk in screened,
                   "rejection_reason": None if chunk in screened
                   else "efficiency-filtered"}
            )

        # Compute the strict lower bound only for screened candidates.  This
        # keeps candidate screening separate from exact DAG evaluation.
        for chunk in sorted(screened):
            candidate = states[chunk]["candidate"]
            try:
                bound = estimate_dspp_compute_lower_bound(
                    groups, chunk_size=chunk, pipeline_size=pipeline_size,
                    packing_strategy=packing_strategy,
                    execution_profile_path=execution_profile_path,
                    batch_plans=batch_plans_by_chunk[chunk],
                )
            except (ValueError, RuntimeError) as error:
                states[chunk]["candidate"] = DsppPerIterationCandidate(
                    **{**candidate.__dict__, "feasible": False,
                       "rejection_reason": f"lower-bound-failed: {str(error).splitlines()[0][:160]}"})
                continue
            coverage_ok = (
                bound.profiled_compute_fraction >= min_profiled_compute_fraction
                and bound.low_confidence_compute_fraction <= max_low_confidence_compute_fraction
            )
            if require_profile_coverage and not coverage_ok:
                states[chunk]["candidate"] = DsppPerIterationCandidate(
                    **{**candidate.__dict__, "feasible": False,
                       "compute_lower_bound_ms": bound.lower_bound_ms,
                       "profiled_compute_fraction": bound.profiled_compute_fraction,
                       "extrapolated_compute_fraction": bound.extrapolated_compute_fraction,
                       "low_confidence_compute_fraction": bound.low_confidence_compute_fraction,
                       "rejection_reason": "insufficient-profile-coverage"})
                continue
            states[chunk]["bound"] = bound
            states[chunk]["candidate"] = DsppPerIterationCandidate(
                **{**candidate.__dict__, "compute_lower_bound_ms": bound.lower_bound_ms,
                   "profiled_compute_fraction": bound.profiled_compute_fraction,
                   "extrapolated_compute_fraction": bound.extrapolated_compute_fraction,
                   "low_confidence_compute_fraction": bound.low_confidence_compute_fraction})

        eligible = sorted(
            (state for state in states.values()
             if state["candidate"].feasible and state["bound"] is not None
             and state["candidate"].selected_by_screen),
            key=lambda state: (state["candidate"].compute_lower_bound_ms,
                               state["candidate"].chunk_size),
        )
        incumbent_ms = float("inf")
        for state in eligible:
            candidate = state["candidate"]
            if (selection_strategy == "branch-and-bound"
                    and candidate.compute_lower_bound_ms > incumbent_ms):
                state["candidate"] = DsppPerIterationCandidate(
                    **{**candidate.__dict__,
                       "rejection_reason": "pruned-by-compute-lower-bound"})
                continue
            full_start = time.perf_counter()
            try:
                plan = plan_dspp_iteration(
                    groups, chunk_size=candidate.chunk_size,
                    pipeline_size=pipeline_size, ordering_mode=ordering_mode,
                    packing_strategy=packing_strategy,
                    execution_profile_path=execution_profile_path,
                    ordering_profile_path=ordering_profile_path,
                    p2p_profile_path=p2p_profile_path,
                    hidden_size=hidden_size, dtype_bytes=dtype_bytes,
                    p2p_percentile=p2p_percentile,
                    communication_cost_mode=communication_cost_mode,
                    p2p_bandwidth_bytes_per_ms=p2p_bandwidth_bytes_per_ms,
                    p2p_latency_ms=p2p_latency_ms,
                    batch_plans=state["bound"].batch_plans,
                )
            except (ValueError, RuntimeError) as error:
                state["candidate"] = DsppPerIterationCandidate(
                    **{**candidate.__dict__, "feasible": False,
                       "fully_planned": True,
                       "planning_ms": candidate.planning_ms +
                       (time.perf_counter() - full_start) * 1000.0,
                       "rejection_reason": f"planner-failed: {str(error).splitlines()[0][:160]}"})
                continue
            state["candidate"] = DsppPerIterationCandidate(
                **{**candidate.__dict__,
                   "predicted_makespan_ms": plan.predicted_makespan_ms,
                   "predicted_p95_ms": plan.predicted_p95_makespan_ms,
                   "planning_ms": candidate.planning_ms +
                   (time.perf_counter() - full_start) * 1000.0,
                   "fully_planned": True,
                   "profiled_compute_fraction": plan.profiled_compute_fraction,
                   "extrapolated_compute_fraction": plan.extrapolated_compute_fraction,
                   "low_confidence_compute_fraction": plan.low_confidence_compute_fraction})
            incumbent_ms = min(incumbent_ms, plan.predicted_makespan_ms)

        candidates = tuple(states[chunk]["candidate"] for chunk in menu)
        feasible_candidates = [candidate for candidate in candidates
                               if candidate.feasible and candidate.fully_planned]
        if not feasible_candidates:
            raise ValueError(f"no feasible chunk candidate at iteration {iteration_index}")
        selected = min(feasible_candidates,
                        key=lambda candidate: (candidate.predicted_makespan_ms,
                                               candidate.predicted_p95_ms,
                                               candidate.chunk_size))
        decisions.append(DsppPerIterationDecision(
            iteration_index=iteration_index,
            selected_chunk_size=selected.chunk_size,
            decision_ms=(time.perf_counter() - decision_start) * 1000.0,
            sequence_count=len(lengths), total_tokens=sum(lengths),
            max_sequence_length=max(lengths),
            mean_sequence_length=sum(lengths) / len(lengths),
            long_sequence_fraction=(
                sum(length >= long_sequence_threshold for length in lengths) / len(lengths)
                if long_sequence_threshold > 0 else 0.0),
            candidates=candidates, selection_strategy=selection_strategy,
            efficiency_screen_ms=screen_ms,
            screened_candidate_count=len(screened),
        ))
    return tuple(decisions)


def build_dspp_chunk_menu(
    length_trace: LengthTrace,
    *,
    min_chunk_size: int,
    max_chunk_size: int,
    alignment: int = 256,
    saturation_chunk_size: Optional[int] = None,
    max_candidates: int = 8,
) -> Tuple[int, ...]:
    """Build a bounded, hardware-agnostic candidate menu.

    Structural breakpoints are supplemented by the calibrated saturation
    point.  The menu is deliberately small; it is not an exhaustive search.
    """
    groups = [group for iteration in length_trace for group in iteration]
    points = list(_structural_chunk_candidates(
        groups, lower=min_chunk_size, upper=max_chunk_size,
        alignment=alignment, max_breakpoints=max_candidates,
    ))
    if saturation_chunk_size is not None:
        points.append(max(min_chunk_size, min(max_chunk_size, int(saturation_chunk_size))))
    points = sorted(set(points))
    if len(points) <= max_candidates:
        return tuple(points)
    # Preserve the lower bound, saturation point, and upper bound, then take
    # evenly spaced structural points for the remaining budget.
    keep = {points[0], points[-1]}
    if saturation_chunk_size is not None:
        keep.add(min(points, key=lambda value: abs(value - saturation_chunk_size)))
    remaining = max(0, max_candidates - len(keep))
    interior = [point for point in points if point not in keep]
    if remaining and interior:
        if remaining == 1:
            keep.add(interior[len(interior) // 2])
        else:
            for index in range(remaining):
                position = round(index * (len(interior) - 1) / (remaining - 1))
                keep.add(interior[position])
    return tuple(sorted(keep))


def select_global_fixed_chunk(
    length_trace: LengthTrace,
    *,
    candidate_chunks: Sequence[int],
    pipeline_size: int,
    ordering_mode: str = "warmup-short-steady-long",
    packing_strategy: str = "baseline",
    execution_profile_path: Optional[str] = None,
    ordering_profile_path: Optional[str] = None,
    p2p_profile_path: Optional[str] = None,
    hidden_size: int = 1,
    dtype_bytes: int = 1,
    p2p_percentile: float = 0.5,
    communication_cost_mode: str = "ignore",
    p2p_bandwidth_bytes_per_ms: float = 25.0e6,
    p2p_latency_ms: float = 0.02,
    require_profile_coverage: bool = False,
    min_profiled_compute_fraction: float = 0.90,
    max_low_confidence_compute_fraction: float = 0.25,
    feasibility_fn: Optional[Callable[[int, Sequence[Sequence[int]]], bool]] = None,
) -> DsppFixedChunkSelection:
    """Evaluate a small menu and choose one chunk for all iterations.

    ``length_trace`` is a sequence of iterations, each containing logical
    sequence-length groups.  The exact same planner is used for the fixed
    candidates and the per-iteration oracle, so regret measures policy loss,
    not differences between estimators.
    """
    menu = tuple(sorted({int(c) for c in candidate_chunks if int(c) > 0}))
    if not menu or not length_trace:
        raise ValueError("chunk menu and length trace must be non-empty")
    plans: dict[Tuple[int, int], DsppDagPlan] = {}
    scores = []
    for chunk in menu:
        values = []
        feasible = True
        rejection_reason = None
        coverage_values = []
        extrapolated_values = []
        low_confidence_values = []
        for index, groups in enumerate(length_trace):
            if feasibility_fn is not None and not feasibility_fn(chunk, groups):
                feasible = False
                continue
            try:
                plan = plan_dspp_iteration(
                    groups,
                    chunk_size=chunk,
                    pipeline_size=pipeline_size,
                    ordering_mode=ordering_mode,
                    packing_strategy=packing_strategy,
                    execution_profile_path=execution_profile_path,
                    ordering_profile_path=ordering_profile_path,
                    p2p_profile_path=p2p_profile_path,
                    hidden_size=hidden_size,
                    dtype_bytes=dtype_bytes,
                    p2p_percentile=p2p_percentile,
                    communication_cost_mode=communication_cost_mode,
                    p2p_bandwidth_bytes_per_ms=p2p_bandwidth_bytes_per_ms,
                    p2p_latency_ms=p2p_latency_ms,
                )
            except (ValueError, RuntimeError):
                feasible = False
                rejection_reason = "planner-failed"
                continue
            plans[(chunk, index)] = plan
            values.append(plan.predicted_makespan_ms)
            coverage_values.append(plan.profiled_compute_fraction)
            extrapolated_values.append(plan.extrapolated_compute_fraction)
            low_confidence_values.append(plan.low_confidence_compute_fraction)
            if require_profile_coverage and (
                plan.profiled_compute_fraction < min_profiled_compute_fraction
                or plan.low_confidence_compute_fraction
                > max_low_confidence_compute_fraction
            ):
                feasible = False
                rejection_reason = "insufficient-profile-coverage"
        if not values:
            scores.append(DsppChunkCandidateScore(
                chunk, 0, float("inf"), float("inf"), float("inf"), False,
                rejection_reason=rejection_reason or "no-valid-iteration",
            ))
            continue
        scores.append(DsppChunkCandidateScore(
            chunk_size=chunk,
            iterations=len(values),
            median_makespan_ms=float(median(values)),
            p95_makespan_ms=_percentile(values, 0.95),
            mean_makespan_ms=float(sum(values) / len(values)),
            feasible=feasible and len(values) == len(length_trace),
            profiled_compute_fraction=min(coverage_values, default=0.0),
            extrapolated_compute_fraction=max(extrapolated_values, default=1.0),
            low_confidence_compute_fraction=max(low_confidence_values, default=1.0),
            rejection_reason=(
                rejection_reason
                if not (feasible and len(values) == len(length_trace)) else None
            ),
        ))
    feasible_scores = [score for score in scores if score.feasible]
    if not feasible_scores:
        raise ValueError("no chunk candidate is feasible for the supplied trace")
    selected = min(feasible_scores, key=lambda score: (score.mean_makespan_ms, score.p95_makespan_ms, score.chunk_size))
    feasible_chunks = {score.chunk_size for score in feasible_scores}
    oracle_values = []
    fixed_values = []
    oracle_choices = []
    for index, _groups in enumerate(length_trace):
        choices = [
            (plans[(chunk, index)].predicted_makespan_ms, chunk)
            for chunk in menu
            if chunk in feasible_chunks and (chunk, index) in plans
        ]
        if not choices:
            continue
        value, chunk = min(choices)
        oracle_values.append(value)
        oracle_choices.append(chunk)
        fixed_values.append(plans[(selected.chunk_size, index)].predicted_makespan_ms)
    oracle_mean = sum(oracle_values) / max(1, len(oracle_values))
    fixed_mean = sum(fixed_values) / max(1, len(fixed_values))
    return DsppFixedChunkSelection(
        selected_chunk_size=selected.chunk_size,
        scores=tuple(scores),
        oracle_median_ms=float(median(oracle_values) if oracle_values else float("inf")),
        fixed_median_ms=float(median(fixed_values) if fixed_values else float("inf")),
        regret=float((fixed_mean - oracle_mean) / oracle_mean) if oracle_mean > 0 else 0.0,
        oracle_choices=tuple(oracle_choices),
    )


def load_dspp_length_trace(
    path: str | Path,
    *,
    sequence_group_size: int = 1,
    groups_per_iteration: int = 1,
    raw_lengths_include_label: bool = False,
    max_iterations: Optional[int] = None,
) -> LengthTrace:
    """Load a representative length trace for offline fixed-chunk selection.

    JSON accepts either ``{"iterations": [...]}`` or a raw nested list with
    shape ``[iteration][logical_group][sequence_length]``.  A text file is
    interpreted as one stream of lengths and grouped into logical batches of
    ``sequence_group_size`` and then into iterations. This is convenient for
    sampled dataset length files used by the training dataloader.
    """
    source = Path(path)
    if source.suffix.lower() in {".json", ".jsonl"}:
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload = payload.get("iterations", payload) if isinstance(payload, dict) else payload
        if not isinstance(payload, list) or not payload:
            raise ValueError("length trace JSON must contain a non-empty list")
        if payload and isinstance(payload[0], int):
            values = [int(item) for item in payload]
            flat_values = values
        else:
            iterations = tuple(
                tuple(
                    tuple(
                        int(length) - (1 if raw_lengths_include_label else 0)
                        for length in group
                        if int(length) >= (2 if raw_lengths_include_label else 1)
                    )
                    for group in iteration
                )
                for iteration in payload[:max_iterations]
            )
            if any(not group for iteration in iterations for group in iteration):
                raise ValueError("length trace contains an empty logical group")
            return iterations
    else:
        flat_values = [
            int(line.strip())
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not flat_values:
        raise ValueError("length trace text file is empty")
    if raw_lengths_include_label:
        flat_values = [value - 1 for value in flat_values if value >= 2]
    sequence_group_size = max(1, int(sequence_group_size))
    groups_per_iteration = max(1, int(groups_per_iteration))
    groups = tuple(
        tuple(flat_values[index:index + sequence_group_size])
        for index in range(0, len(flat_values), sequence_group_size)
        if len(flat_values[index:index + sequence_group_size]) == sequence_group_size
    )
    iterations = tuple(
        tuple(groups[index:index + groups_per_iteration])
        for index in range(0, len(groups), groups_per_iteration)
        if len(groups[index:index + groups_per_iteration]) == groups_per_iteration
    )
    return iterations[:max_iterations]


def select_global_fixed_chunk_from_file(
    trace_path: str | Path,
    *,
    candidate_chunks: Sequence[int],
    pipeline_size: int,
    sequence_group_size: int = 1,
    groups_per_iteration: int = 1,
    raw_lengths_include_label: bool = False,
    max_iterations: Optional[int] = None,
    **kwargs,
) -> DsppFixedChunkSelection:
    trace = load_dspp_length_trace(
        trace_path,
        sequence_group_size=sequence_group_size,
        groups_per_iteration=groups_per_iteration,
        raw_lengths_include_label=raw_lengths_include_label,
        max_iterations=max_iterations,
    )
    return select_global_fixed_chunk(
        trace,
        candidate_chunks=candidate_chunks,
        pipeline_size=pipeline_size,
        **kwargs,
    )
