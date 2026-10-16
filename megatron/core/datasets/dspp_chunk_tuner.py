"""Runtime chunk-size selection for DSPP variable-length training.

The first adaptive iteration uses the configured ``micro_seq_length`` as a
safe bootstrap. Its measured forward/backward peak calibrates a conservative
high-water bytes-per-live-token model. Later iterations choose the largest
chunk that fits the local memory budget, including an explicit bounded P2P
window reserve; distributed callers reduce the choice across pipeline ranks
before materializing the final batches.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

from .dspp_batch_plan import DsppBatchPlan, build_dspp_batch_plan
from .dspp_ordering import build_dspp_ordering_plan
from .dspp_execution_cost import DsppExecutionCostModel, load_execution_cost_model


@dataclass(frozen=True)
class DsppChunkLayoutEstimate:
    chunk_size: int
    max_segment_count: int
    physical_microbatches: int
    schedule_groups: int
    peak_live_slots: int

    @property
    def peak_live_tokens(self) -> int:
        return self.chunk_size * self.peak_live_slots

    @property
    def physical_token_capacity(self) -> int:
        """Tokens processed by fixed-shape model kernels, including padding."""

        return self.chunk_size * self.physical_microbatches


@dataclass(frozen=True)
class DsppChunkSelection:
    chunk_size: int
    reason: str
    layout: DsppChunkLayoutEstimate
    activation_bytes_per_live_token: Optional[float]
    activation_budget_bytes: int
    estimated_peak_activation_bytes: Optional[int]
    estimated_communication_bytes: int
    estimated_peak_dynamic_bytes: Optional[int]


def _structural_chunk_candidates(
    sequence_length_groups: Sequence[Sequence[int]],
    *,
    lower: int,
    upper: int,
    alignment: int,
    max_breakpoints: int = 32,
) -> Tuple[int, ...]:
    """Return a bounded set of segment/bin structure change points.

    We sample quotient regions rather than enumerating every aligned token
    length.  The number of points is bounded by ``max_breakpoints`` plus a
    small number of observed-length quantiles, so a 128K sequence does not
    create hundreds of plans merely because alignment is 256.
    """

    if lower > upper:
        return ()
    points = {lower, upper}
    lengths = sorted({int(length) for group in sequence_length_groups for length in group})
    if lengths:
        stride = max(1, len(lengths) // 8)
        points.update(lengths[::stride])
        points.add(lengths[-1])
    for divisor in range(1, max_breakpoints + 1):
        longest = max(lengths, default=upper)
        q = longest // divisor
        for point in (q, q + 1):
            if lower <= point <= upper:
                points.add(point)
    aligned = set()
    for point in points:
        snapped = max(lower, min(upper, (point // alignment) * alignment))
        aligned.add(snapped)
        if snapped + alignment <= upper:
            aligned.add(snapped + alignment)
        if snapped - alignment >= lower:
            aligned.add(snapped - alignment)
    return tuple(sorted(aligned))


@dataclass(frozen=True)
class _PlanView:
    plan: DsppBatchPlan


def _pad_split_counts(split_counts: Sequence[int], pipeline_size: int) -> list[int]:
    counts = list(split_counts)
    if len(counts) < 2:
        return counts
    while sum(counts) < max(counts) + 2 * pipeline_size - 2:
        smallest = min(range(len(counts)), key=counts.__getitem__)
        counts[smallest] += 1
    return counts


def estimate_stage_peak_live_slots(schedule, schedule_groups) -> int:
    forward_live = set()
    weight_live = set()
    peak = 0
    for node in schedule:
        if node.microbatch >= len(schedule_groups):
            continue
        if node.split >= len(schedule_groups[node.microbatch]):
            continue
        key = (node.chunk, node.microbatch, node.split)
        if node.kind == "F":
            forward_live.add(key)
        elif node.kind == "B":
            forward_live.discard(key)
            weight_live.add(key)
        elif node.kind == "W":
            weight_live.discard(key)
        peak = max(peak, len(forward_live) + len(weight_live))
    return peak


def estimate_dspp_chunk_layout(
    sequence_length_groups: Sequence[Sequence[int]],
    *,
    chunk_size: int,
    pipeline_size: int,
    ordering_mode: str = "warmup-short-steady-long",
    packing_strategy: str = "baseline",
) -> DsppChunkLayoutEstimate:
    """Estimate task count and peak live fixed-shape activation slots."""

    # Keep this import lazy.  pipeline_parallel.__init__ imports schedules,
    # while schedules imports this metadata-only tuner.
    from megatron.core.pipeline_parallel.slice_v import build_slice_v_schedule

    if not sequence_length_groups or any(not group for group in sequence_length_groups):
        raise ValueError("DSPP chunk tuning requires non-empty sequence groups")
    plans = tuple(
        build_dspp_batch_plan(
            lengths, chunk_size=chunk_size, packing_strategy=packing_strategy
        )
        for lengths in sequence_length_groups
    )
    max_segment_count = max(
        math.ceil(length / chunk_size)
        for lengths in sequence_length_groups
        for length in lengths
    )
    physical_microbatches = sum(len(plan.microbatches) for plan in plans)

    if pipeline_size == 1:
        peak_live_slots = max(len(plan.microbatches) for plan in plans)
        return DsppChunkLayoutEstimate(
            chunk_size=chunk_size,
            max_segment_count=max_segment_count,
            physical_microbatches=physical_microbatches,
            schedule_groups=physical_microbatches,
            peak_live_slots=peak_live_slots,
        )

    ordering = build_dspp_ordering_plan(
        [_PlanView(plan) for plan in plans],
        pp_degree=pipeline_size,
        mode=ordering_mode,
    )
    split_counts = [len(group) for group in ordering.schedule_groups]
    if len(split_counts) < 2:
        peak_live_slots = physical_microbatches
    else:
        padded_counts = _pad_split_counts(split_counts, pipeline_size)
        schedules, _ = build_slice_v_schedule(
            pipeline_size, len(padded_counts), padded_counts
        )
        peak_live_slots = max(
            estimate_stage_peak_live_slots(schedule, ordering.schedule_groups)
            for schedule in schedules
        )
    return DsppChunkLayoutEstimate(
        chunk_size=chunk_size,
        max_segment_count=max_segment_count,
        physical_microbatches=physical_microbatches,
        schedule_groups=len(split_counts),
        peak_live_slots=peak_live_slots,
    )


def _chunk_search_bounds(
    max_sequence_length: int,
    *,
    min_chunk_size: int,
    max_chunk_size: int,
    alignment: int,
) -> Tuple[int, int]:
    if min_chunk_size <= 0 or max_chunk_size <= 0 or alignment <= 0:
        raise ValueError("invalid DSPP chunk-size bounds or alignment")
    if max_sequence_length <= 0:
        raise ValueError("maximum sequence length must be positive")
    # A chunk larger than the aligned ceiling of this iteration's longest
    # sequence cannot remove another segment.  Prefer the aligned ceiling when
    # the configured hard maximum permits it; preserve an unaligned hard cap
    # as a boundary candidate otherwise.
    upper = min(
        max_chunk_size,
        math.ceil(max_sequence_length / alignment) * alignment,
    )
    lower = min(min_chunk_size, upper)
    return lower, upper


def select_dspp_chunk_size(
    sequence_length_groups: Sequence[Sequence[int]],
    *,
    pipeline_size: int,
    bootstrap_chunk_size: int,
    min_chunk_size: int,
    max_chunk_size: int,
    alignment: int,
    activation_bytes_per_live_token: Optional[float],
    activation_budget_bytes: int,
    memory_safety_factor: float,
    communication_bytes_per_chunk_token: int = 0,
    ordering_mode: str = "warmup-short-steady-long",
    packing_strategy: str = "baseline",
) -> DsppChunkSelection:
    """Choose the largest memory-feasible chunk.

    The aligned search assumes the estimated peak activation footprint is
    monotonic with chunk size.  It first checks the current longest-sequence
    ceiling, then uses binary search rather than enumerating every alignment
    multiple.  When no runtime activation sample exists, the configured
    bootstrap chunk is used exactly (subject to the current upper bound).
    """

    if pipeline_size <= 0:
        raise ValueError("pipeline_size must be positive")
    if memory_safety_factor < 1.0:
        raise ValueError("memory_safety_factor must be at least 1")
    if communication_bytes_per_chunk_token < 0:
        raise ValueError("communication_bytes_per_chunk_token must be non-negative")
    max_sequence_length = max(
        length for group in sequence_length_groups for length in group
    )
    lower, upper = _chunk_search_bounds(
        max_sequence_length,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        alignment=alignment,
    )

    if activation_bytes_per_live_token is None:
        bootstrap = min(upper, max(1, bootstrap_chunk_size))
        layout = estimate_dspp_chunk_layout(
            sequence_length_groups,
            chunk_size=bootstrap,
            pipeline_size=pipeline_size,
            ordering_mode=ordering_mode,
            packing_strategy=packing_strategy,
        )
        return DsppChunkSelection(
            chunk_size=bootstrap,
            reason="bootstrap-no-activation-sample",
            layout=layout,
            activation_bytes_per_live_token=None,
            activation_budget_bytes=activation_budget_bytes,
            estimated_peak_activation_bytes=None,
            estimated_communication_bytes=(
                bootstrap * communication_bytes_per_chunk_token
            ),
            estimated_peak_dynamic_bytes=None,
        )

    def evaluate(candidate: int):
        layout = estimate_dspp_chunk_layout(
            sequence_length_groups,
            chunk_size=candidate,
            pipeline_size=pipeline_size,
            ordering_mode=ordering_mode,
            packing_strategy=packing_strategy,
        )
        activation_estimate = math.ceil(
            layout.peak_live_tokens
            * activation_bytes_per_live_token
            * memory_safety_factor
        )
        communication_estimate = candidate * communication_bytes_per_chunk_token
        total_estimate = activation_estimate + communication_estimate
        return layout, activation_estimate, communication_estimate, total_estimate

    (
        upper_layout,
        upper_activation_estimate,
        upper_communication_estimate,
        upper_total_estimate,
    ) = evaluate(upper)
    if upper_total_estimate <= activation_budget_bytes:
        return DsppChunkSelection(
            chunk_size=upper,
            reason="largest-memory-feasible",
            layout=upper_layout,
            activation_bytes_per_live_token=activation_bytes_per_live_token,
            activation_budget_bytes=activation_budget_bytes,
            estimated_peak_activation_bytes=upper_activation_estimate,
            estimated_communication_bytes=upper_communication_estimate,
            estimated_peak_dynamic_bytes=upper_total_estimate,
        )

    first_aligned = math.ceil(lower / alignment) * alignment
    last_aligned = upper // alignment * alignment
    if last_aligned == upper:
        last_aligned -= alignment
    if first_aligned > last_aligned:
        smallest = (
            upper_layout,
            upper_activation_estimate,
            upper_communication_estimate,
            upper_total_estimate,
        )
    else:
        smallest = evaluate(first_aligned)
    (
        smallest_layout,
        smallest_activation_estimate,
        smallest_communication_estimate,
        smallest_total_estimate,
    ) = smallest

    if smallest_total_estimate > activation_budget_bytes:
        return DsppChunkSelection(
            chunk_size=0,
            reason="no-memory-feasible-chunk",
            layout=smallest_layout,
            activation_bytes_per_live_token=activation_bytes_per_live_token,
            activation_budget_bytes=activation_budget_bytes,
            estimated_peak_activation_bytes=smallest_activation_estimate,
            estimated_communication_bytes=smallest_communication_estimate,
            estimated_peak_dynamic_bytes=smallest_total_estimate,
        )

    best = smallest
    low_index = first_aligned // alignment + 1
    high_index = last_aligned // alignment
    while low_index <= high_index:
        middle_index = (low_index + high_index) // 2
        candidate = evaluate(middle_index * alignment)
        if candidate[3] <= activation_budget_bytes:
            best = candidate
            low_index = middle_index + 1
        else:
            high_index = middle_index - 1

    (
        best_layout,
        best_activation_estimate,
        best_communication_estimate,
        best_total_estimate,
    ) = best

    return DsppChunkSelection(
        chunk_size=best_layout.chunk_size,
        reason="largest-memory-feasible",
        layout=best_layout,
        activation_bytes_per_live_token=activation_bytes_per_live_token,
        activation_budget_bytes=activation_budget_bytes,
        estimated_peak_activation_bytes=best_activation_estimate,
        estimated_communication_bytes=best_communication_estimate,
        estimated_peak_dynamic_bytes=best_total_estimate,
    )


def select_dspp_structural_chunk_size(
    sequence_length_groups: Sequence[Sequence[int]],
    *,
    pipeline_size: int,
    bootstrap_chunk_size: int,
    min_chunk_size: int,
    max_chunk_size: int,
    alignment: int,
    activation_bytes_per_live_token: Optional[float],
    activation_budget_bytes: int,
    memory_safety_factor: float,
    saturation_chunk_size: int,
    saturation_slack: float = 0.20,
    communication_bytes_per_chunk_token: int = 0,
    ordering_mode: str = "warmup-short-steady-long",
    packing_strategy: str = "attention-critical",
    execution_profile_path: Optional[str] = None,
    p2p_profile_path: Optional[str] = None,
    hidden_size: int = 1,
    dtype_bytes: int = 1,
) -> DsppChunkSelection:
    """Choose a robust point in the compute-saturation window.

    ``saturation_chunk_size`` is a small offline calibration result: the
    smallest chunk whose kernels reach the desired efficiency on the current
    model/hardware.  It is not a L40-specific constant.  Once memory gives an
    upper bound, this selector evaluates only bounded structural candidates,
    keeps candidates whose largest physical task is within ``saturation_slack``
    of the lower-bound candidate, and chooses the one with fewer physical
    tasks.  This expresses a platform choice rather than claiming a precise
    optimum from a synthetic FLOPs model.
    """

    if not 0.0 <= saturation_slack < 1.0:
        raise ValueError("saturation_slack must be in [0, 1)")
    memory_selection = select_dspp_chunk_size(
        sequence_length_groups,
        pipeline_size=pipeline_size,
        bootstrap_chunk_size=bootstrap_chunk_size,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        alignment=alignment,
        activation_bytes_per_live_token=activation_bytes_per_live_token,
        activation_budget_bytes=activation_budget_bytes,
        memory_safety_factor=memory_safety_factor,
        communication_bytes_per_chunk_token=communication_bytes_per_chunk_token,
        ordering_mode=ordering_mode,
        packing_strategy=packing_strategy,
    )
    if memory_selection.chunk_size <= 0 or activation_bytes_per_live_token is None:
        return memory_selection
    lower = max(min_chunk_size, saturation_chunk_size)
    upper = memory_selection.chunk_size
    candidates = _structural_chunk_candidates(
        sequence_length_groups,
        lower=min(lower, upper),
        upper=upper,
        alignment=alignment,
    )
    if not candidates:
        return memory_selection

    execution_model = load_execution_cost_model(execution_profile_path)
    weighted_planner = None
    if execution_profile_path or p2p_profile_path:
        from .dspp_dag import plan_dspp_iteration
        weighted_planner = plan_dspp_iteration

    def evaluate(candidate: int):
        plans = tuple(
            build_dspp_batch_plan(
                lengths,
                chunk_size=candidate,
                packing_strategy=packing_strategy,
            )
            for lengths in sequence_length_groups
        )
        workloads = []
        makespan = None
        for plan in plans:
            for meta in plan.microbatches:
                fallback = sum(float(item.estimated_flops) for item in meta.items)
                predicted, _source, _error = execution_model.predict_task_workload(
                    meta,
                    pipeline_size=pipeline_size,
                    fallback=fallback,
                )
                workloads.append(predicted)
        if weighted_planner is not None:
            try:
                dag = weighted_planner(
                    sequence_length_groups,
                    chunk_size=candidate,
                    pipeline_size=pipeline_size,
                    ordering_mode=ordering_mode,
                    packing_strategy=packing_strategy,
                    execution_profile_path=execution_profile_path,
                    p2p_profile_path=p2p_profile_path,
                    hidden_size=hidden_size,
                    dtype_bytes=dtype_bytes,
                )
                makespan = dag.predicted_p95_makespan_ms
            except (ValueError, RuntimeError):
                makespan = None
        layout = estimate_dspp_chunk_layout(
            sequence_length_groups,
            chunk_size=candidate,
            pipeline_size=pipeline_size,
            ordering_mode=ordering_mode,
            packing_strategy=packing_strategy,
        )
        activation_estimate = math.ceil(
            layout.peak_live_tokens
            * activation_bytes_per_live_token
            * memory_safety_factor
        )
        communication_estimate = candidate * communication_bytes_per_chunk_token
        total_estimate = activation_estimate + communication_estimate
        if total_estimate > activation_budget_bytes:
            return None
        max_work = max(workloads, default=0.0)
        mean_work = sum(workloads) / max(1, len(workloads))
        return (
            candidate,
            layout,
            activation_estimate,
            communication_estimate,
            total_estimate,
            max_work,
            mean_work,
            len(workloads),
            makespan if makespan is not None else max_work,
        )

    evaluated = [item for candidate in candidates if (item := evaluate(candidate))]
    if not evaluated:
        return memory_selection
    baseline = min(evaluated, key=lambda item: (item[0], item[8]))
    max_allowed_work = baseline[8] * (1.0 + saturation_slack)
    plateau = [item for item in evaluated if item[8] <= max_allowed_work]
    best = min(
        plateau or evaluated,
        key=lambda item: (
            item[7],                 # fewer physical tasks
            item[2] + item[3],       # lower dynamic-memory estimate
            item[8],
            -item[0],
        ),
    )
    return DsppChunkSelection(
        chunk_size=best[0],
        reason="structure-aware-saturation-window",
        layout=best[1],
        activation_bytes_per_live_token=activation_bytes_per_live_token,
        activation_budget_bytes=activation_budget_bytes,
        estimated_peak_activation_bytes=best[2],
        estimated_communication_bytes=best[3],
        estimated_peak_dynamic_bytes=best[4],
    )


def update_activation_bytes_per_live_token(
    previous: Optional[float],
    *,
    baseline_allocated_bytes: int,
    peak_allocated_bytes: int,
    peak_live_tokens: int,
    smoothing: float = 0.25,
) -> Optional[float]:
    """Raise the conservative activation envelope from one measured interval."""

    if peak_live_tokens <= 0 or not 0.0 < smoothing <= 1.0:
        return previous
    dynamic_bytes = max(0, peak_allocated_bytes - baseline_allocated_bytes)
    if dynamic_bytes == 0:
        return previous
    sample = dynamic_bytes / peak_live_tokens
    if previous is None:
        return sample
    # Chunk selection is a capacity decision, not a performance forecast.
    # Never let a run of cheap shapes lower the established memory envelope;
    # doing so could make a later high-memory layout OOM before EWMA catches up.
    return max(previous, sample)
