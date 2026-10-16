"""Weighted, metadata-only DSPP critical-path planner.

This module mirrors the training construction path but never materializes
tokens or activations.  Compute and directed P2P nodes occupy separate lanes,
so a communication node can overlap a stage's compute nodes in the predicted
makespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .dspp_batch_plan import DsppBatchPlan, DsppMicrobatchMeta, build_dspp_batch_plan
from .dspp_execution_cost import DsppExecutionCostModel, load_execution_cost_model
from .dspp_p2p import DsppP2PCostModel, load_p2p_cost_model
from .dspp_ordering import build_dspp_ordering_plan
from .dspp_training import DsppTrainingBatch
from megatron.core.pipeline_parallel.slice_v import SliceVNode, build_slice_v_schedule


TaskKey = Tuple[str, Tuple]


@dataclass(frozen=True)
class DsppDagNode:
    key: TaskKey
    kind: str
    stage: int
    duration_ms: float
    percentile_error_ms: float = 0.0
    direction: Optional[str] = None
    payload_bytes: int = 0


@dataclass(frozen=True)
class DsppDagPlan:
    chunk_size: int
    nodes: Tuple[DsppDagNode, ...]
    edges: Tuple[Tuple[TaskKey, TaskKey], ...]
    predicted_makespan_ms: float
    predicted_p95_makespan_ms: float
    stage_compute_ms: Tuple[float, ...]
    critical_path: Tuple[TaskKey, ...]
    cost_source: str
    p2p_source: str
    cost_source_counts: Tuple[Tuple[str, int], ...]
    profiled_compute_fraction: float
    extrapolated_compute_fraction: float
    low_confidence_compute_fraction: float


@dataclass(frozen=True)
class DsppComputeLowerBound:
    """Cheap candidate plan and a strict lower bound on DAG makespan."""

    chunk_size: int
    batch_plans: Tuple[DsppBatchPlan, ...]
    lower_bound_ms: float
    stage_compute_ms: Tuple[float, ...]
    profiled_compute_fraction: float
    extrapolated_compute_fraction: float
    low_confidence_compute_fraction: float


def build_dspp_iteration_batch_plans(
    sequence_length_groups: Sequence[Sequence[int]],
    *,
    chunk_size: int,
    packing_strategy: str = "baseline",
) -> Tuple[DsppBatchPlan, ...]:
    if not sequence_length_groups or any(not group for group in sequence_length_groups):
        raise ValueError("sequence_length_groups must be non-empty")
    return tuple(
        build_dspp_batch_plan(
            lengths, chunk_size=chunk_size, packing_strategy=packing_strategy
        )
        for lengths in sequence_length_groups
    )


def estimate_dspp_compute_lower_bound(
    sequence_length_groups: Sequence[Sequence[int]],
    *,
    chunk_size: int,
    pipeline_size: int,
    packing_strategy: str = "baseline",
    execution_profile_path: Optional[str] = None,
    batch_plans: Optional[Sequence[DsppBatchPlan]] = None,
) -> DsppComputeLowerBound:
    """Sum profiled task costs per stage without building a schedule DAG.

    Every stage serializes its own F/B/W compute tasks, so the largest stage
    sum is a lower bound on any valid schedule makespan. Candidate-dependent
    slicing and packing are already represented in ``batch_plans``; only
    ordering, communication nodes and dependency edges are omitted.
    """

    if pipeline_size <= 0:
        raise ValueError("pipeline_size must be positive")
    plans = (
        tuple(batch_plans)
        if batch_plans is not None
        else build_dspp_iteration_batch_plans(
            sequence_length_groups,
            chunk_size=chunk_size,
            packing_strategy=packing_strategy,
        )
    )
    if len(plans) != len(sequence_length_groups) or any(
        plan.chunk_size != chunk_size for plan in plans
    ):
        raise ValueError("batch_plans do not match iteration groups or chunk size")
    execution_model = load_execution_cost_model(execution_profile_path)
    stage_totals = [0.0] * pipeline_size
    source_counts: Dict[str, int] = {}
    low_confidence = 0
    node_count = 0
    for stage in range(pipeline_size):
        for plan in plans:
            for meta in plan.microbatches:
                fallback = sum(float(item.estimated_flops) for item in meta.items) / 1.0e6
                for op in "FBW":
                    prediction = execution_model.predict_detailed(
                        stage, op, meta, fallback=fallback
                    )
                    stage_totals[stage] += max(0.0, prediction.duration_ms)
                    source_counts[prediction.source] = (
                        source_counts.get(prediction.source, 0) + 1
                    )
                    low_confidence += prediction.confidence in {"low", "none"}
                    node_count += 1
    profiled_sources = {
        "exact-profile", "exact-capacity-profile",
        "interpolated-profile", "fitted-profile",
    }
    extrapolated_sources = {"fitted-extrapolated", "metadata-fallback"}
    return DsppComputeLowerBound(
        chunk_size=chunk_size,
        batch_plans=plans,
        lower_bound_ms=max(stage_totals, default=0.0),
        stage_compute_ms=tuple(stage_totals),
        profiled_compute_fraction=(
            sum(source_counts.get(source, 0) for source in profiled_sources)
            / max(1, node_count)
        ),
        extrapolated_compute_fraction=(
            sum(source_counts.get(source, 0) for source in extrapolated_sources)
            / max(1, node_count)
        ),
        low_confidence_compute_fraction=low_confidence / max(1, node_count),
    )


def _pad_split_counts(split_counts: Sequence[int], pipeline_size: int) -> list[int]:
    counts = list(split_counts)
    if len(counts) < 2:
        return counts
    while sum(counts) < max(counts) + 2 * pipeline_size - 2:
        index = min(range(len(counts)), key=counts.__getitem__)
        counts[index] += 1
    return counts


def _meta_for_node(
    node: SliceVNode,
    groups: Sequence[Sequence[Tuple[int, int]]],
    metadata: Mapping[Tuple[int, int], DsppMicrobatchMeta],
) -> Optional[DsppMicrobatchMeta]:
    if node.microbatch >= len(groups) or node.split >= len(groups[node.microbatch]):
        return None  # communication-only Slice-V padding task
    task = groups[node.microbatch][node.split]
    return metadata.get(task)


def _topological_longest_path(
    nodes: Mapping[TaskKey, DsppDagNode], edges: Sequence[Tuple[TaskKey, TaskKey]]
) -> Tuple[float, Tuple[TaskKey, ...]]:
    outgoing: Dict[TaskKey, list[TaskKey]] = {key: [] for key in nodes}
    indegree: Dict[TaskKey, int] = {key: 0 for key in nodes}
    for source, target in edges:
        outgoing[source].append(target)
        indegree[target] += 1
    ready = [key for key, value in indegree.items() if value == 0]
    distance = {key: nodes[key].duration_ms for key in nodes}
    predecessor: Dict[TaskKey, Optional[TaskKey]] = {key: None for key in nodes}
    visited = 0
    while ready:
        key = ready.pop()
        visited += 1
        for target in outgoing[key]:
            candidate = distance[key] + nodes[target].duration_ms
            if candidate > distance[target]:
                distance[target] = candidate
                predecessor[target] = key
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(nodes):
        blocked = [key for key, value in indegree.items() if value]
        raise ValueError(f"weighted DSPP DAG contains a cycle: {blocked[:8]}")
    end = max(nodes, key=distance.get, default=None)
    if end is None:
        return 0.0, ()
    path = []
    while end is not None:
        path.append(end)
        end = predecessor[end]
    path.reverse()
    return float(max(distance.values(), default=0.0)), tuple(path)


def _build_weighted_dag(
    schedules: Sequence[Sequence[SliceVNode]],
    groups: Sequence[Sequence[Tuple[int, int]]],
    metadata: Mapping[Tuple[int, int], DsppMicrobatchMeta],
    *,
    execution_model: DsppExecutionCostModel,
    p2p_model: DsppP2PCostModel,
    hidden_size: int,
    dtype_bytes: int,
    p2p_percentile: float,
    communication_cost_mode: str,
    p2p_bandwidth_bytes_per_ms: float = 25.0e6,
    p2p_latency_ms: float = 0.02,
) -> Tuple[
    Dict[TaskKey, DsppDagNode],
    list[Tuple[TaskKey, TaskKey]],
    Dict[str, int],
    int,
    int,
    str,
]:
    nodes: Dict[TaskKey, DsppDagNode] = {}
    edges: list[Tuple[TaskKey, TaskKey]] = []
    positions: Dict[Tuple[int, str, int, int, int], TaskKey] = {}
    stage_compute_ms = [0.0 for _ in schedules]
    compute_sources: Dict[str, int] = {}
    low_confidence = 0
    real_compute_nodes = 0
    p2p_source = "ignored" if communication_cost_mode == "ignore" else "analytical-fallback"

    for stage, schedule in enumerate(schedules):
        previous: Optional[TaskKey] = None
        for index, vnode in enumerate(schedule):
            key: TaskKey = ("compute", (stage, index))
            meta = _meta_for_node(vnode, groups, metadata)
            if meta is None:
                duration, source, error = 0.0, "padding", 0.0
                confidence = "high"
            else:
                fallback = sum(float(item.estimated_flops) for item in meta.items) / 1.0e6
                prediction = execution_model.predict_detailed(
                    stage, vnode.kind, meta, fallback=fallback
                )
                upper_prediction = execution_model.predict_detailed(
                    stage, vnode.kind, meta, percentile=0.95, fallback=fallback
                )
                duration, source, error = (
                    prediction.duration_ms,
                    prediction.source,
                    max(0.0, upper_prediction.duration_ms - prediction.duration_ms)
                    + upper_prediction.error_ms,
                )
                confidence = prediction.confidence
                real_compute_nodes += 1
                compute_sources[source] = compute_sources.get(source, 0) + 1
                if confidence in {"low", "none"}:
                    low_confidence += 1
            nodes[key] = DsppDagNode(
                key=key,
                kind=vnode.kind,
                stage=stage,
                duration_ms=max(0.0, duration),
                percentile_error_ms=max(0.0, error),
            )
            stage_compute_ms[stage] += max(0.0, duration)
            positions[(stage, vnode.kind, vnode.chunk, vnode.microbatch, vnode.split)] = key
            if previous is not None:
                edges.append((previous, key))
            previous = key

    # Explicit long-chain data dependencies, independent of incidental order.
    for stage, schedule in enumerate(schedules):
        for node in schedule:
            task = (node.microbatch, node.split)
            if node.kind == "F" and node.chunk in (0, 1) and node.split > 0:
                prev = positions[(stage, node.kind, node.chunk, node.microbatch, node.split - 1)]
                cur = positions[(stage, node.kind, node.chunk, node.microbatch, node.split)]
                edges.append((prev, cur))
            if node.kind == "B" and node.chunk in (0, 1):
                count = len(groups[node.microbatch])
                if node.split + 1 < count:
                    nxt = positions[(stage, node.kind, node.chunk, node.microbatch, node.split + 1)]
                    cur = positions[(stage, node.kind, node.chunk, node.microbatch, node.split)]
                    edges.append((nxt, cur))

    # Four independent directed message lanes per PP edge.  The lane order is
    # producer order; compute and P2P remain separate resources.
    for left in range(len(schedules) - 1):
        right = left + 1
        messages = []
        for node in schedules[left]:
            if (node.kind, node.chunk) in {("F", 0), ("B", 1)}:
                messages.append((left, right, node.kind, node.chunk, node.microbatch, node.split))
        for node in schedules[right]:
            if (node.kind, node.chunk) in {("F", 1), ("B", 0)}:
                messages.append((right, left, node.kind, node.chunk, node.microbatch, node.split))
        lane_previous: Dict[Tuple[int, int, str], TaskKey] = {}
        for sender, receiver, kind, chunk, microbatch, split in messages:
            producer = positions[(sender, kind, chunk, microbatch, split)]
            consumer = positions[(receiver, kind, chunk, microbatch, split)]
            # ``metadata`` is already keyed by the logical/physical task id.
            # Looking the vnode up by scanning the complete stage schedule for
            # every message makes planning quadratic for long segmented
            # sequences, although the message tuple contains the same key.
            meta = metadata.get((microbatch, split))
            payload = int(meta.chunk_size * hidden_size * dtype_bytes) if meta is not None else 0
            direction = f"{sender}->{receiver}"
            if communication_cost_mode == "ignore":
                comm_duration, source = 0.0, "ignored"
            elif communication_cost_mode in {"profile", "analytical"}:
                comm_duration, source = p2p_model.predict(
                    payload,
                    direction=direction,
                    overlap="compute",
                    percentile=p2p_percentile,
                    fallback_bandwidth_bytes_per_ms=p2p_bandwidth_bytes_per_ms,
                    fallback_latency_ms=p2p_latency_ms,
                )
                if source != "analytical-fallback":
                    p2p_source = "profile"
            else:
                raise ValueError(
                    "communication_cost_mode must be 'ignore', 'profile', or 'analytical'"
                )
            ckey: TaskKey = ("p2p", (sender, receiver, kind, chunk, microbatch, split))
            nodes[ckey] = DsppDagNode(
                key=ckey, kind="P2P", stage=sender, duration_ms=comm_duration,
                direction=direction, payload_bytes=payload,
            )
            edges.append((producer, ckey))
            edges.append((ckey, consumer))
            lane = (sender, receiver, kind)
            if lane in lane_previous:
                edges.append((lane_previous[lane], ckey))
            lane_previous[lane] = ckey
    return (
        nodes, edges, compute_sources, low_confidence,
        real_compute_nodes, p2p_source,
    )


def plan_dspp_iteration(
    sequence_length_groups: Sequence[Sequence[int]],
    *,
    chunk_size: int,
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
    batch_plans: Optional[Sequence[DsppBatchPlan]] = None,
) -> DsppDagPlan:
    if not sequence_length_groups or any(not group for group in sequence_length_groups):
        raise ValueError("sequence_length_groups must be non-empty")
    plans = (
        tuple(batch_plans)
        if batch_plans is not None
        else build_dspp_iteration_batch_plans(
            sequence_length_groups,
            chunk_size=chunk_size,
            packing_strategy=packing_strategy,
        )
    )
    if len(plans) != len(sequence_length_groups) or any(
        plan.chunk_size != chunk_size for plan in plans
    ):
        raise ValueError("batch_plans do not match iteration groups or chunk size")
    batches = tuple(type("PlanView", (), {"plan": plan})() for plan in plans)
    ordering = build_dspp_ordering_plan(
        batches, pp_degree=pipeline_size, mode=ordering_mode,
        profile_path=ordering_profile_path,
    )
    split_counts = _pad_split_counts(
        [len(group) for group in ordering.schedule_groups], pipeline_size
    )
    schedules, _ = build_slice_v_schedule(pipeline_size, len(split_counts), split_counts)
    metadata = {
        (logical_id, physical_id): meta
        for logical_id, plan in enumerate(plans)
        for physical_id, meta in enumerate(plan.microbatches)
    }
    (
        nodes, edges, compute_sources, low_confidence,
        real_compute_nodes, p2p_source,
    ) = _build_weighted_dag(
        schedules, ordering.schedule_groups, metadata,
        execution_model=load_execution_cost_model(execution_profile_path),
        p2p_model=load_p2p_cost_model(p2p_profile_path),
        hidden_size=hidden_size, dtype_bytes=dtype_bytes,
        p2p_percentile=p2p_percentile,
        communication_cost_mode=communication_cost_mode,
        p2p_bandwidth_bytes_per_ms=p2p_bandwidth_bytes_per_ms,
        p2p_latency_ms=p2p_latency_ms,
    )
    median_ms, critical_path = _topological_longest_path(nodes, edges)
    # Add the stored residual envelope to compute nodes for a conservative P95.
    p95_nodes = {
        key: DsppDagNode(
            key=node.key, kind=node.kind, stage=node.stage,
            duration_ms=node.duration_ms + node.percentile_error_ms,
            percentile_error_ms=node.percentile_error_ms,
            direction=node.direction, payload_bytes=node.payload_bytes,
        )
        for key, node in nodes.items()
    }
    p95_ms, _ = _topological_longest_path(p95_nodes, edges)
    return DsppDagPlan(
        chunk_size=chunk_size,
        nodes=tuple(nodes.values()), edges=tuple(edges),
        predicted_makespan_ms=median_ms,
        predicted_p95_makespan_ms=p95_ms,
        stage_compute_ms=tuple(
            sum(node.duration_ms for node in nodes.values() if node.stage == stage and node.kind != "P2P")
            for stage in range(pipeline_size)
        ),
        critical_path=critical_path,
        cost_source=(
            next(iter(compute_sources))
            if len(compute_sources) == 1 else "mixed"
        ),
        p2p_source=p2p_source,
        cost_source_counts=tuple(sorted(compute_sources.items())),
        profiled_compute_fraction=(
            sum(
                count for source, count in compute_sources.items()
                if source in {
                    "exact-profile", "exact-capacity-profile",
                    "interpolated-profile", "fitted-profile",
                }
            ) / max(1, real_compute_nodes)
        ),
        extrapolated_compute_fraction=(
            sum(
                count for source, count in compute_sources.items()
                if source in {"fitted-extrapolated", "metadata-fallback"}
            ) / max(1, real_compute_nodes)
        ),
        low_confidence_compute_fraction=(
            low_confidence / max(1, real_compute_nodes)
        ),
    )
