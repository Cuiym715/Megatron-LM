"""Low-overhead DSPP physical-microbatch ordering.

Segments of a long sequence are consecutive in the pipeline entrance order.
They remain in one Slice-V group only because the current scheduler traverses
each group's backward tasks in reverse; this does not make the chain one
warmup unit and does not prohibit B/W or unrelated tasks from interleaving.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .dspp_batch_plan import DsppMicrobatchMeta
from .dspp_training import DsppTrainingBatch


TaskId = Tuple[int, int]


@dataclass(frozen=True)
class _LongChain:
    key: Tuple[int, int]
    sequence_length: int
    tasks: Tuple[TaskId, ...]


@dataclass(frozen=True)
class DsppOrderingPlan:
    mode: str
    entrance_order: Tuple[TaskId, ...]
    schedule_groups: Tuple[Tuple[TaskId, ...], ...]
    workloads: Mapping[TaskId, float]
    warmup_count: int
    warmup_task_count: int
    warmup_short_only_count: int
    cost_source: str


def task_signature(meta: DsppMicrobatchMeta) -> str:
    """Shape/cost signature shared by timeline profiles and later runs."""

    items = ";".join(
        f"{item.token_length}:{item.token_offset + item.token_length}"
        for item in meta.items
    )
    return f"c{meta.chunk_size}|{items}"


@lru_cache(maxsize=8)
def _load_profile_costs(path: str) -> Mapping[str, float]:
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    costs = payload.get("costs", payload)
    return {str(key): float(value) for key, value in costs.items()}


def _task_metadata(
    batches: Sequence[DsppTrainingBatch],
) -> Dict[TaskId, DsppMicrobatchMeta]:
    return {
        (logical_id, physical_id): physical.meta
        for logical_id, batch in enumerate(batches)
        for physical_id, physical in enumerate(batch.physical_microbatches)
    }


def _task_workloads(
    metadata: Mapping[TaskId, DsppMicrobatchMeta],
    profile_path: Optional[str],
) -> Tuple[Dict[TaskId, float], str]:
    profile = _load_profile_costs(profile_path) if profile_path else {}
    result = {}
    profile_hits = 0
    for task, meta in metadata.items():
        signature = task_signature(meta)
        if signature in profile:
            result[task] = profile[signature]
            profile_hits += 1
        else:
            # Metadata-only fallback. It never simulates runtime or delays GPU
            # work; a saved true-machine profile takes precedence when present.
            result[task] = sum(float(item.estimated_flops) for item in meta.items)
    if profile_hits == len(metadata) and metadata:
        source = "profile"
    elif profile_hits:
        source = "profile+metadata-fallback"
    else:
        source = "metadata-proxy"
    return result, source


def _build_long_chains(
    metadata: Mapping[TaskId, DsppMicrobatchMeta],
) -> Tuple[List[_LongChain], set[TaskId]]:
    entries: Dict[Tuple[int, int], List[Tuple[int, TaskId, int]]] = {}
    long_tasks = set()
    for task, meta in metadata.items():
        logical_id, _ = task
        for item in meta.items:
            if item.is_short_sequence:
                continue
            key = (logical_id, item.sequence_id)
            entries.setdefault(key, []).append(
                (item.segment_id, task, item.sequence_length)
            )
            long_tasks.add(task)
    chains = []
    for key, values in entries.items():
        values.sort(key=lambda value: value[0])
        chains.append(
            _LongChain(
                key=key,
                sequence_length=max(value[2] for value in values),
                tasks=tuple(value[1] for value in values),
            )
        )
    return chains, long_tasks


def _closest_workload_window(
    candidates: Sequence[TaskId], workloads: Mapping[TaskId, float], count: int
) -> List[TaskId]:
    if count <= 0 or not candidates:
        return []
    ordered = sorted(candidates, key=lambda task: (workloads[task], task))
    if len(ordered) <= count:
        return ordered
    best = ordered[:count]
    best_key = (
        workloads[best[-1]] - workloads[best[0]],
        sum(workloads[task] for task in best) / count,
        tuple(best),
    )
    for start in range(1, len(ordered) - count + 1):
        window = ordered[start : start + count]
        key = (
            workloads[window[-1]] - workloads[window[0]],
            sum(workloads[task] for task in window) / count,
            tuple(window),
        )
        if key < best_key:
            best, best_key = window, key
    return list(best)


def _input_atomic_units(
    input_order: Sequence[TaskId], chains: Sequence[_LongChain]
) -> Tuple[Tuple[TaskId, ...], ...]:
    """Keep each sequence chain as one Slice-V microbatch.

    Dataloader logical batches are only data-loading containers.  They must
    not become Slice-V microbatches because doing so combines unrelated
    sequences into one oversized split chain and inflates the warmup.
    """

    chain_by_task = {
        task: chain.tasks
        for chain in chains
        for task in chain.tasks
    }
    emitted = set()
    units: List[Tuple[TaskId, ...]] = []
    for task in input_order:
        unit = chain_by_task.get(task, (task,))
        if unit in emitted:
            continue
        units.append(unit)
        emitted.add(unit)
    return tuple(units)


def build_dspp_ordering_plan(
    batches: Sequence[DsppTrainingBatch],
    *,
    pp_degree: int,
    mode: str = "warmup-short-steady-long",
    profile_path: Optional[str] = None,
) -> DsppOrderingPlan:
    if pp_degree <= 0:
        raise ValueError("pp_degree must be positive")
    if not batches:
        raise ValueError("DSPP ordering requires at least one logical batch")
    metadata = _task_metadata(batches)
    workloads, source = _task_workloads(metadata, profile_path)
    input_groups = tuple(
        tuple((logical_id, physical_id) for physical_id in range(len(batch.physical_microbatches)))
        for logical_id, batch in enumerate(batches)
    )
    input_order = tuple(task for group in input_groups for task in group)
    chains, long_tasks = _build_long_chains(metadata)
    if mode == "input":
        schedule_groups = _input_atomic_units(input_order, chains)
        return DsppOrderingPlan(
            mode=mode,
            entrance_order=input_order,
            schedule_groups=schedule_groups,
            workloads=workloads,
            warmup_count=0,
            warmup_task_count=0,
            warmup_short_only_count=0,
            cost_source=source,
        )
    if mode != "warmup-short-steady-long":
        raise ValueError(f"unknown DSPP microbatch order: {mode}")

    # Slice-V split counts describe one sequence chain, not all physical
    # microbatches produced by one dataloader logical batch.  Short/packed
    # standalone tasks have one split.
    max_split_count = max((len(chain.tasks) for chain in chains), default=1)
    warmup_count = min(len(metadata), 2 * (pp_degree - 1) + max_split_count)
    # Match the simulator's ``kind == packed_residual`` predicate: a single
    # exact-chunk short sequence is a full segment, not a warmup candidate.
    short_only = [
        task
        for task in input_order
        if task not in long_tasks
        and (
            len(metadata[task].items) > 1
            or metadata[task].items[0].token_length < metadata[task].chunk_size
        )
    ]
    warmup = _closest_workload_window(short_only, workloads, warmup_count)
    selected = set(warmup)
    units: List[Tuple[TaskId, ...]] = [(task,) for task in warmup]

    if len(selected) < warmup_count:
        for chain in sorted(
            chains,
            key=lambda item: (item.sequence_length, len(item.tasks), item.key),
        ):
            if any(task in selected for task in chain.tasks):
                continue
            units.append(chain.tasks)
            selected.update(chain.tasks)
            if len(selected) >= warmup_count:
                break

    # A selected chain stays consecutive, but the policy phase boundary remains
    # exactly at the simulator's target.  The boundary may therefore fall
    # inside a chain; this is a label used by ordering, not a scheduling fence.
    warmup_task_count = warmup_count

    for chain in sorted(chains, key=lambda item: (-item.sequence_length, item.key)):
        if any(task in selected for task in chain.tasks):
            continue
        units.append(chain.tasks)
        selected.update(chain.tasks)

    for task in sorted(short_only, key=lambda item: (-workloads[item], item)):
        if task not in selected:
            units.append((task,))
            selected.add(task)
    for task in input_order:
        if task not in selected:
            units.append((task,))
            selected.add(task)

    groups = tuple(units)
    order = tuple(task for group in groups for task in group)
    return DsppOrderingPlan(
        mode=mode,
        entrance_order=order,
        schedule_groups=groups,
        workloads=workloads,
        warmup_count=warmup_count,
        warmup_task_count=warmup_task_count,
        warmup_short_only_count=sum(task in short_only for task in warmup),
        cost_source=source,
    )


def _describe_microbatch(logical_id: int, meta: DsppMicrobatchMeta) -> str:
    items = []
    for item in meta.items:
        kind = "short" if item.is_short_sequence else "long"
        items.append(
            f"L{logical_id}/S{item.sequence_id} {kind} "
            f"seg {item.segment_id + 1}/{item.segment_count} "
            f"q[{item.token_offset}:{item.token_offset + item.token_length}] "
            f"len={item.sequence_length}"
        )
    return "<br>".join(items)


def render_dspp_microbatch_report(
    batches: Sequence[DsppTrainingBatch],
    plan: DsppOrderingPlan,
    *,
    schedule_split_counts: Sequence[int],
    model_layout: Optional[Mapping[int, Sequence[str]]] = None,
) -> str:
    """Render the construction and final entrance order as readable Markdown."""

    metadata = _task_metadata(batches)
    input_order = [
        (logical_id, physical_id)
        for logical_id, batch in enumerate(batches)
        for physical_id in range(len(batch.physical_microbatches))
    ]
    group_by_task = {
        task: group_id
        for group_id, group in enumerate(plan.schedule_groups)
        for task in group
    }
    lines = [
        "# DSPP microbatch construction and ordering",
        "",
        f"- Mode: `{plan.mode}`",
        f"- Cost source: `{plan.cost_source}`",
        f"- Physical microbatches: {len(plan.entrance_order)}",
        f"- Warmup boundary: first {plan.warmup_task_count} tasks",
        f"- Warmup short-only tasks: {plan.warmup_short_only_count}",
        "- Real schedule group sizes: " + ", ".join(
            str(len(group)) for group in plan.schedule_groups
        ),
        "- Group sizes including communication-only padding: " + ", ".join(
            str(count) for count in schedule_split_counts
        ),
    ]
    if model_layout:
        lines.extend(["", "## Model layout", ""])
        for stage, descriptions in sorted(model_layout.items()):
            lines.append(f"- Stage {stage}: " + "; ".join(descriptions))

    lines.extend(
        [
            "",
            "## Constructed physical microbatches (before ordering)",
            "",
            "| Input position | TaskId | Valid/padding tokens | Composition |",
            "| ---: | --- | ---: | --- |",
        ]
    )
    for position, task in enumerate(input_order):
        meta = metadata[task]
        lines.append(
            f"| {position} | L{task[0]}:P{task[1]} | "
            f"{meta.valid_token_count}/{meta.padding_token_count} | "
            f"{_describe_microbatch(task[0], meta)} |"
        )

    lines.extend(
        [
            "",
            "## Final pipeline entrance order",
            "",
            "| Order | Phase | Schedule group | TaskId | Workload | Composition |",
            "| ---: | --- | ---: | --- | ---: | --- |",
        ]
    )
    for position, task in enumerate(plan.entrance_order):
        meta = metadata[task]
        if plan.mode == "input":
            phase = "input"
        elif position < plan.warmup_task_count:
            phase = "warmup"
        else:
            phase = "steady"
        lines.append(
            f"| {position} | {phase} | {group_by_task[task]} | "
            f"L{task[0]}:P{task[1]} | {plan.workloads[task]:.4f} | "
            f"{_describe_microbatch(task[0], meta)} |"
        )
    unit = "milliseconds" if plan.cost_source == "profile" else "metadata proxy units"
    lines.extend(
        [
            "",
            f"Workload unit: {unit}. It is used only for ordering; it is not simulated runtime.",
            "Long-sequence segments are consecutive in the entrance F order. The warmup "
            "boundary may fall inside a chain; it neither inserts a fence nor prevents "
            "B/W or unrelated work from interleaving.",
            "",
        ]
    )
    return "\n".join(lines)
