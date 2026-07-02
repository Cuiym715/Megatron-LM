from collections import defaultdict
from dataclasses import dataclass
import heapq
from itertools import permutations
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple, Union


_PERIOD = 6
_KINDS = ("F0", "F1", "B1", "B0", "W1", "W0")
DeltaInput = Union[int, Sequence[int]]
TaskKey = Tuple[str, int, int, int]


@dataclass(frozen=True)
class SliceVPlan:
    delta: Tuple[int, int]
    transition_stage: int
    period: int
    stage_phases: Tuple[Tuple[int, ...], ...]
    stage_offsets: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class SliceVNode:
    kind: str
    stage: int
    microbatch: int
    split: int
    chunk: int
    slot: int
    start_time: float
    completion_time: float


@dataclass(frozen=True)
class _Candidate:
    plan: SliceVPlan
    slots: Dict[TaskKey, int]
    dependencies: Dict[TaskKey, Tuple[Tuple[TaskKey, int], ...]]
    stage_order: Tuple[Tuple[TaskKey, ...], ...]
    score: Tuple[int, int, int, Tuple[Tuple[int, ...], ...]]


def _normalize_delta(delta: DeltaInput) -> Tuple[int, int]:
    if isinstance(delta, int):
        values = (delta, delta)
    else:
        values = tuple(int(value) for value in delta)
        if len(values) == 4:
            if values[0] != values[2] or values[1] != values[3]:
                raise ValueError(
                    "four-value delta must satisfy F0=B1 and F1=B0; "
                    "prefer [delta0, delta1]."
                )
            values = values[:2]
        if len(values) != 2:
            raise ValueError("delta must be an int or [delta0, delta1].")
    if any(value <= 0 for value in values):
        raise ValueError("all delta values must be positive.")
    return values


def _fill_weight_phases(phases: List[int]) -> Optional[List[int]]:
    used = set(phases[:4])
    for backward_index, weight_index in ((2, 4), (3, 5)):
        for distance in range(_PERIOD):
            candidate = (phases[backward_index] + distance) % _PERIOD
            if candidate not in used:
                phases[weight_index] = candidate
                used.add(candidate)
                break
        else:
            return None
    return phases


def _candidate_stage_zero_patterns() -> Iterable[Tuple[int, ...]]:
    for f1, b1, b0 in permutations(range(1, _PERIOD), 3):
        phases = _fill_weight_phases([0, f1, b1, b0, -1, -1])
        if phases is not None:
            yield tuple(phases)


def _build_plan(*,
                num_stages: int,
                delta: Tuple[int, int],
                transition_stage: int,
                stage_zero_phases: Tuple[int, ...]) -> Optional[SliceVPlan]:
    phase_rows: List[Tuple[int, ...]] = [stage_zero_phases]
    offset_rows: List[Tuple[int, ...]] = [stage_zero_phases]

    for boundary in range(num_stages - 1):
        forward_delta = delta[0] if boundary < transition_stage else delta[1]
        signed_delta = (forward_delta, -1, forward_delta, -1)
        previous_offsets = offset_rows[-1]
        dependency_offsets = [
            previous_offsets[index] + signed_delta[index]
            for index in range(4)
        ]
        dependency_phases = [offset % _PERIOD for offset in dependency_offsets]
        if len(set(dependency_phases)) != 4:
            return None
        phases = _fill_weight_phases(dependency_phases + [-1, -1])
        if phases is None:
            return None

        offsets = dependency_offsets + [0, 0]
        for backward_index, weight_index in ((2, 4), (3, 5)):
            distance = (phases[weight_index] - phases[backward_index]) % _PERIOD
            offsets[weight_index] = offsets[backward_index] + distance
        phase_rows.append(tuple(phases))
        offset_rows.append(tuple(offsets))

    return SliceVPlan(
        delta=delta,
        transition_stage=transition_stage,
        period=_PERIOD,
        stage_phases=tuple(phase_rows),
        stage_offsets=tuple(offset_rows),
    )


def _forward_positions(num_stages: int) -> List[Tuple[int, int]]:
    return (
        [(stage, 0) for stage in range(num_stages)]
        + [(stage, 1) for stage in range(num_stages - 1, -1, -1)]
    )


def _task_occurrences(split_counts: Sequence[int],
                      reverse_splits: bool) -> Iterable[Tuple[int, int]]:
    for microbatch, seq_splits in enumerate(split_counts):
        split_range = (
            range(seq_splits - 1, -1, -1)
            if reverse_splits
            else range(seq_splits)
        )
        for split in split_range:
            yield microbatch, split


def _key(kind: str, stage: int, microbatch: int, split: int) -> TaskKey:
    return kind, stage, microbatch, split


def _add_dependency(dependencies: DefaultDict[TaskKey, Dict[TaskKey, int]],
                    child: TaskKey,
                    parent: TaskKey,
                    slot_distance: int) -> None:
    dependencies[child][parent] = max(
        dependencies[child].get(parent, 0),
        slot_distance,
    )


def _build_dependency_graph(*,
                            num_stages: int,
                            split_counts: Sequence[int],
                            plan: SliceVPlan
                            ) -> Tuple[List[TaskKey], Dict[TaskKey, Tuple[Tuple[TaskKey, int], ...]]]:
    keys: List[TaskKey] = []
    mutable_dependencies: DefaultDict[TaskKey, Dict[TaskKey, int]] = defaultdict(dict)

    for stage in range(num_stages):
        for kind in _KINDS:
            previous: Optional[TaskKey] = None
            for microbatch, split in _task_occurrences(
                    split_counts, reverse_splits=kind[0] in {"B", "W"}):
                task = _key(kind, stage, microbatch, split)
                keys.append(task)
                if previous is not None:
                    _add_dependency(mutable_dependencies, task, previous, _PERIOD)
                previous = task

    forward_path = _forward_positions(num_stages)
    backward_path = list(reversed(forward_path))

    for microbatch, seq_splits in enumerate(split_counts):
        for split in range(seq_splits):
            for op_type, path in (("F", forward_path), ("B", backward_path)):
                for parent_position, child_position in zip(path, path[1:]):
                    parent_stage, parent_chunk = parent_position
                    child_stage, child_chunk = child_position
                    if parent_chunk != child_chunk:
                        distance = 1
                    elif (op_type, parent_chunk) in {("F", 0), ("B", 1)}:
                        boundary = min(parent_stage, child_stage)
                        distance = (
                            plan.delta[0]
                            if boundary < plan.transition_stage
                            else plan.delta[1]
                        )
                    else:
                        distance = 1
                    _add_dependency(
                        mutable_dependencies,
                        _key(f"{op_type}{child_chunk}", child_stage, microbatch, split),
                        _key(f"{op_type}{parent_chunk}", parent_stage, microbatch, split),
                        distance,
                    )

            for stage, chunk in forward_path:
                if split > 0:
                    _add_dependency(
                        mutable_dependencies,
                        _key(f"F{chunk}", stage, microbatch, split),
                        _key(f"F{chunk}", stage, microbatch, split - 1),
                        1,
                    )

            for stage, chunk in backward_path:
                if split < seq_splits - 1:
                    _add_dependency(
                        mutable_dependencies,
                        _key(f"B{chunk}", stage, microbatch, split),
                        _key(f"B{chunk}", stage, microbatch, split + 1),
                        1,
                    )

            if split == seq_splits - 1:
                final_forward_stage, final_forward_chunk = forward_path[-1]
                first_backward_stage, first_backward_chunk = backward_path[0]
                _add_dependency(
                    mutable_dependencies,
                    _key(f"B{first_backward_chunk}", first_backward_stage, microbatch, split),
                    _key(f"F{final_forward_chunk}", final_forward_stage, microbatch, split),
                    1,
                )

            for stage in range(num_stages):
                for chunk in (0, 1):
                    _add_dependency(
                        mutable_dependencies,
                        _key(f"W{chunk}", stage, microbatch, split),
                        _key(f"B{chunk}", stage, microbatch, split),
                        1,
                    )

    return keys, {
        key: tuple(mutable_dependencies.get(key, {}).items())
        for key in keys
    }


def _assign_absolute_slots(*,
                           plan: SliceVPlan,
                           keys: Sequence[TaskKey],
                           dependencies: Dict[TaskKey, Tuple[Tuple[TaskKey, int], ...]]
                           ) -> Optional[Dict[TaskKey, int]]:
    kind_index = {kind: index for index, kind in enumerate(_KINDS)}
    indegree = {key: len(dependencies[key]) for key in keys}
    children: DefaultDict[TaskKey, List[TaskKey]] = defaultdict(list)
    for child, parents in dependencies.items():
        for parent, _ in parents:
            children[parent].append(child)

    ready = sorted(key for key, degree in indegree.items() if degree == 0)
    slots: Dict[TaskKey, int] = {}
    while ready:
        key = ready.pop(0)
        kind, stage, _, _ = key
        phase = plan.stage_phases[stage][kind_index[kind]]
        lower_bound = phase
        for parent, distance in dependencies[key]:
            lower_bound = max(lower_bound, slots[parent] + distance)
        slots[key] = lower_bound + (phase - lower_bound) % _PERIOD
        for child in children.get(key, []):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort()

    if len(slots) != len(keys):
        return None
    return slots


def _segment_stage_order(*,
                         num_stages: int,
                         slots: Dict[TaskKey, int]
                         ) -> Optional[Tuple[Tuple[TaskKey, ...], ...]]:
    rows: List[Tuple[TaskKey, ...]] = []
    for stage in range(num_stages):
        row = sorted(
            (key for key in slots if key[1] == stage),
            key=lambda key: (
                slots[key],
                _KINDS.index(key[0]),
                key[2],
                key[3],
            ),
        )
        row_slots = [slots[key] for key in row]
        if len(row_slots) != len(set(row_slots)):
            return None
        rows.append(tuple(row))
    return tuple(rows)


def _candidate_score(*,
                     stage_order: Tuple[Tuple[TaskKey, ...], ...],
                     slots: Dict[TaskKey, int]) -> Tuple[int, int, int, Tuple[Tuple[int, ...], ...]]:
    boundary_bubbles: List[int] = []
    peak_memory = 0
    phase_rows: List[Tuple[int, ...]] = []
    for row in stage_order:
        row_slots = [slots[key] for key in row]
        boundary_bubbles.append(row_slots[-1] - row_slots[0] + 1 - len(row_slots))
        pending = 0
        stage_peak = 0
        for key in row:
            if key[0][0] == "F":
                pending += 1
            elif key[0][0] == "W":
                pending -= 1
            stage_peak = max(stage_peak, pending)
        peak_memory = max(peak_memory, stage_peak)
        phase_rows.append(tuple(slot % _PERIOD for slot in row_slots[:_PERIOD]))
    return (
        max(boundary_bubbles),
        peak_memory,
        sum(boundary_bubbles),
        tuple(phase_rows),
    )


def _build_candidate(*,
                     plan: SliceVPlan,
                     num_stages: int,
                     split_counts: Sequence[int]) -> Optional[_Candidate]:
    keys, dependencies = _build_dependency_graph(
        num_stages=num_stages,
        split_counts=split_counts,
        plan=plan,
    )
    slots = _assign_absolute_slots(
        plan=plan,
        keys=keys,
        dependencies=dependencies,
    )
    if slots is None:
        return None
    stage_order = _segment_stage_order(num_stages=num_stages, slots=slots)
    if stage_order is None:
        return None
    score = _candidate_score(stage_order=stage_order, slots=slots)
    return _Candidate(
        plan=plan,
        slots=slots,
        dependencies=dependencies,
        stage_order=stage_order,
        score=score,
    )


def _task_cost(key: TaskKey, split_counts: Sequence[int],
               cost: Sequence[float]) -> float:
    kind, _, microbatch, _ = key
    return float(cost[{"F": 0, "B": 1, "W": 2}[kind[0]]]) / float(
        2 * split_counts[microbatch]
    )


def _schedule_order(*,
                    stage_order: Sequence[Sequence[TaskKey]],
                    dependencies: Dict[TaskKey, Tuple[Tuple[TaskKey, int], ...]],
                    split_counts: Sequence[int],
                    cost: Sequence[float],
                    comm_cost: float
                    ) -> Optional[Tuple[Dict[TaskKey, float], Dict[TaskKey, float]]]:
    parents: Dict[TaskKey, Dict[TaskKey, float]] = {
        key: {
            parent: float(comm_cost) if parent[1] != key[1] else 0.0
            for parent, _ in task_dependencies
        }
        for key, task_dependencies in dependencies.items()
    }
    local_rank: Dict[TaskKey, Tuple[int, int]] = {}
    for stage, row in enumerate(stage_order):
        for index, key in enumerate(row):
            local_rank[key] = (index, stage)
        for previous, current in zip(row, row[1:]):
            parents[current][previous] = 0.0

    children: DefaultDict[TaskKey, List[TaskKey]] = defaultdict(list)
    indegree = {key: len(task_parents) for key, task_parents in parents.items()}
    for child, task_parents in parents.items():
        for parent in task_parents:
            children[parent].append(child)

    ready: List[Tuple[Tuple[int, int], TaskKey]] = [
        (local_rank[key], key)
        for key, degree in indegree.items()
        if degree == 0
    ]
    heapq.heapify(ready)
    start_times: Dict[TaskKey, float] = {}
    completion: Dict[TaskKey, float] = {}
    while ready:
        _, key = heapq.heappop(ready)
        start = 0.0
        for parent, communication in parents[key].items():
            start = max(start, completion[parent] + communication)
        start_times[key] = start
        completion[key] = start + _task_cost(key, split_counts, cost)

        for child in children.get(key, []):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, (local_rank[child], child))

    if len(completion) != len(parents):
        return None
    return start_times, completion


def _move_task(row: Sequence[TaskKey],
               source_index: int,
               destination_index: int) -> Tuple[TaskKey, ...]:
    moved = list(row)
    task = moved.pop(source_index)
    moved.insert(destination_index, task)
    return tuple(moved)


def _global_completion(completion: Dict[TaskKey, float]) -> float:
    return max(completion.values(), default=0.0)


def _communication_order_matches(
        stage_order: Sequence[Sequence[TaskKey]]) -> bool:
    """Keep untagged NCCL messages in the same order on both edge endpoints."""
    for left_stage in range(len(stage_order) - 1):
        right_stage = left_stage + 1

        left_to_right = [
            (kind, microbatch, split)
            for kind, _, microbatch, split in stage_order[left_stage]
            if kind in {"F0", "B1"}
        ]
        received_from_left = [
            (kind, microbatch, split)
            for kind, _, microbatch, split in stage_order[right_stage]
            if kind in {"F0", "B1"}
        ]
        if left_to_right != received_from_left:
            return False

        right_to_left = [
            (kind, microbatch, split)
            for kind, _, microbatch, split in stage_order[right_stage]
            if kind in {"F1", "B0"}
        ]
        received_from_right = [
            (kind, microbatch, split)
            for kind, _, microbatch, split in stage_order[left_stage]
            if kind in {"F1", "B0"}
        ]
        if right_to_left != received_from_right:
            return False
    return True


def _fill_boundary_bubbles(*,
                           stage_order: Sequence[Sequence[TaskKey]],
                           dependencies: Dict[TaskKey, Tuple[Tuple[TaskKey, int], ...]],
                           split_counts: Sequence[int],
                           cost: Sequence[float],
                           comm_cost: float,
                           warmup: bool) -> Tuple[Tuple[TaskKey, ...], ...]:
    rows = [tuple(row) for row in stage_order]
    epsilon = 1e-9

    for stage in range(len(rows)):
        while True:
            timing = _schedule_order(
                stage_order=rows,
                dependencies=dependencies,
                split_counts=split_counts,
                cost=cost,
                comm_cost=comm_cost,
            )
            if timing is None:
                raise RuntimeError("SliceV boundary reordering introduced a dependency cycle.")
            start_times, completion = timing
            current_global_completion = _global_completion(completion)
            row = rows[stage]

            if warmup:
                boundary = next(
                    (index for index, key in enumerate(row) if key[0] == "B0"),
                    len(row),
                )
                candidate_kinds = {"F0", "F1"}
                gap_indices = range(max(0, boundary - 1))
            else:
                last_forward = max(
                    (index for index, key in enumerate(row) if key[0][0] == "F"),
                    default=-1,
                )
                candidate_kinds = {"B0", "B1", "W0", "W1"}
                gap_indices = range(last_forward, len(row) - 1)

            accepted = False
            for gap_index in gap_indices:
                gap_start = completion[row[gap_index]]
                gap_end = start_times[row[gap_index + 1]]
                if gap_end - gap_start <= epsilon:
                    continue

                for source_index in range(gap_index + 2, len(row)):
                    candidate = row[source_index]
                    if candidate[0] not in candidate_kinds:
                        continue
                    if _task_cost(candidate, split_counts, cost) > gap_end - gap_start + epsilon:
                        continue

                    trial_rows = list(rows)
                    trial_rows[stage] = _move_task(row, source_index, gap_index + 1)
                    if not _communication_order_matches(trial_rows):
                        continue
                    trial_timing = _schedule_order(
                        stage_order=trial_rows,
                        dependencies=dependencies,
                        split_counts=split_counts,
                        cost=cost,
                        comm_cost=comm_cost,
                    )
                    if trial_timing is None:
                        continue
                    trial_start, trial_completion = trial_timing
                    if trial_start[candidate] > gap_start + epsilon:
                        continue
                    if trial_completion[candidate] > gap_end + epsilon:
                        continue
                    if _global_completion(trial_completion) > current_global_completion + epsilon:
                        continue

                    rows = trial_rows
                    accepted = True
                    break
                if accepted:
                    break
            if not accepted:
                break

    return tuple(rows)


def _reorder_cooldown_backward_before_weight(*,
                                             stage_order: Sequence[Sequence[TaskKey]],
                                             dependencies: Dict[TaskKey, Tuple[Tuple[TaskKey, int], ...]],
                                             split_counts: Sequence[int],
                                             cost: Sequence[float],
                                             comm_cost: float
                                             ) -> Tuple[Tuple[TaskKey, ...], ...]:
    rows = [tuple(row) for row in stage_order]
    epsilon = 1e-9

    for stage in range(len(rows)):
        source_index = 0
        while source_index < len(rows[stage]):
            row = rows[stage]
            last_forward = max(
                (index for index, key in enumerate(row) if key[0][0] == "F"),
                default=-1,
            )
            source_index = max(source_index, last_forward + 1)
            if source_index >= len(row):
                break
            if row[source_index][0][0] != "B":
                source_index += 1
                continue

            destination_index = source_index
            while destination_index > last_forward + 1 and row[destination_index - 1][0][0] == "W":
                destination_index -= 1
            if destination_index == source_index:
                source_index += 1
                continue

            current_timing = _schedule_order(
                stage_order=rows,
                dependencies=dependencies,
                split_counts=split_counts,
                cost=cost,
                comm_cost=comm_cost,
            )
            if current_timing is None:
                raise RuntimeError("SliceV cooldown reordering introduced a dependency cycle.")
            trial_rows = list(rows)
            trial_rows[stage] = _move_task(row, source_index, destination_index)
            if not _communication_order_matches(trial_rows):
                source_index += 1
                continue
            trial_timing = _schedule_order(
                stage_order=trial_rows,
                dependencies=dependencies,
                split_counts=split_counts,
                cost=cost,
                comm_cost=comm_cost,
            )
            if (
                trial_timing is not None
                and _global_completion(trial_timing[1])
                <= _global_completion(current_timing[1]) + epsilon
            ):
                rows = trial_rows
                source_index = destination_index + 1
            else:
                source_index += 1

    return tuple(rows)


def _reorder_boundaries(*,
                        stage_order: Sequence[Sequence[TaskKey]],
                        dependencies: Dict[TaskKey, Tuple[Tuple[TaskKey, int], ...]],
                        split_counts: Sequence[int],
                        cost: Sequence[float],
                        comm_cost: float) -> Tuple[Tuple[TaskKey, ...], ...]:
    warmup_reordered = _fill_boundary_bubbles(
        stage_order=stage_order,
        dependencies=dependencies,
        split_counts=split_counts,
        cost=cost,
        comm_cost=comm_cost,
        warmup=True,
    )
    cooldown_reordered = _reorder_cooldown_backward_before_weight(
        stage_order=warmup_reordered,
        dependencies=dependencies,
        split_counts=split_counts,
        cost=cost,
        comm_cost=comm_cost,
    )
    return _fill_boundary_bubbles(
        stage_order=cooldown_reordered,
        dependencies=dependencies,
        split_counts=split_counts,
        cost=cost,
        comm_cost=comm_cost,
        warmup=False,
    )


def _materialize_schedule(*,
                          candidate: _Candidate,
                          num_stages: int,
                          split_counts: Sequence[int],
                          cost: Sequence[float],
                          comm_cost: float) -> List[List[SliceVNode]]:
    stage_order = _reorder_boundaries(
        stage_order=candidate.stage_order,
        dependencies=candidate.dependencies,
        split_counts=split_counts,
        cost=cost,
        comm_cost=comm_cost,
    )
    timing = _schedule_order(
        stage_order=stage_order,
        dependencies=candidate.dependencies,
        split_counts=split_counts,
        cost=cost,
        comm_cost=comm_cost,
    )
    if timing is None:
        raise RuntimeError("SliceV reordering introduced a dependency cycle.")
    if not _communication_order_matches(stage_order):
        raise RuntimeError("SliceV generated a non-executable P2P message order.")
    start_times, completion = timing

    result: List[List[SliceVNode]] = [[] for _ in range(num_stages)]
    for stage, row in enumerate(stage_order):
        for key in row:
            kind, _, microbatch, split = key
            result[stage].append(SliceVNode(
                kind=kind[0],
                stage=stage,
                microbatch=microbatch,
                split=split,
                chunk=int(kind[1]),
                slot=candidate.slots[key],
                start_time=start_times[key],
                completion_time=completion[key],
            ))
    return result


def build_slice_v_schedule(num_stages: int,
                           num_microbatches: int,
                           split_counts: Sequence[int],
                           delta: DeltaInput = (2, 2),
                           cost: Sequence[float] = (6.0, 6.0, 6.0),
                           comm_cost: float = 0.0) -> Tuple[List[List[SliceVNode]], SliceVPlan]:
    if num_stages <= 0 or num_microbatches <= 0:
        raise ValueError("num_stages and num_microbatches must be positive.")
    if len(split_counts) != num_microbatches:
        raise ValueError("split_counts length must equal num_microbatches.")
    if any(split_count <= 0 for split_count in split_counts):
        raise ValueError("all split counts must be positive.")
    if len(cost) < 3:
        raise ValueError("cost must contain F, B, and W costs.")
    if any(float(value) <= 0 for value in cost[:3]):
        raise ValueError("F, B, and W costs must be positive.")

    normalized_delta = _normalize_delta(delta)
    candidates: List[_Candidate] = []
    transition_stages = range(1, num_stages) if num_stages > 1 else (0,)
    for transition_stage in transition_stages:
        for stage_zero_phases in _candidate_stage_zero_patterns():
            plan = _build_plan(
                num_stages=num_stages,
                delta=normalized_delta,
                transition_stage=transition_stage,
                stage_zero_phases=stage_zero_phases,
            )
            if plan is None:
                continue
            candidate = _build_candidate(
                plan=plan,
                num_stages=num_stages,
                split_counts=split_counts,
            )
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:
        raise RuntimeError("No collision-free SliceV schedule exists.")

    best = min(candidates, key=lambda candidate: candidate.score)
    result = _materialize_schedule(
        candidate=best,
        num_stages=num_stages,
        split_counts=split_counts,
        cost=cost,
        comm_cost=comm_cost,
    )
    return result, best.plan
