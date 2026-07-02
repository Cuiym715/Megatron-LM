from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
from typing import Deque, Dict, List, Optional, Sequence, Tuple


TaskId = Tuple[int, int]


@dataclass(frozen=True)
class SliceVPlan:
    split_counts: Tuple[int, ...]
    phase_repeats: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class SliceVNode:
    kind: str
    stage: int
    microbatch: int
    split: int
    chunk: int
    phase: int
    slot: int
    start_time: int
    completion_time: int


@dataclass(frozen=True)
class SliceVAction:
    kind: str
    node: Optional[SliceVNode] = None
    sender: Optional[int] = None
    receiver: Optional[int] = None
    message: Optional[Tuple[str, int, int, int]] = None


def _forward_task_ids(split_counts: Sequence[int]) -> List[TaskId]:
    return [
        (microbatch, split)
        for microbatch, count in enumerate(split_counts)
        for split in range(count)
    ]


def _backward_task_ids(split_counts: Sequence[int]) -> List[TaskId]:
    return [
        (microbatch, split)
        for microbatch, count in enumerate(split_counts)
        for split in range(count - 1, -1, -1)
    ]


def _phase_repeats(num_stages: int, stage: int,
                   total_segments: int,
                   max_segments: int) -> Tuple[int, ...]:
    n1 = (num_stages - stage - 1) * 2
    n2 = max_segments + stage
    n3 = num_stages - stage - 1
    n4 = total_segments - n1 - n2
    n5 = num_stages - stage - 1
    n6 = max_segments + stage
    n7 = num_stages - stage - 1
    if n4 < 0:
        minimum = n1 + n2
        raise ValueError(
            f"SliceV requires at least {minimum} sequence segments on stage {stage}; "
            f"got {total_segments}."
        )
    return n1, n2, n3, n4, n5, n6, n7, 1


def _build_stage_schedule(stage: int,
                          num_stages: int,
                          split_counts: Sequence[int]) -> Tuple[List[SliceVNode], Tuple[int, ...]]:
    total_segments = sum(split_counts)
    repeats = _phase_repeats(
        num_stages, stage, total_segments, max(split_counts)
    )
    forward_ids = _forward_task_ids(split_counts)
    backward_ids = _backward_task_ids(split_counts)
    queues: Dict[str, Deque[TaskId]] = {
        'F0': deque(forward_ids),
        'F1': deque(forward_ids),
        'B0': deque(backward_ids),
        'B1': deque(backward_ids),
    }
    ready_weights: Deque[Tuple[int, TaskId]] = deque()
    schedule: List[SliceVNode] = []

    def emit(kind: str, chunk: int, task: TaskId, phase: int) -> None:
        slot = len(schedule)
        schedule.append(SliceVNode(
            kind=kind,
            stage=stage,
            microbatch=task[0],
            split=task[1],
            chunk=chunk,
            phase=phase,
            slot=slot,
            start_time=slot,
            completion_time=slot + 1,
        ))

    def emit_compute(kind: str, chunk: int, phase: int) -> TaskId:
        task = queues[f'{kind}{chunk}'].popleft()
        emit(kind, chunk, task, phase)
        if kind == 'B':
            ready_weights.append((chunk, task))
        return task

    def emit_weight(phase: int) -> None:
        if not ready_weights:
            raise RuntimeError(
                f"SliceV stage {stage} phase {phase} requested W before any B was ready."
            )
        chunk, task = ready_weights.popleft()
        emit('W', chunk, task, phase)

    def emit_bw(chunk: int, phase: int) -> None:
        emit_compute('B', chunk, phase)
        emit_weight(phase)

    n1, n2, n3, n4, n5, n6, n7, _ = repeats
    for _ in range(n1):
        emit_compute('F', 0, 1)
    for _ in range(n2):
        emit_compute('F', 0, 2)
        emit_compute('F', 1, 2)
    for _ in range(n3):
        emit_bw(1, 3)
        emit_compute('F', 1, 3)
    for _ in range(n4):
        emit_bw(1, 4)
        emit_compute('F', 0, 4)
        emit_bw(0, 4)
        emit_compute('F', 1, 4)
    for _ in range(n5):
        emit_bw(1, 5)
        emit_bw(0, 5)
        emit_compute('F', 1, 5)
    for _ in range(n6):
        emit_compute('B', 1, 6)
        emit_compute('B', 0, 6)
    for _ in range(n7):
        emit_weight(7)
        emit_compute('B', 0, 7)
    while ready_weights:
        emit_weight(8)

    if any(queues.values()):
        remaining = {name: len(queue) for name, queue in queues.items() if queue}
        raise RuntimeError(f"SliceV stage {stage} left compute tasks unscheduled: {remaining}")
    expected = total_segments * 6
    if len(schedule) != expected:
        raise RuntimeError(
            f"SliceV stage {stage} generated {len(schedule)} tasks; expected {expected}."
        )
    return schedule, repeats


def _validate_dependencies(schedules: Sequence[Sequence[SliceVNode]],
                           split_counts: Sequence[int]) -> None:
    positions = {
        (node.stage, node.kind, node.chunk, node.microbatch, node.split): node.slot
        for schedule in schedules
        for node in schedule
    }
    num_stages = len(schedules)

    def pos(stage: int, kind: str, chunk: int, task: TaskId) -> int:
        return positions[(stage, kind, chunk, task[0], task[1])]

    for stage in range(num_stages):
        for microbatch, count in enumerate(split_counts):
            for split in range(count):
                task = (microbatch, split)
                assert pos(stage, 'B', 0, task) < pos(stage, 'W', 0, task)
                assert pos(stage, 'B', 1, task) < pos(stage, 'W', 1, task)
                if stage == 0:
                    assert pos(stage, 'F', 1, task) < pos(stage, 'B', 1, task)
                if stage == num_stages - 1:
                    assert pos(stage, 'F', 0, task) < pos(stage, 'F', 1, task)
                    assert pos(stage, 'B', 1, task) < pos(stage, 'B', 0, task)
                if split > 0:
                    previous = (microbatch, split - 1)
                    assert pos(stage, 'F', 0, previous) < pos(stage, 'F', 0, task)
                    assert pos(stage, 'F', 1, previous) < pos(stage, 'F', 1, task)
                if split + 1 < count:
                    following = (microbatch, split + 1)
                    assert pos(stage, 'B', 0, following) < pos(stage, 'B', 0, task)
                    assert pos(stage, 'B', 1, following) < pos(stage, 'B', 1, task)

    for left_stage in range(num_stages - 1):
        left = schedules[left_stage]
        right = schedules[left_stage + 1]
        left_to_right = [
            (node.kind, node.chunk, node.microbatch, node.split)
            for node in left
            if (node.kind, node.chunk) in {('F', 0), ('B', 1)}
        ]
        received_from_left = [
            (node.kind, node.chunk, node.microbatch, node.split)
            for node in right
            if (node.kind, node.chunk) in {('F', 0), ('B', 1)}
        ]
        if sorted(left_to_right) != sorted(received_from_left):
            raise RuntimeError(
                "SliceV left-to-right message sets differ on edge "
                f"{left_stage}->{left_stage + 1}."
            )

        right_to_left = [
            (node.kind, node.chunk, node.microbatch, node.split)
            for node in right
            if (node.kind, node.chunk) in {('F', 1), ('B', 0)}
        ]
        received_from_right = [
            (node.kind, node.chunk, node.microbatch, node.split)
            for node in left
            if (node.kind, node.chunk) in {('F', 1), ('B', 0)}
        ]
        if sorted(right_to_left) != sorted(received_from_right):
            raise RuntimeError(
                "SliceV right-to-left message sets differ on edge "
                f"{left_stage + 1}->{left_stage}."
            )


def build_slice_v_schedule(num_stages: int,
                           num_microbatches: int,
                           split_counts: Sequence[int]) -> Tuple[List[List[SliceVNode]], SliceVPlan]:
    if num_stages <= 0 or num_microbatches <= 0:
        raise ValueError("num_stages and num_microbatches must be positive.")
    if len(split_counts) != num_microbatches:
        raise ValueError("split_counts length must equal num_microbatches.")
    if any(count <= 0 for count in split_counts):
        raise ValueError("all split counts must be positive.")

    schedules: List[List[SliceVNode]] = []
    phase_repeats: List[Tuple[int, ...]] = []
    for stage in range(num_stages):
        schedule, repeats = _build_stage_schedule(stage, num_stages, split_counts)
        schedules.append(schedule)
        phase_repeats.append(repeats)
    _validate_dependencies(schedules, split_counts)
    return schedules, SliceVPlan(
        split_counts=tuple(split_counts),
        phase_repeats=tuple(phase_repeats),
    )


def build_slice_v_execution_plan(
        schedules: Sequence[Sequence[SliceVNode]]) -> List[SliceVAction]:
    """Build a deadlock-free global action order without reordering computation."""
    graph: Dict[Tuple, List[Tuple]] = defaultdict(list)
    indegree: Dict[Tuple, int] = {}
    priorities: Dict[Tuple, Tuple] = {}
    compute_nodes: Dict[Tuple, SliceVNode] = {}
    message_nodes: Dict[Tuple, Tuple[int, int, Tuple[str, int, int, int]]] = {}

    def add_node(key: Tuple, priority: Tuple) -> None:
        indegree.setdefault(key, 0)
        priorities[key] = priority

    def add_edge(source: Tuple, target: Tuple) -> None:
        graph[source].append(target)
        indegree[target] += 1

    positions = {}
    for stage, schedule in enumerate(schedules):
        previous = None
        for index, node in enumerate(schedule):
            key = ('compute', stage, index)
            add_node(key, (node.slot * 2, stage, index))
            compute_nodes[key] = node
            positions[(stage, node.kind, node.chunk,
                       node.microbatch, node.split)] = key
            if previous is not None:
                add_edge(previous, key)
            previous = key

    def add_message(sender: int, receiver: int, kind: str, chunk: int,
                    microbatch: int, split: int) -> None:
        message = (kind, chunk, microbatch, split)
        producer = positions[(sender, *message)]
        consumer = positions[(receiver, *message)]
        key = ('message', sender, receiver, *message)
        producer_node = compute_nodes[producer]
        add_node(key, (producer_node.slot * 2 + 1, sender, receiver,
                       kind, chunk, microbatch, split))
        message_nodes[key] = (sender, receiver, message)
        add_edge(producer, key)
        add_edge(key, consumer)

    num_stages = len(schedules)
    if not num_stages:
        return []
    first_schedule = schedules[0]
    tasks = [(node.microbatch, node.split) for node in first_schedule
             if node.kind == 'F' and node.chunk == 0]
    for left in range(num_stages - 1):
        right = left + 1
        for microbatch, split in tasks:
            add_message(left, right, 'F', 0, microbatch, split)
            add_message(right, left, 'F', 1, microbatch, split)
            add_message(left, right, 'B', 1, microbatch, split)
            add_message(right, left, 'B', 0, microbatch, split)

    ready = [(priorities[key], key) for key, degree in indegree.items()
             if degree == 0]
    heapq.heapify(ready)
    actions = []
    while ready:
        _, key = heapq.heappop(ready)
        if key[0] == 'compute':
            actions.append(SliceVAction(kind='compute', node=compute_nodes[key]))
        else:
            sender, receiver, message = message_nodes[key]
            actions.append(SliceVAction(
                kind='communication', sender=sender, receiver=receiver,
                message=message,
            ))
        for target in graph[key]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, (priorities[target], target))

    if len(actions) != len(indegree):
        blocked = [key for key, degree in indegree.items() if degree > 0]
        raise RuntimeError(
            "SliceV computation and communication dependencies contain a cycle: "
            f"{blocked[:8]}"
        )
    return actions
