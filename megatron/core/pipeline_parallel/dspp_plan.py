"""CPU-only Plan IR, dependency replay, and configuration A.

No CUDA tensors, research-directory imports, sleeps, or runtime ready policy.
Version 1 retains F/B/W; version 2 adds explicit selective D/H execution.
"""
from dataclasses import asdict, dataclass, field, replace
from copy import deepcopy
import hashlib
import heapq
import json
import math
import time

from .slice_v import SliceVNode


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def node_key(node):
    return (node.stage, node.kind, node.chunk, node.microbatch, node.split)


def incoming_sender(stage, node, pp):
    kind = 'B' if node.kind == 'D' else node.kind
    if (kind, node.chunk) in {('F', 0), ('B', 1)} and stage > 0:
        return stage - 1
    if (kind, node.chunk) in {('F', 1), ('B', 0)} and stage < pp - 1:
        return stage + 1
    return None


def message_key(node):
    return ('B' if node.kind == 'D' else node.kind, node.chunk, node.microbatch, node.split)


def backward_key(keys, stage, chunk, group, split):
    key = (stage, 'D', chunk, group, split)
    return key if key in keys else (stage, 'B', chunk, group, split)


def receive_orders(schedules):
    """One shared F/B FIFO per directed physical edge, in consumer order."""
    result = {}
    for stage, schedule in enumerate(schedules):
        for peer in (stage - 1, stage + 1):
            if 0 <= peer < len(schedules):
                result[(peer, stage)] = tuple(message_key(n) for n in schedule
                                             if incoming_sender(stage, n, len(schedules)) == peer)
    return result


@dataclass(frozen=True)
class PlanBinding:
    iteration_id: int
    # P1 input_hash binds CPU layout/length metadata, not token contents or sampler IDs.
    input_hash: str
    layout_version: str
    cost_version: str = 'none'
    capacity_version: str = 'none'
    protocol_version: str = 'stage-local-v1'
    sample_ids: tuple = ()


@dataclass(frozen=True)
class DsppSchedulePlan:
    schema_version: int
    binding: PlanBinding
    algorithm: str
    schedules: tuple
    split_counts: tuple
    # (group, split, logical_batch, physical_task); padding has no binding.
    task_map: tuple
    fallback_reason: str = ''
    reference_ms: float = 0.0
    predicted_ms: float = 0.0
    wire_shape: tuple = ()
    wire_dtype: str = ''
    wire_bytes: int = 0
    # All source -> earlier consumer pairs from the frozen input metadata,
    # independent of selected split mode: (source_group, split, target_group, split).
    history_dependencies: tuple = ()
    planning_info: dict = field(default_factory=dict)

    @property
    def tasks(self):
        mapping = {(g, s): (l, p) for g, s, l, p in self.task_map}
        keys = {node_key(n) for seq in self.schedules for n in seq}
        records = []
        for seq in self.schedules:
            for n in seq:
                real = mapping.get((n.microbatch, n.split))
                base = ((self.binding.iteration_id, *real, n.chunk, n.stage)
                        if real is not None else None)
                source = backward_key(keys, n.stage, n.chunk, n.microbatch, n.split)[1]
                record = dict(key=(*base, n.kind) if base is not None else None,
                                    padding=base is None, stage=n.stage, slot=n.slot,
                                    weight_source=(*base, source) if base is not None and n.kind == 'W' else None)
                if self.schema_version == 2:
                    record['history_source'] = (*base, 'D') if base is not None and n.kind == 'H' else None
                records.append(record)
        return tuple(records)

    @property
    def messages(self):
        keys = {node_key(n) for seq in self.schedules for n in seq}
        def endpoint(stage, msg):
            kind, c, g, s = msg
            return backward_key(keys, stage, c, g, s) if kind == 'B' else (stage, *msg)
        return tuple(dict(sender=a, receiver=b, producer=endpoint(a, msg), consumer=endpoint(b, msg),
                          sequence=i, fifo_lane=(a, b), shape=self.wire_shape,
                          dtype=self.wire_dtype, bytes=self.wire_bytes)
                     for (a, b), lane in receive_orders(self.schedules).items()
                     for i, msg in enumerate(lane))

    def to_dict(self):
        # All structural fields are immutable tuples/frozen records. Avoid
        # recursively deepcopying every integer for each wire/hash export.
        fields = dict(vars(self))
        fields['binding'] = asdict(self.binding)
        fields['schedules'] = tuple(tuple(dict(vars(n)) for n in seq) for seq in self.schedules)
        fields['planning_info'] = deepcopy(self.planning_info)
        if self.schema_version == 1:
            fields.pop('history_dependencies')
        if not self.planning_info:
            fields.pop('planning_info')
        return dict(fields, tasks=self.tasks, messages=self.messages)

    @classmethod
    def from_dict(cls, payload):
        fields = dict(payload)
        supplied_tasks = fields.pop('tasks')
        supplied_messages = fields.pop('messages')
        fields['binding'] = PlanBinding(**{**fields['binding'],
                                         'sample_ids': tuple(fields['binding'].get('sample_ids', ()))})
        fields['schedules'] = tuple(tuple(SliceVNode(**n) for n in seq) for seq in fields['schedules'])
        fields['split_counts'] = tuple(fields['split_counts'])
        fields['task_map'] = tuple(tuple(row) for row in fields['task_map'])
        fields['wire_shape'] = tuple(fields['wire_shape'])
        fields['history_dependencies'] = tuple(tuple(edge) for edge in fields.get('history_dependencies', ()))
        plan = cls(**fields)
        if digest(supplied_tasks) != digest(plan.tasks):
            raise ValueError('stable task keys/padding/W source do not match plan')
        if digest(supplied_messages) != digest(plan.messages):
            raise ValueError('message identity/FIFO/wire contract does not match final plan')
        validate_plan(plan)
        return plan

    @property
    def plan_hash(self):
        return digest(self.to_dict())

    def task_key(self, node):
        task = {(g, s): (l, p) for g, s, l, p in self.task_map}.get((node.microbatch, node.split))
        if task is None:
            return None
        return (self.binding.iteration_id, *task, node.chunk, node.stage)


def template_plan(schedules, split_counts, task_map, binding, *, history_dependencies=None):
    plan = DsppSchedulePlan(1 if history_dependencies is None else 2, binding, 'template',
                            tuple(map(tuple, schedules)), tuple(split_counts), tuple(sorted(task_map)),
                            history_dependencies=tuple(history_dependencies or ()))
    validate_plan(plan)
    return plan


def history_dependencies(task_map, metadata):
    """Derive every history consumer from actual sequence/segment identities."""
    chains = {}
    for g, s, logical, physical in task_map:
        for item in metadata[logical, physical].items:
            chain = chains.setdefault((logical, item.sequence_id), {})
            if item.segment_id in chain:
                raise ValueError('duplicate input sequence segment')
            chain[item.segment_id] = (g, s)
    edges = set()
    for chain in chains.values():
        if sorted(chain) != list(range(len(chain))):
            raise ValueError('incomplete input sequence segment chain')
        for segment, source in chain.items():
            for target in range(segment):
                edges.add((*source, *chain[target]))
    return tuple(sorted(edges))


def split_plan(reference, selected):
    """Make an adjacent-H manual plan; callers can move H and revalidate.

    This is an execution fixture constructor, not algorithm B's split search.
    """
    if reference.schema_version != 2 or reference.algorithm != 'template':
        raise ValueError('selective splitting requires a v2 template')
    selected = set(selected)
    eligible = {(n.stage, 'B', n.chunk, n.microbatch, n.split)
                for seq in reference.schedules for n in seq if n.kind == 'B'
                and any((n.microbatch, n.split) == edge[:2] for edge in reference.history_dependencies)}
    if not selected <= eligible:
        raise ValueError('selected D must be a real task with history')
    schedules = []
    for seq in reference.schedules:
        out = []
        for n in seq:
            out.extend((replace(n, kind='D'), replace(n, kind='H')) if node_key(n) in selected else (n,))
        schedules.append(tuple(replace(n, slot=i) for i, n in enumerate(out)))
    plan = replace(reference, algorithm='explicit-DH', schedules=tuple(schedules))
    validate_plan(plan, reference)
    return plan


def _graph(plan, *, resource=True, fixed_fb=False, fixed_w=False):
    nodes = {node_key(n): n for seq in plan.schedules for n in seq}
    succ = {key: set() for key in nodes}
    pred = {key: set() for key in nodes}

    def add(a, b):
        if a not in succ or b not in pred:
            raise ValueError(f'missing dependency endpoint: {a} -> {b}')
        succ[a].add(b)
        pred[b].add(a)

    def key(stage, kind, chunk, group, split):
        return (backward_key(nodes, stage, chunk, group, split) if kind == 'B'
                else (stage, kind, chunk, group, split))

    for stage, seq in enumerate(plan.schedules):
        previous = None
        previous_w = {}
        for n in seq:
            k = node_key(n)
            if resource or (fixed_fb and n.kind in 'FB'):
                if previous is not None:
                    add(previous, k)
                previous = k
            g, s, c = n.microbatch, n.split, n.chunk
            if n.kind in ('B', 'D'):
                add(key(stage, 'F', c, g, s), k)
            if n.kind == 'H':
                add(key(stage, 'D', c, g, s), k)
            if n.kind == 'W':
                add(key(stage, 'B', c, g, s), k)
                if fixed_w and c in previous_w:
                    add(previous_w[c], k)
                previous_w[c] = k
            if n.kind == 'F' and s > 0:
                add(key(stage, 'F', c, g, s - 1), k)
            if n.kind in ('B', 'D') and s + 1 < plan.split_counts[g]:
                add(key(stage, 'B', c, g, s + 1), k)
            if stage == len(plan.schedules) - 1:
                if (n.kind, c) == ('F', 1):
                    add(key(stage, 'F', 0, g, s), k)
                if n.kind in ('B', 'D') and c == 0:
                    add(key(stage, 'B', 1, g, s), k)
            if stage == 0 and n.kind in ('B', 'D') and c == 1:
                add(key(stage, 'F', 1, g, s), k)

    for source_g, source_s, target_g, target_s in plan.history_dependencies:
        for stage in range(len(plan.schedules)):
            for chunk in range(2):
                source = key(stage, 'B', chunk, source_g, source_s)
                if source[1] == 'D':
                    source = key(stage, 'H', chunk, source_g, source_s)
                add(source, key(stage, 'B', chunk, target_g, target_s))

    messages = {}
    for (sender, receiver), lane in receive_orders(plan.schedules).items():
        previous = None
        for msg in lane:
            m = ('message', sender, receiver, *msg)
            succ[m], pred[m] = set(), set()
            kind, c, g, s = msg
            producer, consumer = key(sender, kind, c, g, s), key(receiver, kind, c, g, s)
            messages[m] = (producer, consumer)
            add(producer, m)
            add(m, consumer)
            if previous is not None:
                add(previous, m)
            previous = m
    return nodes, messages, succ, pred


def _replay(succ, pred, costs):
    degree = {k: len(p) for k, p in pred.items()}
    # Numeric IDs avoid comparing mixed compute/message tuple fields.
    ids = {k: i for i, k in enumerate(succ)}
    ready = [(ids[k], k) for k, d in degree.items() if not d]
    heapq.heapify(ready)
    start, end, order = dict.fromkeys(succ, 0.0), {}, []
    while ready:
        _, k = heapq.heappop(ready)
        order.append(k)
        end[k] = start[k] + costs[k]
        for v in sorted(succ[k], key=ids.__getitem__):
            start[v] = max(start[v], end[k])
            degree[v] -= 1
            if not degree[v]:
                heapq.heappush(ready, (ids[v], v))
    if len(order) != len(succ):
        raise ValueError('plan dependencies/communication FIFO contain a cycle')
    tail = {}
    for k in reversed(order):
        tail[k] = costs[k] + max((tail[v] for v in succ[k]), default=0.0)
    return max(end.values(), default=0.0), start, end, tail


def validate_plan(plan, reference=None, binding=None):
    if (plan.schema_version not in (1, 2)
            or plan.algorithm not in ({'template', 'A'} if plan.schema_version == 1
                                     else {'template', 'explicit-DH', 'B', 'C'})):
        raise ValueError('unsupported Plan schema or algorithm')
    if any(not math.isfinite(t) or t < 0 for t in (plan.reference_ms, plan.predicted_ms)):
        raise ValueError('invalid plan prediction summary')
    if binding is not None and plan.binding != binding:
        raise ValueError('stale or mismatched plan binding')
    if plan.wire_shape:
        dtype_bytes = {'torch.bfloat16': 2, 'torch.float16': 2, 'torch.float32': 4}.get(plan.wire_dtype)
        if (len(plan.wire_shape) != 3 or any(d <= 0 for d in plan.wire_shape)
                or dtype_bytes is None or math.prod(plan.wire_shape) * dtype_bytes != plan.wire_bytes):
            raise ValueError('invalid message shape/dtype/bytes')
    elif plan.wire_dtype or plan.wire_bytes:
        raise ValueError('incomplete wire contract')
    if not plan.schedules or not plan.split_counts or any(c <= 0 for c in plan.split_counts):
        raise ValueError('empty plan or invalid split counts')
    expected = {(r, k, c, g, s) for r in range(len(plan.schedules))
                for k in 'FBW' for c in range(2)
                for g, count in enumerate(plan.split_counts) for s in range(count)}
    actual = []
    real = {(g, s) for g, s, _, _ in plan.task_map}
    dependencies = plan.history_dependencies
    if (plan.schema_version == 1 and dependencies
            or len(set(dependencies)) != len(dependencies)):
        raise ValueError('invalid history dependency metadata')
    bindings = {(g, s): (l, p) for g, s, l, p in plan.task_map}
    for edge in dependencies:
        if len(edge) != 4 or edge[:2] not in real or edge[2:] not in real:
            raise ValueError('invalid history consumer task')
        source, target = bindings[edge[:2]], bindings[edge[2:]]
        if source[0] != target[0] or source[1] <= target[1]:
            raise ValueError('invalid history source/consumer identity')
    for r, seq in enumerate(plan.schedules):
        for slot, n in enumerate(seq):
            allowed = 'FBW' if plan.schema_version == 1 else 'FBDHW'
            if n.stage != r or n.slot != slot or n.kind not in allowed or len(n.kind) != 1:
                raise ValueError('invalid stage/slot/task kind')
            actual.append(node_key(n))
    canonical = {(r, 'B' if k == 'D' else k, c, g, s)
                 for r, k, c, g, s in actual if k != 'H'}
    if (len(actual) != len(set(actual)) or canonical != expected
            or len(canonical) != sum(k[1] != 'H' for k in actual)):
        raise ValueError('missing, duplicate, or unexpected F/backward/W task')
    selected = {k for k in actual if k[1] == 'D'}
    required_h = {(r, 'H', c, g, s) for r, _, c, g, s in selected}
    if {k for k in actual if k[1] == 'H'} != required_h:
        raise ValueError('missing or orphan H task')
    history_sources = {edge[:2] for edge in dependencies}
    if any((k[3], k[4]) not in history_sources for k in selected):
        raise ValueError('D/H must replace a real task with history')
    # The template emits W in B order. A may move W across F/B and across
    # another model chunk, but never reorder accumulation into the same
    # chunk's parameters. Check standalone/serialized plans too, before the
    # executor has an input-bound reference available.
    for seq in plan.schedules:
        for chunk in range(2):
            def order(kind):
                return [(n.microbatch, n.split) for n in seq
                        if (n.kind == kind or kind == 'B' and n.kind == 'D') and n.chunk == chunk]
            if order('W') != order('B'):
                raise ValueError('plan changed per-chunk W order from template B order')
    slots, tasks = set(), set()
    for g, s, l, p in plan.task_map:
        if (g, s) in slots or (l, p) in tasks or min(g, s, l, p) < 0:
            raise ValueError('duplicate or invalid physical task binding')
        if g >= len(plan.split_counts) or s >= plan.split_counts[g]:
            raise ValueError('task binding outside schedule')
        slots.add((g, s))
        tasks.add((l, p))
    # Padding may only follow real tasks, matching the existing executor.
    for g in range(len(plan.split_counts)):
        real = sorted(s for group, s in slots if group == g)
        if real != list(range(len(real))):
            raise ValueError('padding interleaved with real tasks')
    if reference is not None:
        if (plan.schema_version, plan.history_dependencies) != (reference.schema_version, reference.history_dependencies):
            raise ValueError('plan changes schema/history consumer metadata')
        if (plan.binding, plan.split_counts, plan.task_map) != (
                reference.binding, reference.split_counts, reference.task_map):
            raise ValueError('plan changes input/layout identity')
        if (plan.wire_shape, plan.wire_dtype, plan.wire_bytes) != (
                reference.wire_shape, reference.wire_dtype, reference.wire_bytes):
            raise ValueError('plan changes wire contract')
        if plan.algorithm == 'template' and plan.schedules != reference.schedules:
            raise ValueError('template plan differs from original template')
        for seq, original in zip(plan.schedules, reference.schedules):
            def fb_order(nodes):
                return [(n.stage, 'B' if n.kind == 'D' else n.kind, n.chunk, n.microbatch, n.split)
                        for n in nodes if n.kind in ('F', 'B', 'D')]
            if plan.algorithm in ('B', 'C'):
                for kind in ('F', 'B'):
                    if ([k for k in fb_order(seq) if k[1] == kind]
                            != [k for k in fb_order(original) if k[1] == kind]):
                        raise ValueError('B changed internal F/F or B/B order')
            elif fb_order(seq) != fb_order(original):
                raise ValueError('plan changed relative F/B order')
        if len(plan.schedules) != len(reference.schedules):
            raise ValueError('plan changes PP size')
    _, _, succ, pred = _graph(plan)
    _replay(succ, pred, dict.fromkeys(succ, 0.0))


@dataclass(frozen=True)
class PlanOwner:
    """Explicit storage owner; bytes must come from the caller's versioned contract."""
    stage: int
    bytes: float
    born: tuple
    users: tuple


def _owner_peaks(owners, start, end, pp):
    events = [[] for _ in range(pp)]
    for owner in owners:
        begin = min(start[k] for k in owner.born)
        finish = max(end[k] for k in owner.users)
        if finish < begin:
            raise ValueError('owner released before allocation')
        if finish > begin:
            events[owner.stage].extend(((begin, owner.bytes), (finish, -owner.bytes)))
    peaks = []
    for ev in events:
        live = peak = 0.0
        for _, delta in sorted(ev):
            live += delta
            peak = max(peak, live)
        peaks.append(peak)
    return tuple(peaks)


def order_owner_peaks(plan, owners):
    """Linear local-order audit, independent of predicted task durations.

    Reserve at the first producer's start and release after the last user's
    completion. Each owner is an already deduplicated allocation or a
    conservative accounting allowance, never an individual tensor view.
    """
    positions = {node_key(n): i for seq in plan.schedules for i, n in enumerate(seq)}
    changes = [[0] * (len(seq) + 1) for seq in plan.schedules]
    for owner in owners:
        if (not 0 <= owner.stage < len(changes) or not owner.born or not owner.users
                or not math.isfinite(owner.bytes) or owner.bytes < 0
                or any(k not in positions or k[0] != owner.stage
                       for k in (*owner.born, *owner.users))):
            raise ValueError('invalid local-order owner')
        begin = min(positions[k] for k in owner.born)
        finish = max(positions[k] for k in owner.users)
        if finish < max(positions[k] for k in owner.born):
            raise ValueError('owner released before its producers')
        changes[owner.stage][begin] += owner.bytes
        changes[owner.stage][finish + 1] -= owner.bytes
    peaks = []
    for events in changes:
        live = peak = 0
        for delta in events:
            live += delta
            peak = max(peak, live)
        peaks.append(peak)
    return tuple(peaks)


def _token_peaks(plan, start, end, tokens):
    events = [[] for _ in plan.schedules]
    keys = {node_key(n) for seq in plan.schedules for n in seq}
    for seq in plan.schedules:
        for n in seq:
            if n.kind == 'F':
                k = node_key(n)
                b = backward_key(keys, n.stage, n.chunk, n.microbatch, n.split)
                # Match the prototype's end-before-start convention at a tie.
                if end[b] > start[k]:
                    events[n.stage].extend(((start[k], tokens.get(k, 0.0)),
                                            (end[b], -tokens.get(k, 0.0))))
    peaks = []
    for ev in events:
        live = peak = 0.0
        for _, delta in sorted(ev):
            live += delta
            peak = max(peak, live)
        peaks.append(peak)
    return tuple(peaks)


def replay_plan(plan, compute_costs, message_costs, owners=(), tokens=None):
    validate_plan(plan)
    nodes, messages, succ, pred = _graph(plan)
    costs = {k: compute_costs[k] for k in nodes}
    costs.update({m: message_costs[m] for m in messages})
    if any(not math.isfinite(c) or c < 0 for c in costs.values()):
        raise ValueError('costs must be finite and nonnegative')
    total, start, end, tail = _replay(succ, pred, costs)
    return dict(makespan_ms=total, start=start, end=end, tail=tail,
                owner_peaks=_owner_peaks(owners, start, end, len(plan.schedules)),
                token_peaks=_token_peaks(plan, start, end, tokens or {}))


def _plan_unsplit(reference, compute_costs, message_costs, *, owners=(),
                  capacities=None, tokens=None, order_based_capacity=False, supply=False, stable_ids=None,
                  feedback=False, token_limits=None, deadline=None, backend='python'):
    """027 construction for A/B; 028 mode 8 for fixed-H C feedback.

    Reference FIFO predicts readiness; final FIFO is rebuilt and replayed.
    A fixes merged F/B and per-chunk W order. B preserves separate F/F and
    B/B order; B/C candidates must satisfy the deployed W accumulation order.
    """
    validate_plan(reference)
    if feedback and not supply:
        raise ValueError('fixed-H feedback requires supply priority')
    tokens = tokens or {}
    nodes, messages, succ, pred = _graph(reference, resource=False,
                                         fixed_fb=not supply, fixed_w=not supply)
    if supply:
        # 027 mode 1 preserves each kind's order, opens their merged interleave.
        for seq in reference.schedules:
            previous = {}
            for n in seq:
                if n.kind in ('F', 'B', 'D'):
                    k = node_key(n)
                    kind = 'B' if n.kind == 'D' else n.kind
                    if kind in previous:
                        a = previous[kind]
                        succ[a].add(k)
                        pred[k].add(a)
                    previous[kind] = k
    for o in owners:
        if (not 0 <= o.stage < len(reference.schedules) or not o.born or not o.users
                or not math.isfinite(o.bytes) or o.bytes < 0
                or any(k not in nodes or nodes[k].stage != o.stage for k in (*o.born, *o.users))):
            raise ValueError('invalid owner contract')
    # The deployed byte contract uses local order. Do not also sort owner
    # events on the predicted millisecond timeline for each complete replay.
    replay_owners = () if order_based_capacity else owners
    ref = replay_plan(reference, compute_costs, message_costs, replay_owners, tokens)
    token_limits = ref['token_peaks'] if token_limits is None else token_limits
    reference_peaks = (order_owner_peaks(reference, owners) if order_based_capacity
                       else ref['owner_peaks'])
    capacities = tuple(capacities) if capacities is not None else tuple(
        max(reference_peaks, default=0.0) * 1.05 for _ in reference.schedules)
    if len(capacities) != len(reference.schedules) or any(not math.isfinite(c) or c < 0 for c in capacities):
        raise ValueError('invalid owner capacities')
    if any(p > c + 1e-8 for p, c in zip(reference_peaks, capacities)):
        raise ValueError('reference exceeds owner capacity')
    costs = {**compute_costs, **message_costs}
    total = ref['makespan_ms']
    priority = {k: total - ref['tail'][k] for k in nodes}
    slack = {k: priority[k] - ref['start'][k] for k in nodes}
    if feedback:
        for k in nodes:
            if k[1] == 'H':
                priority[k] = min((priority[t] for t in succ[k]), default=total) - compute_costs[k]
    outmsg = {k: [] for k in nodes}
    for m, (source, target) in messages.items():
        outmsg[source].append((m, target))
    known_end, frontier, drift = {}, [0.0] * len(reference.schedules), [0.0] * len(reference.schedules)

    def estimate_end(k):
        if k in known_end:
            return known_end[k]
        mapped = relays.get(k, k)
        stage = nodes[mapped].stage if mapped in nodes else nodes[messages[mapped][0]].stage
        return max(now, ref['end'][mapped] + drift[stage])

    # Mode 1 keeps inactive H as zero-duration relays during construction.
    # Forecasting an unresolved H uses reference B end + rank drift, even
    # while that B is running with a known actual end. Collapsing H into B
    # here changes the supply priority (though it preserves DAG readiness).
    relays = {}
    if supply:
        for sg,ss,tg,ts in reference.history_dependencies:
            for stage in range(len(reference.schedules)):
                for chunk in range(2):
                    b = (stage,'B',chunk,sg,ss)
                    h = (stage,'H',chunk,sg,ss)
                    target = (stage,'B',chunk,tg,ts)
                    if feedback and b not in nodes:
                        continue  # Selected D/H already has its real dependency edges.
                    if target not in nodes:
                        target = (stage,'D',chunk,tg,ts)
                    if h not in relays:
                        relays[h] = b
                        succ[h], pred[h] = set(), {b}
                        succ[b].add(h)
                    succ[b].discard(target)
                    pred[target].discard(b)
                    succ[h].add(target)
                    pred[target].add(h)
        # F/F and B/B edges are separate constraints in 027; restore any
        # B/B predecessor also present among the history edges above.
        for seq in reference.schedules:
            backward = [node_key(n) for n in seq if n.kind in ('B', 'D')]
            for a,b in zip(backward,backward[1:]):
                succ[a].add(b)
                pred[b].add(a)
    ids = stable_ids or {k: i for i, k in enumerate(succ)}
    if set(ids) != set(succ) or len(set(ids.values())) != len(ids):
        raise ValueError('invalid stable input node IDs')
    if backend not in ('python', 'cpp'):
        raise ValueError('unknown DSPP planner backend')
    if backend == 'cpp':
        from .dspp_native import construct
        orders, reason = construct(nodes, messages, succ, relays, ids, costs,
            priority, slack, ref['end'], owners, capacities, tokens, token_limits,
            supply=supply, feedback=feedback, deadline=deadline)
    else:
        degree = {k: len(v) for k, v in pred.items()}
        pp = len(reference.schedules)
        ready, orders = [[] for _ in range(pp)], [[] for _ in range(pp)]
        busy, live, tlive = [False] * pp, [0.0] * pp, [0.0] * pp
        born, users = {k: [] for k in nodes}, {k: [] for k in nodes}
        remaining, allocated = [], set()
        for i, o in enumerate(owners):
            remaining.append(len(set(o.users)))
            for k in set(o.born):
                born[k].append(i)
            for k in set(o.users):
                users[k].append(i)
        events, done, now = [], 0, 0.0

        def activate(k):
            if k in messages or k in relays:
                known_end[k] = now + (0.0 if k in relays else costs[k])
                heapq.heappush(events, (known_end[k], ids[k], k))
            else:
                ready[nodes[k].stage].append(k)

        for k, d in degree.items():
            if not d:
                activate(k)
        reason = ''
        while done < len(succ):
            if deadline is not None and time.perf_counter() >= deadline:
                reason = 'feedback-deadline'
                break
            for r in range(pp):
                if busy[r]:
                    continue
                candidates = []
                for k in ready[r]:
                    new = set(born[k]) - allocated
                    extra = sum(owners[i].bytes for i in new)
                    if live[r] + extra > capacities[r] + 1e-8:
                        continue
                    if nodes[k].kind == 'F' and tlive[r] + tokens.get(k, 0.0) > token_limits[r] + 1e-8:
                        continue
                    freed = sum(owners[i].bytes for i in users[k] if remaining[i] == 1 and i in allocated)
                    date = priority[k]
                    if supply and nodes[k].kind in ('F', 'B', 'D'):
                        for m, target in outmsg[k]:
                            receiver = nodes[target].stage
                            due = max(now, frontier[receiver],
                                      max((estimate_end(p) for p in pred[target] if p != m), default=0.0))
                            date = min(date, due - costs[m] - costs[k] + slack[k])
                    if freed and live[r] > 0.95 * capacities[r]:
                        date = min(date, now - freed / max(1.0, capacities[r]))
                    candidates.append((date, ids[k], k, new))
                if not candidates:
                    continue
                # The tolerance/tie rule is the same as mode 6.
                best = candidates[0]
                for candidate in candidates[1:]:
                    if candidate[0] < best[0] - 1e-8 or (abs(candidate[0] - best[0]) < 1e-8 and candidate[1] < best[1]):
                        best = candidate
                _, _, k, new = best
                ready[r].remove(k)
                orders[r].append(nodes[k])
                busy[r] = True
                allocated.update(new)
                live[r] += sum(owners[i].bytes for i in new)
                if nodes[k].kind == 'F':
                    tlive[r] += tokens.get(k, 0.0)
                known_end[k] = frontier[r] = now + costs[k]
                heapq.heappush(events, (known_end[k], ids[k], k))
            if not events:
                reason = 'construction-capacity-deadlock'
                break
            now = events[0][0]
            # Drain zero-duration messages too before dispatching another compute.
            while events and events[0][0] <= now + 1e-9:
                _, _, k = heapq.heappop(events)
                done += 1
                if k in nodes:
                    r = nodes[k].stage
                    busy[r] = False
                    if not feedback or nodes[k].kind != 'H':
                        drift[r] = now - ref['end'][k]
                    if nodes[k].kind in ('B', 'D'):
                        f = (r, 'F', nodes[k].chunk, nodes[k].microbatch, nodes[k].split)
                        tlive[r] -= tokens.get(f, 0.0)
                    for i in users[k]:
                        remaining[i] -= 1
                        if remaining[i] == 0 and i in allocated:
                            allocated.remove(i)
                            live[r] -= owners[i].bytes
                for v in sorted(succ[k], key=ids.__getitem__):
                    degree[v] -= 1
                    if not degree[v]:
                        activate(v)
    candidate = None
    if not reason:
        candidate = replace(reference, algorithm='C' if feedback else 'B' if supply else 'A', schedules=tuple(
            tuple(replace(n, slot=i) for i, n in enumerate(seq)) for seq in orders))
        try:
            validate_plan(candidate, reference)
            actual = replay_plan(candidate, compute_costs, message_costs, replay_owners, tokens)
            final_peaks = (order_owner_peaks(candidate, owners) if order_based_capacity
                           else actual['owner_peaks'])
            if any(p > c + 1e-8 for p, c in zip(final_peaks, capacities)):
                reason = 'final-owner-capacity'
            elif any(p > c + 1e-8 for p, c in zip(actual['token_peaks'], token_limits)):
                reason = 'final-forward-token-capacity'
            elif actual['makespan_ms'] >= total - 1e-8:
                reason = 'no-makespan-improvement'
        except ValueError as exc:
            reason = ('per-chunk-W-order' if 'W order' in str(exc) else 'rebuilt-FIFO-cycle')
    if reason:
        return replace(reference, algorithm='C' if feedback else 'B' if supply else 'A', fallback_reason=reason,
                       reference_ms=total, predicted_ms=total)
    return replace(candidate, reference_ms=total, predicted_ms=actual['makespan_ms'])


def plan_configuration_a(reference, compute_costs, message_costs, **kwargs):
    """Fixed merged F/B and per-chunk W order (027 mode 6 plus W/W guard)."""
    return _plan_unsplit(reference, compute_costs, message_costs, **kwargs)


def preserves_retention_envelope(plan, reference):
    """Conservative P1 deployment guard, independent of predicted milliseconds.

    At every merged F/B frontier, all W already consumed by the template must
    also have run. Thus no template-freed W closure is retained across another
    F/B. This is a protocol/storage restriction, not a measured byte contract.
    """
    def frontiers(seq):
        consumed, result = set(), []
        for n in seq:
            if n.kind == 'W':
                consumed.add(node_key(n))
            else:
                result.append(frozenset(consumed))
        return result
    return all(all(old <= new for old, new in zip(frontiers(a), frontiers(b)))
               for a, b in zip(reference.schedules, plan.schedules))
