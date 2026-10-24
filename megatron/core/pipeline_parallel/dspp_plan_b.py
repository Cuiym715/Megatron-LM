"""Algorithm B split search, ported from archived 030/011/010 (read-only).

Candidate order, exposed-path pairing, 32 accepted rounds, soft deadlines and
H prefix/suffix scoring are retained. The adapter binds the deployed Plan/FIFO;
no research-directory import, kernel change, beam, top-k or feedback pass.
"""
from collections import deque
from dataclasses import dataclass, replace
import math
import time

from .dspp_plan import (_graph, _plan_unsplit, node_key, replay_plan,
                        validate_plan)


@dataclass(frozen=True)
class State:
    selected: frozenset
    orders: tuple
    makespan: float


class FixedGraph:
    """003/010 recurrence with arbitrary PP, stable IDs and exact parent ties."""
    def __init__(self, keys, edges, costs, ranks, release, targets):
        self.names = tuple(keys)
        self.idx = {k:i for i,k in enumerate(keys)}
        self.n = len(keys)
        self.edges = tuple((self.idx[a], self.idx[b]) for a,b in edges)
        self.succ = [[] for _ in keys]
        self.indeg = [0] * self.n
        for a,b in self.edges:
            self.succ[a].append(b)
            self.indeg[b] += 1
        self.cost = [costs[k] for k in keys]
        self.meta = [dict(rank=ranks.get(k, -1)) for k in keys]
        self.hs = [self.idx[h] for h in release]
        self.release = {self.idx[h]:self.idx[b] for h,b in release.items()}
        self.targets = {self.idx[h]:[self.idx[t] for t in ts] for h,ts in targets.items()}

    def evaluate(self, orders, detail=False):
        degree = self.indeg.copy()
        following = [-1] * self.n
        for order in orders:
            for a,b in zip(order, order[1:]):
                following[a] = b
                degree[b] += 1
        ready = deque(i for i,d in enumerate(degree) if not d)
        start, end, parent = [0.] * self.n, [0.] * self.n, [-1] * self.n
        topo = []
        while ready:
            a = ready.popleft()
            topo.append(a)
            end[a] = start[a] + self.cost[a]
            successors = self.succ[a] + ([following[a]] if following[a] >= 0 else [])
            for b in successors:
                if end[a] > start[b]:
                    start[b], parent[b] = end[a], a
                degree[b] -= 1
                if not degree[b]:
                    ready.append(b)
        if len(topo) != self.n:
            return None
        tip = max(range(self.n), key=end.__getitem__)
        value = end[tip]
        if not detail:
            return value
        path = []
        while tip != -1:
            path.append(tip)
            tip = parent[tip]
        tail = [0.] * self.n
        for a in reversed(topo):
            suffix = tail[following[a]] if following[a] >= 0 else 0.
            tail[a] = self.cost[a] + max(suffix, max((tail[b] for b in self.succ[a]), default=0.))
        return dict(makespan_ms=value, makespan=value, end=end, tail=tail,
                    critical_path=list(reversed(path)))


class Backend:
    # Same graph API as 010, using the existing Python recurrence, no compiler.
    def evaluate(self, model, orders):
        return model.evaluate(orders)

    def profile(self, model, orders):
        return model.evaluate(orders, detail=True)


def initial(o, backend):
    o.configure(set())
    orders = o.base_orders
    value = backend.evaluate(o.m, orders)
    if value is None:
        raise ValueError('invalid split-search base')
    return State(frozenset(), orders, value)

def score_positions(optional, state, h, predecessors, backend):
    """Exact makespan for a feasible *single* H insertion, before validation.

    Temporarily shorten B and leave the new H as a zero-cost dependency relay.
    Any newly lengthened path must then pass through H.  Its prefix is the
    longest predecessor completion, and its suffix is the longest successor
    path.  Resource neighbors are added to H's data predecessors/successors.
    """
    model = optional.m
    selected = set(state.selected) | {h}
    optional.configure(selected)
    h_cost = model.cost[h]
    model.cost[h] = 0.0
    current = backend.profile(model, state.orders)
    model.cost[h] = h_cost
    if current is None:
        return []
    rank = model.meta[h]['rank']
    order = state.orders[rank]
    lo = order.index(model.release[h]) + 1
    hi = min(order.index(target) for target in model.targets[h])
    data_prefix = max(current['end'][p] for p in predecessors[h])
    data_suffix = max(current['tail'][s] for s in model.succ[h])
    scores = []
    for position in range(lo, hi + 1):
        previous = order[position - 1] if position else None
        following = order[position] if position < len(order) else None
        prefix = max(data_prefix, current['end'][previous] if previous is not None else 0.0)
        suffix = max(data_suffix, current['tail'][following] if following is not None else 0.0)
        scores.append((max(current['makespan'], prefix + h_cost + suffix), position))
    return scores



def predecessors(model):
    out = [[] for _ in range(model.n)]
    for a, b in model.edges:
        out[b].append(a)
    return out


def best_insertion(optional, backend, current, h, preds, counters):
    """One-H best legal placement, including zero-gain/worse moves for diagnosis."""
    model = optional.m
    scores = score_positions(optional, current, h, preds, backend)
    counters['positions_scored'] += len(scores)
    for predicted, position in sorted(scores):
        orders = [list(rank) for rank in current.orders]
        orders[model.meta[h]['rank']].insert(position, h)
        selected = current.selected | {h}
        optional.configure(selected)
        actual = backend.evaluate(model, orders)
        counters['positions_replayed'] += 1
        if actual is None:
            continue
        assert abs(actual - predicted) < 1e-6
        return State(selected, tuple(tuple(rank) for rank in orders), actual), position
    return None, None


def critical_path(optional, state):
    optional.configure(set(state.selected))
    detail = optional.m.evaluate(state.orders, detail=True)
    assert detail and abs(detail['makespan_ms'] - state.makespan) < 1e-6
    return detail['critical_path']



def step_cp(o, backend, current, deadline):
    m = o.m
    path = set(critical_path(o, current))
    eligible = [h for h in m.hs if h not in current.selected
                and o.fused_cost[m.release[h]] > o.split_cost[h]]
    firsts = [h for h in eligible if m.release[h] in path]
    preds = predecessors(m)
    counters = dict(on_path=len(firsts), firsts=len(firsts), singles_tried=0,
                    second_candidates=0, second_trials=0, positions_scored=0,
                    positions_replayed=0)
    singles = []

    def best_single():
        improving = [s for s, h in singles if s.makespan < current.makespan-1e-8]
        return min(improving, key=lambda s:s.makespan) if improving else None

    for h in firsts:
        if time.perf_counter() >= deadline:
            return best_single(), 'deadline', counters
        one, _ = best_insertion(o, backend, current, h, preds, counters)
        counters['singles_tried'] += 1
        if one is not None:
            singles.append((one, h))
    best = best_single()
    if best is not None:
        return best, 'single', counters
    for one, h1 in singles:
        if time.perf_counter() >= deadline:
            return best, 'deadline', counters
        exposed = set(critical_path(o, one))-path
        seconds = [h for h in eligible if h != h1 and m.release[h] in exposed]
        counters['second_candidates'] += len(seconds)
        for h2 in seconds:
            if time.perf_counter() >= deadline:
                return best, 'deadline', counters
            two, _ = best_insertion(o, backend, one, h2, preds, counters)
            counters['second_trials'] += 1
            if two is not None and two.makespan < current.makespan-1e-8 and (
                    best is None or two.makespan < best.makespan):
                best = two
    return best, 'pair' if best is not None else 'stalled', counters


def run_cp(o, backend, budget_ms):
    state = initial(o, backend)
    begin = time.perf_counter()
    deadline = begin+budget_ms/1000
    records = []
    stop = 'step_limit'
    for _ in range(32):
        candidate, kind, counters = step_cp(o, backend, state, deadline)
        records.append(dict(kind=kind, before_ms=state.makespan,
                            after_ms=None if candidate is None else candidate.makespan,
                            **counters))
        if candidate is None:
            stop = kind
            break
        state = candidate
        if time.perf_counter() >= deadline:
            stop = 'deadline'
            break
    return state, dict(stop=stop, decision_ms=1000*(time.perf_counter()-begin),
                       budget_ms=budget_ms, max_steps=32, steps=records,
                       candidate_order='input-node-order', grouping=None, top_k=None)



class OptionalPlan:
    """Bind 010's inactive-H relay graph to the deployed Plan's actual FIFO."""
    def __init__(self, base, compute, messages, split_costs):
        nodes, network, succ, pred = _graph(base, resource=False)
        keys = list(succ)
        edges = {(a,b) for a, targets in succ.items() for b in targets}
        costs = {**compute, **messages}
        ranks = {k:n.stage for k,n in nodes.items()}
        by_source = {}
        for g,s,tg,ts in base.history_dependencies:
            by_source.setdefault((g,s), set()).add((tg,ts))
        release, targets = {}, {}
        for b in nodes:
            if b[1] != 'B' or b not in split_costs:
                continue
            h = (b[0], 'H', *b[2:])
            if b[3:] not in by_source:
                raise ValueError('split cost supplied for history-free task')
            keys.append(h)
            ranks[h] = b[0]
            costs[h] = 0.
            release[h] = b
            targets[h] = tuple((b[0], 'B', b[2], *slot) for slot in sorted(by_source[b[3:]]))
            edges.add((b,h))
            for consumer in targets[h]:
                edges.discard((b,consumer))
                edges.add((h,consumer))
        index = {k:i for i,k in enumerate(keys)}
        self.m = FixedGraph(keys, sorted(edges, key=lambda ab:(index[ab[0]],index[ab[1]])),
                            costs, ranks, release, targets)
        self.fused_cost = self.m.cost.copy()
        self.split_cost = {self.m.idx[h]:split_costs[b][0] for h,b in release.items()}
        self.hcost = {self.m.idx[h]:max(split_costs[b][1],compute[b]-split_costs[b][0])
                      for h,b in release.items()}
        self.base_orders = tuple(tuple(self.m.idx[node_key(n)] for n in seq) for seq in base.schedules)
        self.base, self.nodes = base, nodes

    def configure(self, selected):
        self.m.cost = self.fused_cost.copy()
        for h in selected:
            self.m.cost[self.m.release[h]] = self.split_cost[h]
            self.m.cost[h] = self.hcost[h]

    def materialize(self, state):
        selected_b = {self.m.names[self.m.release[h]] for h in state.selected}
        schedules = []
        for order in state.orders:
            seq = []
            for i in order:
                k = self.m.names[i]
                source = (k[0], 'B', *k[2:]) if k[1] == 'H' else k
                node = self.nodes[source]
                kind = 'H' if k[1] == 'H' else ('D' if k in selected_b else node.kind)
                seq.append(replace(node, kind=kind, slot=len(seq)))
            schedules.append(tuple(seq))
        return replace(self.base, algorithm='B', schedules=tuple(schedules))


def plan_configuration_b(reference, compute, messages, split_costs, *, owners=(),
                         capacities=None, tokens=None, memory=None, budget_fraction=.60,
                         planning_started=None, feedback_reserve_ms=0., backend='python'):
    """027 supply -> 030 cp-all -> full audit, with 031's soft full budget.

    Deployed per-chunk W order is retained as a final execution restriction;
    the research supply priority itself is unchanged. Invalid supply output
    falls back to the same-input unsplit template before the split search.
    """
    begin = time.perf_counter() if planning_started is None else planning_started
    if reference.schema_version != 2:
        raise ValueError('B needs input-bound history metadata (Plan v2)')
    if not math.isfinite(budget_fraction) or not 0 <= budget_fraction <= 1:
        raise ValueError('invalid B planning budget fraction')
    for b,(d,h) in split_costs.items():
        if (b not in compute or b[1] != 'B'
                or any(not math.isfinite(c) or c <= 0 for c in (d,h))):
            raise ValueError('invalid split cost')
    original = replay_plan(reference, compute, messages, tokens=tokens)
    total = original['makespan_ms']
    base = _plan_unsplit(reference, compute, messages, owners=owners,
                         capacities=capacities, tokens=tokens,
                         order_based_capacity=memory is not None, supply=True, backend=backend)
    # A counted byte model audits every accepted base; split trials are audited
    # once at the end, as in 030/031, rather than changing the search algorithm.
    base_reason = base.fallback_reason
    if memory is not None:
        memory.check(base)
    optional = OptionalPlan(base, compute, messages, split_costs)
    prepared = time.perf_counter()
    full_budget = budget_fraction * total
    reserve = .10 * (len(compute) + len(optional.m.hs))
    preparation = (prepared-begin)*1000
    decision_budget = max(0., full_budget-preparation-reserve-feedback_reserve_ms)
    state, decision = run_cp(optional, Backend(), decision_budget)
    searched = time.perf_counter()
    candidate = optional.materialize(state)
    actual_compute = dict(compute)
    for h in state.selected:
        hk = optional.m.names[h]
        bk = optional.m.names[optional.m.release[h]]
        actual_compute[(bk[0], 'D', *bk[2:])] = optional.split_cost[h]
        actual_compute[hk] = optional.hcost[h]
    reject = ''
    validate_plan(candidate, reference)
    actual = replay_plan(candidate, actual_compute, messages, tokens=tokens)
    if abs(actual['makespan_ms']-state.makespan) > 1e-6:
        raise ValueError('split fast score differs from materialized replay')
    try:
        if memory is not None:
            memory.check(candidate)
        if any(a>b+1e-8 for a,b in zip(actual['token_peaks'],original['token_peaks'])):
            reject = 'final-forward-token-capacity'
    except ValueError as exc:
        reject = f'final-capacity: {exc}'
    if reject:
        candidate = base
    del optional
    finish = time.perf_counter()
    final_ms = base.predicted_ms if reject else actual['makespan_ms']
    info = dict(base='supply', backend=backend, base_fallback_reason=base_reason,
        seed_makespan_ms=base.predicted_ms, selected_count=sum(n.kind=='H' for seq in candidate.schedules for n in seq),
        decision=decision, fraction=budget_fraction, full_budget_ms=full_budget,
        preparation_ms=preparation, tail_reserve_ms=reserve,
        feedback_reserve_ms=feedback_reserve_ms,
        allocated_decision_budget_ms=decision_budget,
        tail_actual_ms=(finish-searched)*1000, total_ms=(finish-begin)*1000,
        budget_met=(finish-begin)*1000<=full_budget,
        budget_overrun_ms=max(0., (finish-begin)*1000-full_budget),
        fallback_reason=reject, cost_policy='fused_fw/nonnegative_overhead',
        execution_constraint='per-chunk W accumulation order',
        search_capacity_policy='final audit, otherwise unsplit base')
    return replace(candidate, algorithm='B', reference_ms=total, predicted_ms=final_ms,
                   fallback_reason=reject, planning_info=info)
