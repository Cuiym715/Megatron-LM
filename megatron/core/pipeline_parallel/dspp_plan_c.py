"""028 mode 8: one supply reconstruction, fixed B split set, no withdrawal."""
from dataclasses import replace
import math
import time

from .dspp_plan import _plan_unsplit, node_key, replay_plan, validate_plan
from .dspp_plan_b import plan_configuration_b


def feedback_configuration_c(reference, seed, compute, messages, split_costs, *,
                             tokens=None, memory=None, owners=(), capacities=None,
                             deadline=None, backend='python'):
    """Recompute reference dates using actual D/H costs, then construct once."""
    validate_plan(seed, reference)
    actual = {node_key(n): compute[node_key(n)] for seq in seed.schedules for n in seq
              if n.kind not in ('D', 'H')}
    selected = set()
    for seq in seed.schedules:
        for n in seq:
            if n.kind == 'D':
                d = node_key(n)
                b = (d[0], 'B', *d[2:])
                h = (d[0], 'H', *d[2:])
                cd, ch = split_costs[b]
                actual[d], actual[h] = cd, max(ch, compute[b] - cd)
                selected.add(h)
    limits = replay_plan(reference, compute, messages, tokens=tokens)['token_peaks']
    before = replay_plan(seed, actual, messages, tokens=tokens)['makespan_ms']
    if memory is not None:
        memory.check(seed)
        _, owners, _ = memory.records(seed)
        capacities = memory.capacities
    candidate = _plan_unsplit(seed, actual, messages, owners=owners, capacities=capacities,
        tokens=tokens, token_limits=limits, order_based_capacity=memory is not None,
        supply=True, feedback=True, deadline=deadline, backend=backend)
    reason = candidate.fallback_reason
    if {node_key(n) for seq in candidate.schedules for n in seq if n.kind == 'H'} != selected:
        raise ValueError('feedback changed fixed H set')
    validate_plan(candidate, reference)
    if not reason:
        try:
            if memory is not None:
                memory.check(candidate)
        except ValueError as exc:
            reason = f'feedback-capacity: {exc}'
    if reason:
        candidate = seed
    info = dict(accepted=not reason, reason=reason, passes=1, withdraw=False,
                seed_ms=before, candidate_ms=candidate.predicted_ms,
                selected_count=len(selected))
    return replace(candidate, algorithm='C', reference_ms=seed.reference_ms,
                   fallback_reason=reason, planning_info={**seed.planning_info, 'feedback': info})


def plan_configuration_c(reference, compute, messages, split_costs, *, tokens=None,
                         memory=None, owners=(), capacities=None, budget_fraction=.50,
                         b_budget_fraction=.30,
                         planning_started=None, backend='python'):
    """Share one soft deadline: B gets 30% of reference time, all C gets 50%."""
    if (not math.isfinite(budget_fraction) or not math.isfinite(b_budget_fraction)
            or not 0 <= b_budget_fraction <= budget_fraction <= 1):
        raise ValueError('C planning budgets require 0 <= B fraction <= C fraction <= 1')
    begin = time.perf_counter() if planning_started is None else planning_started
    # Both fractions use the original unsplit iteration, including preparation
    # since planning_started. Tune these provisional allocations in P7.
    seed = plan_configuration_b(reference, compute, messages, split_costs, tokens=tokens,
        memory=memory, owners=owners, capacities=capacities, budget_fraction=b_budget_fraction,
        planning_started=begin, backend=backend)
    full_budget = budget_fraction * seed.reference_ms
    deadline = begin + full_budget / 1000
    start = time.perf_counter()
    if start >= deadline:
        candidate = replace(seed, algorithm='C', planning_info={**seed.planning_info,
            'feedback': dict(accepted=False, reason='feedback-budget-unavailable',
                             passes=0, withdraw=False)})
    else:
        candidate = feedback_configuration_c(reference, seed, compute, messages, split_costs,
            tokens=tokens, memory=memory, owners=owners, capacities=capacities, deadline=deadline, backend=backend)
    elapsed = (time.perf_counter() - begin) * 1000
    candidate.planning_info.update(
        b_budget_fraction=b_budget_fraction,
        b_full_budget_ms=seed.planning_info['full_budget_ms'],
        b_total_ms=seed.planning_info['total_ms'],
        b_budget_met=seed.planning_info['budget_met'],
        b_budget_overrun_ms=seed.planning_info['budget_overrun_ms'],
        fraction=budget_fraction, full_budget_ms=full_budget,
        feedback_ms=(time.perf_counter()-start)*1000,
        total_ms=elapsed, budget_met=elapsed <= full_budget,
        budget_overrun_ms=max(0., elapsed-full_budget))
    return candidate
