"""Metadata adapter for synchronous template/A/B/C plan execution.

Input packets and cost/owner versions remain CPU objects. P6 can supply an
already verified plan through the same executor argument; no prefetch here.
"""
from dataclasses import asdict, replace
import math
import time
from pathlib import Path

from megatron.core.datasets.dspp_calibration import (
    INTERPOLATION_POLICY, StageCalibration, load_runtime_calibration, stage_context,
)
from megatron.core.datasets.dspp_p2p import load_p2p_cost_model
from megatron.core.datasets.dspp_profile_snapshot import profile_snapshot
from .dspp_plan import (
    PlanBinding, digest, node_key, receive_orders, template_plan,
    history_dependencies, replay_plan,
)
from .dspp_memory_counts import CountedMemory
from .dspp_plan_b import plan_configuration_b
from megatron.core.datasets.dspp_theoretical_cost import TheoreticalCostModel


def _file_version(path):
    if not path:
        return 'none'
    # Profiles are fixed for the training process; this is an in-memory lookup.
    return profile_snapshot(path).version


def runtime_inputs(args, pp, schedule_groups, batches, dtype, tensor_shape, chunk_size,
                   forward_only=False):
    """Input/configuration binding without constructing or replaying a task graph."""
    mode = getattr(args, 'dspp_schedule', 'template')
    task_map = tuple((g, s, *task) for g, group in enumerate(schedule_groups)
                     for s, task in enumerate(group))
    metadata = {(l, p): meta for l, batch in enumerate(batches)
                for p, meta in enumerate(batch.plan.microbatches)}
    slot_meta = {(g, s): metadata[(l, p)] for g, s, l, p in task_map}
    execution_version = _file_version(getattr(args, 'dspp_execution_profile', None))
    # Iteration routing uses its ordinal. No input/layout/cost/protocol hashes.
    binding = PlanBinding(iteration_id=int(getattr(args, 'curr_iteration', -1)) + 1,
                          input_hash='', layout_version='', cost_version='',
                          capacity_version='', protocol_version='stage-local-v1')
    return binding, task_map, metadata, slot_meta, execution_version


def check_interpolated_improvement(reference, candidate, compute, messages, errors,
                                  split_costs=None, split_errors=None, *, enforce=True):
    """Require improvement under the empirical interpolation error envelope.

    Legacy profiles enforce their empirical error envelope. Current v3 profiles
    retain the solver's nominal acceptance and only publish interpolation counts.
    """
    if not enforce:
        return replace(candidate, planning_info={**candidate.planning_info,
            'cost_acceptance_policy': 'nominal-makespan-v1',
            'interpolated_compute_costs': len(errors),
            'interpolated_split_costs': len(split_errors or {})})
    if not errors and not split_errors:
        return candidate
    upper = dict(compute)
    used_errors = {k: e for k, e in errors.items() if e > 0}
    for k, e in used_errors.items():
        upper[k] += e
    for seq in candidate.schedules:
        for n in seq:
            if n.kind != 'D':
                continue
            d = node_key(n)
            b, h = (d[0], 'B', *d[2:]), (d[0], 'H', *d[2:])
            cd, ch = split_costs[b]
            ed, eh = (split_errors or {}).get(b, (0., 0.))
            upper[d] = cd + ed
            # H's nonnegative-overhead normalization grows when D shrinks.
            # Bound each node separately: using D's upper endpoint here would
            # underestimate H on paths that do not include its D predecessor.
            upper[h] = max(ch + eh, upper[b] - max(0., cd-ed))
            if ed: used_errors[d] = ed
            if eh: used_errors[h] = eh
    if not used_errors or candidate.schedules == reference.schedules:
        return candidate
    lower = {k: max(0., c-errors.get(k, 0.)) for k, c in compute.items()}
    reference_lower = replay_plan(reference, lower, messages)['makespan_ms']
    candidate_upper = replay_plan(candidate, upper, messages)['makespan_ms']
    accepted = candidate_upper < reference_lower - 1e-8
    info = dict(interpolated_costs=len(used_errors), reference_lower_ms=reference_lower,
                candidate_upper_ms=candidate_upper, accepted=accepted,
                policy='empirical-error-envelope-v1')
    if not accepted:
        rejected_info = {**candidate.planning_info, 'cost_uncertainty': info,
                         'fallback_reason': 'interpolation-uncertain-improvement'}
        if 'selected_count' in rejected_info:
            info['rejected_selected_count'] = rejected_info['selected_count']
            rejected_info['selected_count'] = 0
        if 'feedback' in rejected_info:
            rejected_info['feedback'] = {**rejected_info['feedback'], 'accepted': False,
                                        'reason': 'interpolation-uncertain-improvement'}
        return replace(reference, algorithm=candidate.algorithm,
            reference_ms=candidate.reference_ms, predicted_ms=candidate.reference_ms,
            fallback_reason='interpolation-uncertain-improvement',
            planning_info=rejected_info)
    return replace(candidate, planning_info={**candidate.planning_info, 'cost_uncertainty': info})


def build_runtime_plan(args, schedules, split_counts, schedule_groups, batches,
                       dtype, tensor_shape, chunk_size, forward_only=False, optimize=True,
                       calibration_contexts=None, explicit_history=False, planning_started=None):
    """Bind same-input fallback before optimizing. Never return a partial plan."""
    planning_begin = time.perf_counter() if planning_started is None else planning_started
    mode = getattr(args, 'dspp_schedule', 'template')
    explicit_history = explicit_history or (mode in ('b', 'c') and not forward_only)
    # A CP prefetch template with the same split/profile configuration is the
    # unsplit v2 fallback used by B, suitable for a like-for-like comparison.
    explicit_history = explicit_history or (not forward_only
        and getattr(args, 'context_parallel_size', 1) > 1
        and getattr(args, 'dspp_memory_profile', None)
        and getattr(args, 'dspp_split_attention_backward', False)
        and getattr(args, 'dspp_history_dkv_after_dx', False))
    binding, task_map, metadata, slot_meta, execution_version = runtime_inputs(
        args, len(schedules), schedule_groups, batches, dtype, tensor_shape,
        chunk_size, forward_only)
    if explicit_history and (forward_only or not getattr(args, 'dspp_history_dkv_after_dx', False)
            or not getattr(args, 'dspp_split_attention_backward', False)
            or not getattr(args, 'dspp_memory_profile', None)):
        raise ValueError('explicit D/H requires fixed capacity, after-dx and a memory profile')
    reference = template_plan(schedules, split_counts, task_map, binding,
        history_dependencies=history_dependencies(task_map, metadata) if explicit_history else None)
    local_capacity = chunk_size // getattr(args, 'context_parallel_size', 1)
    payload_bytes = local_capacity * args.hidden_size * (2 if str(dtype) in {'torch.bfloat16', 'torch.float16'} else 4)
    reference = replace(reference, wire_shape=(local_capacity, 1, args.hidden_size),
                        wire_dtype=str(dtype), wire_bytes=payload_bytes)
    memory_path = getattr(args, 'dspp_memory_profile', None)
    memory = (CountedMemory.load(memory_path, args, reference, slot_meta, dtype, chunk_size)
              if memory_path and not forward_only else None)
    args._dspp_counted_memory = memory
    args._dspp_memory_plan = None
    if memory is not None:
        reference = replace(reference, binding=replace(binding, capacity_version=memory.version))
        args._dspp_memory_plan = memory.check(reference)
    if not optimize or mode not in ('a', 'b', 'c') or forward_only:
        return reference, reference
    if mode == 'a' and getattr(args, 'dspp_split_attention_backward', False):
        raise ValueError('P1 A requires fused B, without dKV splitting')
    theoretical = getattr(args, 'dspp_cost_model', 'profile') == 'theoretical'
    execution_path = getattr(args, 'dspp_execution_profile', None)
    if not theoretical and not execution_path:
        raise ValueError('schedule A/B/C requires an explicit execution profile')
    model = TheoreticalCostModel(args, metadata) if theoretical else load_runtime_calibration(execution_path, execution_version)
    stage_model = isinstance(model, (StageCalibration, TheoreticalCostModel))
    if mode in ('b', 'c') and not stage_model:
        raise ValueError('B/C requires a versioned complete-stage calibration')
    enforce_uncertainty = not (theoretical or (isinstance(model, StageCalibration) and
        model.interpolation_policy.get('acceptance') == 'nominal-makespan-v1'))
    contexts = calibration_contexts or ({(stage, chunk): (
                    dict(stage=stage, virtual_chunk=chunk) if theoretical else stage_context(args, stage, chunk))
                for stage in range(len(schedules)) for chunk in range(2)
                } if stage_model else {})
    compute, tokens, errors, unreliable = {}, {}, {}, False
    for seq in schedules:
        for n in seq:
            key, meta = node_key(n), slot_meta.get((n.microbatch, n.split))
            if meta is None:
                compute[key] = 0.0
                continue
            if stage_model:
                prediction = model.predict_detailed(contexts[n.stage, n.chunk],
                    'B_fused' if n.kind == 'B' else n.kind, meta)
            else:
                prediction = model.predict_detailed(n.stage, n.kind, meta)
            compute[key] = prediction.duration_ms
            if prediction.source == 'validated-query-interpolation':
                errors[key] = prediction.error_ms
            if (prediction.duration_ms is None or not math.isfinite(prediction.duration_ms) or prediction.duration_ms <= 0 or prediction.extrapolated
                    or prediction.confidence not in ({'assumed'} if theoretical else {'high', 'medium'})):
                unreliable = True
            if n.kind == 'F':
                tokens[key] = meta.valid_token_count
    if unreliable:
        return reference, replace(reference, algorithm=mode.upper(), fallback_reason='unreliable-compute-cost')
    p2p = None if theoretical else load_p2p_cost_model(getattr(args, 'dspp_p2p_profile', None))
    messages = {}
    # Wire tensors have fixed dense capacity even for communication-only padding.
    for (sender, receiver), lane in receive_orders(schedules).items():
        if theoretical:
            duration = model.p2p_ms(payload_bytes)
        elif stage_model:
            prediction = p2p.predict_detailed(payload_bytes,
                direction=f'{sender}->{receiver}', overlap='compute')
            if prediction.extrapolated:
                return reference, replace(reference, algorithm=mode.upper(), fallback_reason='unreliable-p2p-cost')
            duration = prediction.duration_ms
        else:
            duration, _ = p2p.predict(
                payload_bytes, direction=f'{sender}->{receiver}', overlap='compute',
                fallback_bandwidth_bytes_per_ms=getattr(args, 'dspp_p2p_bandwidth_gbps', 25.0) * 1e6,
                fallback_latency_ms=getattr(args, 'dspp_p2p_latency_us', 20.0) / 1000.0)
        for msg in lane:
            messages[('message', sender, receiver, *msg)] = duration
    if mode in ('b', 'c'):
        split_costs, split_errors, unavailable = {}, {}, []
        history_sources = {edge[:2] for edge in reference.history_dependencies}
        for seq in reference.schedules:
            for n in seq:
                if n.kind != 'B' or (n.microbatch, n.split) not in history_sources:
                    continue
                meta = slot_meta[n.microbatch, n.split]
                estimates = [model.predict_detailed(contexts[n.stage, n.chunk], op, meta)
                             for op in ('D', 'H')]
                if any(p.duration_ms is None or not math.isfinite(p.duration_ms)
                       or p.duration_ms <= 0 or p.extrapolated
                       or p.confidence not in ({'assumed'} if theoretical else {'high', 'medium'}) for p in estimates):
                    unavailable.append(node_key(n))
                else:
                    split_costs[node_key(n)] = tuple(p.duration_ms for p in estimates)
                    uncertainty = tuple(p.error_ms
                        if p.source == 'validated-query-interpolation' else 0. for p in estimates)
                    if any(uncertainty):
                        split_errors[node_key(n)] = uncertainty
        budgets = dict(budget_fraction=getattr(args, 'dspp_b_budget_fraction', .60))
        candidate = plan_configuration_b(reference, compute, messages, split_costs,
            tokens=tokens, owners=memory.owners, capacities=memory.capacities, memory=memory,
            **budgets, slot_meta=slot_meta,
            planning_started=planning_begin,
            backend=getattr(args, 'dspp_planner_backend', 'cpp'))
        candidate = check_interpolated_improvement(reference, candidate, compute, messages,
                                                  errors, split_costs, split_errors,
                                                  enforce=enforce_uncertainty)
        args._dspp_memory_plan = memory.check(candidate)
        candidate.planning_info['unavailable_split_costs'] = unavailable
        if theoretical:
            candidate.planning_info.update(model.assumptions())
        # Export cost belongs to planning; collective publication is recorded
        # separately by the executor and never treated as hidden work.
        elapsed = (time.perf_counter()-planning_begin)*1000
        budget = candidate.planning_info['full_budget_ms']
        candidate.planning_info.update(total_ms=elapsed, budget_met=elapsed<=budget,
                                      budget_overrun_ms=max(0.,elapsed-budget))
        return reference, candidate


def build_distributed_b_plan(args, *plan_args, group, **kwargs):
    """One coordinator chooses deadline-dependent B/C; all ranks commit its plan.

    Reference failures are gathered before publication, so a failing rank never
    leaves peers waiting for a broadcast it cannot reach.
    """
    import torch.distributed as dist
    from .dspp_plan import DsppSchedulePlan
    kwargs.pop('optimize', None)
    planning_started = time.perf_counter()
    error, reference = None, None
    try:
        reference, _ = build_runtime_plan(args, *plan_args, optimize=False, **kwargs)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    checks = [None] * dist.get_world_size(group)
    dist.all_gather_object(checks, (error, reference.plan_hash if reference else None), group=group)
    if any(e for e,h in checks) or len({h for e,h in checks}) != 1:
        raise RuntimeError(f'B reference agreement failed: {checks}')
    result = [None]
    if dist.get_rank(group) == 0:
        try:
            _, candidate = build_runtime_plan(args, *plan_args, optimize=True,
                                              planning_started=planning_started, **kwargs)
            result[0] = (None, candidate.to_dict())
        except Exception as exc:
            result[0] = (f'{type(exc).__name__}: {exc}', None)
    dist.broadcast_object_list(result, src=dist.get_global_rank(group, 0), group=group)
    error, payload = result[0]
    if error:
        raise RuntimeError(f'B coordinator failed: {error}')
    return reference, DsppSchedulePlan.from_dict(payload)
