"""Metadata adapter for synchronous template/A/B/C plan execution.

Input packets and cost/owner versions remain CPU objects. P6 can supply an
already verified plan through the same executor argument; no prefetch here.
"""
from dataclasses import asdict, replace
import math
import time
from pathlib import Path

from megatron.core.datasets.dspp_calibration import (
    StageCalibration, load_runtime_calibration, stage_context,
)
from megatron.core.datasets.dspp_p2p import load_p2p_cost_model
from .dspp_plan import (
    PlanBinding, digest, node_key, receive_orders, template_plan,
    plan_configuration_a, preserves_retention_envelope, history_dependencies,
)
from .dspp_memory_counts import CountedMemory
from .dspp_plan_b import plan_configuration_b
from .dspp_plan_c import plan_configuration_c


def _file_version(path):
    if not path:
        return 'none'
    # Read content even when stat is unchanged: some mounted filesystems
    # have coarse timestamps, including same-size profile rewrites.
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_inputs(args, pp, schedule_groups, batches, dtype, tensor_shape, chunk_size,
                   forward_only=False):
    """Input/configuration binding without constructing or replaying a task graph."""
    mode = getattr(args, 'dspp_schedule', 'template')
    task_map = tuple((g, s, *task) for g, group in enumerate(schedule_groups)
                     for s, task in enumerate(group))
    metadata = {(l, p): meta for l, batch in enumerate(batches)
                for p, meta in enumerate(batch.plan.microbatches)}
    slot_meta = {(g, s): metadata[(l, p)] for g, s, l, p in task_map}
    layout = dict(pp=pp, vpp=2, dtype=str(dtype), shape=list(tensor_shape),
                  chunk_size=chunk_size, model={name: getattr(args, name, None) for name in (
                      'num_layers', 'hidden_size', 'ffn_hidden_size', 'num_attention_heads',
                      'num_query_groups', 'dspp_v_layer_layout', 'kv_channels')})
    execution_version = _file_version(getattr(args, 'dspp_execution_profile', None))
    input_metadata = dict(tasks=task_map, metadata=[(l, p, asdict(m))
                                                  for (l, p), m in sorted(metadata.items())])
    identities = [getattr(batch, 'input_identity', None) for batch in batches]
    if any(identity is not None for identity in identities):
        input_metadata['sources'] = identities
    binding = PlanBinding(
        iteration_id=int(getattr(args, 'curr_iteration', -1)) + 1,
        input_hash=digest(input_metadata),
        layout_version=digest(layout),
        cost_version=digest([execution_version,
                             _file_version(getattr(args, 'dspp_p2p_profile', None))]),
        capacity_version='p1-template-W-retention-envelope-v1',
        protocol_version=digest(dict(protocol='stage-local-v1',
                                     lookahead=getattr(args, 'dspp_recv_lookahead', 2),
                                     depth=getattr(args, 'dspp_send_inflight_depth', 2),
                                     forward_only=forward_only,
                                     split=getattr(args, 'dspp_split_attention_backward', False),
                                     after_dx=getattr(args, 'dspp_history_dkv_after_dx', False))))
    if mode in ('b', 'c'):
        budgets = ([getattr(args, 'dspp_b_budget_fraction', .60)] if mode == 'b' else
                   [getattr(args, 'dspp_c_budget_fraction', .50),
                    getattr(args, 'dspp_c_b_budget_fraction', .30)])
        binding = replace(binding, cost_version=digest([binding.cost_version,
            'supply-cp-all-v1' if mode == 'b' else 'supply-cp-all-fixed-H-feedback-v1',
            *budgets]))
    return binding, task_map, metadata, slot_meta, execution_version


def build_runtime_plan(args, schedules, split_counts, schedule_groups, batches,
                       dtype, tensor_shape, chunk_size, forward_only=False, optimize=True,
                       calibration_contexts=None, explicit_history=False, planning_started=None):
    """Bind same-input fallback before optimizing. Never return a partial plan."""
    planning_begin = time.perf_counter() if planning_started is None else planning_started
    mode = getattr(args, 'dspp_schedule', 'template')
    explicit_history = explicit_history or (mode in ('b', 'c') and not forward_only)
    binding, task_map, metadata, slot_meta, execution_version = runtime_inputs(
        args, len(schedules), schedule_groups, batches, dtype, tensor_shape,
        chunk_size, forward_only)
    if explicit_history and (forward_only or getattr(args, 'dspp_auto_chunk_size', False)
            or not getattr(args, 'dspp_history_dkv_after_dx', False)
            or not getattr(args, 'dspp_split_attention_backward', False)
            or not getattr(args, 'dspp_memory_profile', None)):
        raise ValueError('explicit D/H requires fixed capacity, after-dx and a memory profile')
    reference = template_plan(schedules, split_counts, task_map, binding,
        history_dependencies=history_dependencies(task_map, metadata) if explicit_history else None)
    payload_bytes = chunk_size * args.hidden_size * (2 if str(dtype) in {'torch.bfloat16', 'torch.float16'} else 4)
    reference = replace(reference, wire_shape=(chunk_size, 1, args.hidden_size),
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
    if getattr(args, 'dspp_auto_chunk_size', False):
        raise ValueError('schedule A/B/C requires fixed chunk size')
    execution_path = getattr(args, 'dspp_execution_profile', None)
    if not execution_path:
        raise ValueError('schedule A/B/C requires an explicit execution profile')
    model = load_runtime_calibration(execution_path, execution_version)
    if mode in ('b', 'c') and not isinstance(model, StageCalibration):
        raise ValueError('B/C requires a versioned complete-stage calibration')
    contexts = calibration_contexts or ({(stage, chunk): stage_context(args, stage, chunk)
                for stage in range(len(schedules)) for chunk in range(2)
                } if isinstance(model, StageCalibration) else {})
    compute, tokens, unreliable = {}, {}, False
    for seq in schedules:
        for n in seq:
            key, meta = node_key(n), slot_meta.get((n.microbatch, n.split))
            if meta is None:
                compute[key] = 0.0
                continue
            if isinstance(model, StageCalibration):
                prediction = model.predict_detailed(contexts[n.stage, n.chunk],
                    'B_fused' if n.kind == 'B' else n.kind, meta)
            else:
                prediction = model.predict_detailed(n.stage, n.kind, meta)
            compute[key] = prediction.duration_ms
            if (prediction.duration_ms is None or not math.isfinite(prediction.duration_ms) or prediction.duration_ms <= 0 or prediction.extrapolated
                    or prediction.confidence not in {'high', 'medium'}):
                unreliable = True
            if n.kind == 'F':
                tokens[key] = meta.valid_token_count
    if unreliable:
        return reference, replace(reference, algorithm=mode.upper(), fallback_reason='unreliable-compute-cost')
    p2p = load_p2p_cost_model(getattr(args, 'dspp_p2p_profile', None))
    messages = {}
    # Wire tensors have fixed dense capacity even for communication-only padding.
    for (sender, receiver), lane in receive_orders(schedules).items():
        if isinstance(model, StageCalibration):
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
        split_costs, unavailable = {}, []
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
                       or p.confidence not in {'high', 'medium'} for p in estimates):
                    unavailable.append(node_key(n))
                else:
                    split_costs[node_key(n)] = tuple(p.duration_ms for p in estimates)
        solver = plan_configuration_c if mode == 'c' else plan_configuration_b
        budgets = (dict(budget_fraction=getattr(args, 'dspp_c_budget_fraction', .50),
                        b_budget_fraction=getattr(args, 'dspp_c_b_budget_fraction', .30))
                   if mode == 'c' else
                   dict(budget_fraction=getattr(args, 'dspp_b_budget_fraction', .60)))
        candidate = solver(reference, compute, messages, split_costs,
            tokens=tokens, owners=memory.owners, capacities=memory.capacities, memory=memory,
            **budgets,
            planning_started=planning_begin,
            backend=getattr(args, 'dspp_planner_backend', 'python'))
        args._dspp_memory_plan = memory.check(candidate)
        candidate.planning_info['unavailable_split_costs'] = unavailable
        # Export cost belongs to planning; collective publication is recorded
        # separately by the executor and never treated as hidden work.
        candidate.to_dict()
        elapsed = (time.perf_counter()-planning_begin)*1000
        budget = candidate.planning_info['full_budget_ms']
        candidate.planning_info.update(total_ms=elapsed, budget_met=elapsed<=budget,
                                      budget_overrun_ms=max(0.,elapsed-budget))
        return reference, candidate
    candidate = plan_configuration_a(reference, compute, messages, tokens=tokens,
        owners=memory.owners if memory else (),
        capacities=memory.capacities if memory else None,
        order_based_capacity=memory is not None,
        backend=getattr(args, 'dspp_planner_backend', 'python'))
    if not preserves_retention_envelope(candidate, reference):
        candidate = replace(reference, algorithm='A', fallback_reason='template-W-retention-envelope',
                            reference_ms=candidate.reference_ms, predicted_ms=candidate.reference_ms)
    if memory is not None:
        args._dspp_memory_plan = memory.check(candidate)
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
