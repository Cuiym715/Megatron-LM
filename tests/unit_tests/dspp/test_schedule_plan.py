"""CPU regression for the deployed schedule contract; no CUDA initialization."""
from dataclasses import replace
import json
import random

import pytest

from megatron.core.pipeline_parallel.dspp_plan import (
    DsppSchedulePlan, PlanBinding, PlanOwner, incoming_sender, message_key,
    node_key, plan_configuration_a, preserves_retention_envelope, receive_orders,
    replay_plan, template_plan, validate_plan,
)
from megatron.core.pipeline_parallel.slice_v import build_slice_v_schedule
from megatron.core.pipeline_parallel.schedules import (
    _pad_dspp_v_split_counts, get_forward_backward_func, pipelining_with_dspp_stage_local_v,
)


def case(pp, counts=(3, 1, 1, 1), seed=7):
    padded = _pad_dspp_v_split_counts(counts, pp)
    seqs, _ = build_slice_v_schedule(pp, len(padded), padded)
    # Nonidentity slot mapping exercises logical/physical identity vs phase slots.
    tasks = [(g, s, g % 2, 100 * g + s) for g, c in enumerate(counts) for s in range(c)]
    plan = template_plan(seqs, padded, tasks, PlanBinding(4, 'input', f'pp{pp}'))
    rng = random.Random(seed)
    real = {(g, s) for g, s, _, _ in tasks}
    compute = {node_key(n): rng.uniform(0.2, 4) if (n.microbatch, n.split) in real else 0.0
               for seq in seqs for n in seq}
    comm = {('message', a, b, *msg): 0.13 for (a, b), lane in receive_orders(seqs).items() for msg in lane}
    tokens = {node_key(n): 64 + 7 * n.split for seq in seqs for n in seq
              if n.kind == 'F' and (n.microbatch, n.split) in real}
    return plan, compute, comm, tokens


def simulate_protocol(plan, lookahead, depth):
    """Finite receive credits/queued payloads, with cooperative host progress.

    NCCL/stream progress is not simulated: payload completion is an atomic
    rendezvous. This detects logical wait cycles but cannot certify GPU liveness.
    """
    pp = len(plan.schedules)
    lanes = receive_orders(plan.schedules)
    cursors = {lane: 0 for lane in lanes}
    posted = {lane: set() for lane in lanes}
    delivered = {lane: set() for lane in lanes}
    queued = {lane: set() for lane in lanes}
    positions = [0] * pp
    max_buffers = dict.fromkeys(lanes, 0)
    while any(i < len(seq) for i, seq in zip(positions, plan.schedules)):
        changed = False
        for (a, b), keys in lanes.items():
            cursor = cursors[(a, b)]
            # ready markers gate optional preposting; the demanded recv can
            # post immediately when lookahead==0, matching the executor.
            for j in range(cursor, min(len(keys), cursor + lookahead)):
                if keys[j] not in queued[(a, b)]:
                    break
                posted[(a, b)].add(j)
            i = positions[b]
            if i < len(plan.schedules[b]):
                n = plan.schedules[b][i]
                if incoming_sender(b, n, pp) == a and (lookahead == 0 or message_key(n) in queued[(a, b)]):
                    posted[(a, b)].add(cursor)
            for j in sorted(posted[(a, b)]):
                if j in delivered[(a, b)]:
                    continue
                if keys[j] not in queued[(a, b)] or len(delivered[(a, b)]) >= depth:
                    break
                delivered[(a, b)].add(j)
                changed = True
            max_buffers[(a, b)] = max(max_buffers[(a, b)], len(posted[(a, b)]))
        for r, seq in enumerate(plan.schedules):
            i = positions[r]
            if i == len(seq):
                continue
            n = seq[i]
            peer = incoming_sender(r, n, pp)
            if peer is not None:
                lane = (peer, r)
                cursor = cursors[lane]
                assert lanes[lane][cursor] == message_key(n)
                if cursor not in delivered[lane]:
                    continue
                posted[lane].remove(cursor)
                delivered[lane].remove(cursor)
                cursors[lane] += 1
            # Publish the output to the opposite neighbor, or the same edge
            # direction at the V turn. Local dependencies are in validate_plan.
            if n.kind in 'FBD':
                kind = message_key(n)[0]
                target = r + 1 if (kind, n.chunk) in {('F', 0), ('B', 1)} else r - 1
                if 0 <= target < pp:
                    queued[(r, target)].add(message_key(n))
            positions[r] += 1
            changed = True
        if not changed:
            raise ValueError(f'finite protocol deadlock at {positions}')
    assert all(cursors[lane] == len(keys) for lane, keys in lanes.items())
    assert all(count <= max(1, lookahead) for count in max_buffers.values())


@pytest.mark.parametrize('pp', [3, 4, 8])
@pytest.mark.parametrize('counts', [(1, 1), (3, 2, 1, 1), (5, 4, 3, 2)])
def test_template_roundtrip_and_configuration_a(pp, counts):
    original, compute, comm, tokens = case(pp, counts)
    restored = DsppSchedulePlan.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert restored.plan_hash == original.plan_hash
    assert restored.schedules == original.schedules
    optimized = plan_configuration_a(original, compute, comm, tokens=tokens)
    validate_plan(optimized, original)
    assert optimized.predicted_ms <= optimized.reference_ms + 1e-8
    assert receive_orders(original.schedules) == receive_orders(optimized.schedules)
    assert len([n for seq in optimized.schedules for n in seq if n.kind == 'W']) == 2 * pp * sum(original.split_counts)
    for seq in optimized.schedules:
        for n in seq:
            assert optimized.task_key(n) == original.task_key(n)
        for chunk in range(2):
            assert [node_key(n) for n in seq if n.kind == 'W' and n.chunk == chunk] == [
                node_key(n) for n in original.schedules[seq[0].stage]
                if n.kind == 'W' and n.chunk == chunk]
    for lookahead, depth in [(0, 1), (1, 1), (2, 2), (4, 1)]:
        simulate_protocol(optimized, lookahead, depth)


def test_a_actually_moves_w_and_preserves_fb():
    original, compute, comm, tokens = case(4)
    optimized = plan_configuration_a(original, compute, comm, tokens=tokens)
    assert not optimized.fallback_reason
    assert optimized.schedules != original.schedules
    assert optimized.predicted_ms < optimized.reference_ms - 1e-8
    assert preserves_retention_envelope(optimized, original)


@pytest.mark.parametrize('pp', [3, 4, 8])
def test_reject_ready_w_permutation_including_serialized_plan(pp):
    original, _, _, _ = case(pp)
    seq = list(original.schedules[0])
    # Two cooldown W have both had their B; swapping them does not change
    # merged F/B, continuation dependencies, or communication. It must still
    # be rejected because it changes same-parameter accumulation order.
    candidates = [i for i, n in enumerate(seq)
                  if n.kind == 'W' and n.chunk == 1
                  and all(b.slot < i for b in seq if b.kind == 'B' and b.chunk == 1)]
    assert len(candidates) >= 2
    i, j = candidates[-2:]
    seq[i], seq[j] = seq[j], seq[i]
    changed = replace(original, algorithm='A', schedules=(
        tuple(replace(n, slot=k) for k, n in enumerate(seq)), *original.schedules[1:]))
    with pytest.raises(ValueError, match='per-chunk W order'):
        validate_plan(changed, original)
    with pytest.raises(ValueError, match='per-chunk W order'):
        DsppSchedulePlan.from_dict(json.loads(json.dumps(changed.to_dict())))


@pytest.mark.parametrize('pp', [3, 4, 8])
def test_independent_model_chunks_do_not_acquire_global_w_order(pp):
    original, _, _, _ = case(pp)
    seq = list(original.schedules[0])
    i = next(i for i in range(len(seq) - 2, -1, -1)
             if seq[i].kind == seq[i + 1].kind == 'W'
             and seq[i].chunk != seq[i + 1].chunk
             and all(b.slot < i for b in seq if b.kind == 'B'))
    seq[i], seq[i + 1] = seq[i + 1], seq[i]
    changed = replace(original, algorithm='A', schedules=(
        tuple(replace(n, slot=k) for k, n in enumerate(seq)), *original.schedules[1:]))
    validate_plan(changed, original)


@pytest.mark.parametrize('pp', [3, 4, 8])
@pytest.mark.parametrize('seed', range(12))
def test_a_keeps_w_order_with_heterogeneous_costs(pp, seed):
    original, compute, comm, tokens = case(pp, (5, 4, 3, 2), seed=seed)
    optimized = plan_configuration_a(original, compute, comm, tokens=tokens)
    validate_plan(optimized, original)
    for r, seq in enumerate(optimized.schedules):
        for chunk in range(2):
            assert [node_key(n) for n in seq if n.kind == 'W' and n.chunk == chunk] == [
                node_key(n) for n in original.schedules[r] if n.kind == 'W' and n.chunk == chunk]


@pytest.mark.parametrize('fault', ['missing-W', 'duplicate-W', 'wrong-task', 'D', 'early-W', 'stale', 'changed-FB'])
def test_reject_invalid_plan(fault):
    plan, _, _, _ = case(4)
    seq = list(plan.schedules[0])
    w = next(i for i, n in enumerate(seq) if n.kind == 'W')
    if fault == 'missing-W':
        seq.pop(w)
    elif fault == 'duplicate-W':
        seq[w] = seq[next(i for i, n in enumerate(seq) if n.kind == 'W' and i != w)]
    elif fault == 'wrong-task':
        seq[w] = replace(seq[w], split=999)
    elif fault == 'D':
        seq[w] = replace(seq[w], kind='D')
    elif fault == 'early-W':
        seq.insert(0, seq.pop(w))
    elif fault == 'stale':
        plan = replace(plan, binding=replace(plan.binding, iteration_id=55))
    else:
        seq[0], seq[1] = seq[1], seq[0]
    bad = replace(plan, algorithm='A', schedules=(tuple(replace(n, slot=i) for i, n in enumerate(seq)), *plan.schedules[1:]))
    reference, _, _, _ = case(4)
    with pytest.raises(ValueError):
        validate_plan(bad, reference, binding=reference.binding)


def test_owner_capacity_and_release():
    plan, compute, comm, tokens = case(4)
    owners = tuple(PlanOwner(n.stage, 64.0, (node_key(n),),
                            ((n.stage, 'B', n.chunk, n.microbatch, n.split),
                             (n.stage, 'W', n.chunk, n.microbatch, n.split)))
                   for seq in plan.schedules for n in seq if n.kind == 'F')
    ref = replay_plan(plan, compute, comm, owners, tokens)
    cap = [max(ref['owner_peaks']) * 1.05] * 4
    optimized = plan_configuration_a(plan, compute, comm, owners=owners, capacities=cap, tokens=tokens)
    actual = replay_plan(optimized, compute, comm, owners, tokens)
    assert all(p <= c for p, c in zip(actual['owner_peaks'], cap))
    assert all(p <= c for p, c in zip(actual['token_peaks'], ref['token_peaks']))
    with pytest.raises(ValueError, match='reference exceeds'):
        plan_configuration_a(plan, compute, comm, owners=owners, capacities=[0] * 4)


@pytest.mark.parametrize('pp', [3, 4, 8])
def test_dispatch_reuses_stage_local_v(monkeypatch, pp):
    from megatron.core import parallel_state
    monkeypatch.setattr(parallel_state, 'get_pipeline_model_parallel_world_size', lambda: pp)
    monkeypatch.setattr(parallel_state, 'get_virtual_pipeline_model_parallel_world_size', lambda: 2)
    assert get_forward_backward_func(dspp=True) is pipelining_with_dspp_stage_local_v
    monkeypatch.setattr(parallel_state, 'get_virtual_pipeline_model_parallel_world_size', lambda: 3)
    with pytest.raises(NotImplementedError, match='VPP=2'):
        get_forward_backward_func(dspp=True)


def test_serialized_message_contract_is_checked():
    plan, _, _, _ = case(4)
    plan = replace(plan, wire_shape=(64, 1, 512), wire_dtype='torch.bfloat16', wire_bytes=65536)
    payload = json.loads(json.dumps(plan.to_dict()))
    payload['messages'][0]['sequence'] = 123
    with pytest.raises(ValueError, match='message identity'):
        DsppSchedulePlan.from_dict(payload)
    with pytest.raises(ValueError, match='shape/dtype/bytes'):
        validate_plan(replace(plan, wire_bytes=3))


def test_runtime_metadata_binding_and_cost_fallback(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
    from megatron.core.pipeline_parallel import dspp_plan_runtime as runtime
    batch = SimpleNamespace(plan=build_dspp_batch_plan([129, 33], chunk_size=64))
    groups = [tuple((0, p) for p in range(len(batch.plan.microbatches))), ((1, 0),)]
    second = SimpleNamespace(plan=build_dspp_batch_plan([17], chunk_size=64))
    counts = _pad_dspp_v_split_counts([len(g) for g in groups], 4)
    schedules, _ = build_slice_v_schedule(4, len(counts), counts)
    profile = tmp_path / 'cost.json'
    profile.write_text('{}')
    args = SimpleNamespace(dspp_schedule='a', hidden_size=512, curr_iteration=2,
                           dspp_execution_profile=str(profile))
    def build():
        return runtime.build_runtime_plan(args, schedules, counts, groups, [batch, second],
                                          torch.bfloat16, (64, 1, 512), 64)
    reference, fallback = build()
    assert fallback.fallback_reason == 'unreliable-compute-cost'
    assert fallback.schedules == reference.schedules
    assert len(reference.messages) == 4 * 3 * sum(counts)
    assert all(m['bytes'] == 65536 for m in reference.messages)
    assert reference.task_key(schedules[0][0]) == (3, 0, 0, 0, 0)
    args.curr_iteration = 3
    newer, _ = build()
    assert newer.plan_hash != reference.plan_hash
    with pytest.raises(ValueError, match='binding'):
        validate_plan(reference, binding=newer.binding)
    monkeypatch.setattr(runtime, 'load_runtime_calibration', lambda path, version: SimpleNamespace(
        predict_detailed=lambda *a: SimpleNamespace(duration_ms=1.0, extrapolated=False, confidence='high')))
    reference, candidate = build()
    validate_plan(candidate, reference)
    assert preserves_retention_envelope(candidate, reference)
    assert candidate.predicted_ms <= candidate.reference_ms
    # An external ready plan should not rerun the optimizer at execution time.
    monkeypatch.setattr(runtime, 'load_runtime_calibration', lambda *a: pytest.fail('unexpected planning'))
    runtime.build_runtime_plan(args, schedules, counts, groups, [batch, second],
                               torch.bfloat16, (64, 1, 512), 64, optimize=False)


@pytest.mark.parametrize('pp', [3, 4, 8])
@pytest.mark.parametrize('fault', ['', 'layout', 'vpp', 'dropout'])
def test_distributed_argument_checks_preserve_layout_and_model_contract(monkeypatch, pp, fault):
    from megatron.arguments import parse_args, validate_args
    import sys
    monkeypatch.setenv('WORLD_SIZE', str(pp))
    monkeypatch.setenv('RANK', '0')
    monkeypatch.delenv('OMPI_COMM_WORLD_SIZE', raising=False)
    argv = ['pretrain_llama.py', '--pipeline-model-parallel-size', str(pp),
            '--num-layers', str(2 * pp + (1 if fault == 'layout' else 0)),
            '--hidden-size', '512', '--ffn-hidden-size', '1024', '--num-attention-heads', '4',
            '--seq-length', '256', '--max-position-embeddings', '256', '--micro-seq-length', '64',
            '--micro-batch-size', '4', '--global-batch-size', '8', '--train-iters', '3',
            '--make-main-grad-addresss-divisible-by', '1',
            '--bf16', '--use-flash-attn', '--no-position-embedding', '--use-rotary-position-embeddings',
            '--hidden-dropout', '0.1' if fault == 'dropout' else '0', '--attention-dropout', '0',
            '--dspp', '--dspp-v-layer-layout', 'balanced']
    if fault != 'vpp':
        argv += ['--num-layers-per-virtual-pipeline-stage', '1']
    monkeypatch.setattr(sys, 'argv', argv)
    if fault:
        with pytest.raises(AssertionError):
            validate_args(parse_args())
    else:
        args = validate_args(parse_args())
        assert args.pipeline_model_parallel_size == pp
        assert args.virtual_pipeline_model_parallel_size == 2
