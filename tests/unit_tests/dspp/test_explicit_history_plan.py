"""P3 contract tests: task identity, history gates, selective paths and release."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets.dspp_batch_plan import (
    DsppMicrobatchMeta, DsppSegment, build_dspp_batch_plan,
)
from megatron.core.kv_cache.dspp_packed_attention import DsppSequenceKVState
from megatron.core.pipeline_parallel.dspp_plan import (
    DsppSchedulePlan, PlanBinding, history_dependencies, node_key,
    receive_orders, split_plan, template_plan, validate_plan,
)
from megatron.core.pipeline_parallel.slice_v import build_slice_v_schedule
from megatron.core.pipeline_parallel.schedules import _pad_dspp_v_split_counts
from .test_schedule_plan import simulate_protocol


def fixture(pp):
    batch = build_dspp_batch_plan([193, 129, 33, 17], chunk_size=64)
    groups = [tuple((l, p) for p in range(len(batch.microbatches))) for l in range(2)]
    mapping = tuple((g, s, *task) for g, group in enumerate(groups) for s, task in enumerate(group))
    meta = {(l, p): m for l in range(2) for p, m in enumerate(batch.microbatches)}
    counts = _pad_dspp_v_split_counts(list(map(len, groups)), pp)
    seqs, _ = build_slice_v_schedule(pp, len(counts), counts)
    ref = template_plan(seqs, counts, mapping, PlanBinding(1, 'input', 'layout'),
                        history_dependencies=history_dependencies(mapping, meta))
    return ref


def move(plan, stage, source, target):
    seq = list(plan.schedules[stage])
    n = seq.pop(source)
    seq.insert(target, n)
    schedules = list(plan.schedules)
    schedules[stage] = tuple(replace(n, slot=i) for i, n in enumerate(seq))
    return replace(plan, schedules=tuple(schedules))


def delayed_plan(ref, across):
    sources = {e[:2] for e in ref.history_dependencies}
    selected = {node_key(n) for seq in ref.schedules for n in seq
                if n.kind == 'B' and (n.microbatch, n.split) in sources}
    # Keep genuine fused history tasks too, not merely history-free B nodes.
    selected.remove(sorted(selected)[0])
    adjacent = split_plan(ref, selected)
    for r, seq in enumerate(adjacent.schedules):
        for i, n in enumerate(seq):
            if n.kind != 'H':
                continue
            for j in range(i+1, len(seq)):
                if seq[j].kind not in across:
                    continue
                plan = move(adjacent, r, i, j)
                try:
                    validate_plan(plan, ref)
                except ValueError:
                    continue
                return adjacent, plan
    raise AssertionError(f'no legal H move across {across}')


@pytest.mark.parametrize('pp', [3, 4, 8])
def test_v2_identity_roundtrip_history_moves_and_finite_protocol(pp):
    ref = fixture(pp)
    for across in ('W', 'FB'):
        adjacent, plan = delayed_plan(ref, across)
        restored = DsppSchedulePlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        assert restored == plan
        assert receive_orders(plan.schedules) == receive_orders(ref.schedules)
        assert any(m['producer'][1] == 'D' for m in plan.messages)
        assert all(m['producer'][1] != 'H' for m in plan.messages)
        assert any(t['weight_source'] and t['weight_source'][-1] == 'D' for t in plan.tasks)
        assert any(t['history_source'] for t in plan.tasks)
        simulate_protocol(plan, 2, 2)
        assert plan.schedules != adjacent.schedules


def test_reject_missing_duplicate_late_h_and_changed_metadata():
    ref = fixture(3)
    adjacent, _ = delayed_plan(ref, 'W')
    r = next(r for r, seq in enumerate(adjacent.schedules) if any(n.kind == 'H' for n in seq))
    i = next(i for i, n in enumerate(adjacent.schedules[r]) if n.kind == 'H')
    for duplicate in (False, True):
        seqs = list(adjacent.schedules)
        seq = list(seqs[r]); h = seq.pop(i)
        if duplicate:
            seq[i:i] = [h, h]
        seqs[r] = tuple(replace(n, slot=j) for j, n in enumerate(seq))
        with pytest.raises(ValueError, match='missing|duplicate|orphan'):
            validate_plan(replace(adjacent, schedules=tuple(seqs)), ref)
    with pytest.raises(ValueError, match='cycle'):
        validate_plan(move(adjacent, r, i, len(adjacent.schedules[r])-1), ref)
    with pytest.raises(ValueError, match='consumer metadata'):
        validate_plan(replace(adjacent, history_dependencies=adjacent.history_dependencies[:-1]), ref)
    with pytest.raises(ValueError, match='real task with history'):
        split_plan(ref, {(0, 'B', 0, 0, 0)})


def meta(task, sequence, segment, count=2):
    return DsppMicrobatchMeta(task, 64, (DsppSegment(sequence, segment, segment*64,
        64 if segment+1<count else 2, (count-1)*64+2, count, 0.),))


def test_independent_multiple_h_identity_multilayer_gates_and_release(monkeypatch):
    metas = (meta(0,0,0), meta(1,0,1), meta(2,1,0), meta(3,1,1), meta(4,2,0,1))
    state = DsppSequenceKVState(split_attention_backward=True, history_dkv_after_dx=True)
    state.configure_task_modes(metas, {1,3})
    assert state.task_after_dx(1) and not state.task_after_dx(4)
    calls = []
    monkeypatch.setattr('megatron.core.kv_cache.fa_split.backward_dkv',
        lambda job, *a, **kw: (torch.tensor(job.task), torch.tensor(job.layer)))
    monkeypatch.setattr(state, '_route_key_value_grads',
        lambda layer, layouts, dk, dv, *a, **kw: calls.append((int(dk), layer)))
    for task in (1,3):
        state.begin_backward_task(task)
        for layer in range(2):
            state._history_jobs.append((SimpleNamespace(task=task, layer=layer), layer, (), ()))
        state.mark_dx_ready(task)
    assert state.pending_history_count == 4
    with pytest.raises(RuntimeError, match='unfinished H'):
        state.check_history_consumers(0)
    with pytest.raises(RuntimeError, match='not completed'):
        state.finish_iteration()
    state.run_history_task(3)
    state.check_history_consumers(2)
    with pytest.raises(RuntimeError, match='unfinished H'):
        state.check_history_consumers(0)
    state.run_history_task(1)
    state.check_history_consumers(0)
    assert calls == [(3,0),(3,1),(1,0),(1,1)]
    assert not state._history_records and not state.pending_history_count
    with pytest.raises(RuntimeError, match='completed stage dx'):
        state.run_history_task(1)
    state.finish_iteration()


def test_all_sources_required_and_failed_h_cannot_retry(monkeypatch):
    state = DsppSequenceKVState(split_attention_backward=True, history_dkv_after_dx=True)
    state.configure_task_modes(tuple(meta(i,0,i,3) for i in range(3)), {1,2})
    state.begin_backward_task(2); state.mark_dx_ready(2)
    with pytest.raises(RuntimeError, match='unfinished H'):
        state.begin_backward_task(1)
    state.run_history_task(2)
    state.begin_backward_task(1)
    state._history_jobs.append((None, 0, (), ()))
    state.mark_dx_ready(1)
    def fail(*a, **kw):
        raise RuntimeError('injected H failure')
    monkeypatch.setattr('megatron.core.kv_cache.fa_split.backward_dkv', fail)
    with pytest.raises(RuntimeError, match='injected'):
        state.run_history_task(1)
    with pytest.raises(RuntimeError, match='completed stage dx'):
        state.run_history_task(1)
    with pytest.raises(RuntimeError, match='unfinished H'):
        state.check_history_consumers(0)
    with pytest.raises(RuntimeError, match='unfinished history'):
        state.clear()
