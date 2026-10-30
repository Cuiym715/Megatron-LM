"""Bounded shape admission and error-aware plan acceptance, without CUDA."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.datasets.dspp_calibration import (
    StageCalibration, INTERPOLATION_POLICY, LEGACY_INTERPOLATION_POLICY,
    NOISE_RELAXED_INTERPOLATION_POLICY,
)
from megatron.core.datasets.dspp_execution_cost import meta_from_timeline_record
from megatron.core.datasets.dspp_p2p import DsppP2PCostModel
from megatron.core.pipeline_parallel import dspp_plan_runtime as runtime
from megatron.core.pipeline_parallel.dspp_plan import plan_configuration_a, node_key
from megatron.core.pipeline_parallel.dspp_plan_b import plan_configuration_b
from megatron.core.pipeline_parallel.slice_v import build_slice_v_schedule
from megatron.core.pipeline_parallel.schedules import _pad_dspp_v_split_counts
from .test_stage_calibration import record, context
from .test_schedule_plan import case
from .test_algorithm_b import fixture


def sample(q, *, op='F', capacity=64, history=0, duration=None):
    row = record(op=op, capacity=capacity, history=history,
                 duration=duration if duration is not None else 1+q/64)
    row['items'][0].update(token_length=q, sequence_length=history+q)
    return row


def rows(op='F', history=0):
    return [sample(q, op=op, history=history, duration=(1+q/64)*j)
            for q in (16, 32, 48) for j in (.999, 1., 1.001)]


@pytest.mark.parametrize('op', ['F', 'B_fused', 'D', 'H'])
def test_validated_query_interval_and_profile_roundtrip(tmp_path, op):
    history = 64 if op == 'H' else 0
    model = StageCalibration(rows(op, history))
    p = model.predict_detailed(context(), op, meta_from_timeline_record(sample(24, history=history)))
    assert p.source == 'validated-query-interpolation'
    assert p.duration_ms == pytest.approx(1.375)
    assert 0 < p.error_ms < .05 and p.confidence == 'medium' and not p.extrapolated
    exact = model.predict_detailed(context(), op, meta_from_timeline_record(sample(32, history=history)))
    assert exact.source == 'exact-profile'
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(model.to_dict()))
    assert StageCalibration.from_json(path).predict_detailed(
        context(), op, meta_from_timeline_record(sample(24, history=history))) == p
    assert StageCalibration.from_json(path).version == model.version


@pytest.mark.parametrize('mutation', ['sparse', 'two-shapes', 'wide'])
def test_unvalidated_regions_stay_closed(mutation):
    data = rows()
    query = sample(24)
    if mutation == 'sparse':
        data = [r for i,r in enumerate(data) if i % 3 != 2]
    elif mutation == 'two-shapes':
        data = data[:3] + data[6:]
    else:
        data = [sample(q, capacity=1024) for q in (128,512,896) for _ in range(3)]
        query = sample(256, capacity=1024)
    p = StageCalibration(data).predict_detailed(context(), 'F', meta_from_timeline_record(query))
    assert p.extrapolated and p.confidence == 'low'
    assert 'kernel-region-unverified' in p.out_of_range


def test_noisy_linear_region_admitted_but_noise_remains_in_error_allowance():
    data = rows()
    data[0]['duration_ms'] *= 1.5
    query = meta_from_timeline_record(sample(24))
    clean = StageCalibration(rows()).predict_detailed(context(), 'F', query)
    noisy = StageCalibration(data).predict_detailed(context(), 'F', query)
    assert noisy.source == 'validated-query-interpolation' and not noisy.extrapolated
    assert noisy.error_ms > .6 and noisy.error_ms > clean.error_ms
    legacy = StageCalibration(data, interpolation_policy=LEGACY_INTERPOLATION_POLICY)
    assert legacy.predict_detailed(context(), 'F', query).extrapolated


@pytest.mark.parametrize('explicit_policy', [True, False])
def test_frozen_old_profile_keeps_noise_gate_with_both_loaders(tmp_path, explicit_policy):
    data = rows()
    data[0]['duration_ms'] *= 1.5
    old = StageCalibration(data, interpolation_policy=LEGACY_INTERPOLATION_POLICY)
    value = old.to_dict()
    if not explicit_policy:
        value.pop('interpolation_policy')
    path = tmp_path/'old.json'
    path.write_text(json.dumps(value))
    query = meta_from_timeline_record(sample(24))
    for model in (StageCalibration.from_json(path),
                  runtime.load_runtime_calibration(str(path), runtime._file_version(path))):
        assert model.predict_detailed(context(), 'F', query).extrapolated
        assert model.version == old.version
    assert StageCalibration(data).version != old.version


def test_no_extrapolation_history_capacity_packing_or_identity_borrowing():
    model = StageCalibration(rows())
    for row in (sample(8), sample(56), sample(24, history=64), sample(24, capacity=128)):
        assert model.predict_detailed(context(), 'F', meta_from_timeline_record(row)).extrapolated
    packed = build_dspp_batch_plan([12,12], chunk_size=64).microbatches[0]
    assert model.predict_detailed(context(), 'F', packed).extrapolated
    different = deepcopy(context())
    different['hardware']['kernel'] = 'other'
    assert model.predict_detailed(different, 'F', meta_from_timeline_record(sample(24))).duration_ms is None
    assert model.predict_detailed(context(), 'D', meta_from_timeline_record(sample(24))).duration_ms is None


@pytest.mark.parametrize('mutation', ['changed', 'null'])
def test_changed_policy_rejected_by_both_profile_loaders(tmp_path, mutation):
    value = StageCalibration(rows()).to_dict()
    assert value['interpolation_policy'] == INTERPOLATION_POLICY
    if mutation == 'changed':
        value['interpolation_policy']['max_span'] += 1
    else:
        value['interpolation_policy'] = None
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='policy version'):
        StageCalibration.from_json(path)
    with pytest.raises(ValueError, match='policy version'):
        runtime.load_runtime_calibration(str(path), runtime._file_version(path))


@pytest.mark.parametrize('policy', [INTERPOLATION_POLICY, NOISE_RELAXED_INTERPOLATION_POLICY])
@pytest.mark.parametrize('q, searched, noisy', [(24, True, False), (8, False, False), (24, True, True)])
def test_runtime_searches_covered_nonexact_inputs_only(tmp_path, monkeypatch, q, searched, noisy, policy):
    batch = build_dspp_batch_plan([q], chunk_size=64)
    counts = _pad_dspp_v_split_counts([1,1], 3)
    schedules,_ = build_slice_v_schedule(3,len(counts),counts)
    contexts = {(s,c):context(c,s) for s in range(3) for c in range(2)}
    data=[]
    for (s,c),identity in contexts.items():
        measurements = rows('F') + rows('B_fused') + [record(op='W')]
        if noisy:
            measurements[0]['duration_ms'] *= 1.5
        for row in measurements:
            data.append(dict(row, stage=s, chunk=c, context=identity))
    profile=tmp_path/'stage.json'
    profile.write_text(json.dumps(StageCalibration(data, interpolation_policy=policy).to_dict()))
    p2p=tmp_path/'p2p.json'
    DsppP2PCostModel.from_records([dict(bytes=65536,duration_ms=.1,
        direction=f'{a}->{b}',overlap='compute') for a,b in ((0,1),(1,0),(1,2),(2,1))]).save(p2p)
    args=SimpleNamespace(hidden_size=512,curr_iteration=0,dspp_schedule='a',
        dspp_execution_profile=str(profile),dspp_p2p_profile=str(p2p))
    calls=[]
    def solver(reference, *a, **kw):
        calls.append(reference)
        return reference
    monkeypatch.setattr(runtime,'plan_configuration_a',solver)
    _, plan=runtime.build_runtime_plan(args,schedules,counts,
        [((0,0),),((1,0),)],[SimpleNamespace(plan=batch),SimpleNamespace(plan=batch)],
        torch.bfloat16,(64,1,512),64,calibration_contexts=contexts)
    if searched and policy == INTERPOLATION_POLICY:
        assert plan.planning_info['cost_acceptance_policy'] == 'nominal-makespan-v1'
    else:
        assert 'cost_acceptance_policy' not in plan.planning_info
    assert bool(calls) == searched
    assert (plan.fallback_reason == 'unreliable-compute-cost') != searched


def test_nominal_improvement_must_exceed_interpolation_uncertainty():
    ref,compute,comm,tokens=case(4)
    candidate=plan_configuration_a(ref,compute,comm,tokens=tokens)
    assert not candidate.fallback_reason and candidate.predicted_ms < candidate.reference_ms
    key=next(k for k in compute if k[1]=='F' and compute[k]>0)
    accepted=runtime.check_interpolated_improvement(ref,candidate,compute,comm,{key:1e-7})
    assert accepted.schedules == candidate.schedules
    assert accepted.planning_info['cost_uncertainty']['accepted']
    rejected=runtime.check_interpolated_improvement(ref,candidate,compute,comm,{key:1e4})
    assert rejected.schedules == ref.schedules
    assert rejected.fallback_reason == 'interpolation-uncertain-improvement'
    assert runtime.check_interpolated_improvement(ref,candidate,compute,comm,{}) is candidate


def test_only_selected_split_errors_affect_acceptance(monkeypatch):
    ref,compute,comm,split,tokens=fixture(3)
    candidate=plan_configuration_b(ref,compute,comm,split,tokens=tokens,budget_fraction=1.)
    ds=[node_key(n) for seq in candidate.schedules for n in seq if n.kind=='D']
    assert ds
    d=ds[0]; b=(d[0],'B',*d[2:])
    rejected=runtime.check_interpolated_improvement(ref,candidate,compute,comm,{},split,{b:(0.,1e5)})
    assert rejected.schedules == ref.schedules and rejected.planning_info['selected_count']==0
    unselected=next(k for k in split if (k[0],'D',*k[2:]) not in ds)
    assert runtime.check_interpolated_improvement(ref,candidate,compute,comm,{},split,
                                                 {unselected:(1e5,1e5)}) is candidate
    # The H=max(raw_H,B-D) normalization requires the lower D endpoint when
    # bounding H separately, even while D itself uses its upper endpoint.
    normalized = {**split, b:(compute[b]*.4, compute[b]*.1)}
    replay_costs=[]
    def replay(plan, costs, messages):
        replay_costs.append(costs)
        return {'makespan_ms':100. if len(replay_costs)==1 else 50.}
    monkeypatch.setattr(runtime,'replay_plan',replay)
    runtime.check_interpolated_improvement(ref,candidate,compute,comm,{},normalized,
                                           {b:(compute[b]*.1,0.)})
    assert replay_costs[1][d] == pytest.approx(compute[b]*.5)
    assert replay_costs[1][(d[0],'H',*d[2:])] == pytest.approx(compute[b]*.7)


@pytest.mark.parametrize('policy', [LEGACY_INTERPOLATION_POLICY, NOISE_RELAXED_INTERPOLATION_POLICY])
def test_large_residual_admitted_by_v3_and_frozen_old_policy_preserved(tmp_path, policy):
    data = rows()
    for row in data[3:6]: row['duration_ms'] *= 2
    query = meta_from_timeline_record(sample(24))
    current = StageCalibration(data).predict_detailed(context(), 'F', query)
    assert not current.extrapolated and current.error_ms > 1.
    old = StageCalibration(data, interpolation_policy=policy)
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(old.to_dict()))
    for loaded in (StageCalibration.from_json(path),
                   runtime.load_runtime_calibration(str(path), runtime._file_version(path))):
        assert loaded.interpolation_policy == policy and loaded.version == old.version
        assert loaded.predict_detailed(context(), 'F', query).extrapolated


def test_nominal_acceptance_ignores_compute_and_split_error_envelopes(monkeypatch):
    ref,compute,comm,split,tokens=fixture(3)
    candidate=plan_configuration_b(ref,compute,comm,split,tokens=tokens,budget_fraction=1.)
    assert candidate.predicted_ms < candidate.reference_ms
    key=next(iter(compute)); b=next(iter(split))
    def forbidden(*a, **kw):
        raise AssertionError('nominal acceptance must not replay error envelopes')
    monkeypatch.setattr(runtime, 'replay_plan', forbidden)
    accepted=runtime.check_interpolated_improvement(ref,candidate,compute,comm,
        {key:1e6},split,{b:(1e6,1e6)},enforce=False)
    assert accepted.schedules == candidate.schedules
    assert accepted.fallback_reason == candidate.fallback_reason
    assert accepted.planning_info['cost_acceptance_policy'] == 'nominal-makespan-v1'
