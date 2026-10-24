from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.datasets.dspp_calibration import (
    StageCalibration, holdout_report, operation, stage_context, timeline_records,
)
from megatron.core.datasets.dspp_execution_cost import DsppExecutionCostModel, meta_from_timeline_record
from megatron.core.datasets.dspp_p2p import DsppP2PCostModel
from megatron.core.pipeline_parallel.dspp_memory import (
    MemoryContract, OwnerLifetime, conservative_peak, storage_owners,
)


def context(chunk=0, stage=0):
    return stage_context(SimpleNamespace(num_layers=12, pipeline_model_parallel_size=3),
                         stage, chunk, hardware={'gpu': 'test', 'kernel': 'v1'})


def record(chunk=0, op='F', capacity=64, history=0, duration=1.):
    return dict(stage=0, chunk=chunk, operation=op, context=context(chunk),
                sequence_chunk_size=capacity, duration_ms=duration,
                items=[dict(sequence_id=0, segment_id=int(history > 0),
                            segment_count=2 if history else 1, token_offset=history,
                            token_length=capacity, sequence_length=history+capacity)])


def test_complete_stage_identity_separates_virtual_stage_and_paths(tmp_path):
    rows = [record(0, duration=1), record(1, duration=3),
            record(op='B_fused', duration=5), record(op='D', duration=4), record(op='H', duration=2)]
    model = StageCalibration(rows)
    meta = meta_from_timeline_record(rows[0])
    for chunk, op, cost in [(0, 'F', 1), (1, 'F', 3), (0, 'B_fused', 5), (0, 'D', 4), (0, 'H', 2)]:
        p = model.predict_detailed(context(chunk), op, meta)
        assert p.duration_ms == cost and p.confidence == 'high'
    changed = context()
    changed['hardware']['kernel'] = 'v2'
    assert model.predict_detailed(changed, 'F', meta).duration_ms is None
    assert model.predict_detailed(context(stage=1), 'F', meta).duration_ms is None
    assert StageCalibration(rows[:1]).predict_detailed(context(), 'H', meta).duration_ms is None
    import json
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(model.to_dict()))
    assert StageCalibration.from_json(path).version == model.version


def test_path_labels_cannot_mix_fused_and_split():
    assert operation('B1', 'after_dx') == 'D'
    assert operation('B0', 'fused') == 'B_fused'
    assert operation('B0', 'selective') == 'B_fused'
    assert operation('D1', 'selective') == 'D'
    assert operation('H1', 'selective') == 'H'
    with pytest.raises(ValueError, match='own calibration'):
        operation('B0', 'split_immediate')
    with pytest.raises(ValueError, match='no versioned'):
        list(timeline_records(dict(events=[])))
    with pytest.raises(ValueError, match='missing H'):
        DsppExecutionCostModel.empty().predict_detailed(0, 'H',
            build_dspp_batch_plan([64], chunk_size=64).microbatches[0])


def test_unmeasured_capacity_and_history_are_low_confidence():
    rows = [record(capacity=c, history=h) for c in (64, 128) for h in (0, 64)]
    model = StageCalibration(rows)
    meta = build_dspp_batch_plan([96], chunk_size=96).microbatches[0]
    p = model.predict_detailed(context(), 'F', meta)
    assert p.duration_ms > 0 and p.extrapolated and p.confidence == 'low'
    assert 'capacity-unmeasured' in p.out_of_range
    long = meta_from_timeline_record(record(history=256))
    assert 'history' in model.predict_detailed(context(), 'F', long).out_of_range
    report = holdout_report(rows)
    folds = [r for r in report['folds'] if r['dimension'] == 'capacity']
    assert len(folds) == 2
    assert all(not p['covered'] for r in folds for p in r['operations']['F'])


def test_dense_w_cost_does_not_depend_on_history_or_packing():
    model = StageCalibration([record(op='W', duration=2)])
    meta = meta_from_timeline_record(record(history=256))
    p = model.predict_detailed(context(), 'W', meta)
    assert p.duration_ms == 2 and not p.extrapolated
    short = build_dspp_batch_plan([17, 11], chunk_size=64).microbatches[0]
    assert model.predict_detailed(context(), 'W', short).duration_ms == 2


def test_runtime_uses_frozen_contexts_without_cuda_queries(tmp_path, monkeypatch):
    import json
    from megatron.core.pipeline_parallel import dspp_plan_runtime as runtime
    from megatron.core.pipeline_parallel.slice_v import build_slice_v_schedule
    from megatron.core.pipeline_parallel.dspp_plan import validate_plan
    plan = build_dspp_batch_plan([193, 129, 33, 17], chunk_size=64)
    counts = [len(plan.microbatches)] * 2
    schedules, _ = build_slice_v_schedule(3, 2, counts)
    contexts = {(stage, chunk): context(chunk, stage) for stage in range(3) for chunk in range(2)}
    rows = []
    for (stage, chunk), identity in contexts.items():
        for meta in plan.microbatches:
            for op in ('F', 'B_fused', 'W'):
                row = record(chunk, op)
                row.update(stage=stage, context=identity, items=[dict(
                    sequence_id=i.sequence_id, segment_id=i.segment_id,
                    segment_count=i.segment_count, token_offset=i.token_offset,
                    token_length=i.token_length, sequence_length=i.sequence_length) for i in meta.items])
                rows.append(row)
    profile = tmp_path/'stage.json'
    profile.write_text(json.dumps(StageCalibration(rows).to_dict()))
    communication = tmp_path/'p2p.json'
    DsppP2PCostModel.from_records([dict(bytes=65536, duration_ms=.1,
        direction=f'{sender}->{receiver}', overlap='compute')
        for sender, receiver in ((0, 1), (1, 0), (1, 2), (2, 1))]).save(communication)
    monkeypatch.setattr(runtime, 'stage_context', lambda *a: pytest.fail('CPU planner queried CUDA'))
    args = SimpleNamespace(hidden_size=512, curr_iteration=0, dspp_schedule='a',
                           dspp_execution_profile=str(profile), dspp_p2p_profile=str(communication))
    reference, candidate = runtime.build_runtime_plan(args, schedules, counts,
        [tuple((logical, i) for i in range(len(plan.microbatches))) for logical in range(2)],
        [SimpleNamespace(plan=plan), SimpleNamespace(plan=plan)],
        torch.bfloat16, (64, 1, 512), 64, calibration_contexts=contexts)
    validate_plan(candidate, reference)
    assert not candidate.fallback_reason


def test_p2p_requires_same_overlap_and_measured_size_range():
    model = DsppP2PCostModel.from_records([dict(bytes=64, duration_ms=1, direction='0->1', overlap='isolated')])
    assert model.predict_detailed(64, direction='0->1', overlap='isolated').confidence == 'high'
    assert model.predict_detailed(64, direction='0->1', overlap='compute').extrapolated
    assert model.predict_detailed(128, direction='0->1', overlap='isolated').extrapolated
    with pytest.raises(ValueError):
        DsppP2PCostModel.from_records([dict(bytes=1, duration_ms=float('nan'))])


def test_storage_aliases_closures_saved_tensors_and_copy_coexistence():
    tensor = torch.ones(32, requires_grad=True)
    view = tensor[8:16]
    def closure():
        return view
    graph = tensor.square()
    owners = storage_owners({'F': graph, 'W': closure})
    allocation = next(row for row in owners.values() if 'W' in row['owners'])
    assert allocation['bytes'] == tensor.untyped_storage().nbytes()
    assert allocation['owners'] == ['F', 'W']
    assert any(v[0] == 8 for v in allocation['views'])
    copy = tensor.detach().clone()
    copied = storage_owners({'old': tensor, 'new': copy})
    assert len(copied) == 2
    assert sum(r['bytes'] for r in copied.values()) == 2*tensor.untyped_storage().nbytes()


def test_storage_audit_does_not_retain_tensors_until_cyclic_gc():
    import gc
    import weakref
    gc.disable()
    try:
        tensor = torch.ones(32)
        reference = weakref.ref(tensor)
        storage_owners({'input': tensor})
        del tensor
        assert reference() is None
    finally:
        gc.enable()


def test_order_based_memory_contract_deduplicates_and_checks_increment():
    owners = [OwnerLifetime('F', 'shared', 100, 0, 2),
              OwnerLifetime('W', 'shared', 100, 1, 3),
              OwnerLifetime('copy', 'copy', 100, 1, 2)]
    assert conservative_peak(owners) == 200
    contract = MemoryContract(physical_bytes=2000, resident_bytes=1000,
        physical_reserve_bytes=100, baseline_dynamic_bytes=200,
        certified=True, identity='layout')
    assert contract.validate(owners, version=contract.version, identity='layout',
                             observed_allocated=1200)['dynamic_bytes'] == 200
    with pytest.raises(ValueError, match='allocator peak'):
        contract.validate(owners, version=contract.version, identity='layout', observed_allocated=1201)
    tighter = replace(contract, baseline_dynamic_bytes=180)
    with pytest.raises(ValueError, match='dynamic memory'):
        tighter.validate(owners, version=tighter.version, identity='layout')
    with pytest.raises(ValueError, match='stale'):
        contract.validate(owners, version='old', identity='layout')
    with pytest.raises(ValueError, match='uncertified'):
        replace(contract, certified=False).validate(owners,
            version=replace(contract, certified=False).version, identity='layout')
