"""Targeted input parity, bounded queue, fallback and spawn isolation checks."""
import time
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core.datasets.dspp_training import build_dspp_training_batch, prepare_dspp_cpu_batch, DsppTrainingBatch
from megatron.core.pipeline_parallel import dspp_prefetch as module


def data(offset=0):
    return dict(text=torch.arange(24).reshape(2, 12) + offset,
                lengths=torch.tensor([12, 3]))


def config(**kwargs):
    values = dict(pipeline_model_parallel_size=3, num_layers=12, hidden_size=32,
        micro_seq_length=8, micro_batch_size=2, seq_length=11, params_dtype=torch.bfloat16,
        dspp_schedule='template', dspp_microbatch_order='warmup-short-steady-long',
        dspp_packing_strategy='baseline', variable_seq_pad_token_id=0,
        global_batch_size=4, data_parallel_size=1, context_parallel_size=1,
        rampup_batch_size=None, rank=0, dspp_prefetch_max_bytes=1 << 20,
        dspp_prefetch_timeout=15., train_iters=3)
    return SimpleNamespace(**{**values, **kwargs})


def test_cpu_metadata_and_materialization_match_original_and_detect_corruption():
    raw = data()
    raw['lengths'][1] = 1  # Filtering, truncation, shifted labels and residual packing.
    raw['text'][0, 3] = 0  # A pad value inside valid data is still a token.
    state = torch.get_rng_state().clone()
    cpu = prepare_dspp_cpu_batch(raw, [7, 8], chunk_size=4, max_sequence_length=8)
    original = build_dspp_training_batch(raw['text'], raw['lengths'], chunk_size=4, max_sequence_length=8)
    actual = cpu.materialize('cpu')
    assert actual.plan == original.plan
    assert cpu.host_bytes == 9 * 8
    assert cpu.sequences[0].untyped_storage().nbytes() == cpu.host_bytes
    for left, right in zip(actual.physical_microbatches, original.physical_microbatches):
        for name in ('tokens', 'labels', 'position_ids', 'loss_mask'):
            assert torch.equal(getattr(left, name), getattr(right, name))
    assert torch.equal(state, torch.get_rng_state())
    module.verify_sources([cpu])
    cpu.sequences[0][0] += 1
    with pytest.raises(ValueError, match='content hash'):
        module.verify_sources([cpu])


def request(iteration=1):
    cfg = config(curr_iteration=iteration-1)
    batch = prepare_dspp_cpu_batch(data(), [0, 1], chunk_size=8, max_sequence_length=11)
    metadata = [DsppTrainingBatch(batch.plan, (), input_identity=batch.input_identity)] * 2
    return cfg, metadata, {(r, c): {} for r in range(3) for c in range(2)}, time.perf_counter()


def test_spawn_worker_has_no_cuda_or_collectives_and_returns_verified_plan():
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    planner = module.CpuPlanner()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == visible_devices
    try:
        job = request()
        reference = module._build(job, False)[0]
        assert not planner.submit(1, reference.binding.input_hash, job)
        deadline = time.perf_counter() + 30
        result = None
        while result is None and time.perf_counter() < deadline:
            result = planner.poll()
            time.sleep(.01)
        assert result is not None and not result['error']
        assert not result['worker_cuda_initialized']
        plan = module.DsppSchedulePlan.from_dict(result['plan'])
        module.validate_plan(plan, reference, binding=reference.binding)
        planner.process.terminate()
        planner.process.join(timeout=3)
        assert planner.submit(2, reference.binding.input_hash, job) == 'worker-unavailable'
        assert planner.poll() is None
    finally:
        planner.close()
    assert not planner.process.is_alive()


def test_missing_failed_or_wrong_input_results_keep_same_input_fallback():
    job = request()
    reference = module._build(job, False)[0]
    packet = module.CpuIteration(1, 0, 1, (), reference.to_dict(), reference.binding.input_hash, {})
    for result in (None,
            dict(iteration=1, input_hash=packet.input_hash, error='worker-crashed'),
            dict(iteration=1, input_hash='wrong-input', error=None),
            dict(iteration=0, input_hash=packet.input_hash, error=None)):
        plan, reason = module.choose_plan(packet, result)
        assert reason and plan.schedules == reference.schedules
        assert plan.binding == reference.binding
    plan, reason = module.choose_plan(packet, dict(iteration=1, input_hash=packet.input_hash,
                                                  error=None, plan=reference.to_dict()))
    assert not reason and plan.plan_hash == reference.plan_hash


class DeadPlanner:
    def __init__(self):
        self.process = SimpleNamespace(is_alive=lambda: False)
    def submit(self, *args):
        return 'worker-unavailable'
    def poll(self):
        return None
    def close(self):
        pass


def fake_replica(monkeypatch):
    from megatron.core.datasets import dspp_calibration
    monkeypatch.setattr(dspp_calibration, 'stage_context', lambda *args: {})
    monkeypatch.setattr(module, 'CpuPlanner', DeadPlanner)
    monkeypatch.setattr(module.dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(module.dist, 'get_world_size', lambda *args: 3)
    monkeypatch.setattr(module.dist, 'new_group', lambda **kwargs: object())
    monkeypatch.setattr(module.dist, 'broadcast_object_list', lambda *args, **kwargs: None)
    def gather(output, value, **kwargs):
        output[:] = [value] * 3
    monkeypatch.setattr(module.dist, 'all_gather_object', gather)


def test_one_future_rampup_and_resume_from_consumed_position(monkeypatch):
    fake_replica(monkeypatch)
    get_batch = SimpleNamespace(dspp_materialize_cpu=lambda b: b.materialize('cpu'))
    # Preserve the executor's minimum of two logical batches during ramp-up.
    cfg = config(global_batch_size=8, rampup_batch_size=[4, 4, 4])
    original = module.DsppIterationPrefetch(cfg, iter([data(i*100) for i in range(10)]), get_batch)
    rng = (torch.get_rng_state(), random.getstate(), np.random.get_state())
    try:
        first, plan = original.acquire_iteration(1, 2, 0)
        original.commit(plan)
        future = original.future.result(timeout=10)
        assert torch.equal(rng[0], torch.get_rng_state())
        assert rng[1] == random.getstate()
        assert np.array_equal(rng[2][1], np.random.get_state()[1])
        assert future.num_microbatches == 4 and future.consumed_samples == 4
        with pytest.raises(RuntimeError, match='one future'):
            original._request(3, 6, 2)
        expected_hash = future.input_hash
    finally:
        original.close()
    resumed = module.DsppIterationPrefetch(cfg, iter([data(i*100) for i in range(2, 10)]), get_batch)
    try:
        batches, plan = resumed.acquire_iteration(2, 4, 4)
        assert plan.binding.input_hash == expected_hash
        assert [b.input_identity['sample_ids'] for b in batches] == [[4, 5], [6, 7], [8, 9], [10, 11]]
        resumed.active['states'].append('retired')
    finally:
        resumed.close()


def test_byte_limit_and_partial_iteration_fail_before_publication(monkeypatch):
    fake_replica(monkeypatch)
    get_batch = SimpleNamespace(dspp_materialize_cpu=lambda b: b)
    for cfg, inputs, reason in ((config(dspp_prefetch_max_bytes=1), [data()], 'max-bytes'),
                                (config(), [data()], 'incomplete DSPP iteration')):
        wrapper = module.DsppIterationPrefetch(cfg, iter(inputs), get_batch)
        try:
            with pytest.raises(RuntimeError, match=reason):
                wrapper.acquire_iteration(1, 2, 0)
        finally:
            wrapper.close()


def test_worker_startup_failure_is_a_same_input_fallback(monkeypatch):
    fake_replica(monkeypatch)
    def fail():
        raise OSError('injected spawn failure')
    monkeypatch.setattr(module, 'CpuPlanner', fail)
    wrapper = module.DsppIterationPrefetch(config(), iter([data(), data(100)]),
                                         SimpleNamespace(dspp_materialize_cpu=lambda b: b))
    try:
        batches, plan = wrapper.acquire_iteration(1, 2, 0)
        assert len(batches) == 2 and 'worker-startup' in plan.fallback_reason
    finally:
        wrapper.close()


@pytest.mark.parametrize('pp', [3, 8])
def test_checked_execution_binds_actual_input_plan_and_current_contract(pp, tmp_path):
    from dataclasses import replace
    cfg, batches, contexts, started = request()
    cfg.pipeline_model_parallel_size = pp
    cfg.num_layers = pp * 4
    reference, _ = module._build((cfg, batches, contexts, started), False)
    _, _, groups = module._metadata_inputs(cfg, batches)
    checked = module.CheckedExecution(reference, reference.plan_hash, None, None,
        module.digest([reference.plan_hash, None, None]))
    shape = (cfg.seq_length, cfg.micro_batch_size, cfg.hidden_size)
    args = (groups, batches, cfg.params_dtype, shape, cfg.micro_seq_length)
    assert module.check_execution_inputs(cfg, checked, reference, *args)
    # A received plan or count report changed after the coordinator checked it.
    with pytest.raises(ValueError, match='checksum'):
        module.check_execution_inputs(cfg, checked, replace(reference, predicted_ms=1.), *args)
    with pytest.raises(ValueError, match='checksum'):
        module.check_execution_inputs(cfg, replace(checked, memory_report={}), reference, *args)
    cfg.curr_iteration += 1
    with pytest.raises(ValueError, match='input metadata'):
        module.check_execution_inputs(cfg, checked, reference, *args)
    cfg.curr_iteration -= 1
    cfg.dspp_recv_lookahead = 7
    assert not module.check_execution_inputs(cfg, checked, reference, *args)
    del cfg.dspp_recv_lookahead
    profile = tmp_path / 'cost.json'
    profile.write_text('{}')
    cfg.dspp_execution_profile = str(profile)
    assert not module.check_execution_inputs(cfg, checked, reference, *args)
    # A stale input is rejected even when a profile changed at the same time.
    cfg.curr_iteration += 1
    with pytest.raises(ValueError, match='input metadata'):
        module.check_execution_inputs(cfg, checked, reference, *args)


def test_fast_commit_has_one_agreement_and_catches_corruption_before_materialization(monkeypatch):
    fake_replica(monkeypatch)
    gathered = []
    def gather(output, value, **kwargs):
        gathered.append(value)
        output[:] = [value] * 3
    monkeypatch.setattr(module.dist, 'all_gather_object', gather)
    materialized = []
    def materialize(batch):
        materialized.append(batch)
        return batch.materialize('cpu'), None
    wrapper = module.DsppIterationPrefetch(config(curr_iteration=0), iter([data(), data(100)]),
        SimpleNamespace(dspp_materialize_cpu=materialize))
    try:
        logical, ordering, counts, plan, plan_hash = wrapper.acquire_execution(
            1, 2, 0, torch.bfloat16, (11, 2, 32), 8)
        assert len(logical) == 2 and len(materialized) == 2
        assert len(gathered) == 1 and gathered[0] == (None, plan_hash)
        assert tuple(counts) == plan.split_counts
        assert wrapper.active['validation_path'] == 'coordinator-graph-local-contract'
        assert wrapper.future is not None  # Still prepares exactly one future input.
    finally:
        wrapper.close()
    materialized.clear()
    gathered.clear()
    wrapper = module.DsppIterationPrefetch(config(curr_iteration=0), iter([data(), data(100)]),
        SimpleNamespace(dspp_materialize_cpu=materialize))
    original = wrapper.acquire_iteration
    def corrupt(*args, **kwargs):
        batches, plan = original(*args, **kwargs)
        batches[0].sequences[0][0] += 1
        return batches, plan
    wrapper.acquire_iteration = corrupt
    try:
        with pytest.raises(RuntimeError, match='content hash'):
            wrapper.acquire_execution(1, 2, 0, torch.bfloat16, (11, 2, 32), 8)
        assert not materialized and len(gathered) == 1
        assert gathered[0][0] and gathered[0][1] is None
    finally:
        wrapper.close()


def test_fast_commit_still_checks_live_gpu_capacity(monkeypatch):
    profile = dict(device_name='test', stages=[dict(physical_bytes=1000,
        resident_bytes=100, reserve_bytes=10)])
    report = dict(total_peak_bytes=[300])
    cuda = module.torch.cuda
    monkeypatch.setattr(cuda, 'get_device_name', lambda: 'test')
    monkeypatch.setattr(cuda, 'get_device_properties', lambda _: SimpleNamespace(total_memory=1000))
    monkeypatch.setattr(cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(cuda, 'memory_allocated', lambda: 100)
    monkeypatch.setattr(cuda, 'memory_reserved', lambda: 100)
    monkeypatch.setattr(cuda, 'mem_get_info', lambda: (500, 1000))
    module.check_live_capacity(profile, report, 0)
    monkeypatch.setattr(cuda, 'mem_get_info', lambda: (100, 1000))
    with pytest.raises(ValueError, match='capacity unavailable'):
        module.check_live_capacity(profile, report, 0)
