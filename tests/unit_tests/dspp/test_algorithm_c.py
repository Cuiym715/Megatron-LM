"""Fixed-H feedback: unchanged split identity, actual costs and safe fallback."""
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from megatron.core.pipeline_parallel.dspp_plan import (
    DsppSchedulePlan, node_key, replay_plan, validate_plan,
)
from megatron.core.pipeline_parallel.dspp_plan_b import plan_configuration_b
from megatron.core.pipeline_parallel.dspp_plan_c import feedback_configuration_c, plan_configuration_c
from .test_algorithm_b import fixture
from .test_schedule_plan import simulate_protocol


def selected(plan):
    return {node_key(n) for seq in plan.schedules for n in seq if n.kind in ('D', 'H')}


@pytest.mark.parametrize('pp', [3, 4, 8])
def test_fixed_split_feedback_and_rebuilt_fifo(pp):
    ref, compute, messages, split, tokens = fixture(pp)
    seed = plan_configuration_b(ref, compute, messages, split, tokens=tokens, budget_fraction=1.)
    assert selected(seed)
    plan = feedback_configuration_c(ref, seed, compute, messages, split, tokens=tokens)
    validate_plan(plan, ref)
    assert DsppSchedulePlan.from_dict(plan.to_dict()) == plan
    assert selected(plan) == selected(seed)
    assert plan.predicted_ms <= seed.predicted_ms + 1e-8
    assert plan.planning_info['feedback']['passes'] == 1
    assert not plan.planning_info['feedback']['withdraw']
    if pp == 4:
        assert plan.planning_info['feedback']['accepted']
        assert plan.schedules != seed.schedules
    simulate_protocol(plan, 2, 2)


def test_deadline_does_not_spend_a_second_full_budget():
    ref, compute, messages, split, tokens = fixture()
    plan = plan_configuration_c(ref, compute, messages, split, tokens=tokens,
                                budget_fraction=0., b_budget_fraction=0.)
    assert plan.planning_info['feedback']['reason'] == 'feedback-budget-unavailable'
    assert plan.planning_info['feedback_reserve_ms'] == 0
    seed = plan_configuration_b(ref, compute, messages, split, tokens=tokens)
    expired = feedback_configuration_c(ref, seed, compute, messages, split,
        tokens=tokens, deadline=time.perf_counter()-1)
    assert expired.schedules == seed.schedules
    assert expired.planning_info['feedback']['reason'] == 'feedback-deadline'


@pytest.mark.parametrize('b_elapsed_ms', [20., 51.])
def test_c_percentage_budget_includes_preparation_and_one_shared_deadline(monkeypatch, b_elapsed_ms):
    from megatron.core.pipeline_parallel import dspp_plan_c as module
    ref, compute, messages, split, tokens = fixture()
    calls = []
    def seed(*args, **kwargs):
        assert kwargs['budget_fraction'] == .30
        assert kwargs['planning_started'] == 10.
        assert kwargs.get('feedback_reserve_ms', 0.) == 0.
        return replace(ref, algorithm='B', reference_ms=100., planning_info=dict(
            full_budget_ms=30., total_ms=b_elapsed_ms, budget_met=b_elapsed_ms <= 30.,
            budget_overrun_ms=max(0., b_elapsed_ms - 30.)))
    def feedback(reference, plan, *args, deadline, **kwargs):
        calls.append(deadline)
        return replace(plan, algorithm='C', planning_info={**plan.planning_info,
            'feedback': dict(accepted=True)})
    monkeypatch.setattr(module, 'plan_configuration_b', seed)
    monkeypatch.setattr(module, 'feedback_configuration_c', feedback)
    monkeypatch.setattr(module.time, 'perf_counter', lambda: 10. + b_elapsed_ms / 1000)
    plan = module.plan_configuration_c(ref, compute, messages, split, tokens=tokens,
                                       planning_started=10.)
    info = plan.planning_info
    assert info['fraction'] == .50 and info['full_budget_ms'] == 50.
    assert info['b_full_budget_ms'] == 30. and info['b_total_ms'] == b_elapsed_ms
    assert info['total_ms'] == pytest.approx(b_elapsed_ms)
    assert info['budget_met'] == (b_elapsed_ms <= 50.)
    assert info['budget_overrun_ms'] == pytest.approx(max(0., b_elapsed_ms - 50.))
    if b_elapsed_ms < 50.:
        assert calls == [10.05]
    else:
        assert not calls
        assert plan.planning_info['feedback']['reason'] == 'feedback-budget-unavailable'


@pytest.mark.parametrize('total,b', [(.5, .6), (.5, float('nan')), (float('inf'), .3)])
def test_invalid_c_budget_fractions(total, b):
    with pytest.raises(ValueError, match='C planning budgets'):
        plan_configuration_c(None, {}, {}, {}, budget_fraction=total, b_budget_fraction=b)


def test_runtime_dispatches_c_budgets_and_versions_both_fractions(monkeypatch):
    from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
    from megatron.core.pipeline_parallel import dspp_plan_runtime as runtime
    ref = fixture()[0]
    batch = build_dspp_batch_plan([145, 33], chunk_size=64)
    groups = [tuple((l, p) for p in range(len(batch.microbatches))) for l in range(2)]
    prediction = SimpleNamespace(duration_ms=1., extrapolated=False, confidence='high',
                                 source='exact-profile', error_ms=0.)
    class Model:
        interpolation_policy = {}
        def predict_detailed(self, *args, **kwargs):
            return prediction
    class Memory:
        version, owners, capacities = 'memory-test', (), [1e30] * 4
        @classmethod
        def load(cls, *args):
            return cls()
        def check(self, plan):
            return {}
    monkeypatch.setattr(runtime, '_file_version', lambda path: 'test-profile')
    monkeypatch.setattr(runtime, 'StageCalibration', Model)
    monkeypatch.setattr(runtime, 'load_runtime_calibration', lambda *args: Model())
    monkeypatch.setattr(runtime, 'load_p2p_cost_model', lambda *args: Model())
    monkeypatch.setattr(runtime, 'CountedMemory', Memory)
    calls = []
    def solve(reference, *args, **kwargs):
        calls.append((kwargs['budget_fraction'], kwargs['b_budget_fraction']))
        return replace(reference, algorithm='C', planning_info=dict(full_budget_ms=50.))
    monkeypatch.setattr(runtime, 'plan_configuration_c', solve)
    args = SimpleNamespace(dspp_schedule='c', hidden_size=512,
        dspp_execution_profile='stage', dspp_p2p_profile='p2p', dspp_memory_profile='memory',
        dspp_history_dkv_after_dx=True, dspp_split_attention_backward=True,
        dspp_b_budget_fraction=.99)
    def build():
        return runtime.build_runtime_plan(args, ref.schedules, ref.split_counts, groups,
            [SimpleNamespace(plan=batch)] * 2, 'torch.bfloat16', (64, 1, 512), 64,
            calibration_contexts={(r, c): {} for r in range(4) for c in range(2)})
    original, plan = build()
    assert calls[-1] == (.50, .30)
    args.dspp_c_b_budget_fraction = .20
    changed_b, _ = build()
    args.dspp_c_budget_fraction = .40
    changed_c, _ = build()
    assert calls == [(.50, .30), (.50, .20), (.40, .20)]
    assert len({p.binding.cost_version for p in (original, changed_b, changed_c)}) == 3


def test_counted_feedback_uses_selected_state_owners():
    ref, compute, messages, split, tokens = fixture()
    seed = plan_configuration_b(ref, compute, messages, split, tokens=tokens)
    calls = []
    class Memory:
        capacities = [1e30]*4
        def check(self, plan):
            calls.append(('check', plan))
        def records(self, plan):
            calls.append(('records', plan))
            assert selected(plan) == selected(seed)
            return plan, (), {}
    plan = feedback_configuration_c(ref, seed, compute, messages, split, tokens=tokens, memory=Memory())
    assert any(kind == 'records' for kind, _ in calls)
    assert selected(plan) == selected(seed)


def test_feedback_capacity_failure_preserves_entire_seed():
    ref, compute, messages, split, tokens = fixture(4)
    seed = plan_configuration_b(ref, compute, messages, split, tokens=tokens)
    class Memory:
        capacities = [1e30]*4
        def check(self, plan):
            if plan.schedules != seed.schedules:
                raise ValueError('injected feedback capacity')
        def records(self, plan):
            return plan, (), {}
    result = feedback_configuration_c(ref, seed, compute, messages, split, tokens=tokens, memory=Memory())
    assert result.schedules == seed.schedules
    assert result.planning_info['feedback']['reason'].startswith('feedback-capacity')


@pytest.mark.parametrize('backend', ['python', 'cpp'])
def test_feedback_after_b_fallback_uses_its_own_acceptance_result(backend):
    ref, compute, messages, split, tokens = fixture(4)
    # A valid unsplit B seed with tail W leaves room for one feedback pass.
    # It models the verified base returned after B's final split audit fails.
    orders = tuple(tuple(replace(n, slot=i) for i, n in enumerate(
        [n for n in seq if n.kind != 'W'] + [n for n in seq if n.kind == 'W']))
        for seq in ref.schedules)
    seed = replace(ref, algorithm='B', schedules=orders)
    validate_plan(seed, ref)
    seed = replace(seed,
        predicted_ms=replay_plan(seed, compute, messages, tokens=tokens)['makespan_ms'],
        reference_ms=replay_plan(ref, compute, messages, tokens=tokens)['makespan_ms'])

    class Memory:
        capacities = [1e30] * 4
        reject_feedback = False

        def check(self, plan):
            if self.reject_feedback and plan.schedules != seed.schedules:
                raise ValueError('injected feedback capacity')

        def records(self, plan):
            return plan, (), {}

    memory = Memory()
    def feedback(plan):
        return feedback_configuration_c(ref, plan, compute, messages, split,
            tokens=tokens, memory=memory, backend=backend)

    clean = feedback(seed)
    assert clean.planning_info['feedback']['accepted']
    assert clean.predicted_ms < seed.predicted_ms
    prior_reason = 'final-capacity: injected split capacity'
    seed = replace(seed, fallback_reason=prior_reason,
                   planning_info={'fallback_reason': prior_reason})
    result = feedback(seed)
    assert result.planning_info['feedback']['accepted']
    assert result.fallback_reason == ''
    assert result.schedules == clean.schedules
    assert result.predicted_ms == clean.predicted_ms
    assert selected(result) == selected(seed) == set()
    validate_plan(result, ref)
    # Keep the earlier B diagnostic, while reporting C's decision separately.
    assert seed.fallback_reason == result.planning_info['fallback_reason'] == prior_reason

    memory.reject_feedback = True
    rejected = feedback(seed)
    assert rejected.schedules == seed.schedules
    assert not rejected.planning_info['feedback']['accepted']
    assert rejected.fallback_reason == 'feedback-capacity: injected feedback capacity'
