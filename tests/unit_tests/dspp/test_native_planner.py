"""Native/Python parity at the deployed graph/owner/FIFO boundary."""
from dataclasses import asdict, replace
import time

import pytest
import torch

from megatron.core.pipeline_parallel.dspp_native import library
from megatron.core.pipeline_parallel.dspp_plan import (
    DsppSchedulePlan, _plan_unsplit, node_key, plan_configuration_a,
    receive_orders, validate_plan,
)
from megatron.core.pipeline_parallel.dspp_plan_b import plan_configuration_b
from megatron.core.pipeline_parallel.dspp_plan_c import feedback_configuration_c
from megatron.core.pipeline_parallel.dspp_memory_counts import CountedMemory
from megatron.core.pipeline_parallel.dspp_plan_runtime import _file_version
from .test_algorithm_b import fixture
from .test_memory_counts import case


@pytest.mark.parametrize('feedback', [False, True])
def test_supply_and_feedback_respect_w_chain_before_final_validation(feedback):
    from .test_schedule_plan import case as schedule_case
    from megatron.core.pipeline_parallel.dspp_plan import _graph
    ref, compute, messages, tokens = schedule_case(3)
    ref = replace(ref, schema_version=2)
    # Zero-cost cooldown W tie on due date. Adversarial tie-break IDs would
    # reverse them if construction relied solely on the final validator.
    compute = {k: 0. if k[1] == 'W' else v for k,v in compute.items()}
    _,_,succ,_ = _graph(ref, resource=False)
    ids = {k: len(succ)-i for i,k in enumerate(succ)}
    results = [_plan_unsplit(ref, compute, messages, tokens=tokens, supply=True,
                            feedback=feedback, stable_ids=ids, backend=backend)
               for backend in ('python', 'cpp')]
    assert results[0] == results[1]
    for plan in results:
        assert plan.fallback_reason != 'per-chunk-W-order'
        validate_plan(plan, ref)


@pytest.fixture(scope='module', autouse=True)
def compile_before_timed_planning():
    library()


@pytest.mark.parametrize('pp', [3, 4, 8])
def test_supply_split_and_fixed_h_feedback_match_python(pp):
    ref, compute, messages, split, tokens = fixture(pp)
    # A generous identical time budget avoids comparing different timeout cuts.
    compute = {k: v * 1000 for k, v in compute.items()}
    messages = {k: v * 1000 for k, v in messages.items()}
    split = {k: tuple(v * 1000 for v in costs) for k, costs in split.items()}
    bases = [_plan_unsplit(ref, compute, messages, tokens=tokens, supply=True, backend=b)
             for b in ('python', 'cpp')]
    assert bases[0] == bases[1]
    plans = [plan_configuration_b(ref, compute, messages, split, tokens=tokens,
                                  budget_fraction=1., backend=b) for b in ('python', 'cpp')]
    assert plans[0].schedules == plans[1].schedules
    assert plans[0].predicted_ms == plans[1].predicted_ms
    assert plans[0].planning_info['selected_count'] > 0
    assert plans[0].planning_info['decision']['steps'] == plans[1].planning_info['decision']['steps']
    feedback = [feedback_configuration_c(ref, plans[0], compute, messages, split,
                                        tokens=tokens, backend=b) for b in ('python', 'cpp')]
    assert feedback[0] == feedback[1]
    for plan in (*plans, *feedback):
        validate_plan(plan, ref)
        assert DsppSchedulePlan.from_dict(plan.to_dict()) == plan
    expired = feedback_configuration_c(ref, plans[0], compute, messages, split,
        tokens=tokens, backend='cpp', deadline=time.perf_counter() - 1)
    assert expired.schedules == plans[0].schedules
    assert expired.planning_info['feedback']['reason'] == 'feedback-deadline'


@pytest.mark.parametrize('pp', [3, 8])
def test_fixed_fb_and_owner_capacity_match_python(pp):
    args, _, _, ref, meta, profile = case(pp)
    memory = CountedMemory(profile, args, ref, meta, torch.bfloat16, 64)
    compute = {node_key(n): .2 + i % 7 for seq in ref.schedules for i, n in enumerate(seq)}
    messages = {('message', a, b, *m): .13
                for (a, b), lane in receive_orders(ref.schedules).items() for m in lane}
    plans = [plan_configuration_a(ref, compute, messages, owners=memory.owners,
        capacities=memory.capacities, order_based_capacity=True, backend=b)
        for b in ('python', 'cpp')]
    assert plans[0] == plans[1]
    # The new export has precisely the original dataclass wire/hash structure.
    expected = asdict(plans[0])
    expected.pop('history_dependencies')
    expected.pop('planning_info')
    expected.update(tasks=plans[0].tasks, messages=plans[0].messages)
    assert expected == plans[0].to_dict()
    report = memory.check(plans[0])
    report['owner_peak_bytes'][0] = -1
    assert memory.check(plans[0])['owner_peak_bytes'][0] >= 0
    # Changed structure must never inherit the successful cached count.
    corrupt = replace(plans[0], schedules=(plans[0].schedules[0][1:], *plans[0].schedules[1:]))
    with pytest.raises(ValueError):
        memory.check(corrupt)


def test_profile_version_detects_same_size_rewrite_and_replacement(tmp_path):
    path = tmp_path / 'profile.json'
    path.write_text('first')
    first = _file_version(path)
    path.write_text('other')
    assert _file_version(path) != first
    fresh = tmp_path / 'new'
    fresh.write_text('third')
    previous = _file_version(path)
    fresh.replace(path)
    assert _file_version(path) != previous
