"""CPU numerical check of actual _DsppVTaskBatch B/W binding.

Tiny dense stage only; this does not replace FlashAttention/NCCL/GPU checks.
"""
from collections import deque
from dataclasses import replace
import random

import torch

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.datasets.dspp_training import DsppTrainingBatch
from megatron.core.pipeline_parallel.dspp_plan import (
    PlanBinding, node_key, plan_configuration_a, receive_orders, template_plan,
)
from megatron.core.pipeline_parallel.slice_v import build_slice_v_schedule
from megatron.core.pipeline_parallel.schedules import _DsppVTaskBatch, _pad_dspp_v_split_counts
from megatron.core.zbpp_utils import WeightGradStore


def test_dense_stage_legacy_template_a_over_three_optimizer_steps(monkeypatch):
    monkeypatch.setattr(WeightGradStore, 'cache', [])
    monkeypatch.setattr(WeightGradStore, 'weight_grad_queue', [deque(), deque()])
    monkeypatch.setattr(WeightGradStore, 'keyed_weight_grads', {})
    monkeypatch.setattr(WeightGradStore, 'keyed_task_keys', set())
    monkeypatch.setattr(WeightGradStore, 'split_bw', classmethod(lambda cls: True))
    monkeypatch.setattr('megatron.core.zbpp_utils.parallel_state.get_pipeline_model_parallel_rank', lambda: 0)
    batches = []
    for lengths in ([7, 3, 2], [11, 1, 2]):
        batch = build_dspp_batch_plan(lengths, chunk_size=4)
        batches.append(DsppTrainingBatch(batch, batch.materialize([torch.arange(n + 1) for n in lengths])))
    groups = [tuple((l, p) for p in range(len(b.plan.microbatches))) for l, b in enumerate(batches)]
    counts = _pad_dspp_v_split_counts(list(map(len, groups)), 3)
    schedules, _ = build_slice_v_schedule(3, len(counts), counts)
    mapping = tuple((g, s, *t) for g, group in enumerate(groups) for s, t in enumerate(group))
    plan = template_plan(schedules, counts, mapping, PlanBinding(1, 'input', 'dense'))
    rng = random.Random(9)
    compute = {node_key(n): rng.uniform(.1, 3) for seq in schedules for n in seq}
    comm = {('message', a, b, *m): .1 for (a, b), lane in receive_orders(schedules).items() for m in lane}
    optimized = plan_configuration_a(plan, compute, comm)
    assert optimized.schedules != plan.schedules
    slots = {(g, s): (l, p) for g, s, l, p in mapping}
    initial = [torch.arange(15, dtype=torch.float64).reshape(3, 5) / 19,
               torch.arange(15, dtype=torch.float64).reshape(3, 5) / 23]

    def run(schedule, keyed):
        weights = [w.clone().requires_grad_() for w in initial]
        records = []
        for iteration in range(3):
            grads = [torch.zeros_like(w) for w in weights]
            outputs, dxs = {}, {}
            task_batches = {}
            for l, b in enumerate(batches):
                for chunk in range(2):
                    def forward(model_batch, x, state, c=chunk):
                        return x @ weights[c]
                    def backward(x, y, dy, c=chunk):
                        dx = torch.autograd.grad(y, x, dy)[0]
                        # Real store closure captures this physical task's own
                        # input and upstream gradient after its B graph is freed.
                        captured_x, captured_dy = x.detach(), dy.detach()
                        WeightGradStore.put(
                            lambda async_op=False, x=captured_x, dy=captured_dy: (x, dy),
                            lambda x, dy, c=c: grads[c].add_(x.T @ dy))
                        return dx
                    task_batches[l, chunk] = _DsppVTaskBatch(b, forward, backward, None, 1, False, chunk)
            for n in schedule:
                task = slots.get((n.microbatch, n.split))
                if task is None:
                    continue
                l, p = task
                tb = task_batches[l, n.chunk]
                k = (l, p, n.chunk)
                weight_key = (iteration, *k, 0) if keyed else None
                if n.kind == 'F':
                    physical = batches[l].physical_microbatches[p]
                    x = (physical.tokens.double().T + torch.arange(3).double() / 7).requires_grad_()
                    y = tb.forward(p, x)
                    outputs[k] = y.detach().clone()
                elif n.kind == 'B':
                    dy = torch.full((4, 5), (1 + p + l) / 37, dtype=torch.float64)
                    dxs[k] = tb.backward(p, dy, weight_key).detach().clone()
                else:
                    tb.weight(task_key=weight_key)
            WeightGradStore.assert_empty()
            assert all(not tb.inputs and not tb.outputs for tb in task_batches.values())
            for w, g in zip(weights, grads):
                with torch.no_grad():
                    w.add_(g, alpha=-.001)
            records.append((outputs, dxs, [g.clone() for g in grads], [w.detach().clone() for w in weights]))
        return records
    legacy = run(schedules[0], False)
    for actual in (run(plan.schedules[0], True), run(optimized.schedules[0], True)):
        for ref_step, actual_step in zip(legacy, actual):
            for ref_values, actual_values in zip(ref_step, actual_step):
                if isinstance(ref_values, dict):
                    assert ref_values.keys() == actual_values.keys()
                    pairs = ((ref_values[k], actual_values[k]) for k in ref_values)
                else:
                    pairs = zip(ref_values, actual_values)
                for ref, value in pairs:
                    torch.testing.assert_close(value, ref, atol=0, rtol=0)
