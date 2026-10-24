#!/usr/bin/env python3
"""Paired P1 training diagnostics: legacy -> template/A, or clean memory run.

CPU snapshot copies intentionally synchronize; numerical mode is not a timing
benchmark. memory mode omits tensor hooks and reports whole-step CUDA peaks.
Uses the existing split-training tolerances, fixed before execution.
"""
import json
import os
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
import torch.distributed as dist
from megatron import get_args
import megatron.training as training
from megatron.core.pipeline_parallel import schedules
from megatron.model import llama_model

OUTPUT = Path(os.environ['DSPP_PLAN_CHECK_OUTPUT'])
SNAPSHOTS = Path(os.environ['DSPP_PLAN_CHECK_SNAPSHOTS'])
MODE = os.environ['DSPP_PLAN_CHECK_MODE']
DETERMINISTIC_FA = os.environ.get('DSPP_PLAN_CHECK_DETERMINISTIC_FA', '0') == '1'
PLAN_ROOT = os.environ.get('DSPP_PLAN_CHECK_PLAN_ROOT')
assert MODE in {'reference', 'compare', 'memory'}
OUTPUT.mkdir(parents=True, exist_ok=True)
SNAPSHOTS.mkdir(parents=True, exist_ok=True)
original_train = training.train_step
original_init = schedules._DsppVTaskBatch.__init__
original_forward = schedules._DsppVTaskBatch.forward
original_backward = schedules._DsppVTaskBatch.backward
original_logits = llama_model.parallel_lm_logits
active = None
current_task = None

if PLAN_ROOT:
    # Exercise the public prebuilt-plan input instead of the runtime builder.
    from megatron.core.pipeline_parallel.dspp_plan import DsppSchedulePlan
    original_dispatch = training.get_forward_backward_func

    def dispatch(*args, **kwargs):
        execute = original_dispatch(*args, **kwargs)
        assert execute is schedules.pipelining_with_dspp_stage_local_v

        def execute_plan(*a, **kw):
            iteration = get_args().curr_iteration + 1
            payload = json.loads((Path(PLAN_ROOT) / f'iteration{iteration}.json').read_text())
            return execute(*a, **kw, dspp_plan=DsppSchedulePlan.from_dict(payload))

        return execute_plan

    training.get_forward_backward_func = dispatch

if DETERMINISTIC_FA:
    # Diagnostic only: select the existing fused FA deterministic reduction.
    # Production attention and kernel sources are untouched. Apply identically
    # to the paired reference/candidate; never use this run as default-path timing.
    from megatron.core.kv_cache import dspp_packed_attention as packed
    original_fa_backward = packed._wrapped_flash_attn_varlen_backward
    assert original_fa_backward is not None, 'requires the baseline FA wrapped API'

    def deterministic_fa_backward(*args, **kwargs):
        if len(args) > 20:
            args = list(args)
            assert isinstance(args[20], bool), 'unexpected FA backward ABI'
            args[20] = True
        else:
            kwargs['deterministic'] = True
        return original_fa_backward(*args, **kwargs)

    packed._wrapped_flash_attn_varlen_backward = deterministic_fa_backward


def snapshot(value):
    return value.detach().float().cpu().clone()


def tensor_map(value, prefix=''):
    if value is None:
        return {}
    if isinstance(value, torch.Tensor):
        return {prefix: snapshot(value)}
    return {k: t for i, v in enumerate(value) for k, t in tensor_map(v, f'{prefix}/{i}').items()}


def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if active is not None:
        self._plan_check_logical = active['batch_count'] // 2
        active['batch_count'] += 1


def forward(self, split, tensor):
    global current_task
    current_task = f'l{self._plan_check_logical}:p{split}:c{self.chunk}'
    try:
        result = original_forward(self, split, tensor)
        active['outputs'].update(tensor_map(self.outputs[split], current_task))
        return result
    finally:
        current_task = None


def backward(self, split, grad, *args, **kwargs):
    result = original_backward(self, split, grad, *args, **kwargs)
    key = f'l{self._plan_check_logical}:p{split}:c{self.chunk}'
    active['dx'].update(tensor_map(result, key))
    return result


def logits(*args, **kwargs):
    result = original_logits(*args, **kwargs)
    if active is not None and current_task is not None:
        active['logits'][current_task] = snapshot(result)
    return result


def compare(actual, reference, exact=False, category=''):
    assert actual.keys() == reference.keys(), 'task/tensor set mismatch'
    maximum = 0.0
    for name, a in actual.items():
        b = reference[name]
        assert torch.isfinite(a).all(), name
        torch.testing.assert_close(a, b, atol=0 if exact else .004, rtol=0 if exact else .01,
                                   msg=lambda m: f'{category}/{name}: {m}')
        norm = b.double().norm().item()
        diff = a.double() - b.double()
        if norm > 1e-8 and not exact:
            assert diff.norm().item() / norm <= .02, name
        maximum = max(maximum, diff.abs().max().item())
    return dict(tensors=len(actual), max_abs=maximum)


def parameters(model):
    return {f'{i}:{n}': snapshot(p) for i, m in enumerate(model) for n, p in m.named_parameters()}


def train_step(*args, **kwargs):
    global active
    model = kwargs.get('model', args[3] if len(args) > 3 else None)
    optimizer = kwargs.get('optimizer', args[4] if len(args) > 4 else None)
    config = get_args()
    iteration, rank = config.curr_iteration + 1, dist.get_rank()
    path = SNAPSHOTS / f'iteration{iteration}-rank{rank}.pt'
    record_path = OUTPUT / f'iteration{iteration}-rank{rank}.json'
    assert not record_path.exists(), 'refusing to overwrite validation report'
    if MODE == 'memory':
        torch.cuda.synchronize()
        allocated_start = torch.cuda.memory_allocated()
        reserved_start = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        result = original_train(*args, **kwargs)
        torch.cuda.synchronize()
        record = dict(allocated_start=allocated_start, reserved_start=reserved_start,
                      allocated_peak=torch.cuda.max_memory_allocated(),
                      reserved_peak=torch.cuda.max_memory_reserved(),
                      allocated_end=torch.cuda.memory_allocated(),
                      reserved_end=torch.cuda.memory_reserved())
    else:
        active = dict(batch_count=0, outputs={}, dx={}, logits={})
        reference = torch.load(path, map_location='cpu', weights_only=True) if MODE == 'compare' else None
        values = dict(params_before=parameters(model))
        original_step = optimizer.step
        def step(*a, **kw):
            values['gradients'] = {}
            for i, m in enumerate(model):
                for n, p in m.named_parameters():
                    if p.requires_grad:
                        grad = getattr(p, 'main_grad', None)
                        if grad is None:
                            grad = p.grad
                        assert grad is not None, (i, n)
                        values['gradients'][f'{i}:{n}'] = snapshot(grad)
            return original_step(*a, **kw)
        optimizer.step = step
        try:
            result = original_train(*args, **kwargs)
        finally:
            optimizer.step = original_step
        values.update(params_after=parameters(model), outputs=active['outputs'],
                      dx=active['dx'], logits=active['logits'],
                      loss={k: torch.as_tensor(v).float().cpu() for k, v in result[0].items()})
        assert values['outputs'] and values['gradients']
        if rank == 0:
            assert values['logits'] and values['loss']
        if MODE == 'reference':
            assert not path.exists(), 'refusing to overwrite reference snapshot'
            torch.save(values, path)
            record = dict(reference=True)
        else:
            record = {k: compare(v, reference[k], exact=(k == 'params_before' and iteration == 1), category=k)
                      for k, v in values.items()}
        active = None
    plan = getattr(config, '_dspp_last_schedule_plan', None)
    record.update(rank=rank, iteration=iteration, mode=MODE, schedule=config.dspp_schedule,
                  diagnostic_deterministic_fa=DETERMINISTIC_FA,
                  plan_input='external' if PLAN_ROOT else 'runtime',
                  plan=plan, metrics=getattr(config, '_dspp_last_iteration_metrics', {}),
                  criteria=dict(atol=.004, rtol=.01, relative_l2_max=.02))
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    return result


training.train_step = train_step
if MODE != 'memory':
    schedules._DsppVTaskBatch.__init__ = init
    schedules._DsppVTaskBatch.forward = forward
    schedules._DsppVTaskBatch.backward = backward
    llama_model.parallel_lm_logits = logits
runpy.run_path(str(REPO / 'pretrain_llama.py'), run_name='__main__')
