#!/usr/bin/env python3
"""Diagnostic-only PP3 W identity, operand, accumulation and Adam tracing.

Runs the real training entry point and fused GEMMs; hooks synchronize/copy CPU
tensors and are unsuitable for timing or memory acceptance. No kernel changes.
Use a fresh DSPP_W_DIAG_OUTPUT. Optional DSPP_W_DIAG_PLAN_ROOT supplies existing
serialized per-iteration plans. DSPP_W_DIAG_FREEZE=1 suppresses the inner update
to isolate scheduling from optimizer feedback. All runs use deterministic FA.
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
from megatron.core.zbpp_utils import WeightGradStore
from megatron.core.kv_cache import dspp_packed_attention as packed
from megatron.model import llama_model

ROOT = Path(os.environ['DSPP_W_DIAG_OUTPUT'])
PLAN_ROOT = os.environ.get('DSPP_W_DIAG_PLAN_ROOT')
FREEZE = os.environ.get('DSPP_W_DIAG_FREEZE', '0') == '1'
ROOT.mkdir(parents=True, exist_ok=True)
active = None
backward_task = None
forward_task = None
pop_task = None


def cpu(tensor):
    return tensor.detach().cpu().clone()


def tensors(value, prefix):
    if value is None:
        return {}
    if isinstance(value, torch.Tensor):
        return {prefix: cpu(value)}
    return {k: t for i, v in enumerate(value) for k, t in tensors(v, f'{prefix}/{i}').items()}


original_fa_backward = packed._wrapped_flash_attn_varlen_backward
assert original_fa_backward is not None


def deterministic_fa_backward(*args, **kwargs):
    if len(args) > 20:
        args = list(args)
        assert isinstance(args[20], bool), 'unexpected FA backward ABI'
        args[20] = True
    else:
        kwargs['deterministic'] = True
    return original_fa_backward(*args, **kwargs)


packed._wrapped_flash_attn_varlen_backward = deterministic_fa_backward
if PLAN_ROOT:
    from megatron.core.pipeline_parallel.dspp_plan import DsppSchedulePlan
    original_dispatch = training.get_forward_backward_func

    def dispatch(*args, **kwargs):
        execute = original_dispatch(*args, **kwargs)
        assert execute is schedules.pipelining_with_dspp_stage_local_v

        def execute_plan(*a, **kw):
            iteration = get_args().curr_iteration + 1
            plan = DsppSchedulePlan.from_dict(json.loads(
                (Path(PLAN_ROOT) / f'iteration{iteration}.json').read_text()))
            return execute(*a, **kw, dspp_plan=plan)

        return execute_plan

    training.get_forward_backward_func = dispatch

original_init = schedules._DsppVTaskBatch.__init__
original_forward = schedules._DsppVTaskBatch.forward
original_backward = schedules._DsppVTaskBatch.backward
original_logits = llama_model.parallel_lm_logits
original_put = WeightGradStore.put
original_flush = WeightGradStore.flush
original_pop = WeightGradStore.pop
original_train = training.train_step


def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self._w_diag_logical = active['batches'] // 2
    active['batches'] += 1


def task(self, split):
    return (self._w_diag_logical, split, self.chunk)


def forward(self, split, tensor):
    global forward_task
    forward_task = str(task(self, split))
    try:
        result = original_forward(self, split, tensor)
        active['outputs'].update(tensors(self.outputs[split], forward_task))
        return result
    finally:
        forward_task = None


def backward(self, split, grad, task_key=None):
    global backward_task
    assert backward_task is None
    backward_task = task(self, split)
    active['backward_order'].append(backward_task)
    try:
        result = original_backward(self, split, grad, task_key=task_key)
        active['dx'].update(tensors(result, str(backward_task)))
        return result
    finally:
        backward_task = None


def logits(*args, **kwargs):
    result = original_logits(*args, **kwargs)
    if forward_task is not None:
        active['logits'][forward_task] = cpu(result)
    return result


def metadata(t):
    return dict(shape=list(t.shape), stride=list(t.stride()), offset=t.storage_offset(),
                storage=t.untyped_storage().data_ptr(), version=t._version,
                dtype=str(t.dtype), stream=torch.cuda.current_stream().cuda_stream)


def put(cls, pre_func, func):
    assert backward_task is not None
    if active['iteration'] != 1:
        return original_put(pre_func, func)
    # This fixture requires TP1/no-SP: its pre_func only reads captured tensors.
    args = pre_func(async_op=False)
    assert len(args) == 3
    x, dy, destination = args
    name = active['gradient_names'][(destination.data_ptr(), destination.numel())]
    identity = f'{backward_task}:{name}'
    assert identity not in active['closures'], identity
    record = dict(task=backward_task, name=name, operands=[cpu(x), cpu(dy)],
                  capture=[metadata(x), metadata(dy)], executed=0)
    active['closures'][identity] = record
    active['pending'].append(identity)

    def checked_func(actual_x, actual_dy, actual_destination):
        assert pop_task is not None and identity in pop_task, (identity, pop_task)
        assert actual_destination.data_ptr() == destination.data_ptr(), identity
        record['executed'] += 1
        assert record['executed'] == 1, identity
        record['execute'] = [metadata(actual_x), metadata(actual_dy)]
        record['unchanged'] = [torch.equal(cpu(t), saved)
                               for t, saved in zip((actual_x, actual_dy), record['operands'])]
        assert all(record['unchanged']), f'W operands mutated: {identity}'
        independent = torch.zeros_like(actual_destination)
        func(actual_x, actual_dy, independent)
        record['contribution'] = cpu(independent)
        active['execution_order'].append(identity)
        # Do not replace the production accumulation with independent sums.
        func(actual_x, actual_dy, actual_destination)
        active['functions'][name] = func

    return original_put(pre_func, checked_func)


def flush(cls, chunk=0, task_key=None):
    if active['iteration'] == 1:
        ids = active['pending']
        active['pending'] = []
        if task_key is None:
            active['fifo'][chunk].append(ids)
        else:
            assert tuple(task_key[1:4]) == backward_task, (task_key, backward_task)
            active['keyed'][(chunk, task_key)] = ids
    return original_flush(chunk=chunk, task_key=task_key)


def pop(cls, chunk=0, pop_num=None, timers=None, progress_tracker=None,
        progress_parent=None, task_key=None):
    global pop_task
    if active['iteration'] == 1:
        assert pop_num in (None, 1)
        pop_task = (active['fifo'][chunk].pop(0) if task_key is None
                    else active['keyed'].pop((chunk, task_key)))
    try:
        return original_pop(chunk=chunk, pop_num=pop_num, timers=timers,
                            progress_tracker=progress_tracker,
                            progress_parent=progress_parent, task_key=task_key)
    finally:
        pop_task = None


def optimizer_snapshot(optimizer, names):
    result = {}
    for model_group, main_group in zip(optimizer.float16_groups,
                                      optimizer.fp32_from_float16_groups):
        for model, main in zip(model_group, main_group):
            name = names[id(model)]
            state = optimizer.optimizer.state.get(main, {})
            result[name] = dict(master=cpu(main), model=cpu(model),
                                grad=None if main.grad is None else cpu(main.grad),
                                state={k: cpu(v) if isinstance(v, torch.Tensor) else v
                                       for k, v in state.items()})
    return result


def train_step(*args, **kwargs):
    global active
    model = kwargs.get('model', args[3] if len(args) > 3 else None)
    optimizer = kwargs.get('optimizer', args[4] if len(args) > 4 else None)
    config = get_args()
    iteration, rank = config.curr_iteration + 1, dist.get_rank()
    path = ROOT / f'iteration{iteration}-rank{rank}.pt'
    assert not path.exists(), 'refusing to overwrite diagnostic data'
    assert config.tensor_model_parallel_size == 1 and not config.sequence_parallel
    names = {id(p): f'{i}:{n}' for i, m in enumerate(model) for n, p in m.named_parameters()}
    gradient_names = {(p.main_grad.data_ptr(), p.main_grad.numel()): names[id(p)]
                      for m in model for p in m.parameters() if p.requires_grad}
    active = dict(iteration=iteration, batches=0, outputs={}, dx={}, logits={},
                  backward_order=[], closures={}, execution_order=[], pending=[],
                  fifo={0: [], 1: []}, keyed={}, functions={}, gradient_names=gradient_names)
    values = dict(initial_optimizer=optimizer_snapshot(optimizer, names))
    original_step, original_inner = optimizer.step, optimizer.optimizer.step
    original_reduce = optimizer.reduce_model_grads

    def reduce_model_grads(*a, **kw):
        if iteration == 1:
            values['before_reduce_gradients'] = {
                names[id(p)]: cpu(p.main_grad)
                for m in model for p in m.parameters() if p.requires_grad}
        return original_reduce(*a, **kw)

    def step(*a, **kw):
        values['raw_gradients'] = {names[id(p)]: cpu(p.main_grad)
                                   for m in model for p in m.parameters() if p.requires_grad}
        return original_step(*a, **kw)

    def inner_step(*a, **kw):
        master_names = {id(main): names[id(model)]
                        for models, mains in zip(optimizer.float16_groups,
                                                 optimizer.fp32_from_float16_groups)
                        for model, main in zip(models, mains)}
        values['optimizer_groups'] = [
            {k: [master_names[id(p)] for p in v] if k == 'params' else v
             for k, v in group.items()}
            for group in optimizer.optimizer.param_groups]
        values['before_inner'] = optimizer_snapshot(optimizer, names)
        if not FREEZE:
            result = original_inner(*a, **kw)
        else:
            result = None
        values['after_inner'] = optimizer_snapshot(optimizer, names)
        return result

    optimizer.step, optimizer.optimizer.step = step, inner_step
    optimizer.reduce_model_grads = reduce_model_grads
    try:
        result = original_train(*args, **kwargs)
    finally:
        optimizer.step, optimizer.optimizer.step = original_step, original_inner
        optimizer.reduce_model_grads = original_reduce
    values.update(final_optimizer=optimizer_snapshot(optimizer, names),
                  outputs=active['outputs'], dx=active['dx'], logits=active['logits'],
                  loss={k: cpu(torch.as_tensor(v)) for k, v in result[0].items()},
                  grad_norm=result[2], closures=active['closures'],
                  backward_order=active['backward_order'], execution_order=active['execution_order'])
    if iteration == 1:
        assert not active['pending'] and not active['keyed'] and not any(active['fifo'].values())
        assert all(r['executed'] == 1 for r in active['closures'].values())
        # Independent original-kernel replay: operands are frozen at B capture.
        # Compare replay orders later across paired files; exclude B-direct grads.
        replay = {}
        for identity in active['execution_order']:
            record = active['closures'][identity]
            name = record['name']
            if name not in replay:
                replay[name] = torch.zeros_like(values['raw_gradients'][name], device='cuda')
            x, dy = [t.cuda() for t in record['operands']]
            active['functions'][name](x, dy, replay[name])
        values['kernel_replay'] = {name: cpu(t) for name, t in replay.items()}
    torch.save(values, path)
    summary = dict(iteration=iteration, rank=rank, freeze=FREEZE,
                   closure_count=len(active['closures']),
                   operands_unchanged=all(all(r['unchanged']) for r in active['closures'].values()),
                   grad_norm=float(result[2]), plan=getattr(config, '_dspp_last_schedule_plan', None))
    path.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'W_DIAG saved {path}', flush=True)
    active = None
    return result


training.train_step = train_step
schedules._DsppVTaskBatch.__init__ = init
schedules._DsppVTaskBatch.forward = forward
schedules._DsppVTaskBatch.backward = backward
llama_model.parallel_lm_logits = logits
WeightGradStore.put = classmethod(put)
WeightGradStore.flush = classmethod(flush)
WeightGradStore.pop = classmethod(pop)
runpy.run_path(str(REPO / 'pretrain_llama.py'), run_name='__main__')
