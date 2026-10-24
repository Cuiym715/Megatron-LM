"""Independent full-tensor hash diagnostic; never used for throughput timing."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, '/workspace/src/Megatron-LM-kwai')
import torch
import torch.distributed as dist
from megatron import get_args, training
from megatron.core.pipeline_parallel import schedules
from megatron.core.kv_cache import dspp_packed_attention as packed
from megatron.model import llama_model

root = Path(os.environ['DSPP_HASH_OUTPUT'])
root.mkdir(parents=True, exist_ok=True)
original_train = training.train_step
original_init = schedules._DsppVTaskBatch.__init__
original_forward = schedules._DsppVTaskBatch.forward
original_backward = schedules._DsppVTaskBatch.backward
original_logits = llama_model.parallel_lm_logits
original_plan = schedules.build_runtime_plan
original_fa = packed._wrapped_flash_attn_varlen_backward
active = None
current_task = None

def deterministic_fa(*args, **kwargs):
    if len(args) > 20:
        args = list(args)
        assert isinstance(args[20], bool)
        args[20] = True
    else:
        kwargs['deterministic'] = True
    return original_fa(*args, **kwargs)

def fingerprint(value):
    cpu = value.detach().contiguous().view(torch.uint8).cpu()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(memoryview(cpu.numpy())).hexdigest())

def tensor_map(value, prefix):
    if value is None:
        return {}
    if isinstance(value, torch.Tensor):
        return {prefix: fingerprint(value)}
    return {k: v for i, item in enumerate(value)
            for k, v in tensor_map(item, f'{prefix}/{i}').items()}

def parameters(model, gradients=False):
    result = {}
    for i, module in enumerate(model):
        for name, param in module.named_parameters():
            value = param
            if gradients:
                value = getattr(param, 'main_grad', None)
                if value is None:
                    value = param.grad
                assert value is not None, name
            result[f'{i}:{name}'] = fingerprint(value)
    return result

def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self._hash_logical = active['batch_count'] // 2
    active['batch_count'] += 1

def forward(self, split, tensor):
    global current_task
    current_task = f'l{self._hash_logical}:p{split}:c{self.chunk}'
    try:
        result = original_forward(self, split, tensor)
        active['outputs'].update(tensor_map(self.outputs[split], current_task))
        return result
    finally:
        current_task = None

def backward(self, split, grad, *args, **kwargs):
    result = original_backward(self, split, grad, *args, **kwargs)
    key = f'l{self._hash_logical}:p{split}:c{self.chunk}'
    active['dx'].update(tensor_map(result, key))
    return result

def logits(*args, **kwargs):
    result = original_logits(*args, **kwargs)
    if current_task is not None:
        active['logits'][current_task] = fingerprint(result)
    return result

def plan(*args, **kwargs):
    start = time.perf_counter()
    result = original_plan(*args, **kwargs)
    active['build_runtime_plan_seconds'] = time.perf_counter() - start
    return result

def step(*args, **kwargs):
    global active
    model = kwargs.get('model', args[3] if len(args) > 3 else None)
    optimizer = kwargs.get('optimizer', args[4] if len(args) > 4 else None)
    cfg = get_args()
    active = dict(batch_count=0, outputs={}, dx={}, logits={})
    values = dict(params_before=parameters(model))
    original_step = optimizer.step
    def update(*a, **kw):
        values['gradients'] = parameters(model, gradients=True)
        return original_step(*a, **kw)
    optimizer.step = update
    try:
        result = original_train(*args, **kwargs)
    finally:
        optimizer.step = original_step
    values.update(params_after=parameters(model), outputs=active['outputs'],
                  dx=active['dx'], logits=active['logits'],
                  loss={k: fingerprint(torch.as_tensor(v).reshape(-1))
                        for k, v in result[0].items()})
    record = dict(iteration=cfg.curr_iteration + 1, rank=dist.get_rank(),
                  values=values, skipped=int(result[1]), deterministic_fa=True,
                  build_runtime_plan_seconds=active.get('build_runtime_plan_seconds'),
                  plan=getattr(cfg, '_dspp_last_schedule_plan', None))
    assert values['outputs'] and values['gradients']
    if dist.get_rank() == 0:
        assert values['logits'] and values['loss']
    path = root / f'iteration{record["iteration"]}-rank{record["rank"]}.json'
    assert not path.exists()
    path.write_text(json.dumps(record) + '\n')
    active = None
    return result

packed._wrapped_flash_attn_varlen_backward = deterministic_fa
training.train_step = step
schedules._DsppVTaskBatch.__init__ = init
schedules._DsppVTaskBatch.forward = forward
schedules._DsppVTaskBatch.backward = backward
schedules.build_runtime_plan = plan
llama_model.parallel_lm_logits = logits
runpy.run_path('/workspace/src/Megatron-LM-kwai/pretrain_llama.py', run_name='__main__')
