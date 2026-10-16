#!/usr/bin/env python3
"""Separate diagnostic run: reject instrumentation/sync inside B/H.

Never use this wrapper for reported timings. It adds Python hooks deliberately
in order to audit the production task-only path, then records counts per rank.
"""
import contextlib
import json
import os
from pathlib import Path
import runpy
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from megatron import get_args, profile_utils
from megatron.core import utils
from megatron.core.pipeline_parallel.dspp_timeline import DsppTimelineRecorder
from megatron.timers import Timer

active=None
counts={'B':0,'H':0}
violations=[]
original_task=DsppTimelineRecorder.task


def guard(name,original,cuda_tensor_only=False):
    def wrapped(*args,**kwargs):
        if active and (not cuda_tensor_only or args[0].is_cuda):
            violations.append({'task':active,'operation':name})
            raise AssertionError(f'{name} executed inside {active}')
        return original(*args,**kwargs)
    return wrapped


@contextlib.contextmanager
def task(self,kind,*args,**kwargs):
    global active
    with original_task(self,kind,*args,**kwargs):
        if kind.startswith(('B','H')):
            assert get_args().dspp_task_timing_only
            assert not profile_utils.ANNOTATIONS_ENABLED
            assert active is None
            active=kind
            counts[kind[0]]+=1
        try:
            yield
        finally:
            active=None


DsppTimelineRecorder.task=task
# Boundary records happen before/after the guarded body and remain allowed.
for cls,name in ((torch.cuda.Event,'record'),(torch.cuda.Event,'synchronize'),
                 (Timer,'start'),(Timer,'stop')):
    setattr(cls,name,guard(f'{cls.__name__}.{name}',getattr(cls,name)))
torch.cuda.synchronize=guard('cuda.synchronize',torch.cuda.synchronize)
for name in ('cpu','item','tolist'):
    setattr(torch.Tensor,name,guard(f'Tensor.{name}',getattr(torch.Tensor,name),True))
for name in ('allclose','equal'):
    setattr(torch,name,guard(name,getattr(torch,name)))
torch.testing.assert_close=guard('assert_close',torch.testing.assert_close)
for name in ('range_push','range_pop'):
    setattr(torch.cuda.nvtx,name,guard('nvtx.'+name,getattr(torch.cuda.nvtx,name)))

# These wrappers must never even be inserted during forward in task-only mode.
def forbidden(*args,**kwargs):
    raise AssertionError('annotation/sync-only autograd node was inserted')
for cls in (profile_utils.WrapInputsFunction,profile_utils.WrapOutputsFunction,
            profile_utils.BackwardPushFunction,profile_utils.BackwardPopFunction,
            utils.SyncAtBackwardFunction):
    cls.apply=forbidden

runpy.run_path(str(REPO/'pretrain_llama.py'),run_name='__main__')
assert counts['B']>0 and counts['H']>0 and not violations
root=Path(os.environ['DSPP_TIMING_AUDIT_ROOT']);root.mkdir(parents=True,exist_ok=True)
rank=torch.distributed.get_rank()
(root/f'rank{rank}.json').write_text(json.dumps({'counts':counts,'violations':violations,
    'scope':'Python B/H-body audit; C++ metadata checks remain; not a timing result'},indent=2)+'\n')
