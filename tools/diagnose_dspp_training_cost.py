#!/usr/bin/env python3
"""Separate, intrusive stage-1 profiler run; never use as clean B/H timing.

Enabled only by TRAIN_ENTRYPOINT. Production instrumentation is unchanged.
"""
import contextlib
import functools
import os
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from megatron.core.pipeline_parallel.dspp_timeline import DsppTimelineRecorder
from megatron.core.kv_cache import dspp_packed_attention as packed, fa_split

active = False
original_init = DsppTimelineRecorder.__init__
original_finish = DsppTimelineRecorder.finish
original_task = DsppTimelineRecorder.task


def init(self, *args, **kwargs):
    global active
    original_init(self, *args, **kwargs)
    self.cost_profiler = None
    if self.stage == 1 and self.iteration == 3:
        self.cost_profiler = torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        self.cost_profiler.__enter__()
        active = True


def finish(self):
    global active
    if self.cost_profiler:
        torch.cuda.synchronize()
        active = False
        self.cost_profiler.__exit__(None, None, None)
        root = Path(os.environ['DSPP_COST_DIAG_ROOT'])
        root.mkdir(parents=True, exist_ok=True)
        self.cost_profiler.export_chrome_trace(str(root/'stage1.pt.trace.json'))
    return original_finish(self)


@contextlib.contextmanager
def task(self, kind, task_id, meta, *, chunk):
    with original_task(self, kind, task_id, meta, chunk=chunk):
        if active:
            shape = ','.join(f'{i.token_length}:{i.token_offset}' for i in meta.items)
            with torch.profiler.record_function(f'COST_TASK/{kind}/c{chunk}/m{task_id[0]}:{task_id[1]}/{shape}'):
                yield
        else:
            yield


def wrap(name, fn):
    @functools.wraps(fn)
    def call(*args, **kwargs):
        if not active:
            return fn(*args, **kwargs)
        label = name
        if name == 'dkv': label += '/' + (args[1] if len(args) > 1 else kwargs.get('region', 'all'))
        with torch.profiler.record_function('COST_OP/' + label):
            return fn(*args, **kwargs)
    return call


DsppTimelineRecorder.__init__ = init
DsppTimelineRecorder.finish = finish
DsppTimelineRecorder.task = task
packed._DsppPackedFlashAttention.backward = staticmethod(wrap('attention_backward', packed._DsppPackedFlashAttention.backward))
packed._wrapped_flash_attn_varlen_backward = wrap('fused', packed._wrapped_flash_attn_varlen_backward)
for label, name in [('prepare', 'prepare_backward'), ('dq', 'backward_dq'),
                    ('dq_current', 'backward_dq_current'), ('dkv', 'backward_dkv')]:
    setattr(fa_split, name, wrap(label, getattr(fa_split, name)))
runpy.run_path(str(REPO/'pretrain_llama.py'), run_name='__main__')
