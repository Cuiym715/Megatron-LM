#!/usr/bin/env python3
"""Current-code task collection; diagnostic hooks exist only in this entrypoint.

DSPP_COLLECTION_OUTPUT: per-mode directory. DSPP_COLLECTION_REPLAY: directory
containing input26-rank*.pt for a three-step diagnostic replay. No gradients or
GPU tensor values are copied by formal collection. Raw CPU input references,
event results and metadata are serialized only after training exits.
"""
import builtins
import contextlib
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
import torch.distributed as dist
from megatron import get_args
import megatron.training as training
from megatron.core.pipeline_parallel.dspp_timeline import DsppTimelineRecorder
from megatron.core.kv_cache.dspp_packed_attention import DsppSequenceKVState

ROOT = Path(os.environ['DSPP_COLLECTION_OUTPUT'])
REPLAY = os.environ.get('DSPP_COLLECTION_REPLAY', '')
ROOT.mkdir(parents=True, exist_ok=True)
original_train = training.train_step
records = []
current_inputs = []
current_plans = []
events = []
anchor = None
anchor_cpu_ns = None
replay_batches = None
active_diag = False
active_task = None
diag_records = []


class InputIterator:
    def __init__(self, source):
        self.source = source
    def __iter__(self):
        return self
    def __next__(self):
        if REPLAY:
            data = replay_batches[len(current_inputs)]
        else:
            data = next(self.source)
        current_inputs.append(data)
        return data


def diag_event(kind, *, gpu=False, **details):
    record = dict(kind=kind, cpu_ns=time.perf_counter_ns(), task=active_task, **details)
    if gpu:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        record['_event'] = event
    diag_records.append(record)


def install_diagnostics():
    # Python wrappers, NVTX/profiler labels and extra events are diagnostic only.
    original_task = DsppTimelineRecorder.task
    @contextlib.contextmanager
    def task(self, kind, task_id, meta, *, chunk):
        global active_task
        if not active_diag:
            with original_task(self, kind, task_id, meta, chunk=chunk):
                yield
            return
        active_task = dict(kind=kind, logical=task_id[0], physical=task_id[1], chunk=chunk)
        diag_event('task_host_enter')
        with original_task(self, kind, task_id, meta, chunk=chunk):
            yield
        diag_event('task_host_return')
        active_task = None
    DsppTimelineRecorder.task = task

    original_route = DsppSequenceKVState._route_key_value_grads
    def route(self, layer_idx, layouts, *args, **kwargs):
        if not active_diag:
            return original_route(self, layer_idx, layouts, *args, **kwargs)
        layout = [dict(sequence=x.item.sequence_id, segment=x.item.segment_id,
                       q=x.item.token_length, offset=x.item.token_offset,
                       history=list(x.history)) for x in layouts]
        details = dict(layer=layer_idx, layouts=layout,
                       route_history=kwargs['route_history'],
                       consume_current=kwargs['return_current'])
        diag_event('gradient_route_begin', gpu=True, **details)
        with torch.profiler.record_function('COLLECT/gradient_route/' + str(active_task)):
            result = original_route(self, layer_idx, layouts, *args, **kwargs)
        diag_event('gradient_route_end', gpu=True, **details)
        return result
    DsppSequenceKVState._route_key_value_grads = route

    original_ready = DsppSequenceKVState.mark_dx_ready
    def ready(self, microbatch_id):
        result = original_ready(self, microbatch_id)
        if active_diag:
            diag_event('complete_dx_stream_boundary', gpu=True, microbatch_id=microbatch_id)
        return result
    DsppSequenceKVState.mark_dx_ready = ready

    original_print = builtins.print
    def diagnostic_print(*args, **kwargs):
        if active_diag and args and str(args[0]).startswith('[dspp][p2p-debug]'):
            diag_event('p2p_protocol', message=str(args[0]))
            return
        return original_print(*args, **kwargs)
    builtins.print = diagnostic_print

    # Observe the existing Gloo rendezvous operations, adding no wait calls.
    for name in ('isend', 'recv'):
        original = getattr(dist, name)
        def wrapped(*args, _name=name, _original=original, **kwargs):
            tag = kwargs.get('tag', 0)
            tracked = active_diag and 2_000_000 <= tag < 4_000_100
            if tracked:
                diag_event('control_enter', operation=_name, tag=tag,
                           peer=kwargs.get('src', kwargs.get('dst')))
            result = _original(*args, **kwargs)
            if tracked:
                diag_event('control_return', operation=_name, tag=tag,
                           peer=kwargs.get('src', kwargs.get('dst')),
                           value=args[0].tolist())
            return result
        setattr(dist, name, wrapped)


def train_step(forward_step_func, get_batch_func, data_iterator, *args, **kwargs):
    global current_inputs, current_plans, anchor, anchor_cpu_ns, replay_batches, active_diag
    cfg = get_args()
    iteration = cfg.curr_iteration + 1
    rank = dist.get_rank()
    if anchor is None:
        for _ in range(cfg.train_iters):
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for event in pair:
                event.record()
            events.append(pair)
        torch.cuda.synchronize()
        anchor = torch.cuda.Event(enable_timing=True)
        anchor_cpu_ns = time.perf_counter_ns()
        anchor.record()
        if REPLAY:
            replay_batches = torch.load(Path(REPLAY) / f'input26-rank{rank}.pt',
                                        map_location='cpu', weights_only=True)
    current_inputs = []
    current_plans = []
    def batch(iterator):
        result = get_batch_func(InputIterator(iterator))
        current_plans.append(result[0].plan)
        return result
    active_diag = bool(REPLAY and iteration == 3)
    if active_diag:
        os.environ['DSPP_GPU_PROGRESS_DIR'] = str(ROOT / 'progress')
        os.environ['DSPP_P2P_DEBUG'] = '1'
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                  torch.profiler.ProfilerActivity.CUDA]) if active_diag else None
    if profiler:
        profiler.__enter__()
    begin, end = events[iteration - 1]
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    host_begin = time.perf_counter_ns()
    begin.record()
    result = original_train(forward_step_func, batch, data_iterator, *args, **kwargs)
    end.record()
    host_end = time.perf_counter_ns()
    records.append(dict(iteration=iteration, host_begin_ns=host_begin, host_end_ns=host_end,
                        allocated_before=allocated_before,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        _inputs=current_inputs, _plans=current_plans))
    if profiler:
        profiler.__exit__(None, None, None)
        profiler.export_chrome_trace(str(ROOT / f'rank{rank}.pt.trace.json'))
        os.environ.pop('DSPP_GPU_PROGRESS_DIR', None)
        os.environ.pop('DSPP_P2P_DEBUG', None)
    active_diag = False
    return result


def finish():
    torch.cuda.synchronize()
    rank = dist.get_rank()
    for record in records:
        iteration = record['iteration']
        begin, end = events[iteration - 1]
        record.update(device_train_step_ms=begin.elapsed_time(end),
                      device_begin_offset_ms=anchor.elapsed_time(begin),
                      device_end_offset_ms=anchor.elapsed_time(end))
        inputs = record.pop('_inputs')
        record['inputs'] = []
        for batch in inputs:
            digest = hashlib.sha256()
            for name in sorted(batch):
                t = batch[name]
                assert isinstance(t, torch.Tensor) and t.device.type == 'cpu'
                digest.update(name.encode())
                digest.update(str((tuple(t.shape), str(t.dtype))).encode())
                digest.update(t.contiguous().numpy().tobytes())
            record['inputs'].append(dict(sha256=digest.hexdigest(), lengths=batch['lengths'].tolist()))
        record['plans'] = [dict(sequence_lengths=list(p.sequence_lengths),
                                physical_microbatches=len(p.microbatches),
                                valid_tokens=sum(m.valid_token_count for m in p.microbatches))
                           for p in record.pop('_plans')]
        if iteration == 26 and not REPLAY:
            torch.save(inputs, ROOT / f'input26-rank{rank}.pt')
    for record in diag_records:
        event = record.pop('_event', None)
        if event is not None:
            record['device_offset_ms'] = anchor.elapsed_time(event)
    output = dict(rank=rank, anchor_cpu_ns=anchor_cpu_ns,
                  clock_note='Per-rank CUDA elapsed time; CPU anchor is approximate, not exact cross-GPU clock calibration.',
                  boundary_note='Train-step compute-stream boundary after existing P2P drains and optimizer; no extra per-step device sync. Host times are submission/return, not device completion.',
                  records=records)
    (ROOT / f'completion-rank{rank}.json').write_text(json.dumps(output, indent=2)+'\n')
    if REPLAY:
        (ROOT / f'diagnostic-rank{rank}.json').write_text(json.dumps(diag_records, indent=2)+'\n')


if __name__ == '__main__':
    if REPLAY:
        install_diagnostics()
    training.train_step = train_step
    runpy.run_path(str(REPO / 'pretrain_llama.py'), run_name='__main__')
    finish()
