"""Complete-update CP benchmark / opt-in profile driver, no tensor snapshots.

DSPP_CP_BENCH_OUTPUT must be a new directory. CUDA/CPU waits at the measurement
window boundaries only. Detailed task events / ownership audits require the
normal explicit CLI switches and must not be used as throughput evidence.
"""
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
from megatron import training, get_args
from megatron.core import parallel_state
from megatron.core.datasets.dspp_calibration import stage_context
from megatron.core.pipeline_parallel.dspp_memory_counts import memory_layout

OUTPUT = Path(os.environ['DSPP_CP_BENCH_OUTPUT'])
WARMUP = int(os.environ.get('DSPP_CP_BENCH_WARMUP', '2'))
TIMED = int(os.environ.get('DSPP_CP_BENCH_TIMED', '6'))
AUDIT = os.environ.get('DSPP_CP_BENCH_AUDIT', '')
original = training.train_step
records = []
started = None
summary = None



# Separate diagnostic runs only: no hooks or counting in throughput windows.
IO_DIAGNOSTIC = bool(os.environ.get('DSPP_CP_BENCH_IO'))
attention_io = {}
history_peak = dict(logical_bytes=0, storage_bytes=0, pending_dkv_logical_bytes=0)
if IO_DIAGNOSTIC:
    import weakref
    from megatron.core.kv_cache import dspp_cp_attention as kv_attn
    from megatron.core.kv_cache import dspp_cp_qo_attention as qo_attn
    states = weakref.WeakSet()
    phase = None
    def sample_history():
        logical, pending, storage = 0, 0, {}
        for state in states:
            for entry in state._entries.values():
                pending += sum(t.numel()*t.element_size() for t in (*entry.pending_key_grads.values(), *entry.pending_value_grads.values()))
                for segment in entry.segments:
                    for t in (segment.key,segment.value):
                        logical += t.numel()*t.element_size()
                        st = t.untyped_storage()
                        storage[st.data_ptr()] = st.nbytes()
        history_peak['logical_bytes'] = max(history_peak['logical_bytes'],logical)
        history_peak['pending_dkv_logical_bytes'] = max(history_peak['pending_dkv_logical_bytes'],pending)
        history_peak['storage_bytes'] = max(history_peak['storage_bytes'],sum(storage.values()))
    def scoped(fn, label):
        def run(*args, **kwargs):
            global phase
            previous, phase = phase, label
            try:
                result = fn(*args, **kwargs)
                if label == 'F':
                    state = kwargs.get('state',args[4] if len(args)>4 else None)
                    if state is not None:
                        states.add(state)
                        sample_history()
                return result
            finally:
                phase = previous
        return run
    # Function.forward receives ctx,Q,K,V,state,...; no numerical changes.
    for cls in (kv_attn._DsppCpAttention,qo_attn._QOAttention):
        cls.forward = staticmethod(scoped(cls.forward,'F'))
        cls.backward = staticmethod(scoped(cls.backward,'B'))
    def counted(fn,name):
        def run(*args, **kwargs):
            if phase:
                source = args[0] if name == 'all_reduce' else args[1]
                group = kwargs.get('group')
                size = dist.get_world_size(group)
                nbytes = source.numel()*source.element_size()
                key = phase+'/'+name
                row = attention_io.setdefault(key,dict(calls=0,input_bytes=0,ring_equivalent_send_bytes=0))
                row['calls'] += 1
                row['input_bytes'] += nbytes
                row['ring_equivalent_send_bytes'] += (nbytes*(size-1) if name=='all_gather_into_tensor'
                    else nbytes*(size-1)//size*(2 if name=='all_reduce' else 1))
            return fn(*args, **kwargs)
        return run
    for name in ('all_gather_into_tensor','reduce_scatter_tensor','all_reduce'):
        setattr(dist,name,counted(getattr(dist,name),name))


def train_step(*args, **kwargs):
    global started, summary
    cfg = get_args()
    iteration = cfg.curr_iteration+1
    if iteration == WARMUP+1:
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
    # Two warmed layouts cover the short-only and history-heavy input phases.
    if AUDIT:
        cfg.dspp_memory_audit_dir = AUDIT if WARMUP+1 <= iteration <= WARMUP+2 else None
    result = original(*args, **kwargs)
    assert not int(result[1]), 'optimizer update was skipped'
    if iteration == WARMUP+TIMED:
        torch.cuda.synchronize()
        summary = dict(window_ms=(time.perf_counter()-started)*1000,
            allocated_peak=torch.cuda.max_memory_allocated(),
            reserved_peak=torch.cuda.max_memory_reserved())
    plan = getattr(cfg, '_dspp_last_schedule_plan', {}) or {}
    record = dict(iteration=iteration,
        selected=sum(n['kind']=='D' for seq in plan.get('schedules', ()) for n in seq),
        fallback=plan.get('fallback_reason', ''),
        algorithm=plan.get('algorithm'), predicted_ms=plan.get('predicted_ms'),
        reference_ms=plan.get('reference_ms'), planning_info=plan.get('planning_info', {}),
        metrics=getattr(cfg, '_dspp_last_iteration_metrics', {}))
    records.append(record)
    if cfg.dspp_memory_report_dir:
        record['memory_report'] = getattr(cfg, '_dspp_memory_plan', None)
        record['allocated_peak'] = torch.cuda.max_memory_allocated()
    return result


if __name__ == '__main__':
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert not (OUTPUT/f'rank{os.environ["RANK"]}.json').exists()
    if '--dspp-split-attention-backward' in sys.argv:
        # Prepare once, before PP traffic. Never compile on a first D deadline.
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
        from megatron.core.kv_cache.fa_split_build import load_extension
        load_extension()
    training.train_step = train_step
    runpy.run_path(str(REPO/'pretrain_llama.py'), run_name='__main__')
    assert summary is not None, 'run ended before the measurement window completed'
    cfg = get_args()
    summary.update(rank=dist.get_rank(), stage=parallel_state.get_pipeline_model_parallel_rank(),
        cp_rank=parallel_state.get_context_parallel_rank(), cp=cfg.context_parallel_size,
        warmup=WARMUP, timed=TIMED, ms_per_update=summary['window_ms']/TIMED,
        diagnostic=bool(AUDIT or cfg.dspp_timeline_dir or cfg.dspp_task_timing_only
                        or cfg.dspp_memory_report_dir or IO_DIAGNOSTIC),
        cp_impl=cfg.kaimm_context_parallel_impl,
        attention_io=attention_io if IO_DIAGNOSTIC else None,
        history_kv_peak=history_peak if IO_DIAGNOSTIC else None,
        layout=memory_layout(cfg,cfg.pipeline_model_parallel_size,cfg.params_dtype,cfg.micro_seq_length),
        contexts={c:stage_context(cfg,parallel_state.get_pipeline_model_parallel_rank(),c) for c in range(2)},
        records=records)
    (OUTPUT/f'rank{dist.get_rank()}.json').write_text(json.dumps(summary,indent=2)+'\n')
    dist.barrier()
    dist.destroy_process_group()
