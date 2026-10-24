"""Block wall-time benchmark; no tensor snapshots or per-task timing hooks."""
import atexit
import faulthandler
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time
sys.path.insert(0, '/workspace/src/Megatron-LM-kwai')
import torch
import torch.distributed as dist
from megatron import get_args
from megatron import training
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)
root = Path(os.environ['DSPP_BENCH_OUTPUT'])
root.mkdir(parents=True, exist_ok=True)
warmup = int(os.environ.get('DSPP_BENCH_WARMUP', '4'))
records, report = [], {}
original = training.train_step
start = None

def step(*args, **kwargs):
    global start
    cfg = get_args()
    iteration = cfg.curr_iteration + 1
    if iteration == warmup + 1:
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
    result = original(*args, **kwargs)
    if iteration > warmup:
        if iteration == cfg.train_iters:
            torch.cuda.synchronize()
            report.update(seconds=time.perf_counter()-start,
                          allocated_peak=torch.cuda.max_memory_allocated(),
                          reserved_peak=torch.cuda.max_memory_reserved())
        records.append(dict(iteration=iteration,
                            metrics=dict(getattr(cfg, '_dspp_last_iteration_metrics', {})),
                            plan=getattr(cfg, '_dspp_last_schedule_plan', None),
                            skipped=int(result[1])))
    return result

def finish():
    rank = int(os.environ.get('RANK', '-1'))
    report.update(rank=rank, warmup_iterations=warmup, records=records,
                  scope='wall block includes data, synchronous planning/validation, training/optimizer and intervening normal loop overhead; no per-step added CUDA sync; final reporting outside timed block')
    (root/f'rank{rank}.json').write_text(json.dumps(report)+'\n')
training.train_step = step
atexit.register(finish)
runpy.run_path('/workspace/src/Megatron-LM-kwai/pretrain_llama.py', run_name='__main__')
