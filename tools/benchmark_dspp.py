"""Native complete-update throughput; no per-task probes or numeric overrides.

Set DSPP_BENCH_OUTPUT, DSPP_BENCH_WARMUP and DSPP_BENCH_TIMED, and pass normal
pretrain_llama arguments (train-iters = warmup + timed). Write results at exit.
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


def main():
    output = Path(os.environ['DSPP_BENCH_OUTPUT'])
    warmup = int(os.environ.get('DSPP_BENCH_WARMUP', '5'))
    timed = int(os.environ.get('DSPP_BENCH_TIMED', '20'))
    native = training.train_step
    elapsed = None
    started = None
    tokens = skipped = completed = 0
    updates = []

    def step(*args, **kwargs):
        nonlocal started, elapsed, tokens, skipped, completed
        cfg = get_args()
        iteration = cfg.curr_iteration + 1
        if iteration == warmup + 1:
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
        result = native(*args, **kwargs)
        if warmup < iteration <= warmup + timed:
            completed += 1
            skipped += int(result[1])
            if dist.get_rank() == 0:
                metrics = cfg._dspp_last_iteration_metrics
                tokens += metrics['valid_tokens']
                updates.append(dict(iteration=iteration, **metrics))
            if iteration == warmup + timed:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
        return result

    training.train_step = step
    runpy.run_path(str(REPO / 'pretrain_llama.py'), run_name='__main__')
    local = dict(rank=dist.get_rank(), elapsed_seconds=elapsed,
                 completed_updates=completed, skipped_updates=skipped,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved())
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, local)
    if dist.get_rank() == 0:
        complete = all(r['completed_updates'] == timed and r['elapsed_seconds'] is not None
                       and r['skipped_updates'] == 0 for r in rows)
        seconds = max(r['elapsed_seconds'] for r in rows) if complete else None
        output.mkdir(parents=True, exist_ok=True)
        (output/'summary.json').write_text(json.dumps(dict(
            complete=complete, warmup=warmup, timed_updates=timed,
            effective_tokens=tokens, elapsed_seconds=seconds,
            effective_tokens_per_second=tokens/seconds if seconds else None,
            ranks=rows, updates=updates), indent=2)+'\n')


if __name__ == '__main__':
    main()
