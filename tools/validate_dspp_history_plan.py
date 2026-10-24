"""P3 manual-plan diagnostics, reusing the existing full-tensor comparator.

Select DSPP_P3_HISTORY_PLAN=adjacent/across-w/across-fb. Identical split
selection is frozen before F in all three runs. This constructs fixtures only;
it is neither algorithm B nor a throughput benchmark.
"""
import os
import json
from pathlib import Path
import runpy
import sys
from dataclasses import replace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from megatron.core.pipeline_parallel import schedules
from megatron.core.pipeline_parallel.dspp_plan import digest, node_key, split_plan, validate_plan
from megatron import get_args, training
import torch
import torch.distributed as dist

mode = os.environ['DSPP_P3_HISTORY_PLAN']
assert mode in ('adjacent', 'across-w', 'across-fb')
original_build = schedules.build_runtime_plan


def build(*args, **kwargs):
    kwargs.update(explicit_history=True, optimize=False)
    reference, _ = original_build(*args, **kwargs)
    sources = {edge[:2] for edge in reference.history_dependencies}
    selected = {node_key(n) for seq in reference.schedules for n in seq
                if n.kind == 'B' and (n.microbatch, n.split) in sources}
    assert selected
    selected.remove(sorted(selected)[0])  # Exercise a genuine fused-history B too.
    adjacent = split_plan(reference, selected)
    candidate = adjacent
    if mode != 'adjacent':
        across = 'W' if mode == 'across-w' else 'FB'
        found = False
        for stage, seq in enumerate(adjacent.schedules):
            for i, node in enumerate(seq):
                if node.kind != 'H':
                    continue
                for j in range(i+1, len(seq)):
                    if seq[j].kind not in across:
                        continue
                    changed = list(seq)
                    changed.insert(j, changed.pop(i))
                    stages = list(adjacent.schedules)
                    stages[stage] = tuple(replace(n, slot=slot) for slot,n in enumerate(changed))
                    trial = replace(adjacent, schedules=tuple(stages))
                    try:
                        validate_plan(trial, reference)
                    except ValueError:
                        continue
                    candidate, found = trial, True
                    break
                if found:
                    break
            if found:
                break
        assert found, f'no legal {mode} fixture'
        assert candidate.schedules != adjacent.schedules
    validate_plan(candidate, reference)
    return reference, candidate


schedules.build_runtime_plan = build
events = []
original_backward = schedules._DsppVTaskBatch.backward
original_history = schedules._DsppVTaskBatch.history
original_train = training.train_step


def backward(self, split, *args, **kwargs):
    result = original_backward(self, split, *args, **kwargs)
    task = self.training_batch.physical_microbatches[split].meta.microbatch_id
    events.append(dict(kind='D' if self.kv_state.task_after_dx(task) else 'B',
        logical=self._plan_check_logical, chunk=self.chunk, physical=split,
        pending_tasks=len(self.kv_state._history_records)))
    return result


def history(self, split):
    task = self.training_batch.physical_microbatches[split].meta.microbatch_id
    before = self.kv_state.history_job_count(task)
    result = original_history(self, split)
    assert task not in self.kv_state._history_records
    events.append(dict(kind='H', logical=self._plan_check_logical, chunk=self.chunk,
        physical=split, history_layers=before, pending_tasks=len(self.kv_state._history_records)))
    return result


def train_step(*args, **kwargs):
    events.clear()
    result = original_train(*args, **kwargs)
    torch.cuda.synchronize()
    config, rank = get_args(), dist.get_rank()
    peak = torch.cuda.max_memory_allocated()
    predicted = config._dspp_memory_plan['total_peak_bytes'][rank]
    assert not int(result[1]), 'skipped optimizer step'
    assert peak <= predicted, (peak, predicted)
    directory = Path(os.environ['DSPP_PLAN_CHECK_OUTPUT'])
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(mode=mode, rank=rank, iteration=config.curr_iteration+1,
        allocated_peak_bytes=peak, predicted_total_bytes=predicted,
        events=list(events), plan_hash=digest(config._dspp_last_schedule_plan))
    (directory/f'p3-iteration{config.curr_iteration+1}-rank{rank}.json').write_text(json.dumps(record, indent=2))
    return result


schedules._DsppVTaskBatch.backward = backward
schedules._DsppVTaskBatch.history = history
training.train_step = train_step
runpy.run_path(str(REPO/'tools/validate_dspp_plan_training.py'), run_name='__main__')
