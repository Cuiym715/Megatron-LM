"""Small B training comparison, with optional fused v2 reference execution.

DSPP_B_CHECK_REFERENCE=1 only selects the same-input unsplit v2 template.
The B arm invokes the real distributed coordinator and unmodified solver.
"""
import json
import os
from pathlib import Path
import runpy
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
import torch.distributed as dist
from megatron import get_args, training
from megatron.core.pipeline_parallel import schedules

if os.environ.get('DSPP_B_CHECK_REFERENCE')=='1':
    original=schedules.build_runtime_plan
    def fused_reference(*args,**kwargs):
        kwargs.update(explicit_history=True,optimize=False)
        reference,_=original(*args,**kwargs)
        return reference,reference
    schedules.build_runtime_plan=fused_reference

if os.environ.get('DSPP_B_CHECK_SPLIT_FIXTURE') == '1':
    # Diagnostic costs force an automatic split; they are not measurements
    # and this arm must never be used to report throughput or model accuracy.
    from megatron.core.pipeline_parallel import dspp_plan_runtime as runtime
    real_solver = runtime.plan_configuration_b
    def split_fixture(reference, compute, messages, split_costs, **kwargs):
        compute = {k:1. if c else 0. for k,c in compute.items()}
        source = next((k for k in split_costs if k[:3] == (0,'B',1)), None)
        if source is not None:
            compute[source] = 100.
            compute[(1,*source[1:])] = 100.
        split_costs = {k:(compute[k]*.3,compute[k]*.8) for k in split_costs}
        if source is not None:
            split_costs[source] = (20.,90.)
        messages = dict.fromkeys(messages,3.)
        plan = real_solver(reference,compute,messages,split_costs,**kwargs)
        plan.planning_info['diagnostic_cost_fixture'] = True
        return plan
    runtime.plan_configuration_b = split_fixture

original_train=training.train_step

def train_step(*args,**kwargs):
    result=original_train(*args,**kwargs)
    torch.cuda.synchronize()
    config=get_args();rank=dist.get_rank()
    peak=torch.cuda.max_memory_allocated()
    predicted=config._dspp_memory_plan['total_peak_bytes'][rank]
    assert peak<=predicted,(rank,peak,predicted)
    assert not int(result[1]),'skipped optimizer step'
    plan=config._dspp_last_schedule_plan
    info=dict(iteration=config.curr_iteration+1,rank=rank,allocated_peak_bytes=peak,
              predicted_total_bytes=predicted,planning=plan.get('planning_info'),
              selected_count=sum(n['kind']=='D' for seq in plan['schedules'] for n in seq),
              fallback=plan['fallback_reason'])
    out=Path(os.environ['DSPP_PLAN_CHECK_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    (out/f'b-iteration{config.curr_iteration+1}-rank{rank}.json').write_text(json.dumps(info,indent=2))
    return result

training.train_step=train_step
runpy.run_path(str(REPO/'tools/validate_dspp_plan_training.py'),run_name='__main__')
