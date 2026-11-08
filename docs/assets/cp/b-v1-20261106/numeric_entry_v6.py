"""Isolated target CP numerical wrapper; never used for timed throughput.

Native diagnostics capture every parameter gradient, valid-token output/dx,
logit and loss. Add complete FP32 master and actual master-update comparisons;
parameters-after alone can hide a wrong small update. Preserve native gates.
"""
import json, os, runpy, sys
from pathlib import Path
SRC=Path(os.environ['CP_V1_SOURCE'])
sys.path.insert(0,str(SRC))
import torch
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
import torch.distributed as dist
from megatron import training,get_args
from zero_dropout import install
zero_stats,_,_=install()
original=training.train_step
master_failed=False
from megatron.core.kv_cache.dspp_cp_qo_attention import _QOHistoryJob
native_history=_QOHistoryJob.run
history_counts=dict(calls=0,nonempty_parts=0,history_tokens=0)
def history_run(self):
    history_counts['calls']+=1
    history_counts['nonempty_parts']+=sum(h>0 for _,_,h in self.parts)
    history_counts['history_tokens']+=sum(h for _,_,h in self.parts)
    return native_history(self)
_QOHistoryJob.run=history_run

def masters(optimizer):
    assert hasattr(optimizer,'fp32_from_float16_groups'),'unexpected optimizer'
    return {f'{i}:{j}':p.detach().float().cpu().clone()
            for i,g in enumerate(optimizer.fp32_from_float16_groups) for j,p in enumerate(g)}

def compare(a,b,exact=False):
    assert a.keys()==b.keys() and a
    ss=rr=0.;maximum=0.;bad=elements=0
    for key,x in a.items():
        y=b[key];assert x.shape==y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
        d=x.double()-y.double();ss+=float(d.square().sum());rr+=float(y.double().square().sum())
        maximum=max(maximum,float(d.abs().max()));elements+=x.numel()
        bad+=int((~torch.isclose(x,y,atol=0 if exact else .004,rtol=0 if exact else .01)).sum())
    rel=(ss/max(rr,1e-24))**.5
    return dict(passed=bad==0 and (rr<=1e-16 or rel<=.02),elements=elements,
                mismatched_elements=bad,max_abs=maximum,relative_l2=rel,exact=exact)

def train_step(*args,**kwargs):
    global master_failed
    optimizer=kwargs.get('optimizer',args[4] if len(args)>4 else None)
    iteration=get_args().curr_iteration+1;rank=dist.get_rank()
    history_before=dict(history_counts)
    native=optimizer.step;captured={}
    def step(*a,**kw):
        before=masters(optimizer);result=native(*a,**kw);after=masters(optimizer)
        captured.update(before=before,after=after,update={k:after[k]-before[k] for k in before})
        return result
    optimizer.step=step
    try:result=original(*args,**kwargs)
    finally:optimizer.step=native
    assert captured and not int(result[1]),'missing or skipped update'
    reference_rank=rank
    if os.environ.get('DSPP_PLAN_CHECK_CP_REFERENCE')=='1':
        from megatron.core import parallel_state
        reference_rank=parallel_state.get_pipeline_model_parallel_rank()
    reference=Path(os.environ.get('CP_V1_MASTER_REFERENCE',os.environ['DSPP_PLAN_CHECK_SNAPSHOTS']))/f'full-master-iteration{iteration}-rank{reference_rank}.pt'
    capture=Path(os.environ['DSPP_PLAN_CHECK_OUTPUT'])/f'full-master-iteration{iteration}-rank{rank}.pt'
    assert not capture.exists();torch.save(captured,capture)
    if os.environ['DSPP_PLAN_CHECK_MODE']=='reference':
        assert not reference.exists();torch.save(captured,reference);report=dict(reference=True)
    else:
        expected=torch.load(reference,map_location='cpu',weights_only=True)
        report={k:compare(v,expected[k],exact=k=='before') for k,v in captured.items()}
    report.update(rank=rank,iteration=iteration,zero_dropout=dict(zero_stats),
                  history_executed={k:history_counts[k]-history_before[k] for k in history_counts},
                  scope='all FP32 masters and actual updates; per-iteration aligned, not continuous trajectory')
    output=Path(os.environ['DSPP_PLAN_CHECK_OUTPUT'])/f'full-master-iteration{iteration}-rank{rank}.json'
    assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n')
    master_failed |= not all(v['passed'] for v in report.values() if isinstance(v,dict) and 'passed' in v)
    return result

training.train_step=train_step
assert '--dspp-split-attention-backward' not in sys.argv,'CP1 attribution is fused only'
setup=runpy.run_path(str(SRC/'tools/validate_dspp_plan_training.py'),run_name='cp_reference_setup')
native_check=training.train_step
check_globals=native_check.__globals__
def bootstrap_reference(*args,**kwargs):
    origin=os.environ.get('CP_V1_BOOTSTRAP_FROM')
    if origin:
        from megatron.core import parallel_state
        model=kwargs.get('model',args[3] if len(args)>3 else None)
        optimizer=kwargs.get('optimizer',args[4] if len(args)>4 else None)
        assert os.environ['DSPP_PLAN_CHECK_MODE']=='reference' and get_args().context_parallel_size==1
        source_rank=2*parallel_state.get_pipeline_model_parallel_rank()
        root=Path(origin);it=get_args().curr_iteration+1;assert it==1
        values=torch.load(root/f'iteration{it}-rank{source_rank}.pt',map_location='cpu',weights_only=True)
        optimizer.load_state_dict(torch.load(root/f'optimizer-iteration{it}-rank{source_rank}.pt',map_location='cpu',weights_only=True))
        with torch.no_grad():
            for i,chunk in enumerate(model):
                for name,p in chunk.named_parameters():p.copy_(values['params_before'][f'{i}:{name}'])
        report=Path(os.environ['DSPP_PLAN_CHECK_OUTPUT'])/f'bootstrap-rank{dist.get_rank()}.json'
        report.write_text(json.dumps(dict(source=str(root),source_CP2_rank=source_rank,
            target_CP1_rank=dist.get_rank(),scope='inject SAME failed-case initial weights and optimizer before native reference captures them'))+'\n')
    return native_check(*args,**kwargs)
training.train_step=bootstrap_reference
runpy.run_path(str(SRC/'pretrain_llama.py'),run_name='__main__')
if master_failed or check_globals['comparison_failed']:
    raise SystemExit('numerical attribution completed with failed comparisons; all reports retained')
