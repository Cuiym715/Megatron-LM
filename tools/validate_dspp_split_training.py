#!/usr/bin/env python3
"""Instrument actual pretrain_llama training for split-vs-fused correctness.

Use DSPP_SPLIT_CHECK_OUTPUT and DSPP_SPLIT_CHECK_SNAPSHOTS. In fused mode save
complete local model gradients and stage input gradients outside Git; in split
mode compare every tensor against that run. This is not a timing benchmark.
Tolerances are fixed before execution: atol=.004, rtol=.01, relative L2<=.02.
"""
from pathlib import Path
import json
import os
import runpy
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import torch
import torch.distributed as dist
from megatron import get_args
import megatron.training as training
from megatron.core.pipeline_parallel import schedules
from megatron.core.kv_cache import dspp_packed_attention as packed

OUTPUT=Path(os.environ['DSPP_SPLIT_CHECK_OUTPUT'])
SNAPSHOTS=Path(os.environ['DSPP_SPLIT_CHECK_SNAPSHOTS'])
OUTPUT.mkdir(parents=True,exist_ok=True);SNAPSHOTS.mkdir(parents=True,exist_ok=True)
original_train=training.train_step;original_backward=schedules.backward_step
original_attn=packed._DsppPackedFlashAttention.backward
original_finish=packed.DsppSequenceKVState.finish_iteration
original_history=packed.DsppSequenceKVState.run_history_task
active=None


def flatten_tensors(value):
    if isinstance(value,torch.Tensor):return [value]
    if value is None:return []
    return [t for v in value for t in flatten_tensors(v)]


def backward(*args,**kwargs):
    result=original_backward(*args,**kwargs)
    if active is not None:
        active['dx'].extend(t.detach().float().cpu() for t in flatten_tensors(result))
    return result


def attention_backward(ctx,grad):
    if active is not None:
        active['attention_calls']+=1
        active['split_calls']+=int(ctx.split_attention_backward)
        active['shapes'].add((tuple(ctx.saved_tensors[0].shape),tuple(ctx.saved_tensors[1].shape)))
        active['mixed_packs']+=int(len(ctx.layouts)>1 and any(x.history for x in ctx.layouts))
    return original_attn(ctx,grad)


def finish(self):
    result=original_finish(self)
    if active is not None:
        active['drained_states']+=1
        assert not self._entries and not self._prepared_cu_seqlens
    return result


def history(self, microbatch_id):
    assert self._history_task_ready, 'H started before stage backward returned'
    count = self.pending_history_count
    result = original_history(self, microbatch_id)
    assert result == count and not self.pending_history_count
    if active is not None:
        active['history_tasks'] += int(count > 0)
        active['history_layers'] += count
    return result


def compare(actual,reference,exact=False):
    assert set(actual)==set(reference),(set(actual)-set(reference),set(reference)-set(actual))
    result={}
    for name,a in actual.items():
        b=reference[name]
        assert a.shape==b.shape and torch.isfinite(a).all(),name
        torch.testing.assert_close(a,b,atol=0 if exact else .004,rtol=0 if exact else .01,
                                   msg=lambda m:f'{name}: {m}')
        diff=a.double()-b.double();norm=b.double().norm().item()
        rel=diff.norm().item()/norm if norm>1e-8 else None
        if rel is not None and not exact:assert rel<=.02,(name,rel)
        result[name]={'max_abs':diff.abs().max().item(),'relative_l2':rel,'numel':a.numel()}
    return result


def train_step(*args,**kwargs):
    global active
    model=kwargs.get('model',args[3] if len(args)>3 else None)
    optimizer=kwargs.get('optimizer',args[4] if len(args)>4 else None)
    config=get_args();iteration=config.curr_iteration+1;rank=dist.get_rank()
    split=config.dspp_split_attention_backward
    active={'dx':[],'attention_calls':0,'split_calls':0,'drained_states':0,'shapes':set(),'mixed_packs':0,
            'history_tasks':0,'history_layers':0}
    base_path=SNAPSHOTS/f'iteration{iteration}-rank{rank}.pt'
    reference=torch.load(base_path,map_location='cpu',weights_only=True) if split else None
    params={f'{i}:{n}':p.detach().float().cpu() for i,m in enumerate(model) for n,p in m.named_parameters()}
    param_checks=compare(params,reference['params'],exact=iteration==1) if split else {}
    step=optimizer.step;snapshot={};gradient_checks={}
    def capture_step(*a,**kw):
        for i,m in enumerate(model):
            for n,p in m.named_parameters():
                if not p.requires_grad:continue
                grad=getattr(p,'main_grad',None)
                if grad is None:grad=p.grad
                assert grad is not None,(i,n)
                snapshot[f'{i}:{n}']=grad.detach().float().cpu()
        if split:gradient_checks.update(compare(snapshot,reference['gradients']))
        return step(*a,**kw)
    optimizer.step=capture_step
    try:result=original_train(*args,**kwargs)
    finally:optimizer.step=step
    dx={str(i):t for i,t in enumerate(active['dx'])}
    dx_checks=compare(dx,reference['dx']) if split else {}
    loss={k:float(v) if isinstance(v,(float,int,torch.Tensor)) else str(v) for k,v in result[0].items()}
    if split:
        assert active['attention_calls']>0 and active['split_calls']==active['attention_calls']
        compare({k:torch.tensor(v) for k,v in loss.items()}, {k:torch.tensor(v) for k,v in reference['loss'].items()})
    else:
        assert active['attention_calls']>0 and active['split_calls']==0
        assert not base_path.exists(),'refusing to overwrite baseline'
        torch.save(dict(params=params,gradients=snapshot,dx=dx,loss=loss),base_path)
    assert active['drained_states']>0
    if config.dspp_history_dkv_after_dx:
        assert active['history_tasks'] > 0 and active['history_layers'] > 0
    record={'rank':rank,'iteration':iteration,'split':split,'loss':loss,
            'after_dx':config.dspp_history_dkv_after_dx,
            'history_tasks':active['history_tasks'],'history_layers':active['history_layers'],
            'attention_calls':active['attention_calls'],'split_calls':active['split_calls'],
            'drained_states':active['drained_states'],'mixed_pack_calls':active['mixed_packs'],
            'attention_shapes':sorted(active['shapes']),'stage_dx_count':len(dx),
            'parameter_tensor_count':len(params),'gradient_tensor_count':len(snapshot),
            'parameter_checks':param_checks,'gradient_checks':gradient_checks,'stage_dx_checks':dx_checks,
            'criteria':{'atol':.004,'rtol':.01,'relative_l2_max':.02,'initial_parameters':'exact'}}
    (OUTPUT/f'iteration{iteration}-rank{rank}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'SPLIT_TRAIN_CHECK rank={rank} iteration={iteration} split={split} gradients={len(snapshot)} dx={len(dx)} attention_calls={active["attention_calls"]}',flush=True)
    active=None
    return result


training.train_step=train_step
schedules.backward_step=backward
packed._DsppPackedFlashAttention.backward=staticmethod(attention_backward)
packed.DsppSequenceKVState.finish_iteration=finish
packed.DsppSequenceKVState.run_history_task=history
runpy.run_path(str(REPO/'pretrain_llama.py'),run_name='__main__')
