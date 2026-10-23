"""No-history FA selection and paired dKV output: numerical/dispatch contracts."""
import json

import pytest
import torch

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.kv_cache import dspp_packed_attention as packed, fa_split
from .test_packed_split import physical
from .test_fa_split import fa_forward, fa_backward, errors
from .test_fa_split_regions import make_inputs, math_ref, check_accuracy, SCALE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('hk', [4,2,1])
@pytest.mark.parametrize('chunk', [63,64])
@pytest.mark.parametrize('lengths', [(17,3),(145,70,17,3),(193,129,33,17),(64,64)])
def test_nohistory_dispatch_and_gradients(hk, chunk, lengths, monkeypatch):
    plan = build_dspp_batch_plan(lengths, chunk_size=chunk)
    gen = torch.Generator(device='cuda').manual_seed(2047)
    source = [tuple(torch.randn(n,h,128,device='cuda',generator=gen).bfloat16()
                    for h in (4,hk,hk)) for n in lengths]
    upstream = [torch.randn(n,4,128,device='cuda',generator=gen).bfloat16() for n in lengths]
    original_fused = packed._wrapped_flash_attn_varlen_backward
    original_prepare = fa_split.prepare_backward
    calls = {'fused':0,'split':0,'mixed':0}
    expect_history = None
    def fused(*args, **kwargs):
        assert expect_history is False, 'history-bearing task incorrectly took full fused backward'
        calls['fused'] += 1
        return original_fused(*args, **kwargs)
    def prepare(*args, **kwargs):
        assert expect_history is True, 'history-free task still prepared split state'
        if kwargs.get('cu_seqlens_q') is not None: calls['split'] += 1
        return original_prepare(*args, **kwargs)
    results=[]
    for after_dx in (False,True):
        state=packed.DsppSequenceKVState(split_attention_backward=after_dx,history_dkv_after_dx=after_dx)
        inputs=[];outputs=[]
        for meta in plan.microbatches:
            inp=[physical([v[j] for v in source],meta).unsqueeze(1).detach().requires_grad_() for j in range(3)]
            inputs.append(inp)
            outputs.append(packed.dspp_packed_flash_attention(*inp,state=state,meta=meta,layer_idx=0))
        state.set_microbatch(plan.microbatches[0])
        with monkeypatch.context() as patch:
            if after_dx:
                patch.setattr(packed,'_wrapped_flash_attn_varlen_backward',fused)
                patch.setattr(fa_split,'prepare_backward',prepare)
            for meta,inp,out in reversed(list(zip(plan.microbatches,inputs,outputs))):
                expect_history=any(i.token_offset>0 for i in meta.items)
                if after_dx:
                    state.begin_backward_task(meta.microbatch_id)
                    calls['mixed'] += int(expect_history and any(i.token_offset==0 for i in meta.items))
                # Deliberately nonzero dO on padding; output grads must still be zero there.
                grad=torch.ones_like(out);cursor=0
                for item in meta.items:
                    grad[cursor:cursor+item.token_length,0]=upstream[item.sequence_id][item.token_offset:item.token_offset+item.token_length]
                    cursor+=item.token_length
                out.backward(grad)
                if after_dx:
                    assert state.pending_history_count==int(expect_history)
                    before=[t.grad.clone() for t in inp]
                    state.mark_dx_ready(meta.microbatch_id)
                    assert state.run_history_task(meta.microbatch_id)==int(expect_history)
                    for t,b in zip(inp,before):torch.testing.assert_close(t.grad,b,atol=0,rtol=0)
                for t in inp:assert torch.count_nonzero(t.grad[meta.valid_token_count:])==0
        state.finish_iteration()
        assert not state._entries and not state._history_jobs
        results.append([[t.grad for t in inp] for inp in inputs])
    nh=sum(not any(i.token_offset>0 for i in m.items) for m in plan.microbatches)
    assert calls['fused']==nh and calls['split']==len(plan.microbatches)-nh
    checks={}
    for j,name in enumerate(('dq','dk','dv')):
        a=torch.cat([r[j].flatten() for r in results[1]]);b=torch.cat([r[j].flatten() for r in results[0]])
        torch.testing.assert_close(a,b,atol=.004,rtol=.01);checks[name]=errors(a,b)
    print('OUTPUT_OPT_ROUTE '+json.dumps(dict(lengths=lengths,chunk=chunk,hk=hk,calls=calls,checks=checks)))


@pytest.mark.parametrize('hq,hk', [(6,2),(10,2),(16,1),(32,1)])
@pytest.mark.parametrize('nq,nk', [(17,80),(65,130)])
def test_joint_output_nontrivial_groups(hq,hk,nq,nk):
    q,k,v,dout=make_inputs(nq,nk,hq,hk)
    out,lse,rng=fa_forward(q,k,v,SCALE)
    state=fa_split.prepare_backward(q,k,v,out,dout,lse)
    dq,ck,cv=fa_split.backward_dq_current(state)
    hist=fa_split.backward_dkv(state,'history',compact=True)
    actual=(dq,torch.cat((hist[0],ck)),torch.cat((hist[1],cv)))
    fused=fa_backward(q,k,v,out,dout,lse,rng,SCALE)
    _,ref=math_ref(q,k,v,dout,SCALE)
    checks=check_accuracy(actual,fused,ref)
    print('OUTPUT_OPT_GROUP '+json.dumps(dict(hq=hq,hk=hk,nq=nq,nk=nk,checks=checks)))


@pytest.mark.parametrize('hk', [4,2,1])
def test_joint_output_launch_count(hk,tmp_path):
    q,k,v,dout=make_inputs(65,194,4,hk)
    out,lse,_=fa_forward(q,k,v,SCALE)
    state=fa_split.prepare_backward(q,k,v,out,dout,lse)
    fa_split.backward_dq_current(state);fa_split.backward_dkv(state,'history',compact=True)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        fa_split.backward_dq_current(state)
        fa_split.backward_dkv(state,'history',compact=True)
        torch.cuda.synchronize()
    path=tmp_path/'joint-output.pt.trace.json';prof.export_chrome_trace(str(path))
    kernels=[e for e in json.loads(path.read_text())['traceEvents'] if e.get('cat')=='kernel']
    joint=[e for e in kernels if 'reduce_dkv_output_kernel' in e['name']]
    assert len(joint)==2, [e['name'] for e in kernels]  # one B output pass and one H pass
    assert not any('reduce_kernel' in e['name'] for e in kernels)
    # dQ still needs an FP32 zero-initialization pass; this is not a dKV copy.
    elementwise=[e for e in kernels if 'elementwise_kernel' in e['name']]
    assert all('FillFunctor<float>' in e['name'] for e in elementwise)
    print('OUTPUT_OPT_LAUNCH '+json.dumps(dict(hk=hk,kernels=[e['name'] for e in kernels])))
