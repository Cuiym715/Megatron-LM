"""One-pass all-context dQ/current dKV: numerical and boundary contracts."""
import json

import pytest
import torch

from megatron.core.kv_cache.fa_split import prepare_backward, backward_dq_current, backward_dkv
from .test_fa_split import fa_forward, fa_backward, errors
from .test_fa_split_regions import (make_inputs, math_ref, check_accuracy,
                                    packed_forward, packed_backward, SCALE)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('hk', [4, 2, 1])
@pytest.mark.parametrize('nq,nk', [(1,1), (17,17), (1,130), (17,80),
                                  (64,128), (65,130), (129,257), (65,194)])
def test_combined_dense(nq, nk, hk):
    q,k,v,dout = make_inputs(nq,nk,4,hk)
    out,lse,rng = fa_forward(q,k,v,SCALE)
    state = prepare_backward(q,k,v,out,dout,lse)
    saved = [t.clone() for t in state.tensors()]
    hist = backward_dkv(state,'history',compact=True)
    dq,ck,cv = backward_dq_current(state)
    assert ck.shape == (nq,hk,128) and cv.shape == ck.shape
    actual = dq,torch.cat((hist[0],ck)),torch.cat((hist[1],cv))
    fused = fa_backward(q,k,v,out,dout,lse,rng,SCALE)
    _,refs = math_ref(q,k,v,dout,SCALE)
    checks = check_accuracy(actual,fused,refs)
    # H may execute before or after the combined call, without mutating state.
    for a,b in zip(backward_dkv(state,'history',compact=True),hist):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    for a,b in zip(state.tensors(),saved):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    print('COMBINED_CHECK '+json.dumps(dict(nq=nq,nk=nk,hk=hk,checks=checks)))


@pytest.mark.parametrize('hk',[4,2,1])
def test_combined_packed_isolation(hk):
    qlens=(17,65,1,33,1);klens=(17,128,65,98,130)
    cq=torch.tensor([0,*torch.tensor(qlens).cumsum(0).tolist()],device='cuda',dtype=torch.int32)
    ck=torch.tensor([0,*torch.tensor(klens).cumsum(0).tolist()],device='cuda',dtype=torch.int32)
    mq,mk=tuple(cq.tolist()),tuple(ck.tolist())
    q,k,v,dout=make_inputs(sum(qlens),sum(klens),4,hk)
    out,lse,_,rng=packed_forward(q,k,v,cq,ck,SCALE)
    fused=packed_backward(q,k,v,out,dout,lse,cq,ck,rng,SCALE)
    state=prepare_backward(q,k,v,out,dout,lse,cu_seqlens_q=mq,cu_seqlens_k=mk)
    dq,current_k,current_v=backward_dq_current(state)
    hk_full,hv_full=backward_dkv(state,'history')
    refs=[[],[],[]]
    for i,(nq,nk) in enumerate(zip(qlens,klens)):
        qa,qb=mq[i:i+2];ka,kb=mk[i:i+2]
        hk_full[kb-nq:kb]=current_k[qa:qb];hv_full[kb-nq:kb]=current_v[qa:qb]
        _,ref=math_ref(q[qa:qb],k[ka:kb],v[ka:kb],dout[qa:qb],SCALE)
        for dst,t in zip(refs,ref):dst.append(t)
    checks=check_accuracy((dq,hk_full,hv_full),fused,[torch.cat(t) for t in refs])
    isolated=torch.zeros_like(dout);isolated[mq[1]:mq[2]]=dout[mq[1]:mq[2]]
    iso=prepare_backward(q,k,v,out,isolated,lse,cu_seqlens_q=mq,cu_seqlens_k=mk)
    for grad in backward_dq_current(iso):
        assert torch.count_nonzero(grad[:mq[1]])==0
        assert torch.count_nonzero(grad[mq[2]:])==0
    print('COMBINED_PACKED_CHECK '+json.dumps(dict(hk=hk,checks=checks)))


@pytest.mark.parametrize('hq,hk,nk',[(4,4,6144),(4,2,6144),(4,1,6144),
                                   (16,8,6144),(4,2,14336),(16,8,14336),
                                   (32,16,6144),(32,16,14336)])
def test_combined_long(hq,hk,nk):
    q,k,v,dout=make_inputs(2048,nk,hq,hk)
    out,lse,rng=fa_forward(q,k,v,SCALE)
    state=prepare_backward(q,k,v,out,dout,lse)
    dq,ck,cv=backward_dq_current(state)
    history_k,history_v=backward_dkv(state,'history',compact=True)
    fused=fa_backward(q,k,v,out,dout,lse,rng,SCALE)
    checks={}
    offset=nk-2048
    for name,a,b in zip(('dq','current_dk','current_dv','history_dk','history_dv'),
                        (dq,ck,cv,history_k,history_v),
                        (fused[0],fused[1][offset:],fused[2][offset:],
                         fused[1][:offset],fused[2][:offset])):
        torch.testing.assert_close(a,b,atol=.004,rtol=.01)
        checks[name]=errors(a,b)
        assert checks[name]['relative_l2']<=.012
    print('COMBINED_LONG_CHECK '+json.dumps(dict(hq=hq,hk=hk,nk=nk,checks=checks)))


@pytest.mark.parametrize('mutated',['q','k','v','lse','delta','metadata'])
def test_combined_state_guards(mutated):
    q,k,v,dout=make_inputs(17,80,4,2)
    out,lse,_=fa_forward(q,k,v,SCALE)
    cq=torch.tensor([0,8,17],dtype=torch.int32);ck=torch.tensor([0,40,80],dtype=torch.int32)
    state=prepare_backward(q,k,v,out,dout,lse,cu_seqlens_q=cq,cu_seqlens_k=ck)
    if mutated=='metadata':ck[1]=39
    elif mutated=='delta':state.sequences[0].delta.add_(1)
    else:getattr(state,mutated).add_(1)
    with pytest.raises(RuntimeError,match='modified'):backward_dq_current(state)


def test_combined_stream_and_zero():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        q,k,v,dout=make_inputs(65,194,4,1)
        out,lse,_=fa_forward(q,k,v,SCALE)
        state=prepare_backward(q,k,v,out,torch.zeros_like(dout),lse)
        assert all(torch.count_nonzero(g)==0 for g in backward_dq_current(state))
    with pytest.raises(RuntimeError,match='same CUDA stream'):backward_dq_current(state)


def test_combined_single_main_launch(tmp_path):
    q,k,v,dout=make_inputs(65,130,4,2)
    out,lse,_=fa_forward(q,k,v,SCALE)
    state=prepare_backward(q,k,v,out,dout,lse)
    backward_dq_current(state);torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        backward_dq_current(state)
        torch.cuda.synchronize()
    path=tmp_path/'combined.pt.trace.json';prof.export_chrome_trace(str(path))
    kernels=[e for e in json.loads(path.read_text())['traceEvents'] if e.get('cat')=='kernel']
    mains=[e for e in kernels if 'dq_current_kernel' in e['name']]
    assert len(mains)==1 and mains[0]['args']['grid']==[3,1,4]
    assert not any('split_kernel<' in e['name'] or 'flash_bwd_dq_dk_dv' in e['name'] for e in kernels)
