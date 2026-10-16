"""Stage B: real CUDA region/GQA/varlen correctness, no performance claims.

Criteria fixed before execution: split vs fused BF16 atol=.004, rtol=.01;
both vs FP64 atol=.04, rtol=.04 and relative L2<=.012 (math-zero abs<=1e-5).
Region outputs are disjoint and must sum EXACTLY to the complete dKV output.
"""
import json

import pytest
import torch

from megatron.core.kv_cache.fa_split import prepare_backward, backward_dq, backward_dkv
from .test_fa_split import fa_forward, fa_backward, errors

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
SCALE = 128**-0.5


def make_inputs(nq, nk, hq, hk, seed=2026):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(torch.randn(n, h, 128, device="cuda", generator=gen).to(torch.bfloat16)
                 for n, h in ((nq, hq), (nk, hk), (nk, hk), (nq, hq)))


def math_ref(q, k, v, dout, scale):
    leaves = [t.detach().double().requires_grad_(True) for t in (q, k, v)]
    qt = leaves[0].transpose(0, 1)
    ratio = q.size(1) // k.size(1)
    # Independent reference: explicitly expand KV heads, autograd sums their
    # gradients. The production extension does not expand input KV heads.
    kt, vt = [t.repeat_interleave(ratio, dim=1).transpose(0, 1) for t in leaves[1:]]
    mask = (torch.arange(k.size(0), device=q.device)[None, :] <=
            torch.arange(q.size(0), device=q.device)[:, None] + k.size(0)-q.size(0))
    probs = (qt @ kt.transpose(-1, -2) * scale).masked_fill(~mask, -torch.inf).softmax(-1)
    out = (probs @ vt).transpose(0, 1)
    return out.detach(), torch.autograd.grad(out, leaves, dout.double())


def check_accuracy(actual, fused, refs):
    record = {}
    for name, got, base, ref in zip(("dq", "dk", "dv"), actual, fused, refs):
        assert got.shape == base.shape and got.dtype == torch.bfloat16
        assert torch.isfinite(got).all() and not got.requires_grad
        torch.testing.assert_close(got, base, atol=.004, rtol=.01)
        for value in (got, base):
            torch.testing.assert_close(value.double(), ref, atol=.04, rtol=.04)
            err = errors(value, ref)
            assert (err['relative_l2'] <= .012 if err['relative_l2'] is not None
                    else err['max_abs'] <= 1e-5)
        record[name] = {"split_vs_fused": errors(got, base),
                        "split_vs_fp64": errors(got, ref),
                        "fused_vs_fp64": errors(base, ref)}
    return record


def check_regions(state, boundaries):
    # History first, then dQ/current: no dependency on a previous full dKV call.
    hist = backward_dkv(state, "history")
    dq = backward_dq(state)
    current = backward_dkv(state, "current")
    all_kv = backward_dkv(state)
    mask = torch.zeros(state.k.size(0), dtype=torch.bool, device=state.k.device)
    for begin, split, end in boundaries:
        mask[begin:split] = True
    for h, c, full in zip(hist, current, all_kv):
        assert torch.count_nonzero(h[~mask]).item() == 0
        assert torch.count_nonzero(c[mask]).item() == 0
        torch.testing.assert_close(h + c, full, atol=0, rtol=0)
    return (dq, *(h+c for h, c in zip(hist, current)))


@pytest.mark.parametrize("hq,hk", [(16,16), (16,8), (16,4), (16,1)])
@pytest.mark.parametrize("nq,nk,scale", [
    (1,1,SCALE), (17,17,SCALE), (1,130,SCALE), (17,80,SCALE),
    (64,128,SCALE), (65,130,SCALE), (129,257,.037),
])
def test_regions_gqa_fused_and_fp64(nq, nk, hq, hk, scale):
    q, k, v, dout = make_inputs(nq, nk, hq, hk)
    out, lse, rng = fa_forward(q, k, v, scale)
    fused = fa_backward(q, k, v, out, dout, lse, rng, scale)
    state = prepare_backward(q, k, v, out, dout, lse, softmax_scale=scale)
    actual = check_regions(state, [(0,nk-nq,nk)])
    ref_out, refs = math_ref(q,k,v,dout,scale)
    torch.testing.assert_close(out.double(), ref_out, atol=.04, rtol=.04)
    record = check_accuracy(actual, fused, refs)
    # A deliberate missing-history dQ must differ for nontrivial historical cases.
    if nk > nq and nq > 1:
        _, wrong = math_ref(q, k[-nq:], v[-nq:], dout, scale)
        assert not torch.allclose(actual[0].double(), wrong[0], atol=.004, rtol=.01)
    print('REGION_CHECK ' + json.dumps(dict(nq=nq,nk=nk,hq=hq,hk=hk,scale=scale,gradients=record)))


def packed_forward(q,k,v,cq,ck,scale):
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_varlen_forward
    return _wrapped_flash_attn_varlen_forward(
        q,k,v,cq,ck,int((cq[1:]-cq[:-1]).max()),int((ck[1:]-ck[:-1]).max()),
        0.,scale,True,-1,-1,0.,None,False,None,None,None,False)


def packed_backward(q,k,v,out,dout,lse,cq,ck,rng,scale):
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_varlen_backward
    grads = tuple(torch.empty_like(t) for t in (q,k,v))
    _wrapped_flash_attn_varlen_backward(
        dout,q,k,v,out,lse,*grads,cq,ck,
        int((cq[1:]-cq[:-1]).max()),int((ck[1:]-ck[:-1]).max()),
        0.,scale,True,-1,-1,0.,None,False,rng,False)
    return grads


@pytest.mark.parametrize("hq,hk", [(16,16),(16,8),(16,4),(16,1)])
@pytest.mark.parametrize("metadata", ["cuda", "cpu", "tuple"])
def test_varlen_regions_gqa_and_isolation(hq,hk,metadata):
    # Independent sequences: no history; 63/64/65-token history; one-token tail.
    qlens = (17,65,1,33,1)
    klens = (17,128,65,98,130)
    cq = torch.tensor([0,*torch.tensor(qlens).cumsum(0).tolist()], device='cuda',dtype=torch.int32)
    ck = torch.tensor([0,*torch.tensor(klens).cumsum(0).tolist()], device='cuda',dtype=torch.int32)
    q,k,v,dout = make_inputs(sum(qlens),sum(klens),hq,hk)
    out,lse,_,rng = packed_forward(q,k,v,cq,ck,SCALE)
    fused = packed_backward(q,k,v,out,dout,lse,cq,ck,rng,SCALE)
    mq,mk = (cq,ck) if metadata=='cuda' else (cq.cpu(),ck.cpu())
    if metadata=='tuple': mq,mk = tuple(mq.tolist()),tuple(mk.tolist())
    state=prepare_backward(q,k,v,out,dout,lse,cu_seqlens_q=mq,cu_seqlens_k=mk)
    boundaries=[]; refs=[[],[],[]]; qa=ka=0
    for nq,nk in zip(qlens,klens):
        boundaries.append((ka,ka+nk-nq,ka+nk))
        _,r=math_ref(q[qa:qa+nq],k[ka:ka+nk],v[ka:ka+nk],dout[qa:qa+nq],SCALE)
        for dst,t in zip(refs,r): dst.append(t)
        qa+=nq; ka+=nk
    actual=check_regions(state,boundaries)
    record=check_accuracy(actual,fused,[torch.cat(t) for t in refs])
    # Set dO to zero for all except sequence 2: no gradient may leak across it.
    isolated=torch.zeros_like(dout); isolated[17:82]=dout[17:82]
    iso=prepare_backward(q,k,v,out,isolated,lse,cu_seqlens_q=mq,cu_seqlens_k=mk)
    iso_grads=backward_dq(iso),*backward_dkv(iso)
    for g,a,b in zip(iso_grads,(17,17,17),(82,145,145)):
        assert torch.count_nonzero(g[:a]).item()==0
        assert torch.count_nonzero(g[b:]).item()==0
    print('VARLEN_CHECK '+json.dumps(dict(hq=hq,hk=hk,metadata=metadata,qlens=qlens,klens=klens,gradients=record)))


def test_region_repeat_order_zero_and_lifetime():
    q,k,v,dout=make_inputs(65,194,16,4)
    out,lse,_=fa_forward(q,k,v,SCALE)
    state=prepare_backward(q,k,v,out,dout,lse)
    snapshots=[t.clone() for t in state.tensors()]
    baseline=check_regions(state,[(0,129,194)])
    for _ in range(3):
        current=backward_dkv(state,'current'); hist=backward_dkv(state,'history')
        for got,expected in zip((backward_dq(state),*(c+h for c,h in zip(current,hist))),baseline):
            torch.testing.assert_close(got,expected,atol=.004,rtol=.01)
    zero=prepare_backward(q,k,v,out,torch.zeros_like(dout),lse)
    for region in ('history','current','all'):
        for g in backward_dkv(zero,region): assert torch.count_nonzero(g).item()==0
    assert torch.count_nonzero(backward_dq(zero)).item()==0
    for t,s in zip(state.tensors(),snapshots): torch.testing.assert_close(t,s,atol=0,rtol=0)
    state.k.add_(1)
    with pytest.raises(RuntimeError,match='modified'): backward_dkv(state,'history')


@pytest.mark.parametrize('bad', ['missing','count','nonmonotonic','end','negative_history','float','bool','empty','wrong_device'])
def test_reject_invalid_varlen_metadata(bad):
    q,k,v,dout=make_inputs(17,80,16,4)
    out,lse,_=fa_forward(q,k,v,SCALE)
    cq,ck=(0,17),(0,80)
    if bad=='missing': ck=None
    if bad=='count': ck=(0,10,80)
    if bad=='nonmonotonic': cq=(0,18,17);ck=(0,60,80)
    if bad=='end': ck=(0,79)
    if bad=='negative_history': cq=(0,16,17);ck=(0,15,80)
    if bad=='float': cq=(0,17.)
    if bad=='bool': cq=(False,17)
    if bad=='empty': cq=(0,0,17);ck=(0,10,80)
    if bad=='wrong_device': cq=torch.tensor([0,17],device='meta',dtype=torch.int32)
    with pytest.raises(ValueError):
        prepare_backward(q,k,v,out,dout,lse,cu_seqlens_q=cq,cu_seqlens_k=ck)


@pytest.mark.parametrize('mutated,metadata',[
    ('lse','tensor'),('cu','tensor'),('delta','tensor'),('lse','tuple'),('delta','tuple')])
def test_varlen_mutation_guard(mutated,metadata):
    q,k,v,dout=make_inputs(17,80,16,4)
    out,lse,_=fa_forward(q,k,v,SCALE)
    cq=torch.tensor([0,8,17],dtype=torch.int32);ck=torch.tensor([0,40,80],dtype=torch.int32)
    # Values of O/LSE need not be recomputed: this test never launches backward.
    state=prepare_backward(q,k,v,out,dout,lse,
                           cu_seqlens_q=tuple(cq.tolist()) if metadata=='tuple' else cq,
                           cu_seqlens_k=tuple(ck.tolist()) if metadata=='tuple' else ck)
    if mutated=='lse': lse.add_(1)
    if mutated=='cu': ck[1]=39
    if mutated=='delta': state.sequences[0].delta.add_(1)
    with pytest.raises(RuntimeError,match='modified'): backward_dkv(state,'current')


def test_region_stream_and_invalid_region():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        q,k,v,dout=make_inputs(17,80,16,1)
        out,lse,rng=fa_forward(q,k,v,SCALE)
        state=prepare_backward(q,k,v,out,dout,lse)
        got=check_regions(state,[(0,63,80)])
        fused=fa_backward(q,k,v,out,dout,lse,rng,SCALE)
        for a,b in zip(got,fused): torch.testing.assert_close(a,b,atol=.004,rtol=.01)
        with pytest.raises(ValueError,match='region'): backward_dkv(state,'invalid')
    with pytest.raises(RuntimeError,match='same CUDA stream'): backward_dkv(state,'history')
    stream.synchronize()


@pytest.mark.parametrize('nq,nk',[(129,1025),(129,1026),(17,17)])
def test_region_launches_only_intersecting_tiles(nq,nk,tmp_path):
    q,k,v,dout=make_inputs(nq,nk,16,4)
    out,lse,_=fa_forward(q,k,v,SCALE)
    state=prepare_backward(q,k,v,out,dout,lse)
    backward_dkv(state)  # Warm extension/allocator before inspecting launches.
    torch.cuda.synchronize()
    launches={}
    for region in ('history','current','all'):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            backward_dkv(state,region)
            torch.cuda.synchronize()
        path=tmp_path/f'{region}.json'
        prof.export_chrome_trace(str(path))
        events=json.loads(path.read_text())['traceEvents']
        kernels=[e for e in events if e.get('cat')=='kernel']
        mains=[e for e in kernels if 'dspp_split_detail::split_kernel<' in e['name']]
        h=nk-nq
        begin=h if region=='current' else 0
        end=h if region=='history' else nk
        expected=(end+63)//64-begin//64 if begin<end else 0
        assert len(mains)==int(expected>0)
        assert not any('flash_bwd_dq_dk_dv' in e['name'] or 'convert_dq_kernel' in e['name']
                       for e in kernels)
        if mains:
            assert 'false' in mains[0]['name'] or '(bool)0' in mains[0]['name']
            assert mains[0]['args']['grid']==[expected,1,16]
        launches[region]={'main_launches':len(mains),'kv_tiles':expected,
                          'grid':mains[0]['args']['grid'] if mains else None}
    print('REGION_LAUNCH_CHECK '+json.dumps(dict(nq=nq,nk=nk,hq=16,hk=4,regions=launches)))


def test_varlen_gqa_multiple_devices_and_stream_guard():
    if torch.cuda.device_count()<2:
        pytest.skip('requires a second GPU')
    with torch.cuda.device(1):
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            q,k,v,dout=make_inputs(18,82,16,4)
            cq=torch.tensor([0,1,18],device=q.device,dtype=torch.int32)
            ck=torch.tensor([0,2,82],device=q.device,dtype=torch.int32)
            out,lse,_,rng=packed_forward(q,k,v,cq,ck,SCALE)
            state=prepare_backward(q,k,v,out,dout,lse,cu_seqlens_q=cq,cu_seqlens_k=ck)
            # Switch the current device while preserving device 1's stream.
            with torch.cuda.device(0):
                actual=check_regions(state,[(0,1,2),(2,65,82)])
                fused=packed_backward(q,k,v,out,dout,lse,cq,ck,rng,SCALE)
                for a,b in zip(actual,fused):
                    assert a.device.index==1
                    torch.testing.assert_close(a,b,atol=.004,rtol=.01)
        with pytest.raises(RuntimeError,match='same CUDA stream'):
            backward_dq(state)
        stream.synchronize()


@pytest.mark.parametrize('hk',[4,1])
def test_long_continuation_gqa_against_fused(hk):
    # The earlier research example: current 2048 tokens with two historical
    # chunks (6144 KV tokens). FP64 oracle is covered on small shapes above;
    # here check the production-sized shape against original fused FA2.
    q,k,v,dout=make_inputs(2048,6144,16,hk)
    out,lse,rng=fa_forward(q,k,v,SCALE)
    fused=fa_backward(q,k,v,out,dout,lse,rng,SCALE)
    state=prepare_backward(q,k,v,out,dout,lse)
    actual=check_regions(state,[(0,4096,6144)])
    record={}
    for name,got,base in zip(('dq','dk','dv'),actual,fused):
        torch.testing.assert_close(got,base,atol=.004,rtol=.01)
        record[name]={'split_vs_fused':errors(got,base)}
    print('LONG_CHECK '+json.dumps(dict(nq=2048,nk=6144,hq=16,hk=hk,gradients=record)))
