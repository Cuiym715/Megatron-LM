#!/usr/bin/env python3
"""Stage B attention-only timing, real GPU trace, and chunk-chain evidence.

No production model/autograd/scheduler integration. GPU-event timings include
all kernels and submission gaps in the current eager API; forward is excluded.
Profiler samples are separate from the repeated, unprofiled timing samples.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import statistics

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
import flash_attn
from flash_attn.flash_attn_interface import _wrapped_flash_attn_forward, _wrapped_flash_attn_backward
from megatron.core.kv_cache.fa_split import prepare_backward, backward_dq, backward_dkv

SCALE = 128**-.5
VARIANTS = {'MHA':16, 'GQA':4, 'MQA':1}
CASES = ('fused','dq_all','regional')


def forward(q,k,v):
    out,lse,_,rng = _wrapped_flash_attn_forward(
        q[None],k[None],v[None],0.,SCALE,True,-1,-1,0.,None,False)
    return out[0],lse[0],rng


def fused_backward(q,k,v,dout,out,lse,rng):
    grads = tuple(torch.empty_like(t) for t in (q,k,v))
    _wrapped_flash_attn_backward(dout[None],q[None],k[None],v[None],out[None],lse[None],
                                *(t[None] for t in grads),0.,SCALE,True,-1,-1,0.,None,False,rng)
    return grads


def run(case, inputs, saved, phase=lambda name,fn: fn()):
    q,k,v,dout=inputs;out,lse,rng=saved
    if case=='fused':
        return phase('fused_B',lambda:fused_backward(q,k,v,dout,out,lse,rng))
    state=phase('prepare',lambda:prepare_backward(q,k,v,out,dout,lse))
    if case=='reverse':
        hist=phase('history',lambda:backward_dkv(state,'history'))
        dq=phase('dQ',lambda:backward_dq(state))
        current=phase('current',lambda:backward_dkv(state,'current'))
        return dq,current,hist
    dq=phase('dQ',lambda:backward_dq(state))
    if case=='dq_all':
        dk,dv=phase('all_dKV',lambda:backward_dkv(state))
        return dq,dk,dv
    current=phase('current',lambda:backward_dkv(state,'current'))
    hist=phase('history',lambda:backward_dkv(state,'history'))
    return dq,current,hist


def grads(case,result):
    if case in ('regional','reverse'):
        dq,c,h=result
        return dq,*(a+b for a,b in zip(c,h))
    return result


def metrics(a,b):
    diff=a.double()-b.double(); norm=b.double().norm().item()
    return {'max_abs':diff.abs().max().item(),'relative_l2':diff.norm().item()/norm if norm>1e-8 else None}


def check(case,result,base,hist_len):
    actual=grads(case,result)
    record={}
    for n,a,b in zip(('dQ','dK','dV'),actual,base):
        torch.testing.assert_close(a,b,atol=.004,rtol=.01)
        record[n]=metrics(a,b)
    if case in ('regional','reverse'):
        _,current,history=result
        assert all(torch.count_nonzero(x[:hist_len]).item()==0 for x in current)
        assert all(torch.count_nonzero(x[hist_len:]).item()==0 for x in history)
        record['outside_region_nonzero']=0
    return record


def stats(values):
    ordered=sorted(values)
    def quantile(p):
        x=(len(ordered)-1)*p;i=int(x); f=x-i
        return ordered[i]*(1-f)+ordered[min(i+1,len(ordered)-1)]*f
    return {'median_us':statistics.median(values),'p10_us':quantile(.1),'p90_us':quantile(.9),'samples_us':values}


def benchmark(inputs,saved,repeats,warmup):
    for _ in range(warmup):
        for case in CASES:run(case,inputs,saved)
    torch.cuda.synchronize()
    names={'fused':['fused_B'],'dq_all':['prepare','dQ','all_dKV'],
           'regional':['prepare','dQ','current','history']}
    events={c:{n:(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
               for n in ['total',*names[c]]} for c in CASES}
    data={c:{n:[] for n in events[c]} for c in CASES}
    data['regional']['through_current']=[]
    for i in range(repeats):
        # Rotate order so one implementation is not always first/last.
        for case in CASES[i%3:]+CASES[:i%3]:
            def phase(n,fn):
                start,end=events[case][n];start.record();result=fn();end.record();return result
            start,end=events[case]['total']
            start.record();result=run(case,inputs,saved,phase);end.record();end.synchronize()
            for n,(a,b) in events[case].items():data[case][n].append(a.elapsed_time(b)*1000)
            if case=='regional':
                data[case]['through_current'].append(start.elapsed_time(events[case]['current'][1])*1000)
            del result
    return {c:{n:stats(v) for n,v in phases.items()} for c,phases in data.items()}


def parse_trace(path):
    events=json.loads(path.read_text())['traceEvents']
    ranges=[e for e in events if e.get('ph')=='X' and e.get('name','').startswith('B_EVIDENCE/')]
    runtimes={e['args']['correlation']:e for e in events if e.get('cat')=='cuda_runtime' and 'correlation' in e.get('args',{})}
    result={variant:{case:[] for case in (*CASES,'reverse')} for variant in VARIANTS}
    for e in events:
        if e.get('ph')!='X' or e.get('cat') not in ('kernel','gpu_memset','gpu_memcpy'):continue
        runtime=runtimes.get(e.get('args',{}).get('correlation'))
        owners=[] if runtime is None else [r for r in ranges if r['pid']==runtime['pid'] and r['tid']==runtime['tid'] and r['ts']<=runtime['ts']<=r['ts']+r['dur']]
        if not owners:raise RuntimeError('Unattributed GPU event: '+e['name'])
        owner=min(owners,key=lambda r:r['dur'])
        _,variant,case,phase=owner['name'].split('/')
        result[variant][case].append({'phase':phase,'name':e['name'],'start_us':e['ts'],'duration_us':e['dur'],
                                    'category':e['cat'],'grid':e.get('args',{}).get('grid'),
                                    'stream':e.get('args',{}).get('stream'),'correlation':e.get('args',{}).get('correlation')})
    for cases in result.values():
        for rows in cases.values():
            rows.sort(key=lambda x:x['start_us']);origin=rows[0]['start_us']
            for row in rows:row['start_us']-=origin
    return result


def launch_checks(timeline):
    checks={}
    expected_tiles={'dQ':96,'all_dKV':96,'current':32,'history':64}
    for variant,cases in timeline.items():
        checks[variant]={}
        for case,rows in cases.items():
            mains=[e for e in rows if 'dspp_split_detail::split_kernel<' in e['name']]
            if case=='fused':
                assert not mains
                assert sum('flash_bwd_dq_dk_dv' in e['name'] for e in rows)==1
            else:
                assert not any('flash_bwd_dq_dk_dv' in e['name'] for e in rows)
                assert len(mains)==(2 if case=='dq_all' else 3)
                for e in mains:
                    assert e['grid']==[expected_tiles[e['phase']],1,16]
                    want='true' if e['phase']=='dQ' else 'false'
                    assert f'<{want}>' in e['name'] or f'<(bool){int(want=="true")}>' in e['name']
            checks[variant][case]=[{'phase':e['phase'],'grid':e['grid'],'duration_us':e['duration_us']} for e in mains]
    return checks


def math_chain(q,k,v,dout):
    leaves=[t.detach().double().requires_grad_(True) for t in (q,k,v)]
    qt=leaves[0].transpose(0,1);ratio=q.size(1)//k.size(1)
    kt,vt=[t.repeat_interleave(ratio,1).transpose(0,1) for t in leaves[1:]]
    mask=torch.ones(q.size(0),q.size(0),device=q.device,dtype=torch.bool).tril()
    probs=(qt@kt.transpose(-1,-2)*SCALE).masked_fill(~mask,-torch.inf).softmax(-1)
    out=(probs@vt).transpose(0,1)
    return torch.autograd.grad(out,leaves,dout.double())


def chain_evidence():
    records={}
    for variant,hk in VARIANTS.items():
        gen=torch.Generator(device='cuda').manual_seed(2026)
        q,k,v,dout=[torch.randn(145,h,128,device='cuda',generator=gen).to(torch.bfloat16)
                   for h in (16,hk,hk,16)]
        saved=forward(q,k,v)
        full=fused_backward(q,k,v,dout,*saved)
        ref=math_chain(q,k,v,dout)
        dq_parts=[];dq_fused=[]
        dk=torch.zeros_like(k,dtype=torch.float32);dv=torch.zeros_like(v,dtype=torch.float32)
        base_k=torch.zeros_like(dk);base_v=torch.zeros_like(dv)
        per_chunk=[]
        for begin,end in ((0,63),(63,128),(128,145)):
            inp=(q[begin:end],k[:end],v[:end],dout[begin:end]);sv=forward(*inp[:3])
            base=run('fused',inp,sv);split=run('regional',inp,sv)
            per_chunk.append(check('regional',split,base,begin))
            dqi,current,history=split;dq_parts.append(dqi);dq_fused.append(base[0])
            for dst,c,h,b in zip((dk,dv),current,history,base[1:]):
                dst[:end].add_(c.float()).add_(h.float())
            base_k[:end].add_(base[1].float());base_v[:end].add_(base[2].float())
        combined=(torch.cat(dq_parts),dk,dv)
        chunk_fused=(torch.cat(dq_fused),base_k,base_v)
        record={'lengths':[63,65,17],'per_chunk_vs_fused':per_chunk,'combined':{}}
        for name,a,b,f,r in zip(('dQ','dK','dV'),combined,chunk_fused,full,ref):
            torch.testing.assert_close(a,b,atol=.004,rtol=.01)
            # Chunk-local BF16 rounding and FP32 cross-chunk accumulation differ
            # from single fused backward. Evaluate both against FP64 under the
            # already-declared mathematical accuracy criteria, not exact equality.
            for value in (a,b,f):
                torch.testing.assert_close(value.double(),r,atol=.04,rtol=.04)
                assert metrics(value,r)['relative_l2']<=.012
            record['combined'][name]={'vs_chunk_fused':metrics(a,b),'vs_full_fused':metrics(a,f),
                                      'vs_fp64':metrics(a,r),'full_fused_vs_fp64':metrics(f,r)}
        records[variant]=record
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--warmup',type=int,default=10);args=p.parse_args()
    if args.repeats<10 or args.warmup<3:p.error('at least 10 repeats / 3 warmups required')
    args.output.mkdir(parents=True,exist_ok=False)
    result={'scope':'Isolated chunk attention, not stage B task or end-to-end training.',
            'environment':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'fa':flash_attn.__version__},
            'config':{'chunk_size':2048,'q_heads':16,'head_dim':128,'dtype':'BF16','causal':True,'dropout':0,'seed':1234,'repeats':args.repeats,'warmup':args.warmup},
            'timing_method':'CUDA events, unprofiled eager API, median and p10/p90 over repeats; forward and result combination excluded. Includes allocation-triggered device work, zeroing, conversion, GQA reduction, launch/submission gaps. Total measured directly, not sum of phase medians. Cases rotated each repeat.',
            'benchmarks':{},'accuracy':{}}
    last_inputs={}
    for variant,hk in VARIANTS.items():
        result['benchmarks'][variant]={};result['accuracy'][variant]={}
        # Full 6144-token input sliced into three chunks; prefix KV grows.
        gen=torch.Generator(device='cuda').manual_seed(1234)
        full=[torch.randn(6144,h,128,generator=gen,device='cuda').to(torch.bfloat16) for h in (16,hk,hk,16)]
        for chunk in (1,2,3):
            begin=(chunk-1)*2048;end=chunk*2048
            inputs=(full[0][begin:end],full[1][:end],full[2][:end],full[3][begin:end])
            saved=forward(*inputs[:3]);base=run('fused',inputs,saved)
            result['accuracy'][variant][str(chunk)]={case:check(case,run(case,inputs,saved),base,begin) for case in ('dq_all','regional','reverse')}
            result['benchmarks'][variant][str(chunk)]=benchmark(inputs,saved,args.repeats,args.warmup)
            print('TIMING',variant,chunk,{c:round(result['benchmarks'][variant][str(chunk)][c]['total']['median_us'],2) for c in CASES},flush=True)
            if chunk==3:last_inputs[variant]=inputs
    result['chunk_chain']=chain_evidence()
    # Profiler collected separately, after unprofiled timing.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        for variant,inputs in last_inputs.items():
            for case in (*CASES,'reverse'):
                def phase(n,fn):
                    with torch.profiler.record_function(f'B_EVIDENCE/{variant}/{case}/{n}'):return fn()
                saved=phase('F_attn',lambda:forward(*inputs[:3]))
                out=run(case,inputs,saved,phase)
                torch.cuda.synchronize()
    raw=args.output/'execution.pt.trace.json';prof.export_chrome_trace(str(raw))
    result['timeline']=parse_trace(raw);result['launch_checks']=launch_checks(result['timeline'])
    files=[REPO/'megatron/core/kv_cache/fa_split.py',REPO/'megatron/core/kv_cache/fa_split_build.py',
           *sorted((REPO/'megatron/core/kv_cache/fa_split_src').glob('*')),Path(__file__).resolve()]
    result['source_sha256']={str(f.relative_to(REPO)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}
    result['raw_trace_sha256']=hashlib.sha256(raw.read_bytes()).hexdigest()
    (args.output/'evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    print('All timing/gradient/region launch checks passed.',flush=True)

if __name__=='__main__':main()
