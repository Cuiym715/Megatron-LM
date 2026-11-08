"""CPU reference for captured first attention only; no model/GPU execution."""
import fcntl,json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/workspace/experiments/dspp-cp-v1-20261106')
lock=(ROOT/'numeric.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
import torch
torch.set_num_threads(1)
BASE=Path('/workspace/outputs/dspp-cp-v1-20261106')
OUT=BASE/'numeric-v5-first-attention-CPU-v1'
def read(p):return json.loads(p.read_text())
def error(x,y):
    x,y=x.double(),y.double();d=x-y
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    return dict(exact=torch.equal(x,y),different_elements=int((x!=y).sum()),
                relative_l2=float(d.norm()/y.norm().clamp_min(1e-24)),max_abs=float(d.abs().max()))
states={k:read(BASE/f'numeric-v5-first-attention-{k}-v1/manifest.json') for k in ['kv','qo']}
assert all(s['status']=='complete' and s['production_patch_unchanged'] and s['probe_reports']==8 for s in states.values())
assert states['kv']['source']==states['qo']['source'] and states['kv']['input_identity']==states['qo']['input_identity']
probes={k:[torch.load(BASE/f'numeric-v5-first-attention-{k}-v1/first-attention/probe-iteration1-rank{r}.pt',map_location='cpu',weights_only=True,mmap=True) for r in [0,1]] for k in ['kv','qo']}
task='l0:p0:c0';prefix=task+':layer1:'
meta=probes['kv'][0]['meta'][task]
assert all(p['meta'][task]==meta for group in probes.values() for p in group)
capacity=meta['chunk_size'];valid=sum(i['token_length'] for i in meta['items'])
indices=[probes['kv'][r]['indices'][task] for r in [0,1]]
assert sorted(i for row in indices for i in row)==list(range(valid))
assert all(probes['qo'][r]['indices'][task]==indices[r] for r in [0,1])
assembled={}
for backend,pair in probes.items():
    tensors={}
    for name in ['q','k','v','out']:
        first=pair[0]['attention'][prefix+name]
        target=first.new_zeros((capacity,*first.shape[1:]))
        for rank in [0,1]:
            values=pair[rank]['attention'][prefix+name]
            target[torch.tensor(indices[rank])]=values[:len(indices[rank])]
        tensors[name]=target[:valid]
    assembled[backend]=tensors
linear=[]
for rank in [0,1]:
    a,b=probes['kv'][rank]['tensors'],probes['qo'][rank]['tensors'];assert a.keys()==b.keys() and a
    for name in a:
        for part in ['input','output','weight']:
            linear.append(dict(rank=rank,name=name,part=part,**error(a[name][part],b[name][part])))
result=dict(scope='First actual layer1 attention/task only, not full-model certification or a throughput test',
    source=states['kv']['source'],input_identity=states['kv']['input_identity'],meta=meta,indices=indices,
    QKV={name:error(assembled['qo'][name],assembled['kv'][name]) for name in ['q','k','v']},
    native_output_QO_vs_KV=error(assembled['qo']['out'],assembled['kv']['out']),linear=linear)
if all(x['exact'] for x in result['QKV'].values()) and all(i['token_offset']==0 and i['segment_id']==0 for i in meta['items']):
    q,k,v=(assembled['kv'][name][:,0].transpose(0,1).double() for name in ['q','k','v'])
    assert q.shape[0]%k.shape[0]==0
    k=k.repeat_interleave(q.shape[0]//k.shape[0],dim=0);v=v.repeat_interleave(q.shape[0]//v.shape[0],dim=0)
    reference=torch.zeros_like(q);ideal=torch.zeros_like(q);rounded=torch.zeros_like(q,dtype=torch.float32);transport=torch.zeros_like(q,dtype=torch.float32)
    start=0
    for item in meta['items']:
        n=item['token_length'];qq,kk,vv=q[:,start:start+n],k[:,start:start+n],v[:,start:start+n]
        score=(qq@kk.transpose(-2,-1))*(q.shape[-1]**-.5)
        score.masked_fill_(torch.ones(n,n,dtype=torch.bool).triu(1),-torch.inf)
        reference[:,start:start+n]=score.softmax(-1)@vv
        partials=[];lses=[]
        for row in indices:
            ids=torch.tensor([i-start for i in row if start<=i<start+n],dtype=torch.long)
            if len(ids):
                s=(qq@kk[:,ids].transpose(-2,-1))*(q.shape[-1]**-.5)
                s.masked_fill_(ids[None,:]>torch.arange(n)[:,None],-torch.inf)
                lse=s.logsumexp(-1)
                out=torch.where(torch.isfinite(lse)[...,None],s.softmax(-1)@vv[:,ids],0.)
            else:
                lse=qq.new_full(qq.shape[:-1],-torch.inf);out=torch.zeros_like(qq)
            partials.append(out);lses.append(lse)
        total=torch.stack(lses).logsumexp(0)
        for out,lse in zip(partials,lses):
            weight=(lse-total).exp()[...,None]
            ideal[:,start:start+n]+=out*weight
            weighted=out.to(torch.bfloat16).float()*weight.float()
            rounded[:,start:start+n]+=weighted
            transport[:,start:start+n]+=weighted.to(torch.bfloat16).float()
        start+=n
    def rows(x):return x.transpose(0,1).unsqueeze(1)
    ideal_error=error(rows(ideal),rows(reference));assert ideal_error['relative_l2']<1e-10
    result['FP64_reference']=dict(ideal_partition_consistency=ideal_error,
        KV=error(assembled['kv']['out'],rows(reference)),QO=error(assembled['qo']['out'],rows(reference)),
        KV_vs_rounded_full=error(assembled['kv']['out'],rows(reference.to(torch.bfloat16))),
        QO_vs_partialBF16_and_transportBF16=error(assembled['qo']['out'],rows(transport.to(torch.bfloat16))),
        QO_vs_partialBF16_transportFP32=error(assembled['qo']['out'],rows(rounded.to(torch.bfloat16))),
        limitation='Rounding simulation of first attention only; native LSE/FA reductions may differ. Does not prove full-model gradient origin or adopt any source change.')
else:
    result['FP64_reference']=dict(skipped='QKV not exact or first captured task has history; no unsupported reference claim')
assert not torch.cuda.is_initialized();result['cuda_initialized']=False
OUT.mkdir(exist_ok=False);(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['source','input_identity','meta','indices','linear']}))
