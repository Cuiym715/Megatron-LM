"""Locate the captured output difference relative to actual CP ownership."""
import os,fcntl,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/workspace/experiments/dspp-cp-v1-20261106')
lock=(ROOT/'numeric.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
import torch
torch.set_num_threads(1)
BASE=Path('/workspace/outputs/dspp-cp-v1-20261106');OUT=BASE/'numeric-v5-first-attention-CPU-v1'
prior=json.loads((OUT/'result.json').read_text());meta=prior['meta'];indices=prior['indices'];n=sum(i['token_length'] for i in meta['items'])
assert len(meta['items'])==1 and meta['items'][0]['token_offset']==0 and all(v['exact'] for v in prior['QKV'].values())
values={}
for backend in ['kv','qo']:
    full=None
    for rank in [0,1]:
        p=BASE/f'numeric-v5-first-attention-{backend}-v1/first-attention/probe-iteration1-rank{rank}.pt'
        d=torch.load(p,map_location='cpu',weights_only=True,mmap=True)['attention']['l0:p0:c0:layer1:out']
        if full is None:full=d.new_zeros((n,*d.shape[1:]))
        full[torch.tensor(indices[rank],dtype=torch.long)]=d[:len(indices[rank])]
    values[backend]=full
owner={i:r for r,row in enumerate(indices) for i in row};bounds=[0]+[i for i in range(1,n) if owner[i]!=owner[i-1]]+[n]
rows=[]
for a,b in zip(bounds,bounds[1:]):
    x,y=values['qo'][a:b].double(),values['kv'][a:b].double();delta=x-y
    rows.append(dict(begin=a,end=b,query_owner=owner[a],key_owners_in_causal_prefix=sorted({owner[i] for i in range(b)}),
                     exact=torch.equal(x,y),relative_l2=float(delta.norm()/y.norm().clamp_min(1e-24)),max_abs=float(delta.abs().max())))
changed=(values['qo']!=values['kv']).flatten(1).any(1).nonzero().flatten()
assert not torch.cuda.is_initialized()
result=dict(status='complete-CPU-owner-boundary-review',cuda_initialized=False,
            first_differing_global_query=int(changed[0]) if changed.numel() else None,ranges=rows,
            scope='Same captured first attention only; relation to ownership is evidence, not proof of full-model root cause or correctness.')
(OUT/'owner-range-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
