"""Combine independently admitted old/resumed windows, preserving raw audits."""
import hashlib,json,sys
from pathlib import Path

old,new=map(Path,sys.argv[1:3])
a=json.loads((old/'interruption-review.json').read_text())
b=json.loads((new/'timing-review.json').read_text())
assert a['status']=='timing-partial-reviewed-external-interruption'
assert b['status']=='timing-resume-subset-reviewed-await-merge'
for key in ['source','arithmetic','runtime_correction','runtime_entry_sha256','model','workload','input_identity']:
    assert a[key]==b[key], key
assert set(a['valid_job_names'])==set(b['resume']['valid_job_names'])
assert set(b['resume']['expected_job_names'])=={w['name'] for w in b['windows']}
old_inputs=json.loads((old/a['windows'][0]['name']/'wrapper-rank0.json').read_text())['inputs']
new_inputs=json.loads((new/b['windows'][0]['name']/'wrapper-rank0.json').read_text())['inputs']
assert old_inputs==new_inputs
rows=a['windows']+b['windows'];assert len(rows)==len({w['name'] for w in rows})==16
groups={}
for head in ['mha','gqa1']:
    windows=sorted([w for w in rows if w['name'].startswith(head+'-')],key=lambda w:w['name'])
    assert len(windows)==8
    methods={}
    for backend in ['KV','QO']:
        w=[x for x in windows if x['name'].endswith('-'+backend)];assert len(w)==4
        methods[backend]=dict(tokens_per_second=sum(x['valid_tokens'] for x in w)/sum(x['seconds'] for x in w),
                             peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in w),
                             peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in w))
    ratios=[]
    for block in [windows[:4],windows[4:]]:
        rates={k:sum(x['valid_tokens'] for x in block if x['name'].endswith('-'+k))/sum(x['seconds'] for x in block if x['name'].endswith('-'+k)) for k in ['KV','QO']}
        ratios.append(rates['QO']/rates['KV']-1)
    groups[head]=dict(methods=methods,QO_vs_KV=methods['QO']['tokens_per_second']/methods['KV']['tokens_per_second']-1,pair_block_QO_vs_KV=ratios)
result=dict(b,status='timing-reviewed-numerical-errors-unresolved',windows=rows,groups=groups,
            scope='11 independently reviewed original windows plus5 admitted resume windows; original contaminated audit and excluded row retained unchanged. Numerical errors remain, not numerical certification.',
            provenance=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [old/'interruption-review.json',new/'timing-review.json']])
(new/'timing-review-combined.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(groups,indent=2))
