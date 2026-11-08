"""CPU-only review of already completed control; never repeat GPU for metadata."""
from pathlib import Path
import fcntl,json,os,time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/workspace/experiments/dspp-cp-v1-20261106')
lock=(ROOT/'numeric.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
import torch
torch.set_num_threads(1)
OUT=Path('/workspace/outputs/dspp-cp-v1-20261106/numeric-v5-fp32-o-v1')
OLD=Path('/workspace/outputs/dspp-cp-v1-20261106/numeric-v5-longmix512')
state=json.loads((OUT/'manifest.json').read_text())
assert state['status']=='failed' and state['error']=="KeyError('valid_tokens')" and state['production_patch_unchanged']
path=OUT/'qo-fp32-o'
native=[json.loads((path/f'iteration1-rank{r}.json').read_text()) for r in range(8)]
masters=[json.loads((path/f'full-master-iteration1-rank{r}.json').read_text()) for r in range(8)]
assert native[0]['metrics']['valid_tokens']==88064 and all(n['diagnostic_deterministic_fa'] for n in native)
assert all(m['before']['passed'] and m['before']['exact'] and m['before']['relative_l2']==0 for m in masters)
assert len(list(path.glob('fp32-o-patch-rank*.json')))==8
ss=rr=elements=0;per_stage=[]
for rank in (0,2,4,6):
 reference=torch.load(OLD/f'mha/snapshots/full-master-iteration1-rank{rank}.pt',map_location='cpu',weights_only=True,mmap=True)
 norm=sum(float(x.double().square().sum()) for x in reference['update'].values())
 rel=masters[rank]['update']['relative_l2'];ss+=norm*rel**2;rr+=norm;elements+=masters[rank]['update']['elements']
 per_stage.append(dict(rank=rank,update_relative_l2=rel,reference_update_squared_norm=norm))
 del reference
assert elements==206603264 and not torch.cuda.is_initialized()
result=dict(status='complete-CPU-reviewed-diagnostic',cuda_initialized=False,numerically_certified=False,
 actual_update_relative_l2=(ss/rr)**.5,old_update_relative_l2=0.04123245281043918,
 elements=elements,native_reports=8,master_reports=8,initial_masters_exact=True,
 dx_max_relative_l2=max(n['dx']['max_relative_l2'] for n in native),
 logits_pass=all(n['logits']['passed'] for n in native),per_stage=per_stage,
 retained_criteria=dict(actual_update_relative_l2_max=.02),
 numerical_pass=False,controller_postcheck_error=state['error'],
 correction='Only source pipeline stage0 reports valid_tokens; controller incorrectly required it on every rank. Full GPU evidence already existed, CPU review corrected admission; no GPU rerun.',
 interpretation='FP32 weightedO CPsum alone does not resolve original QO error; no performance/source adoption, shared wall clock is not throughput.')
(OUT/'CPU-review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
