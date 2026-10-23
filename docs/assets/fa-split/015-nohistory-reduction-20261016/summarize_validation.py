import json
from pathlib import Path
root=Path(__file__).resolve().parent
result={}
for hk in (4,2,1):
    files=sorted((root/'pp-validation'/f'hkv{hk}-split').glob('iteration*-rank*.json'))
    assert len(files)==9,(hk,len(files))
    records=[json.loads(p.read_text()) for p in files]
    maxima={}
    for field in ('gradient_checks','stage_dx_checks'):
        checks=[v for r in records for v in r[field].values()]
        maxima[field]=dict(max_abs=max(v['max_abs'] for v in checks),
            max_relative_l2=max(v['relative_l2'] or 0 for v in checks))
    result[str(hk)]=dict(records=len(records),
        parameter_gradient_tensors=sum(r['gradient_tensor_count'] for r in records),
        stage_dx_tensors=sum(r['stage_dx_count'] for r in records),
        actual_split_calls=sum(r['split_calls'] for r in records),
        nohistory_fused_calls=sum(r['nohistory_fused_calls'] for r in records),
        H_tasks=sum(r['history_tasks'] for r in records),
        maxima=maxima)
(root/'validation-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
