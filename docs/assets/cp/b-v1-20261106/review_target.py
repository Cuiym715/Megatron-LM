"""JSON-only review of target smoke, after GPU controller and audit close."""
import json,os,time
from pathlib import Path
ROOT=Path('/workspace/outputs/dspp-cp-v1-20261106/numeric-v4-target')
def main():
    path=ROOT/'target-review.json';assert not path.exists()
    start=time.time();s=json.loads((ROOT/'manifest.json').read_text())
    assert s['status']=='complete' and s['production_patch_unchanged']
    a=json.loads((ROOT/'host-ownership/manifest.json').read_text());assert a['status']=='complete' and not a['unknown_observed']
    summaries=[]
    for job in s['jobs']:
        p=Path(job['output']);records=[json.loads(f.read_text()) for f in p.glob('iteration*-rank*.json')]
        masters=[json.loads(f.read_text()) for f in p.glob('full-master-iteration*-rank*.json')]
        assert len(records)==len(masters)==24
        assert {(r['iteration'],r['rank']) for r in records}=={(i,r) for i in [1,2,3] for r in range(8)}
        assert all(v['passed'] for r in records for v in r.values() if isinstance(v,dict) and 'passed' in v)
        det='kv-repeat' not in job['name']
        if det:
            assert job['exit_code']==0 and job['validated']
            assert all(v['passed'] for r in masters for v in r.values() if isinstance(v,dict) and 'passed' in v)
            assert all(r['diagnostic_deterministic_fa'] for r in records)
        history={str(rank):sum(r['history_executed']['nonempty_parts'] for r in masters if r['rank']==rank) for rank in range(8)}
        if 'immediate' in job['name']:assert all(history.values())
        assert all(r['criteria']['relative_l2_max']==.02 and r['aligned_optimizer_and_weights'] for r in records)
        summaries.append(dict(name=job['name'],deterministic_fa=det,exit_code=job['exit_code'],
            reference_only=all(r.get('reference',False) for r in masters),
            actual_update_max_per_rank=max((r['update']['relative_l2'] for r in masters if 'update' in r),default=None),
            actual_update_pass=all(r['update']['passed'] for r in masters if 'update' in r) if any('update' in r for r in masters) else None,
            nonempty_H_parts_by_rank=history))
    assert len(summaries)==5
    result=dict(status='accepted-target-smoke',pid=os.getpid(),started=start,ended=time.time(),jobs=summaries,
        source_archive=s['source']['sources']['current']['archive_sha256'],
        arithmetic='Shared full GEMM reduction, exact p0 residual, deterministic FA2 backward; default-FA selfrepeat retained separately, not certified',
        next='Actual32K longmix-v512 full-update gap before performance; this review is not throughput or a continuous-trajectory proof')
    path.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
