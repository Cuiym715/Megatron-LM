"""JSON-only gate for the actual baseline domain; no training or CUDA."""
import json,os,time
from pathlib import Path
ROOT=Path('/workspace/outputs/dspp-cp-v1-20261106/numeric-v5-longmix512')
def main():
    out=ROOT/'actual-domain-review.json';assert not out.exists()
    start=time.time();s=json.loads((ROOT/'manifest.json').read_text())
    assert s['status']=='complete' and s['production_patch_unchanged'] and len(s['jobs'])==6
    audit=json.loads((ROOT/'host-ownership/manifest.json').read_text());assert audit['status']=='complete' and not audit['unknown_observed']
    assert s['actual_domain']['valid_tokens']==88064 and s['actual_domain']['max_length']==32768 and s['actual_domain']['vocab']==512
    rows=[]
    for j in s['jobs']:
        assert j['exit_code']==0 and j['validated'] and not j['polluted']
        p=Path(j['output']);native=[json.loads(f.read_text()) for f in p.glob('iteration*-rank*.json')]
        masters=[json.loads(f.read_text()) for f in p.glob('full-master-iteration*-rank*.json')]
        assert len(native)==len(masters)==8 and {r['rank'] for r in native}==set(range(8))
        assert all(r['iteration']==1 and r['criteria']['relative_l2_max']==.02 and r['diagnostic_deterministic_fa'] for r in native)
        assert next(r for r in native if r['rank']==0)['metrics']['valid_tokens']==88064
        assert all(v['passed'] for r in native+masters for v in r.values() if isinstance(v,dict) and 'passed' in v)
        history={r['rank']:r['history_executed']['nonempty_parts'] for r in masters}
        if 'immediate' in j['name']:assert all(history.values())
        rows.append(dict(name=j['name'],max_actual_update_relative_l2=max((r['update']['relative_l2'] for r in masters if 'update' in r),default=None),history_parts=history))
    out.write_text(json.dumps(dict(status='accepted-actual-domain',pid=os.getpid(),started=start,ended=time.time(),
        jobs=rows,actual_domain=s['actual_domain'],data_sha256=s['data_sha256'],source=s['source']['sources']['current'],
        arithmetic='exact-p0/full-GEMM-reduction/deterministic-FA2-backward',scope='first aligned complete update only; not continuous trajectory or throughput'),indent=2)+'\n')
if __name__=='__main__':main()
