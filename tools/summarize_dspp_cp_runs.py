"""Offline CP evidence index. Only complete rank sets become timing rows."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    cfg=p.parse_args()
    rows,operators,updates=[],[],[]
    for path in sorted(cfg.root.rglob('rank0.json')):
        first=json.loads(path.read_text())
        if first.get('reference')=='independent-fp64':
            reports=[json.loads(f.read_text()) for f in path.parent.glob('rank[0-9].json')]
            operators.append(dict(run=str(path.parent.relative_to(cfg.root)),
                statuses=[r.get('status','incomplete') for r in reports],
                cases_per_rank=[len(r['cases']) for r in reports]))
        if 'ms_per_update' not in first:
            continue
        reports=[json.loads(f.read_text()) for f in path.parent.glob('rank*.json')
                 if f.stem[4:].isdigit()]
        if len(reports)!=first['layout']['pp']*first['cp']:
            continue
        plans=[[(r['iteration'],r['selected'],r['fallback']) for r in v['records']]
               for v in reports]
        assert all(v==plans[0] for v in plans),'rank plan disagreement'
        elapsed=max(v['window_ms'] for v in reports)
        valid=sum(r['metrics'].get('valid_tokens',0) for r in first['records'][first['warmup']:])
        rows.append(dict(run=str(path.parent.relative_to(cfg.root)),
            ms_per_update=elapsed/first['timed'],tokens_per_second=valid*1000/elapsed,
            warmup=first['warmup'],timed=first['timed'],diagnostic=first['diagnostic'],
            cp=first['cp'],pp=first['layout']['pp'],model=first['layout']['model'],
            cp_implementation=first['layout'].get('cp_implementation','unversioned'),
            allocated_peak=max(v['allocated_peak'] for v in reports),
            reserved_peak=max(v['reserved_peak'] for v in reports),
            selected=[r['selected'] for r in first['records']],
            fallback=[r['fallback'] for r in first['records']]))
    groups={}
    for path in cfg.root.rglob('iteration*-rank*.json'):
        relative=path.relative_to(cfg.root)
        if relative.parts[0].startswith('update-'):
            groups.setdefault(str(relative.parent),[]).append(json.loads(path.read_text()))
    for folder,reports in sorted(groups.items()):
        comparisons=[v['passed'] for r in reports for v in r.values()
                     if isinstance(v,dict) and 'passed' in v]
        updates.append(dict(run=folder,reports=len(reports),
                            comparisons=len(comparisons),passed=sum(comparisons)))
    result=dict(root=str(cfg.root),benchmarks=rows,operators=operators,updates=updates,
                numerical_scope='completed JSON comparisons only; failed/incomplete runs retain separate logs')
    with cfg.output.open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps([dict(run=r['run'],ms=r['ms_per_update'],diagnostic=r['diagnostic'],
                          selected=max(r['selected'],default=0)) for r in rows],indent=2))


if __name__=='__main__':
    main()
