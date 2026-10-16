#!/usr/bin/env python3
"""Combine independent clean B/H runs by identical stage and Q/history layout."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    records=[]
    for repeat in (1,2,3):
        payload=json.loads((args.root/f'repeat{repeat}/task-times.json').read_text())
        for r in payload['records']:
            records.append(dict(r,repeat=repeat))
        for mode in ('fused','immediate','after_dx'):
            for p in (args.root/f'repeat{repeat}/{mode}/timeline').glob('*_rank_*.json'):
                raw=json.loads(p.read_text())
                assert raw['task_timing_only']
                assert all(e['kind'][0] in 'BH' for e in raw['events'])
    groups={}
    for r in records:
        k=json.dumps([r['stage'],r['model_chunk'],r['layout_q_history']])
        groups.setdefault(k,[]).append(r)
    quantities=('fused_B_ms','immediate_B_ms','after_dx_B_ms','H_ms','B_plus_H_ms')
    result=[]
    for group in groups.values():
        medians={q:statistics.median(r[q] for r in group) for q in quantities}
        ranges={q:[min(r[q] for r in group),max(r[q] for r in group)] for q in quantities}
        repeat_medians={str(n):{q:statistics.median(r[q] for r in group if r['repeat']==n)
                                for q in quantities} for n in (1,2,3)}
        result.append({'stage':group[0]['stage'],'model_chunk':group[0]['model_chunk'],
                       'layout_q_history':group[0]['layout_q_history'],'count_per_mode':len(group),
                       'median_ms':medians,'min_max_ms':ranges,'repeat_median_ms':repeat_medians,
                       'after_dx_B_vs_fused_fraction':medians['after_dx_B_ms']/medians['fused_B_ms']-1})
    out={'scope':'B/H boundary events only; LR=0, same seed/data; 3 independent runs per mode with rotated order. '
                 'Each run warms up two iterations and measures iteration 3. No internal profiling/audit hooks.',
         'groups':result,'records':records}
    (args.root/'combined-task-times.json').write_text(json.dumps(out,indent=2)+'\n')
    for r in result:
        if r['stage']==1 and r['model_chunk']==0 and r['layout_q_history']==[[2048,4096]]:
            print(json.dumps(r,indent=2))


if __name__=='__main__':main()
