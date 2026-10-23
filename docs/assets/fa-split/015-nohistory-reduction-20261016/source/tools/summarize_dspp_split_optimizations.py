#!/usr/bin/env python3
"""Compare baseline SPLIT vs optimized SPLIT; task and e2e runs kept separate."""
import argparse
import json
from pathlib import Path
import re
import statistics as st

VERSIONS=('baseline','nohistory','both')
FIELDS=('B_ms','H_ms','B_plus_H_ms')

def identity(e):
    return tuple(e[k] for k in ('stage','chunk','logical_microbatch','physical_microbatch'))

def layout(e):return [(i['token_length'],i['token_offset']) for i in e['items']]

def stats(rows):
    result={'n':len(rows),'versions':{}}
    for mode in VERSIONS:
        result['versions'][mode]={}
        for field in FIELDS:
            values=[r[mode][field] for r in rows]
            result['versions'][mode][field]=dict(median=st.median(values),range=[min(values),max(values)],sum=sum(values))
    result['paired_deltas_vs_baseline']={mode:{field:[r[mode][field]-r['baseline'][field] for r in rows] for field in FIELDS} for mode in VERSIONS[1:]}
    return result

def task_summary(path):
    runs={};counts={};orders={}
    for mode in VERSIONS:
        bs={};hs={};order=[]
        files=sorted((path/mode/'timeline').glob('iteration_*_rank_*.json'));assert len(files)==3
        for f in files:
            d=json.loads(f.read_text());assert d['task_timing_only'] and d['backward_mode']=='after_dx'
            ev=d['events']
            order.append((f.name,[(e['kind'],identity(e)) for e in ev]))
            for i,e in enumerate(ev):
                k=identity(e)
                if e['kind'].startswith('B'):assert k not in bs;bs[k]=e
                else:
                    assert e['kind'].startswith('H') and k not in hs
                    assert i>0 and ev[i-1]['kind'].startswith('B') and identity(ev[i-1])==k
                    assert ev[i-1]['end_ms']<=e['start_ms'];hs[k]=e
        assert all((k in hs)==any(i['token_offset']>0 for i in b['items']) for k,b in bs.items())
        counts[mode]={'B':len(bs),'H':len(hs)};runs[mode]=(bs,hs);orders[mode]=order
    assert all(set(v[0])==set(runs['baseline'][0]) for v in runs.values())
    assert all(order==orders['baseline'] for order in orders.values()), 'B/H task order changed'
    rows=[]
    for k,b in runs['baseline'][0].items():
        row=dict(stage=k[0],chunk=k[1],logical=k[2],physical=k[3],layout=layout(b))
        for mode,(bs,hs) in runs.items():
            assert layout(bs[k])==row['layout']
            bt=bs[k]['duration_ms'];ht=hs[k]['duration_ms'] if k in hs else 0
            row[mode]=dict(B_ms=bt,H_ms=ht,B_plus_H_ms=bt+ht)
        rows.append(row)
    groups={}
    for r in rows:groups.setdefault(json.dumps([r['stage'],r['chunk'],r['layout']]),[]).append(r)
    selected={}
    for h in (0,4096,12288):
        rr=[r for r in rows if r['stage']==1 and r['chunk']==0 and r['layout']==[(2048,h)]]
        if rr:selected[str(h)]=stats(rr)
    return dict(counts=counts,selected_stage1_chunk0=selected,all_tasks=stats(rows),
        nohistory=stats([r for r in rows if not any(h>0 for q,h in r['layout'])]),
        history=stats([r for r in rows if any(h>0 for q,h in r['layout'])]),
        all_shape_stage={k:stats(v) for k,v in groups.items()},records=rows)

def parse_logs(path):
    logs={}
    for mode in VERSIONS:
        log=(path/mode/'train.log').read_text();rr=[]
        for line in log.splitlines():
            m=re.search(r'iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\): ([\d.]+)',line)
            if not m:continue
            loss=re.search(r'lm loss:\s*([\d.E+\-]+)',line).group(1)
            assert 'number of nan iterations:   0' in line and 'number of skipped iterations:   0' in line
            rr.append(dict(iteration=int(m[1]),ms=float(m[2]),loss=loss))
        logs[mode]=rr
    assert all([r['iteration'] for r in logs[m]]==[r['iteration'] for r in logs['baseline']] for m in VERSIONS)
    # Fixed weights/data: compare paired iterations, never unlike variable-length batches.
    for mode in VERSIONS:
        assert all(abs(float(a['loss'])-float(b['loss']))<1e-4 for a,b in zip(logs[mode],logs['baseline']))
    return logs

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    tasks={};e2e={}
    for path in sorted(args.root.iterdir()):
        if not path.is_dir() or not all((path/m/'train.log').exists() for m in VERSIONS):continue
        if path.name.startswith('tasks'):
            tasks[path.name]=task_summary(path);tasks[path.name]['logs']=parse_logs(path)
            print(path.name,'counts',tasks[path.name]['counts'])
            for h,v in tasks[path.name]['selected_stage1_chunk0'].items():
                print('history',h,{m:{f:round(v['versions'][m][f]['median'],4) for f in FIELDS} for m in VERSIONS})
        if path.name.startswith('e2e'):
            assert not any((path/m/'timeline').exists() for m in VERSIONS)
            logs=parse_logs(path);assert all(len(v)==6 for v in logs.values())
            # iterations 1-2 are excluded; iteration count/length sequence match.
            means={m:st.mean(r['ms'] for r in rr if r['iteration']>=3) for m,rr in logs.items()}
            deltas={m:[dict(iteration=a['iteration'],delta_ms=a['ms']-b['ms'],change_percent=100*(a['ms']/b['ms']-1)) for a,b in zip(logs[m],logs['baseline']) if a['iteration']>=3] for m in VERSIONS[1:]}
            e2e[path.name]=dict(records=logs,mean_ms_iterations3_to6=means,paired_changes=deltas,
                mean_change_percent_vs_baseline={m:100*(means[m]/means['baseline']-1) for m in VERSIONS[1:]})
            print(path.name,'means',means,'change%',e2e[path.name]['mean_change_percent_vs_baseline'])
    pooled_rows=[r for run in tasks.values() for r in run['records']]
    pooled={}
    if pooled_rows:
        pooled['all_tasks']=stats(pooled_rows)
        pooled['nohistory']=stats([r for r in pooled_rows if not any(h>0 for q,h in r['layout'])])
        pooled['history']=stats([r for r in pooled_rows if any(h>0 for q,h in r['layout'])])
        pooled['selected_stage1_chunk0']={}
        for h in (0,4096,12288):
            rr=[r for r in pooled_rows if r['stage']==1 and r['chunk']==0 and r['layout']==[(2048,h)]]
            if rr:pooled['selected_stage1_chunk0'][str(h)]=stats(rr)
    result=dict(baseline='9b9fc8f96 committed after-dx split, NOT fused backward',
        scope='Tasks: boundary CUDA events. E2E: separate runs without timeline/profiler/output collection; iteration log interval=1. All-task sums across ranks are NOT iteration time.',
        task_runs=tasks,e2e_runs=e2e,pooled_task_runs=pooled)
    (args.root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
