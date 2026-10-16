#!/usr/bin/env python3
"""Check real B/H task boundaries and summarize matched shape/stage timings.

Input is the three-mode output of measure_dspp_history_after_dx.sh. Numbers
are CUDA-event stream intervals, not pure-kernel sums or throughput claims.
"""
import argparse
import json
from pathlib import Path
import statistics


def key(event):
    return tuple(event[n] for n in ('stage','chunk','logical_microbatch','physical_microbatch'))


def layout(event):
    return [(i['token_length'], i['token_offset']) for i in event['items']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=('fused','immediate','after_dx'),
                        default=['fused','immediate','after_dx'])
    args = parser.parse_args()
    if not {'fused','after_dx'} <= set(args.modes):
        parser.error('--modes must include fused and after_dx')
    runs = {}
    for mode in args.modes:
        events=[]
        paths=sorted((args.root/mode/'timeline').glob('iteration_*_rank_*.json'))
        assert len(paths)==3,(mode,paths)
        for p in paths:
            payload=json.loads(p.read_text())
            local=payload['events']
            for e in local:
                assert e['duration_ms']>=0
            if mode=='after_dx':
                for index,e in enumerate(local):
                    if e['kind'].startswith('H'):
                        assert index>0 and local[index-1]['kind'].startswith('B')
                        b=local[index-1]
                        assert key(e)==key(b) and b['end_ms']<=e['start_ms']
                        if index+1<len(local):
                            assert e['end_ms']<=local[index+1]['start_ms']
            events.extend(local)
        runs[mode] = events
    by_mode={mode:{key(e):e for e in events if e['kind'].startswith('B')}
             for mode,events in runs.items()}
    assert all(set(by_mode[m])==set(by_mode['fused']) for m in by_mode)
    histories={key(e):e for e in runs['after_dx'] if e['kind'].startswith('H')}
    rows=[]
    for k,b in by_mode['after_dx'].items():
        has_history=any(i['token_offset']>0 for i in b['items'])
        assert (k in histories)==has_history
        for m in by_mode:
            assert layout(by_mode[m][k])==layout(b)
        h=histories.get(k)
        rows.append({'stage':k[0],'model_chunk':k[1],'logical':k[2],'physical':k[3],
                     'layout_q_history':layout(b),
                     'fused_B_ms':by_mode['fused'][k]['duration_ms'],
                     **({'immediate_B_ms':by_mode['immediate'][k]['duration_ms']} if 'immediate' in by_mode else {}),
                     'after_dx_B_ms':b['duration_ms'],
                     'H_ms':h['duration_ms'] if h else 0,
                     'B_plus_H_ms':b['duration_ms']+(h['duration_ms'] if h else 0),
                     'B_start_to_H_end_ms':h['end_ms']-b['start_ms'] if h else b['duration_ms'],
                     'dx_ready_ms':b['end_ms'],
                     'H_start_ms':h['start_ms'] if h else None})
    groups={}
    for row in rows:
        g=json.dumps([row['stage'],row['model_chunk'],row['layout_q_history']])
        groups.setdefault(g,[]).append(row)
    summary=[]
    quantities=tuple(q for q in ('fused_B_ms','immediate_B_ms','after_dx_B_ms','H_ms','B_plus_H_ms','B_start_to_H_end_ms')
                     if rows and q in rows[0])
    for group in groups.values():
        summary.append({'stage':group[0]['stage'],'model_chunk':group[0]['model_chunk'],
                        'layout_q_history':group[0]['layout_q_history'],'count':len(group),
                        'median':{q:statistics.median(r[q] for r in group) for q in quantities}})
    result={'measurement':'Iteration 3 after two warmup iterations; CUDA-event compute-stream intervals. '
                            'B ends after complete stage dx and excludes separate W. H includes history dKV and routing. '
                            'No CPU gradient snapshot hooks. Instrumented single-iteration task samples, not speedup results.',
            'boundary_checks':{'B_count':len(rows),'H_count':len(histories),
                               'all_H_immediately_follow_matching_B':True,
                               'no_H_for_no_history_B':True,'all_mode_layouts_match':True},
            'by_shape_stage':summary,'records':rows}
    target=args.root/'task-times.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['boundary_checks']))
    print(target)


if __name__=='__main__':
    main()
