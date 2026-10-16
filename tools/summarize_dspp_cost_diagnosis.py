#!/usr/bin/env python3
"""Attribute diagnostic GPU events to CPU ranges using launch correlation IDs.

Kernel time sums and launch spans are diagnostic observations under profiling,
not clean task timings. Unassigned GPU events are explicitly counted.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def contains(a, b):
    # B waits for autograd work submitted by another CPU thread. Serial task
    # ownership is temporal across threads; nested operator ownership is exact.
    thread_matches = a['name'].startswith('COST_TASK/') or a['tid'] == b['tid']
    return a['pid'] == b['pid'] and thread_matches and a['ts'] <= b['ts'] < a['ts'] + a['dur']


def summarize(path):
    events = json.loads(path.read_text())['traceEvents']
    ranges = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'user_annotation'
              and e['name'].startswith(('COST/', 'COST_TASK/', 'COST_OP/'))]
    runtimes = {e['args']['correlation']: e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver')
                and e.get('ph') == 'X' and 'correlation' in e.get('args', {})}
    devices = [e for e in events if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    attributed = {id(r): [] for r in ranges}
    unassigned = []
    for e in devices:
        launch = runtimes.get(e.get('args', {}).get('correlation'))
        owners = [r for r in ranges if launch and contains(r, launch)]
        if not owners: unassigned.append(e['name'])
        for r in owners: attributed[id(r)].append(e)
    rows = []
    for r in ranges:
        gpu = attributed[id(r)]
        if not gpu: continue
        intervals = sorted((e['ts'], e['ts']+e['dur']) for e in gpu)
        merged = []
        for a,b in intervals:
            if merged and a <= merged[-1][1]: merged[-1][1] = max(merged[-1][1],b)
            else: merged.append([a,b])
        busy = sum(b-a for a,b in merged)
        span = max(b for a,b in intervals)-intervals[0][0]
        kernels = [{k:e[k] for k in ('name','cat','ts','dur','args')} for e in gpu]
        rows.append({'range':r['name'], 'cpu_duration_us':r['dur'], 'gpu_events':len(gpu),
                     'gpu_time_sum_us':sum(e['dur'] for e in gpu), 'gpu_busy_union_us':busy,
                     'first_to_last_gpu_us':span, 'gap_inside_gpu_span_us':span-busy,
                     'kernels':kernels})
    return {'source':str(path),'scope':'Intrusive profiler, not clean timing; nested ranges overlap. Task ownership uses temporal containment across autograd threads; operator ownership matches thread.',
            'unassigned_event_count':len(unassigned),'unassigned_names':dict(Counter(unassigned)), 'ranges':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=summarize(args.trace)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in result['ranges']:
        name=r['range']
        if name.startswith('COST/') or (name.startswith('COST_TASK/B') and name.endswith('/2048:4096')):
            print({k:v for k,v in r.items() if k!='kernels'})


if __name__=='__main__':main()
