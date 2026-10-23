#!/usr/bin/env python3
"""Offline identities/consumer links; diagnostic durations are not cost samples."""
import argparse
import ast
import json
from pathlib import Path
import re
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--raw-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=a.raw_root/'diagnostic'
messages=[];history=[];clocks=[];slots=[]
for rank in range(3):
    records=json.loads((root/f'diagnostic-rank{rank}.json').read_text())
    trace=json.loads((root/f'rank{rank}.pt.trace.json').read_text())['traceEvents']
    # The two clocks below are CPU perf_counter and this rank's Kineto clock.
    # Their mapping is diagnostic-only; it is NOT a cross-GPU calibration.
    enters=[e for e in records if e['kind']=='task_host_enter']
    ranges=sorted([e for e in trace if e.get('cat')=='user_annotation' and re.match(r'COLLECT/[FBWH][01]/s',e['name'])],key=lambda e:e['ts'])
    assert len(enters)==len(ranges)
    offsets=[]
    for e,t in zip(enters,ranges):
        task=e['task'];assert t['name']==f"COLLECT/{task['kind']}/s{rank}/c{task['chunk']}/mb{task['logical']}:{task['physical']}"
        offsets.append(e['cpu_ns']/1000-t['ts'])
    offset=statistics.median(offsets)
    clocks.append(dict(rank=rank,host_minus_kineto_us=offset,minimum=min(offsets),maximum=max(offsets),note='Approximate local CPU alignment from task annotations; not device completion calibration.'))
    kernels=[e for e in trace if e.get('cat')=='kernel' and 'ncclDevKernel_SendRecv' in e['name']]
    launches={e.get('args',{}).get('correlation'):e for e in trace if e.get('cat') in ('cuda_driver','cuda_runtime') and 'Launch' in e.get('name','')}
    pairs=sorted([(launches[k['args']['correlation']],k) for k in kernels],key=lambda x:x[0]['ts'])
    posts=[e for e in records if e['kind']=='p2p_protocol' and re.search(r'\] post-(send|recv) ',e['message'])]
    assert len(pairs)==len(posts)
    slotmap={};active=None
    for e in records:
        if e['kind']=='p2p_protocol':
            m=re.search(r'node slot=(\d+) kind=([FBW])([01]) group=(\d+) split=(\d+)',e['message'])
            if m:
                active=(m[2],int(m[3]),int(m[4]),int(m[5]));slotmap[active]=None
        elif e['kind']=='task_host_enter' and not e['task']['kind'].startswith('H'):
            assert active[0]==e['task']['kind'][0]
            slotmap[active]=e['task']
    for key,value in slotmap.items():slots.append(dict(rank=rank,message_key=key,task=value,padding=value is None))
    for e,(launch,kernel) in zip(posts,pairs):
        m=re.search(r'post-(send|recv) (?:peer|sender)=(\d+) key=(\(.+?\))',e['message']);key=ast.literal_eval(m[3])
        delta=launch['ts']-(e['cpu_ns']/1000-offset)
        # Correspondence: same training-thread order, one launch per logged op,
        # then exact CUDA correlation id to the device kernel. Check clock bound.
        assert -100<=delta<5000,(rank,key,delta)
        messages.append(dict(rank=rank,operation=m[1],peer=int(m[2]),message_key=key,
             task=slotmap.get(key),host_post_ns=e['cpu_ns'],launch_minus_marker_us=delta,
             launch=launch,kernel=kernel,
             note='Diagnostic SendRecv residency includes peer wait/contention; not a reusable transfer service cost.'))
    consumers={}
    for e in records:
        if e['kind']=='gradient_route_begin' and e['consume_current']:
            for item in e['layouts']:
                consumers[(e['task']['logical'],e['task']['chunk'],e['layer'],item['sequence'],item['segment'])]=e
    for e in records:
        if e['kind']!='gradient_route_end' or not e['route_history'] or not e['task']['kind'].startswith('H'):continue
        for item in e['layouts']:
            for segment,length in item['history']:
                target=consumers[(e['task']['logical'],e['task']['chunk'],e['layer'],item['sequence'],segment)]
                assert e['device_offset_ms']<=target['device_offset_ms']+.001
                history.append(dict(rank=rank,source=e['task'],consumer=target['task'],layer=e['layer'],sequence=item['sequence'],segment=segment,tokens=length,
                    accumulated_by_device_ms=e['device_offset_ms'],consumer_route_begin_device_ms=target['device_offset_ms']))
# Each ordered NCCL payload has both source and receiver records.
keys={}
for m in messages:
    source=m['rank'] if m['operation']=='send' else m['peer']
    dest=m['peer'] if m['operation']=='send' else m['rank']
    key=(source,dest,tuple(m['message_key']))
    keys.setdefault(key,set()).add(m['operation'])
assert all(v=={'send','recv'} for v in keys.values())
out=dict(accepted=True,message_records=len(messages),matched_payloads=len(keys),history_links=len(history),messages=messages,history=history,slot_map=slots,local_clock_alignment=clocks,
    limitations=['No precise cross-GPU clock calibration.','Consumer route entry is conservative; additions are not individually timestamped.','Credit arrival and counterfactual host progress costs remain unknown.','Diagnostic profiler overhead excludes these durations from formal task costs.'])
(a.output/'diagnostic-links.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('accepted','message_records','matched_payloads','history_links','local_clock_alignment')},indent=2))
