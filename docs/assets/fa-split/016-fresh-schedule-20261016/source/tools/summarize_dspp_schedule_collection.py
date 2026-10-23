#!/usr/bin/env python3
"""Validate fresh current-code collection and export conservative replay inputs.

Does not simulate reordered H, fit missing communication time, or predict speedup.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics as st

MODES = ('fused', 'after_dx')

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

def stat(values):
    return dict(n=len(values), mean=st.mean(values), median=st.median(values),
                minimum=min(values), maximum=max(values))

def identity(e):
    return (e['stage'],e['chunk'],e['logical_microbatch'],e['physical_microbatch'])

def node_id(e, kind=None, stage=None, chunk=None):
    s,c,l,p=identity(e)
    return f"s{s if stage is None else stage}:c{c if chunk is None else chunk}:{kind or e['kind'][0]}:l{l}:p{p}"

def layout(e):
    return tuple(tuple(i[k] for k in ('sequence_id','segment_id','token_offset','token_length','sequence_length')) for i in e['items'])

def parse_log(path):
    result=[]
    for line in path.read_text().splitlines():
        m=re.search(r'iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\): ([\d.]+)',line)
        if not m:continue
        assert 'number of nan iterations:   0' in line
        assert 'number of skipped iterations:   0' in line
        loss=float(re.search(r'lm loss:\s*([\d.E+\-]+)',line)[1])
        result.append(dict(iteration=int(m[1]),iteration_log_ms=float(m[2]),loss=loss))
    return result

def replay_input(payloads, mode, iteration):
    nodes={};orders={};items={}
    for p in payloads:
        orders[str(p['rank'])]=[]
        for e in p['events']:
            name=node_id(e);assert name not in nodes
            nodes[name]=dict(cost_ms=e['duration_ms'],rank=e['stage'],vpp=e['chunk'],
                             kind=e['kind'][0],logical=e['logical_microbatch'],
                             physical=e['physical_microbatch'],items=e['items'])
            orders[str(p['rank'])].append(name)
            if e['kind'].startswith('B'):
                for item in e['items']:
                    items[(e['stage'],e['chunk'],e['logical_microbatch'],item['sequence_id'],item['segment_id'])]=e
    edges=set();communication=[]
    def edge(a,b,reason):
        assert a in nodes and b in nodes,(a,b,reason)
        edges.add((a,b,reason))
    for p in payloads:
        for e in p['events']:
            kind=e['kind'][0];s,c,l,_=identity(e)
            if kind=='B':
                edge(node_id(e,'F'),node_id(e),'saved_activation')
                for item in e['items']:
                    if item['segment_id']>0:
                        previous=items[(s,c,l,item['sequence_id'],item['segment_id']-1)]
                        edge(node_id(previous,'F'),node_id(e,'F'),'forward_KV_continuation')
                    for j in range(item['segment_id']):
                        consumer=items[(s,c,l,item['sequence_id'],j)]
                        producer=node_id(e,'H' if mode=='after_dx' else 'B')
                        edge(producer,node_id(consumer),'history_complete_before_target_B_conservative')
            elif kind=='H':edge(node_id(e,'B'),node_id(e),'full_B_release')
            elif kind=='W':edge(node_id(e,'B'),node_id(e),'W_ready_after_B')
            if kind in ('F','B'):
                chain=[(0,0),(1,0),(2,0),(2,1),(1,1),(0,1)]
                if kind=='B':chain.reverse()
                index=chain.index((s,c))
                if index+1<len(chain):
                    dst_s,dst_c=chain[index+1]
                    a=node_id(e);b=node_id(e,stage=dst_s,chunk=dst_c)
                    edge(a,b,'local_V_boundary' if s==dst_s else 'PP_payload_dependency')
                    if s!=dst_s:communication.append(dict(source=a,target=b,payload_bytes=2048*4096*2))
    # Data-dependency graph only. The baseline resource order remains a separate
    # field so a future H scheduler cannot mistake its current position for an
    # immutable data dependency.
    indegree={n:0 for n in nodes};successors=defaultdict(list)
    for a,b,_ in edges:indegree[b]+=1;successors[a].append(b)
    ready=[n for n,d in indegree.items() if d==0];visited=0
    while ready:
        a=ready.pop();visited+=1
        for b in successors[a]:
            indegree[b]-=1
            if indegree[b]==0:ready.append(b)
    assert visited==len(nodes),'conservative data dependencies contain a cycle'
    return dict(mode=mode,iteration=iteration,nodes=nodes,data_edges=sorted(edges),
                baseline_resource_orders=orders,communication_dependencies=communication,
                ordering=[p['ordering'] for p in payloads],
                contract='One compute resource per rank; no assumed overlap or preemption. H is a whole stage task and targets must wait before B starts. Communication edges are dependencies, NOT zero-latency transfers. Padding slots and credit/host/transport costs still require runtime constraints.',
                unknowns=['precise cross-GPU clock offsets','communication service/host costs outside diagnostic sample','extra retained-state memory under delayed H','counterfactual task costs after changed contention'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.raw_root;out=args.output;out.mkdir(parents=True,exist_ok=True)
    completions={};logs={};raw_rows=[];counts={};timeline={}
    for mode in MODES:
        logs[mode]=parse_log(root/mode/'train.log');assert len(logs[mode])==30
        completions[mode]={}
        for rank in range(3):
            p=read(root/mode/f'completion-rank{rank}.json');assert p['rank']==rank
            assert [r['iteration'] for r in p['records']]==list(range(1,31))
            completions[mode][rank]=p
            for record in p['records']:
                assert len(record['plans'])==2
                assert all(len(plan['sequence_lengths'])==5 for plan in record['plans'])
                assert all(n in (14336,4097,1537,257,2048) for plan in record['plans'] for n in plan['sequence_lengths'])
        counts[mode]=Counter();timeline[mode]={}
        files=list((root/mode/'timeline').glob('iteration_*_rank_*.json'));assert len(files)==15
        for iteration in range(26,31):
            payloads=[]
            for rank in range(3):
                p=read(root/mode/'timeline'/f'iteration_{iteration:06d}_rank_{rank}.json')
                assert p['task_timing_only'] and p['task_timing_scope']=='all'
                ev=p['events'];seen=defaultdict(set);by_id={}
                for i,e in enumerate(ev):
                    key=identity(e);kind=e['kind'][0]
                    assert key not in seen[kind];seen[kind].add(key)
                    counts[mode][kind]+=1
                    assert e['duration_ms']>=0 and e['end_ms']>=e['start_ms']
                    if i:assert ev[i-1]['end_ms']<=e['start_ms']+.001
                    if kind=='B':by_id[key]=e
                    if kind=='H':
                        assert mode=='after_dx' and i and ev[i-1]['kind'].startswith('B')
                        assert identity(ev[i-1])==key
                    raw_rows.append(dict(mode=mode,iteration=iteration,**e))
                expected_tasks={(rank,c,l,physical) for c in range(2)
                                for l,plan in enumerate(completions[mode][rank]['records'][iteration-1]['plans'])
                                for physical in range(plan['physical_microbatches'])}
                assert seen['F']==seen['B']==seen['W']==expected_tasks
                expected={key for key,e in by_id.items() if any(i['token_offset']>0 for i in e['items'])}
                assert seen['H']==(expected if mode=='after_dx' else set())
                payloads.append(p)
            timeline[mode][iteration]=payloads
            write(out/f'replay-{mode}-{iteration}.json',replay_input(payloads,mode,iteration))
    for mode in MODES:
        for rank in (1,2):
            for a,b in zip(completions[mode][0]['records'],completions[mode][rank]['records']):
                assert a['inputs']==b['inputs'] and a['plans']==b['plans']
    paired=[]
    for rank in range(3):
        for a,b in zip(completions['fused'][rank]['records'],completions['after_dx'][rank]['records']):
            assert a['inputs']==b['inputs'] and a['plans']==b['plans'],(rank,a['iteration'],'input mismatch')
    for a,b in zip(logs['fused'],logs['after_dx']):
        assert a['iteration']==b['iteration'] and abs(a['loss']-b['loss'])<1e-4
        if 6<=a['iteration']<=25:
            tokens=sum(p['valid_tokens'] for p in completions['fused'][0]['records'][a['iteration']-1]['plans'])
            paired.append(dict(iteration=a['iteration'],valid_tokens=tokens,fused_ms=a['iteration_log_ms'],
                               after_dx_ms=b['iteration_log_ms'],change_percent=100*(b['iteration_log_ms']/a['iteration_log_ms']-1)))
    for iteration in range(26,31):
        for a,b in zip(timeline['fused'][iteration],timeline['after_dx'][iteration]):
            aa=[(e['kind'],identity(e),layout(e)) for e in a['events']]
            bb=[(e['kind'],identity(e),layout(e)) for e in b['events'] if not e['kind'].startswith('H')]
            assert aa==bb and a['ordering']==b['ordering']
    groups=defaultdict(list)
    for row in raw_rows:
        key=json.dumps([row['mode'],row['stage'],row['chunk'],row['kind'],
                        [(i['token_length'],i['token_offset'],i['sequence_length']) for i in row['items']]])
        groups[key].append(row['duration_ms'])
    e_count=sum(r['kind'].startswith('B') and any(i['sequence_length']==2048 for i in r['items']) for r in raw_rows if r['mode']=='after_dx')
    mixed_count=sum(r['kind'].startswith('B') and len(r['items'])>1 and any(i['token_offset']>0 for i in r['items']) for r in raw_rows if r['mode']=='after_dx')
    assert e_count>0 and mixed_count>0
    # Diagnostic input must be EXACTLY the saved iteration26 batch on each rank.
    diagnostic_counts=Counter();routes=[];dx=[];protocol=[]
    assert len(parse_log(root/'diagnostic'/'train.log'))==3
    diag_events=[]
    for rank in range(3):
        p=read(root/'diagnostic'/f'completion-rank{rank}.json')
        assert len(p['records'])==3
        for record in p['records']:
            assert record['inputs']==completions['after_dx'][rank]['records'][25]['inputs']
            assert record['plans']==completions['after_dx'][rank]['records'][25]['plans']
        trace=root/'diagnostic'/f'rank{rank}.pt.trace.json';assert trace.exists() and trace.stat().st_size>0
        events=read(root/'diagnostic'/f'diagnostic-rank{rank}.json')
        for e in events:
            diagnostic_counts[e['kind']]+=1
            if e['kind'].startswith('gradient_route'):routes.append(dict(rank=rank,**e))
            elif e['kind']=='complete_dx_stream_boundary':dx.append(dict(rank=rank,**e))
            elif e['kind'] in ('p2p_protocol','control_enter','control_return'):protocol.append(dict(rank=rank,**e))
        d=read(root/'diagnostic'/'timeline'/f'iteration_000003_rank_{rank}.json')
        assert [(e['kind'],identity(e),layout(e)) for e in d['events']]==[(e['kind'],identity(e),layout(e)) for e in timeline['after_dx'][26][rank]['events']]
        diag_events.extend(dict(rank=rank,**e) for e in d['events'])
    assert routes and dx and protocol
    means={m:st.mean(p[m+'_ms'] for p in paired) for m in MODES}
    summary=dict(accepted=True,counts=counts,completed_iterations=63,paired_e2e=paired,
                 mean_iteration_log_ms=means,mean_change_percent=100*(means['after_dx']/means['fused']-1),
                 total_tokens=sum(p['valid_tokens'] for p in paired),
                 throughput_tokens_per_second={m:1000*sum(p['valid_tokens'] for p in paired)/sum(p[m+'_ms'] for p in paired) for m in MODES},
                 per_rank_device_step_ms={m:{str(rank):stat([r['device_train_step_ms'] for r in p['records'][5:25]]) for rank,p in v.items()} for m,v in completions.items()},
                 peak_memory_bytes={m:{str(rank):{'allocated':max(r['peak_allocated_bytes'] for r in p['records']), 'reserved':max(r['peak_reserved_bytes'] for r in p['records'])} for rank,p in v.items()} for m,v in completions.items()},
                 exact_2048_nohistory_B_samples=e_count,mixed_history_B_samples=mixed_count,
                 diagnostic_counts=diagnostic_counts,
                 scope='Current working tree only. Rank0 iteration log is wall-clock training-loop measurement; per-rank device train-step boundaries are separate. No exact global GPU clock claimed. Batch variation is not repeated-sample noise. No replay/makespan prediction performed.')
    write(out/'summary.json',summary);write(out/'task-samples.json',raw_rows)
    write(out/'costs-by-shape.json',{k:stat(v) for k,v in groups.items()})
    write(out/'diagnostic-dependencies.json',dict(routes=routes,dx_boundaries=dx,protocol=protocol,task_events=diag_events,
        note='Route event brackets are conservative GPU consumption boundaries, not per-addition timestamps. Control/queue host timestamps are not device completion. Use raw profiler for transport correlation; H-delayed memory and host submission remain constraints.'))
    print(json.dumps({k:summary[k] for k in ('accepted','counts','mean_iteration_log_ms','mean_change_percent','throughput_tokens_per_second','exact_2048_nohistory_B_samples','mixed_history_B_samples','diagnostic_counts')},indent=2))

if __name__=='__main__':main()
