#!/usr/bin/env python3
"""Read-only comparison: two archived PP4/8 supply cases and 030 decisions.

The original recurrence/model code is executed in memory with the same PP
size adaptation already used by 022. No archived scripts or binaries are
written or compiled. Production never imports these research directories.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from megatron.core.pipeline_parallel.dspp_plan import (
    PlanBinding, PlanOwner, _graph, _plan_unsplit, node_key, receive_orders,
    replay_plan, template_plan,
)
from megatron.core.pipeline_parallel.dspp_plan_b import (
    Backend, FixedGraph, State, run_cp,
)
from megatron.core.pipeline_parallel.slice_v import build_slice_v_schedule


def adapt(fn, old, new):
    source=textwrap.dedent(inspect.getsource(fn)).replace(old,new)
    namespace=fn.__globals__
    exec(compile(source,'<read-only PP adapter>','exec'),namespace)
    return namespace[fn.__name__]


def check(root,label):
    archive=root/'027-slack-generated-20261022'
    path=archive/'inputs'/f'input-{label}.json'
    raw=json.loads(path.read_text());fused=raw['fused']
    counts=raw['ordering']['schedule_group_sizes_with_padding']
    real_counts=raw['ordering']['schedule_group_sizes']
    entrance=iter(raw['ordering']['entrance_order'])
    mapping=[(g,s,*next(entrance)) for g,c in enumerate(real_counts) for s in range(c)]
    tasks={(g,s):(l,p) for g,s,l,p in mapping}
    schedules,_=build_slice_v_schedule(raw['pp'],len(counts),counts)
    names={node_key(n):f's{n.stage}:c{n.chunk}:{n.kind}:l{tasks[n.microbatch,n.split][0]}:p{tasks[n.microbatch,n.split][1]}' for seq in schedules for n in seq}
    inverse={name:k for k,name in names.items()}
    deps=set()
    for a,b,why in raw['after_dx']['data_edges']:
        if ':H:' in a and ':B:' in b:
            source=inverse[a.replace(':H:',':B:')];target=inverse[b]
            deps.add((*source[3:],*target[3:]))
    reference=template_plan(schedules,counts,mapping,PlanBinding(0,label,'research'),history_dependencies=tuple(sorted(deps)))
    compute={k:fused['nodes'][name]['cost_ms'] for k,name in names.items()}
    messages={('message',a,b,*m):1.0337119698524475 for (a,b),lane in receive_orders(schedules).items() for m in lane}
    owners=[];tokens={}
    for f,name in names.items():
        meta=fused['nodes'][name]
        if f[1]!='F': continue
        q=sum(i['token_length'] for i in meta['items']);tokens[f]=q
        b=inverse[name.replace(':F:',':B:')];w=inverse[name.replace(':F:',':W:')]
        owners.append(PlanOwner(f[0],32*q*4096,(f,),(w,)))
        for item in meta['items']:
            future={k for k,other in names.items() if k[1]=='F' and k[0]==f[0] and k[2]==f[2]
                    and fused['nodes'][other]['logical']==meta['logical']
                    and any(z['sequence_id']==item['sequence_id'] and z['segment_id']>item['segment_id'] for z in fused['nodes'][other]['items'])}
            owners.append(PlanOwner(f[0],item['token_length']*8192,(f,),tuple({b}|future)))
            if future:
                writers=tuple((r,'B',c,g,s) for r,_,c,g,s in future)
                owners.append(PlanOwner(f[0],item['token_length']*16384,writers,(w,)))
    source_ids={name:i for i,name in enumerate(raw['after_dx']['nodes'])}
    stable_ids={k:source_ids[name] for k,name in names.items()}
    stable_ids.update({(k[0],'H',*k[2:]):source_ids[name.replace(':B:',':H:')] for k,name in names.items() if k[1]=='B' and name.replace(':B:',':H:') in source_ids})
    msg_index=0
    for a,b,why in raw['after_dx']['data_edges']:
        if why=='PP_payload_dependency':
            ak,bk=inverse[a],inverse[b]
            stable_ids[('message',ak[0],bk[0],*bk[1:])]=len(source_ids)+msg_index
            msg_index+=1
    result=_plan_unsplit(reference,compute,messages,owners=owners,tokens=tokens,supply=True,stable_ids=stable_ids)
    frozen=json.loads((archive/f'result-{label}-supply-r0.json').read_text())
    actual_orders=[[names[node_key(n)] for n in seq] for seq in result.schedules]
    assert actual_orders==frozen['resource_orders'],(label,result.fallback_reason)
    assert abs(result.predicted_ms-frozen['makespan_ms'])<1e-6

    # Load only original pure-Python models; bypass 022's file-writing adapter.
    sys.path.insert(0,str(root/'006-selective-split-20261017'))
    import analyze_selective as selective
    import analyze as original_model
    import analyze_comm as comm
    if not getattr(original_model,'_p4_adapted',False):
        original_model.Model.__init__=adapt(original_model.Model.__init__,'range(3)',"range(len(data['baseline_resource_orders']))")
        source=textwrap.dedent(inspect.getsource(comm.add_network))
        source=source.replace('    graph=copy.deepcopy(original);', "    pp=len(original['baseline_resource_orders'])\n    directions=[p for r in range(pp-1) for p in ((r,r+1),(r+1,r))]\n    graph=copy.deepcopy(original);")
        source=source.replace('range(3)','range(pp)').replace('DIRECTIONS','directions').replace('rank=3+','rank=pp+')
        exec(compile(source,'<022 dimension-only network adapter>','exec'),comm.__dict__)
        original_model._p4_adapted=True
    # Use the frozen supply order as the identical split-search input.
    fused=json.loads(json.dumps(fused));after=json.loads(json.dumps(raw['after_dx']))
    fused['baseline_resource_orders']={str(r):order for r,order in enumerate(frozen['resource_orders'])}
    after['baseline_resource_orders']={str(r):[x for n in order for x in ([n,n.replace(':B:',':H:')] if n.replace(':B:',':H:') in after['nodes'] and ':B:' in n else [n])] for r,order in enumerate(frozen['resource_orders'])}
    o=selective.OptionalModel(fused,after,1.0337119698524475,'fused_fw','nonnegative_overhead')
    o.base_orders=tuple(tuple(seq) for seq in o.m.base)
    # Source 010's pure profiler and 011's exact helper implementations.
    beam=(root/'010-path-beam-20261018/path_beam.py').read_text()
    audit=(root/'011-pair-audit-20261018/audit.py').read_text()
    source030=(root/'030-critical-path-candidates-20261022/compare.py').read_text()
    namespace=dict(State=State,time=__import__('time'),deque=__import__('collections').deque)
    exec(beam[beam.index('def profile('):beam.index('\ndef score_positions(')],namespace)
    exec(beam[beam.index('def score_positions('):beam.index('\ndef refine(')],namespace)
    exec(audit[audit.index('def value('):audit.index('\ndef state_from(')],namespace)
    exec(audit[audit.index('def predecessors('):audit.index('\ndef fixed_order_ablation(')],namespace)
    original_initial=lambda o,b:State(frozenset(),o.base_orders,b.evaluate(o.m,o.base_orders))
    namespace.update(initial=original_initial,e=SimpleNamespace(**namespace))
    exec(source030[source030.index('def step_cp('):source030.index('\ndef plan(')],namespace)
    backend=SimpleNamespace(evaluate=lambda m,orders:m.evaluate(orders),profile=namespace['profile'])
    original,old_info=namespace['run_cp'](o,backend,10000)
    # Rebuild the same graph in the production recurrence, keeping source IDs,
    # edge traversal order and stable critical-path tie rule exactly identical.
    old=o.m
    keys=old.names
    costs={k:o.fused_cost[i] for i,k in enumerate(keys)}
    graph=FixedGraph(keys,[(keys[a],keys[b]) for a,b in old.edges],costs,
                     {k:old.meta[i]['rank'] for i,k in enumerate(keys)},
                     {keys[h]:keys[b] for h,b in old.release.items()},
                     {keys[h]:tuple(keys[t] for t in ts) for h,ts in old.targets.items()})
    o.m=graph
    actual,info=run_cp(o,Backend(),10000)
    assert actual==original,(label,actual.makespan,original.makespan)
    assert [s['kind'] for s in info['steps']]==[s['kind'] for s in old_info['steps']]
    return dict(label=label,pp=raw['pp'],input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                supply_orders_identical=True,supply_ms=result.predicted_ms,
                cp_orders_identical=True,cp_ms=actual.makespan,selected=len(actual.selected),
                stop=info['stop'],decision_ms=info['decision_ms'],original_decision_ms=old_info['decision_ms'])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--research',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();assert not args.output.exists()
    rows=[check(args.research,label) for label in ('pp4-L8k-extra0','pp8-L8kx3-tokenmatch')]
    args.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
