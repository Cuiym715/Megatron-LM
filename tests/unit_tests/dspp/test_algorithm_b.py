"""Few targeted B checks: deployed graph, exact slots/pairing and fallbacks."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.pipeline_parallel.dspp_plan import (
    DsppSchedulePlan, PlanBinding, history_dependencies, node_key,
    receive_orders, replay_plan, template_plan, validate_plan,
)
from megatron.core.pipeline_parallel.dspp_plan_b import (
    Backend, FixedGraph, OptionalPlan, State, best_insertion, initial,
    plan_configuration_b, predecessors, run_cp, score_positions,
)
from megatron.core.pipeline_parallel.slice_v import build_slice_v_schedule
from megatron.core.pipeline_parallel.schedules import _pad_dspp_v_split_counts
from .test_schedule_plan import simulate_protocol


def fixture(pp=4):
    batch = build_dspp_batch_plan([145,33], chunk_size=64)
    groups = [tuple((l,p) for p in range(len(batch.microbatches))) for l in range(2)]
    mapping = tuple((g,s,*task) for g,group in enumerate(groups) for s,task in enumerate(group))
    metadata = {(l,p):m for l in range(2) for p,m in enumerate(batch.microbatches)}
    counts = _pad_dspp_v_split_counts(list(map(len,groups)), pp)
    seqs,_ = build_slice_v_schedule(pp,len(counts),counts)
    ref = template_plan(seqs,counts,mapping,PlanBinding(1,'input','layout'),
                        history_dependencies=history_dependencies(mapping,metadata))
    real = {(g,s) for g,s,l,p in mapping}
    compute = {node_key(n):1. if (n.microbatch,n.split) in real else 0.
               for seq in seqs for n in seq}
    messages = {('message',a,b,*m):3. for (a,b),lane in receive_orders(seqs).items() for m in lane}
    sources = {edge[:2] for edge in ref.history_dependencies}
    source = next(k for k in compute if k[:3]==(0,'B',1) and k[3:] in sources)
    compute[source] = 100.
    compute[(1,*source[1:])] = 100.
    split = {k:(c*.3,c*.8) for k,c in compute.items() if k[1]=='B' and k[3:] in sources}
    split[source] = (20.,90.)
    tokens = {node_key(n):metadata[groups[n.microbatch][n.split]].valid_token_count
              for seq in seqs for n in seq if n.kind=='F' and (n.microbatch,n.split) in {(g,s) for g,s,l,p in mapping}}
    return ref,compute,messages,split,tokens


@pytest.mark.parametrize('pp',[4,8])
def test_b_plan_replay_fifo_split_identity_and_internal_orders(pp):
    ref,compute,messages,split,tokens = fixture(pp)
    plan = plan_configuration_b(ref,compute,messages,split,tokens=tokens)
    validate_plan(plan,ref)
    assert plan.planning_info['selected_count']>0
    assert plan.predicted_ms<plan.reference_ms
    restored = DsppSchedulePlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert restored==plan
    simulate_protocol(plan,2,2)
    for seq,old in zip(plan.schedules,ref.schedules):
        for kind in ('F','B'):
            assert [(n.chunk,n.microbatch,n.split) for n in seq if n.kind==kind or kind=='B' and n.kind=='D']==[(n.chunk,n.microbatch,n.split) for n in old if n.kind==kind]
    assert plan.planning_info['decision']['candidate_order']=='input-node-order'
    assert plan.planning_info['decision']['top_k'] is None
    assert plan.planning_info['decision']['grouping'] is None


def pair_optional():
    # Two equal critical branches. Each single keeps T=10; the pair gives T=7.
    keys=('b0','c0','b1','c1','tail','h0','h1')
    edges=(('b0','h0'),('h0','c0'),('b1','h1'),('h1','c1'),('b0','tail'),('b1','tail'))
    costs=dict(zip(keys,(5.,0.,5.,0.,5.,0.,0.)))
    graph=FixedGraph(keys,edges,costs,dict(zip(keys,(0,0,1,1,2,0,1))),
                     {'h0':'b0','h1':'b1'},{'h0':('c0',),'h1':('c1',)})
    o=SimpleNamespace(m=graph,fused_cost=graph.cost.copy(),split_cost={5:2.,6:2.},hcost={5:3.5,6:3.5},
                      base_orders=((0,1),(2,3),(4,)))
    def configure(selected):
        graph.cost=o.fused_cost.copy()
        for h in selected:
            graph.cost[graph.release[h]]=o.split_cost[h]
            graph.cost[h]=o.hcost[h]
    o.configure=configure
    return o


def test_exposed_path_pair_and_fast_slot_formula():
    o=pair_optional();backend=Backend();base=initial(o,backend)
    assert base.makespan==10
    preds=predecessors(o.m)
    for h in o.m.hs:
        scores=score_positions(o,base,h,preds,backend)
        for predicted,pos in scores:
            orders=[list(x) for x in base.orders]
            orders[o.m.meta[h]['rank']].insert(pos,h)
            o.configure({h})
            actual=backend.evaluate(o.m,orders)
            if actual is not None:
                assert predicted==actual==10
    result,info=run_cp(o,backend,1000)
    assert result.makespan==7 and result.selected==frozenset({5,6})
    assert info['steps'][0]['kind']=='pair'
    assert info['steps'][0]['second_trials']>0


def test_deadline_and_unprofitable_costs_preserve_unsplit():
    ref,compute,messages,split,tokens=fixture()
    plan=plan_configuration_b(ref,compute,messages,split,tokens=tokens,budget_fraction=0)
    assert plan.planning_info['decision']['stop']=='deadline'
    assert plan.planning_info['selected_count']==0
    assert not plan.planning_info['budget_met']
    assert plan.planning_info['budget_overrun_ms']>0
    bad={k:(compute[k]*2,compute[k]) for k in split}
    other=plan_configuration_b(ref,compute,messages,bad,tokens=tokens)
    assert other.planning_info['decision']['stop']=='stalled'
    assert other.planning_info['selected_count']==0


def test_final_counted_capacity_rejection_restores_verified_base():
    ref,compute,messages,split,tokens=fixture()
    class Memory:
        def check(self,plan):
            if any(n.kind=='D' for seq in plan.schedules for n in seq):
                raise ValueError('counted memory capacity exceeded')
    plan=plan_configuration_b(ref,compute,messages,split,tokens=tokens,memory=Memory())
    assert plan.planning_info['decision']['steps'][0]['after_ms'] is not None
    assert plan.fallback_reason.startswith('final-capacity')
    assert plan.planning_info['selected_count']==0
    assert plan.predicted_ms==plan.planning_info['seed_makespan_ms']


def test_history_cost_normalization_and_missing_split_samples():
    ref,compute,messages,split,_=fixture()
    one=next(iter(split))
    optional=OptionalPlan(ref,compute,messages,{one:(compute[one]*.4,.01)})
    assert len(optional.m.hs)==1
    h=optional.m.hs[0]
    assert optional.hcost[h]==compute[one]-optional.split_cost[h]
    baseline=initial(optional,Backend())
    assert abs(baseline.makespan-replay_plan(ref,compute,messages)['makespan_ms'])<1e-6
    with pytest.raises(ValueError,match='invalid split cost'):
        plan_configuration_b(ref,compute,messages,{one:(float('nan'),1)})


@pytest.mark.parametrize('failure',['reference','coordinator'])
def test_distributed_rejection_is_published_before_raise(monkeypatch, failure):
    import torch.distributed as dist
    from megatron.core.pipeline_parallel import dspp_plan_runtime as runtime
    reference = fixture()[0]
    calls = []
    def build(*args, optimize, **kwargs):
        if failure == 'reference' or optimize:
            raise ValueError('injected failure')
        return reference, reference
    def gather(out, value, **kwargs):
        calls.append('gather')
        out[:] = [value, value]
    def broadcast(value, **kwargs):
        calls.append('broadcast')
        assert value[0][0] == 'ValueError: injected failure'
    monkeypatch.setattr(runtime, 'build_runtime_plan', build)
    monkeypatch.setattr(dist, 'get_world_size', lambda group:2)
    monkeypatch.setattr(dist, 'get_rank', lambda group:0)
    monkeypatch.setattr(dist, 'get_global_rank', lambda group, rank:rank)
    monkeypatch.setattr(dist, 'all_gather_object', gather)
    monkeypatch.setattr(dist, 'broadcast_object_list', broadcast)
    with pytest.raises(RuntimeError, match='agreement failed' if failure=='reference' else 'coordinator failed'):
        runtime.build_distributed_b_plan(None, group=object())
    assert calls == (['gather'] if failure=='reference' else ['gather','broadcast'])
