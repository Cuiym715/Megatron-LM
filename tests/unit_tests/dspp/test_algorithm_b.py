"""CPU-only tail-B regression: real Plan IR, dependencies, budget and capacity.

Run with python3 tests/unit_tests/dspp/test_algorithm_b.py (no CUDA/PyTorch needed).
"""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
if __name__ == '__main__':
    # Load pure scheduling modules without Megatron's CUDA package bootstrap.
    for name in ('megatron', 'megatron.core', 'megatron.core.datasets', 'megatron.core.pipeline_parallel'):
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT / name.replace('.', '/'))]
        sys.modules[name] = module
    if importlib.util.find_spec('torch') is None:
        sys.modules['torch'] = types.ModuleType('torch')  # metadata builder never calls torch

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.pipeline_parallel.dspp_plan import (
    PlanBinding, history_dependencies, node_key, receive_orders,
    replay_plan, template_plan, validate_plan,
)
from megatron.core.pipeline_parallel.dspp_plan_b import (
    Backend, FixedGraph, OptionalPlan, initial, plan_configuration_b, predecessors,
    run_tail, score_positions, tail_frontiers,
)
from megatron.core.pipeline_parallel.slice_v import build_slice_v_schedule


def fixture(pp=4, capacity=64):
    batch = build_dspp_batch_plan([capacity*3+17, capacity+9], chunk_size=capacity, validate=False)
    groups = [tuple((l,p) for p in range(len(batch.microbatches))) for l in range(2)]
    mapping = tuple((g,s,*task) for g,group in enumerate(groups) for s,task in enumerate(group))
    metadata = {(l,p):m for l in range(2) for p,m in enumerate(batch.microbatches)}
    slots = {(g,s): metadata[l,p] for g,s,l,p in mapping}
    counts = list(map(len, groups))
    seqs,_ = build_slice_v_schedule(pp,len(counts),counts)
    ref = template_plan(seqs,counts,mapping,PlanBinding(1,'input','layout'),
                        history_dependencies=history_dependencies(mapping,metadata))
    compute = {node_key(n):1. if (n.microbatch,n.split) in slots else 0. for seq in seqs for n in seq}
    messages = {('message',a,b,*m):3. for (a,b),lane in receive_orders(seqs).items() for m in lane}
    sources = {edge[:2] for edge in ref.history_dependencies}
    source = next(k for k in compute if k[:3]==(0,'B',1) and k[3:] in sources)
    compute[source] = 100.; compute[(1,*source[1:])] = 100.
    split = {k:(c*.3,c*.8) for k,c in compute.items() if k[1]=='B' and k[3:] in sources}
    tokens = {node_key(n):slots[n.microbatch,n.split].valid_token_count
              for seq in seqs for n in seq if n.kind=='F' and (n.microbatch,n.split) in slots}
    return ref,compute,messages,split,tokens,slots


class TailBTests(unittest.TestCase):
    def test_tail_order_and_longest_only(self):
        ref,c,m,s,t,slots=fixture()
        optional=OptionalPlan(ref,c,m,s)
        chains=tail_frontiers(optional,slots)
        self.assertEqual(len(chains),4*2*2) # PP × VPP × tied longest documents
        for chain in chains:
            segments=[]
            for h in chain:
                key=optional.m.names[optional.m.release[h]]
                items=[i for i in slots[key[3:]].items if i.has_history]
                longest=max(items,key=lambda i:i.sequence_length)
                self.assertEqual(longest.sequence_length,209)
                segments.append(longest.segment_id)
            self.assertEqual(segments,sorted(segments,reverse=True))

    def test_real_plans_and_arbitrary_capacity(self):
        for pp,capacity in ((4,64),(8,64),(4,3058),(4,4096)):
            ref,c,m,s,t,slots=fixture(pp,capacity)
            with patch('megatron.core.pipeline_parallel.dspp_plan_b.time.perf_counter',return_value=0.):
                plan=plan_configuration_b(ref,c,m,s,tokens=t,slot_meta=slots)
            validate_plan(plan,ref)
            self.assertLessEqual(plan.predicted_ms,plan.reference_ms)
            self.assertEqual(plan.planning_info['decision']['candidate_order'],'longest-sequence-tail-frontier')
            actual=dict(c)
            for seq in plan.schedules:
                for n in seq:
                    if n.kind in ('D','H'):
                        b=(n.stage,'B',n.chunk,n.microbatch,n.split)
                        actual[node_key(n)]=s[b][0] if n.kind=='D' else max(s[b][1],c[b]-s[b][0])
            self.assertAlmostEqual(replay_plan(plan,actual,m)['makespan_ms'],plan.predicted_ms)

    def test_budget_and_memory_rejection(self):
        ref,c,m,s,t,slots=fixture()
        plan=plan_configuration_b(ref,c,m,s,tokens=t,slot_meta=slots,budget_fraction=0)
        self.assertEqual(plan.planning_info['decision']['stop'],'deadline')
        self.assertEqual(plan.planning_info['selected_count'],0)
        class Memory:
            def check(self,plan):
                if any(n.kind=='H' for seq in plan.schedules for n in seq):
                    raise ValueError('capacity')
        with patch('megatron.core.pipeline_parallel.dspp_plan_b.time.perf_counter',return_value=0.):
            plan=plan_configuration_b(ref,c,m,s,tokens=t,slot_meta=slots,memory=Memory())
        self.assertEqual(plan.planning_info['selected_count'],0)
        validate_plan(plan,ref)

    def test_cpp_supply_backend(self):
        ref,c,m,s,t,slots=fixture()
        with patch('megatron.core.pipeline_parallel.dspp_plan_b.time.perf_counter',return_value=0.):
            plan=plan_configuration_b(ref,c,m,s,tokens=t,slot_meta=slots,backend='cpp')
        validate_plan(plan,ref)
        self.assertEqual(plan.planning_info['backend'],'cpp')

    def test_tail_split_improves_and_respects_feasible_position(self):
        # B=5 followed by work=5; downstream tail=5. Moving H after work
        # reduces 15->12; when that lifetime is infeasible, H before work gives14.
        keys=((0,'B',0,0,1),(0,'W',0,0,1),(0,'B',0,0,0),
              (1,'B',0,0,0),(0,'H',0,0,1))
        b,x,c,tail,h=keys
        graph=FixedGraph(keys,((b,h),(h,c),(x,tail)),
                         dict(zip(keys,(5.,5.,0.,5.,0.))),
                         dict(zip(keys,(0,0,0,1,0))),{h:b},{h:(c,)})
        o=types.SimpleNamespace(m=graph,fused_cost=graph.cost.copy(),split_cost={4:2.},
            hcost={4:2.},base_orders=((0,1,2),(3,)),
            base=types.SimpleNamespace(task_map=((0,0,0,0),(0,1,0,1))))
        def configure(selected):
            graph.cost=o.fused_cost.copy()
            if 4 in selected:graph.cost[0],graph.cost[4]=2.,2.
        o.configure=configure
        result,_=run_tail(o,Backend(),1000)
        self.assertEqual(result.makespan,12.)
        result,_=run_tail(o,Backend(),1000,lambda t:t.orders[0].index(4)<t.orders[0].index(1))
        self.assertEqual(result.makespan,14.)

    def test_worker_request_uses_resident_configuration(self):
        import json
        import pickle
        import queue
        import time
        path=ROOT/'megatron/core/pipeline_parallel/dspp_prefetch.py'
        fn=next(n for n in ast.parse(path.read_text()).body
                if isinstance(n,ast.FunctionDef) and n.name=='_planner_main')
        cfg=types.SimpleNamespace(curr_iteration=0)
        contexts={'resident':'stage contexts'}
        seen=[]
        def build(request,optimize):
            seen.append(request)
            return None,types.SimpleNamespace(to_dict=lambda:{'schedule':'result'})
        fake_torch=types.SimpleNamespace(set_num_threads=lambda n:None,
            set_num_interop_threads=lambda n:None,cuda=types.SimpleNamespace(is_initialized=lambda:False))
        ns=dict(torch=fake_torch,pickle=pickle,json=json,time=time,_build=build)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
        jobs,results=queue.Queue(),queue.Queue()
        jobs.put(pickle.dumps((8,'input8',(('metadata',),1.,('ordering','counts'),7))))
        jobs.put(None)
        ns['_planner_main'](jobs,results,cfg,contexts)
        result=json.loads(results.get())
        self.assertIs(seen[0][0],cfg)
        self.assertIs(seen[0][2],contexts)
        self.assertEqual(cfg.curr_iteration,7)
        self.assertEqual(result['iteration'],8)
        self.assertIsNone(result['error'])

    def test_fast_insertion_matches_full_graph(self):
        ref,c,m,s,t,slots=fixture()
        o=OptionalPlan(ref,c,m,s);state=initial(o,Backend());preds=predecessors(o.m)
        for chain in tail_frontiers(o,slots):
            h=chain[0]
            for predicted,pos in score_positions(o,state,h,preds,Backend()):
                orders=[list(x) for x in state.orders]
                orders[o.m.meta[h]['rank']].insert(pos,h);o.configure({h})
                actual=o.m.evaluate(orders)
                if actual is not None:self.assertAlmostEqual(actual,predicted)

if __name__=='__main__':
    unittest.main()
