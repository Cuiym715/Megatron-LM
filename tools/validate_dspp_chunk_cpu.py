#!/usr/bin/env python3
"""CPU-only integration/latency checks for adaptive chunks; no training launch.

Load metadata modules without Megatron's training package initialization.
The production padding function is loaded verbatim from its AST.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import ast
from collections import Counter
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import types
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for name in ('megatron','megatron.core','megatron.core.datasets','megatron.core.pipeline_parallel','megatron.core.kv_cache'):
    package = types.ModuleType(name)
    package.__path__ = [str(REPO / name.replace('.','/'))]
    sys.modules[name] = package
path = REPO/'megatron/core/pipeline_parallel/schedules.py'
function = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n,ast.FunctionDef) and n.name=='_pad_dspp_v_split_counts')
module = types.ModuleType('megatron.core.pipeline_parallel.schedules')
exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),module.__dict__)
sys.modules[module.__name__] = module

from megatron.core.datasets.dspp_chunk_cost import ChunkCostModel
from megatron.core.datasets.dspp_chunk_search import SearchOptions, sample_capacities, aggregate_features, ChunkSearch
from megatron.core.pipeline_parallel.dspp_chunk_planner import ChunkPlanner
from megatron.core.pipeline_parallel.dspp_chunk_prefetch import AdaptiveChunkPrefetch, choose_chunk_plan, chunk_worker
from megatron.core.pipeline_parallel.dspp_prefetch import CpuPlanner, CpuIteration
from megatron.core.pipeline_parallel.dspp_plan import validate_plan, DsppSchedulePlan
from megatron.core.kv_cache.dspp_chunk_attention import query_tiles, adapt_backward_source


def summarize(samples):
    ordered = sorted(samples)
    return dict(n=len(samples),p50=statistics.median(ordered),p95=ordered[math.ceil(.95*len(ordered))-1],samples=samples)


def config(model):
    return types.SimpleNamespace(**model.payload['layout']['model'], params_dtype=torch.bfloat16,
        pipeline_model_parallel_size=model.pp,dspp_packing_strategy='baseline',
        dspp_microbatch_order='warmup-short-steady-long',dspp_recv_lookahead=2,dspp_send_inflight_depth=2,
        dspp_history_dkv_after_dx=True,dspp_split_attention_backward=True,curr_iteration=0,
        seq_length=32768,micro_seq_length=2048,micro_batch_size=17,dspp_prefetch_max_bytes=64<<20,
        variable_seq_pad_token_id=0,dspp_b_budget_fraction=.60)


def identity(lengths):
    return ({'sample_ids':list(range(len(lengths))),'raw_lengths':[n+1 for n in lengths]},)


def selection_checks(model):
    checked = 0
    for spacing in ('log','linear'):
        for step in (1,64,257):
            for points in (2,16,64,128):
                options = SearchOptions(points,step,spacing)
                # Domain deliberately different from the exported PP8 domain.
                caps = sample_capacities(257,10**9,3058,options)
                assert len(caps)<=points and 3058 in caps
                assert len(set(caps))==len(caps) and all(257<=c<=10**9 for c in caps)
                checked += 1
    # Coarse exact moments compared with explicit slicing (packing count is a proxy).
    rng = random.Random(11)
    for _ in range(50):
        lengths = [rng.randint(1,32768) for _ in range(17)]
        c = rng.randint(200,9000)
        x,w = aggregate_features(tuple(Counter(lengths).items()),[c])
        shape = [(min(c,n-offset),offset) for n in lengths for offset in range(0,n,c)]
        expected = [sum(q for q,h in shape),sum(q*(q+1)/2 for q,h in shape),
                    sum(q*h for q,h in shape),sum(h for q,h in shape),len(shape)]
        assert np.array_equal(x[0,2:],expected)
    for n in (1,2047,2048,2049,3058,3072,3840,8191,8192):
        tiles = list(query_tiles(n))
        assert [i for lo,hi in reversed(tiles) for i in range(lo,hi)] == list(range(n))
        assert all(hi-lo<=2048 for lo,hi in tiles)
    native=(REPO/'megatron/core/kv_cache/dspp_packed_attention.py').read_text()
    cls=next(n for n in ast.parse(native).body if isinstance(n,ast.ClassDef) and n.name=='_DsppPackedFlashAttention')
    backward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='backward')
    import textwrap
    source=ast.get_source_segment(native,backward)
    # AST segments omit the leading indentation of the first line only.
    source='    '+source
    adapted=adapt_backward_source(source)
    compile(adapted,'<CPU-checked-canonical-adapter>','exec')
    class FakeModel:
        domain = (100,1000)
        def coarse(self,hist,caps,mode):
            return [float(c) for c in caps]
    called = []
    def fine(c):
        called.append(c)
        if c==100:
            raise ValueError('capacity infeasible')
        return types.SimpleNamespace(capacity=c,predicted_ms=abs(c-500))
    _, metrics = ChunkSearch(FakeModel(),SearchOptions(5,100,'linear')).select([1234],500,fine)
    assert len(called)<=3 and called==metrics['shortlist']
    # Raw records are not needed after model construction.
    with patch.object(Path,'read_text',side_effect=AssertionError('runtime file read')):
        for c in (3058,3072,3840):
            for op in ('F','B_fused','D','H','W'):
                assert math.isfinite(model.predict(0,0,op,c,((c,c),)))
    return dict(grid_cases=checked,moment_cases=50,tile_cases=9,adapter_source_compiles=True,
                bounded_fine=True,no_runtime_reads=True)


def load_legacy(runtime, profiles):
    sys.path.insert(0,str(runtime))
    from bank import TargetBank
    from frozen_bank import FrozenCalibration
    from affine_coarse import AffineComputeProxy
    from megatron.core.datasets.dspp_p2p import DsppP2PCostModel,DsppP2PPoint
    bank = TargetBank.__new__(TargetBank)
    for name in ('models','proxies','memories','contexts','model_hashes','memory_hashes','mixed'):
        setattr(bank,name,{})
    def read(suffix):
        return json.loads(next(profiles.glob('*-'+suffix)).read_text())
    for c in (2048,4096,6144,8192):
        payload=read(f'c{c}-compact.json')
        bank.models[c]=FrozenCalibration(payload);bank.proxies[c]=AffineComputeProxy(payload)
        bank.memories[c]=read(f'c{c}-fused-memory.json')
        bank.contexts[c]={(s,v):bank.models[c].contexts[((s,v),'F')] for s in range(8) for v in range(2)}
        bank.model_hashes[c]='test';bank.memory_hashes[c]='test'
    bank.mixed[2048]=read('mixed-confirmed-memory.json')
    bank.p2p=DsppP2PCostModel(tuple(DsppP2PPoint(**p) for p in read('p2p-combined.json')['points']))
    # Original snapshots predate an optional CP-only config field. Its absence
    # does not change CP1; remove it only from this legacy check fixture.
    from megatron.core.pipeline_parallel.dspp_memory_counts import memory_layout
    return bank


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--legacy-runtime',type=Path)
    parser.add_argument('--legacy-profiles',type=Path)
    parser.add_argument('--unit-tests',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    begin=time.perf_counter();model=ChunkCostModel.load(args.model);cold=(time.perf_counter()-begin)*1000
    cfg=config(model);planner=ChunkPlanner(model,cfg,SearchOptions())
    checks=selection_checks(model)
    if args.unit_tests:
        import pytest
        tests=REPO/'tests/unit_tests/dspp'
        code=pytest.main(['-q','--confcutdir='+str(tests),str(tests/'test_chunk_search.py'),
                          str(tests/'test_prefetch.py')])
        assert code==0, f'unit regression exit {code}'
        checks['unit_regression']='passed'
    cases={'controlled':[2048]*16+[16384], 'uneven':[3058]*16+[12233],
           'tails':[1024,1537,2501,3071]*4+[16381]}
    plans=[]
    for name,lengths in cases.items():
        for c in (2048,3058,3072,3840,8191):
            try:
                p=planner.fine((tuple(lengths),),identity(lengths),1,'1:0:1',c)
            except ValueError as exc:
                plans.append(dict(case=name,capacity=c,rejected=str(exc)));continue
            validate_plan(p.reference)
            assert sum(sum(b.plan.sequence_lengths) for b in p.batches)==sum(lengths)
            for batch in p.batches:
                batch.plan.validate()
            plans.append(dict(case=name,capacity=c,predicted_ms=p.predicted_ms,
                              tasks=sum(len(s) for s in p.reference.schedules)))
    assert any(r['capacity']==3058 and 'rejected' not in r for r in plans)
    lengths=cases['controlled']; ids=identity(lengths)
    with patch.object(Path,'read_text',side_effect=AssertionError('per-iteration file read')):
        planner.prepare((tuple(lengths),),ids,1,0,2048)
    checks['complete_selection_no_file_reads']=True
    timings={}; equivalent=[]
    if args.legacy_runtime:
        legacy=load_legacy(args.legacy_runtime,args.legacy_profiles)
        from megatron.core.pipeline_parallel.dspp_memory_counts import memory_layout
        from copy import copy
        fused=copy(cfg);fused.dspp_history_dkv_after_dx=False;fused.dspp_split_attention_backward=False
        for c in legacy.memories:
            # Compatible layout metadata, unchanged measured bytes and costs.
            legacy.memories[c]['layout']=memory_layout(fused,8,cfg.params_dtype,c)
        for c in (2048,4096,6144,8192):
            old=legacy.fine(cfg,(tuple(lengths),),ids,legacy.contexts[c],c,'1:0:1')
            new=planner.fine((tuple(lengths),),ids,1,'1:0:1',c)
            assert old.reference.schedules==new.reference.schedules and old.reference.task_map==new.reference.task_map
            assert old.compute==new.compute and old.messages==new.messages
            assert old.predicted_ms==new.predicted_ms
            equivalent.append(c)
    jobs={}
    for points in (32,64,128):
        for score in ('work','pipeline'):
            engine=ChunkPlanner(model,cfg,SearchOptions(points,64,'log',score))
            jobs[f'adaptive-{points}-{score}']=lambda e=engine:e.prepare((tuple(lengths),),ids,1,0,2048)
    jobs['fixed-2048']=lambda:planner.prepare((tuple(lengths),),ids,1,0,2048,fixed=2048)
    if args.legacy_runtime:
        jobs['legacy-auto']=lambda:legacy.decide(cfg,(tuple(lengths),),ids,legacy.contexts[2048],2048,'1:0:1')
        jobs['legacy-fixed']=lambda:legacy.decide(cfg,(tuple(lengths),),ids,legacy.contexts[2048],2048,'1:0:1',fixed=2048)
    samples={k:[] for k in jobs}; selected={}
    for fn in jobs.values():fn();fn()
    rng=random.Random(7)
    for _ in range(11):
        names=list(jobs);rng.shuffle(names)
        for name in names:
            start=time.perf_counter();p,m=jobs[name]()
            samples[name].append((time.perf_counter()-start)*1000)
            selected[name]=dict(capacity=p.capacity,shortlist=m['shortlist'],coarse_ms=m.get('coarse_ms'))
    timings={name:dict(**summarize(values),**selected[name]) for name,values in samples.items()}
    # Actual parent source preparation + spawned B once at an unmeasured capacity.
    worker=CpuPlanner(target=chunk_worker,target_args=(model.payload,cfg,SearchOptions(),.60))
    try:
        frontend=AdaptiveChunkPrefetch.__new__(AdaptiveChunkPrefetch)
        frontend.cfg=cfg;frontend.selector=planner;frontend.current_capacity=2048
        frontend.fixed_capacity=3058;frontend.run_b=True;frontend.planner=worker;frontend.closed=False
        raw=torch.zeros((17,16385),dtype=torch.int64)
        lens=torch.tensor([n+1 for n in lengths])
        for i,n in enumerate(lens):raw[i,:n]=torch.arange(n)%32001
        frontend.iterator=iter([dict(text=raw,lengths=lens)])
        packet=frontend._prepare(1,0,1);reference=frontend.prepared_reference[1]
        assert packet.metrics['selected_chunk']==3058
        request=planner.b_request(planner.fine((tuple(lengths),),ids,1,'1:0:1',3058),time.perf_counter())
        assert 'args' not in request and 'memory_profile' not in request
        deadline=time.monotonic()+30;result=None
        while result is None and time.monotonic()<deadline:
            result=worker.poll()
            if result is None:time.sleep(.01)
        assert result is not None and not result['error'],result
        assert result['b_calls']==1 and not result['worker_cuda_initialized']
        plan,reason=choose_chunk_plan(packet,result,reference)
        assert not reason and plan.wire_shape[0]==3058
        validate_plan(plan,reference,reference.binding)
        for failure in (None,dict(result,iteration=0),dict(result,error='test-worker-failure')):
            fallback,reason=choose_chunk_plan(packet,failure,reference)
            assert reason and fallback.schedules==reference.schedules and fallback.binding==reference.binding
        # End-to-end CPU materialization preserves token/label content at c3058.
        materialized=[b.materialize('cpu') for b in packet.batches]
        assert sum(b.valid_token_count for b in materialized)==sum(lengths)
        for batch in materialized:
            for physical in batch.physical_microbatches:
                cursor=0
                for item in physical.meta.items:
                    q,h,s=item.token_length,item.token_offset,item.sequence_id
                    assert torch.equal(physical.tokens[0,cursor:cursor+q],raw[s,h:h+q])
                    assert torch.equal(physical.labels[0,cursor:cursor+q],raw[s,h+1:h+q+1])
                    assert torch.equal(physical.position_ids[0,cursor:cursor+q],torch.arange(h,h+q))
                    assert torch.all(physical.loss_mask[0,cursor:cursor+q]==1)
                    cursor+=q
                assert torch.all(physical.loss_mask[0,cursor:]==0)
        checks['materialized_tokens_labels_positions_masks']=True
        b_result=dict(capacity=3058,calls=result['b_calls'],H=sum(n.kind=='H' for s in plan.schedules for n in s),
                      planning_info=plan.planning_info,request_bytes=len(__import__('pickle').dumps(request)))
    finally:
        worker.close()
    assert not torch.cuda.is_initialized()
    output=dict(status='passed',scope='CPU integration only; no GPU numerics/throughput claim',
        cold_model_ms=cold,model_bytes=args.model.stat().st_size,checks=checks,plans=plans,
        anchor_cost_and_base_equivalence=equivalent,timings=timings,B=b_result,
        elapsed_s=time.perf_counter()-begin,cuda_initialized=False)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('plans','B')}))


if __name__=='__main__':main()
