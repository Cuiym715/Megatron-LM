"""Storage births, aliases and last consumers in the opt-in v3 accounting."""
from dataclasses import replace

import pytest
import torch

from megatron.core.pipeline_parallel.dspp_memory_counts import CountedMemory, memory_layout
from megatron.core.pipeline_parallel.dspp_memory_lifetimes import ACCOUNTING, stage_attention_sizes
from megatron.core.pipeline_parallel.dspp_plan import (
    history_dependencies, node_key, order_owner_peaks, receive_orders, split_plan, validate_plan,
)
from tests.unit_tests.dspp.test_memory_counts import split_case
from tools.calibrate_dspp_memory_lifetimes import partition


def case(pp=3, gqa=False):
    args, batch, groups, ref, meta, profile = split_case(pp)
    args.group_query_attention = gqa
    ref = replace(ref, schema_version=2, history_dependencies=history_dependencies(
        ref.task_map, {(l,p): meta[g,s] for g,s,l,p in ref.task_map}))
    profile.update(schema='dspp-counted-memory-v3', accounting=ACCOUNTING,
                   layout=memory_layout(args, pp, torch.bfloat16, 64))
    for r, row in enumerate(profile['stages']):
        row.update(weight_forward_bytes=[3 << 20]*2, weight_backward_bytes=[5 << 20]*2,
                   capture_bytes=[stage_attention_sizes(args,pp,r,c,64)[0] for c in range(2)],
                   pending_bytes_per_token=[stage_attention_sizes(args,pp,r,c,64)[1] for c in range(2)])
        row['attention_bytes_per_key_token'] = row['pending_bytes_per_token'][:]
    return args, ref, meta, profile


def memory_case(pp=3, gqa=False):
    args, ref, meta, profile = case(pp, gqa)
    return CountedMemory(profile,args,ref,meta,torch.bfloat16,64), ref, meta


@pytest.mark.parametrize('pp,gqa', [(3,False),(3,True),(4,False),(8,True)])
def test_peaks_and_w_births(pp, gqa):
    memory, ref, meta = memory_case(pp,gqa)
    plan, owners, sizes = memory.records(ref)
    assert 'shared_kv' not in sizes and 'history_scratch' not in sizes
    for owner in owners:
        if owner.bytes == 5 << 20:
            assert owner.born[0][1] == 'B' and owner.users[0][1] == 'W'
        if owner.bytes == 3 << 20:
            assert owner.born[0][1] == 'F' and owner.users[0][1] == 'W'
    positions = {node_key(n): i for seq in plan.schedules for i,n in enumerate(seq)}
    expected = tuple(max(sum(o.bytes for o in owners if o.stage == r
        and positions[o.born[0]] <= i <= max(positions[u] for u in o.users))
        for i in range(len(seq))) for r,seq in enumerate(plan.schedules))
    assert order_owner_peaks(plan,owners) == expected
    assert memory.check(plan)['owner_peak_bytes'] == list(expected)


def test_capture_shared_across_w_and_delayed_h_scratch_only_at_tasks():
    memory, ref, meta = memory_case()
    selected = next(node_key(n) for seq in ref.schedules for n in seq if n.kind == 'B'
        and (n.microbatch,n.split) in meta and any(i.has_history for i in meta[n.microbatch,n.split].items))
    plan = split_plan(ref,{selected})
    r,_,c,g,s = selected
    seq = list(plan.schedules[r]); h = next(n for n in seq if node_key(n) == (r,'H',c,g,s))
    seq.remove(h)
    windex = next(i for i,n in enumerate(seq) if node_key(n) == (r,'W',c,g,s))
    seq.insert(windex+1,h)
    plan = replace(plan,schedules=tuple(tuple(seq) if i == r else x for i,x in enumerate(plan.schedules)))
    _,owners,sizes = memory.records(plan)
    capture = [o for o in owners if o.born == ((r,'F',c,g,s),) and (r,'W',c,g,s) in o.users
               and (r,'H',c,g,s) in o.users]
    assert len(capture) == 1
    assert capture[0].bytes == memory.stages[r]['capture_bytes'][c]
    scratch = [o for o in owners if o.born == o.users and o.born[0][1] in 'DH']
    assert len(scratch) == 2
    assert all(o.born[0][1] in 'DH' for o in scratch)
    assert 'history_source' not in sizes and sizes['history_created'] > 0
    assert memory.fixed == CountedMemory(memory.profile,memory.args,ref,meta,torch.bfloat16,64).fixed
    seq.remove(h); seq.append(h)
    bad = replace(plan,schedules=(tuple(seq),*plan.schedules[1:]))
    with pytest.raises(ValueError,match='consumer'):
        memory.records(bad)


def test_first_pending_charges_whole_backing_once_then_target_additions():
    memory,ref,meta = memory_case()
    _,owners,_ = memory.records(ref)
    # First microbatch contains sequence 0 split across four tasks. Its latest
    # backward writes three history targets, sharing one complete dKV backing.
    source_slot = next(k for k,m in meta.items() if k[0] == 0
                       and any(i.sequence_id == 0 and i.segment_id == 3 for i in m.items))
    g,s = source_slot
    source = (0,'B',0,g,s)
    pending = [o for o in owners if o.born == (source,) and any(u[1] == 'B' for u in o.users)]
    assert len(pending) == 1
    assert len(pending[0].users) >= 3
    unit = memory.stages[0]['pending_bytes_per_token'][0]
    assert pending[0].bytes == sum(meta[g,s].key_lengths)*unit
    assert pending[0].bytes > sum(i.token_offset for i in meta[g,s].items)*unit
    target = next((g,s) for (g,s),m in meta.items() if g == 0
                  and any(i.sequence_id == 0 and i.segment_id == 0 for i in m.items))
    target_b = (0,'B',0,*target)
    extra = [o for o in owners if o.users == (target_b,) and o.born[0][1] == 'B'
             and o.born != (source,) and o.bytes == 64*unit]
    assert extra and extra[0].born[0] != target_b
    # Splitting the same source uses H's compact history allocation instead.
    selected = split_plan(ref,{source})
    _, split_owners, _ = memory.records(selected)
    h = (0,'H',0,g,s)
    compact = [o for o in split_owners if o.born == (h,) and any(u[1] == 'B' for u in o.users)]
    assert len(compact) == 1
    assert compact[0].bytes == sum(i.token_offset for i in meta[g,s].items)*unit


@pytest.mark.parametrize('backend',['python','cpp'])
def test_v3_owners_in_supply_and_feedback_construction(backend):
    from megatron.core.pipeline_parallel.dspp_plan_b import plan_configuration_b
    from megatron.core.pipeline_parallel.dspp_plan_c import feedback_configuration_c
    memory,ref,meta = memory_case()
    compute = {node_key(n):1. if (n.microbatch,n.split) in meta else 0.
               for seq in ref.schedules for n in seq}
    source = next(k for k in compute if k[:3] == (0,'B',1)
                  and k[3:] in meta and any(i.has_history for i in meta[k[3:]].items))
    compute[source] = compute[(1,*source[1:])] = 100.
    messages = {('message',a,b,*msg):3. for (a,b),lane in receive_orders(ref.schedules).items() for msg in lane}
    costs = {k:(v*.3,v*.8) for k,v in compute.items() if k[1]=='B' and k[3:] in meta
             and any(i.has_history for i in meta[k[3:]].items)}
    seed = plan_configuration_b(ref,compute,messages,costs,memory=memory,
        owners=memory.owners,capacities=memory.capacities,backend=backend)
    feedback = feedback_configuration_c(ref,seed,compute,messages,costs,memory=memory,backend=backend)
    for plan in (seed,feedback):
        validate_plan(plan,ref)
        report = memory.check(plan)
        assert all(p<=c for p,c in zip(report['owner_peak_bytes'],memory.capacities))
    assert {node_key(n) for seq in feedback.schedules for n in seq if n.kind=='H'} == {
        node_key(n) for seq in seed.schedules for n in seq if n.kind=='H'}


def test_v3_identity_and_layout_cannot_silently_reuse_other_profiles():
    args,ref,meta,profile = case()
    memory = CountedMemory(profile,args,ref,meta,torch.bfloat16,64)
    profile['stages'][0]['weight_backward_bytes'][0] += 1
    newer = CountedMemory(profile,args,ref,meta,torch.bfloat16,64)
    assert memory.version != newer.version
    profile['accounting'] = 'other-lifetimes'
    with pytest.raises(ValueError,match='versioned'):
        CountedMemory(profile,args,ref,meta,torch.bfloat16,64)


@pytest.mark.parametrize('field', ['capture_bytes','attention_bytes_per_key_token','pending_bytes_per_token'])
def test_underestimated_analytic_floors_rejected(field):
    args,ref,meta,profile = case(gqa=True)
    profile['stages'][0][field][0] = 0
    with pytest.raises(ValueError,match='floor'):
        CountedMemory(profile,args,ref,meta,torch.bfloat16,64)


def test_inadequate_physical_budget_rejected_before_execution():
    args,ref,meta,profile = case()
    memory = CountedMemory(profile,args,ref,meta,torch.bfloat16,64)
    profile['stages'][0]['physical_bytes'] = (profile['stages'][0]['resident_bytes']
        + profile['stages'][0]['reserve_bytes'] + memory.check(ref)['dynamic_peak_bytes'][0] - 1)
    with pytest.raises(ValueError,match='capacity'):
        CountedMemory(profile,args,ref,meta,torch.bfloat16,64)


def test_profile_partition_removes_aliases_from_w_but_keeps_backward_storage():
    def storage(size,shape):
        return dict(bytes=size,views=[[0,shape,[], 'torch.bfloat16']],owners=[])
    f = dict(q=storage(100,[64,4,128]), assembled=storage(200,[192,1,128]),
             matmul=storage(300,[64,512]),activation=storage(400,[64,512]))
    w = dict(q=f['q'],matmul=f['matmul'],grad=storage(500,[64,512]))
    assert partition(f,w,(4,1),64) == dict(activation=400,weight_forward=300,
        weight_backward=500,capture=100)


def test_b_publication_reuses_final_memory_count(monkeypatch):
    from collections import Counter
    from megatron.core.pipeline_parallel.dspp_plan_b import plan_configuration_b
    memory,ref,meta=memory_case()
    memory.capacities=(10**15,)*3
    compute={node_key(n):1. for seq in ref.schedules for n in seq}
    sources={edge[:2] for edge in ref.history_dependencies}
    source=next(k for k in compute if k[:3]==(0,'B',1) and k[3:] in sources)
    compute[source]=compute[(1,*source[1:])]=100.
    split={k:(c*.3,c*.8) for k,c in compute.items() if k[1]=='B' and k[3:] in sources}
    split[source]=(20.,90.)
    messages={('message',a,b,*m):3. for (a,b),lane in receive_orders(ref.schedules).items() for m in lane}
    from megatron.core.pipeline_parallel.dspp_memory_lifetimes import CompiledStorage
    original=CompiledStorage.evaluate; calls=Counter()
    def evaluate(self,plan):
        calls[plan.schedules]+=1
        return original(self,plan)
    monkeypatch.setattr(CompiledStorage,'evaluate',evaluate)
    plan=plan_configuration_b(ref,compute,messages,split,memory=memory,budget_fraction=1.)
    assert plan.planning_info['selected_count']>0
    before=calls.copy()
    memory.check(plan)
    assert calls==before and calls[plan.schedules]==1


def test_v3_initial_capacity_reuses_reference_records(monkeypatch):
    calls=[];original=CountedMemory.records
    def records(self,plan,**kw):
        calls.append(plan);return original(self,plan,**kw)
    monkeypatch.setattr(CountedMemory,'records',records)
    memory,ref,_=memory_case()
    assert len(calls)==1
    memory.check(ref)
    assert len(calls)==1


@pytest.mark.parametrize('pp,gqa', [(3,False),(3,True),(4,False),(8,True)])
def test_compiled_counter_matches_storage_audit_and_rejection(pp,gqa):
    import random
    from megatron.core.pipeline_parallel.dspp_memory_lifetimes import CompiledStorage
    memory,ref,meta=memory_case(pp,gqa)
    counter=CompiledStorage(memory)
    rng=random.Random(20261024)
    eligible=[node_key(n) for seq in ref.schedules for n in seq if n.kind=='B'
              and (n.microbatch,n.split) in meta
              and any(i.has_history for i in meta[n.microbatch,n.split].items)]
    plans=[ref,split_plan(ref,set(eligible))]
    for _ in range(24):
        selected=set(rng.sample(eligible,rng.randint(1,min(6,len(eligible)))))
        plan=split_plan(ref,selected)
        schedules=list(plan.schedules)
        stage=rng.choice(sorted({k[0] for k in selected}))
        seq=list(schedules[stage])
        h=rng.choice([n for n in seq if n.kind=='H'])
        seq.remove(h);seq.insert(rng.randrange(len(seq)+1),h)
        schedules[stage]=tuple(seq)
        plans.append(replace(plan,schedules=tuple(schedules)))
    # Duplicate, missing, unknown tasks and invalid stage placement.
    seq=ref.schedules[0]
    plans.extend(replace(ref,schedules=(bad,*ref.schedules[1:])) for bad in (
        (seq[0],*seq),seq[1:],(replace(seq[0],microbatch=999),*seq[1:]),
        (replace(seq[0],stage=1),*seq[1:])))
    valid=invalid=0
    for plan in plans:
        try:
            accounting,owners,sizes=memory.records(plan)
            expected=order_owner_peaks(accounting,owners)
        except ValueError:
            invalid+=1
            with pytest.raises(ValueError):counter.evaluate(plan)
            continue
        valid+=1
        peaks,count,actual_sizes=counter.evaluate(plan)
        assert peaks==expected and count==len(owners) and actual_sizes==sizes
        # Capacity decisions remain byte-exact, including the boundary.
        memory.capacities=peaks
        assert memory.check(plan)['owner_peak_bytes']==list(peaks)
        memory.capacities=(peaks[0]-1,*peaks[1:])
        with pytest.raises(ValueError,match='capacity'):memory.check(plan)
    assert valid>=2 and invalid>=4


def test_compiled_counter_only_recounts_changed_stage(monkeypatch):
    from megatron.core.pipeline_parallel.dspp_memory_lifetimes import CompiledStorage
    memory,ref,meta=memory_case()
    counter=CompiledStorage(memory)
    counter.evaluate(ref)
    calls=[];original=CompiledStorage._stage
    def evaluate(order,rows,targets):
        calls.append(order)
        return original(order,rows,targets)
    monkeypatch.setattr(CompiledStorage,'_stage',staticmethod(evaluate))
    source=next(node_key(n) for n in ref.schedules[1] if n.kind=='B'
                and (n.microbatch,n.split) in meta
                and any(i.has_history for i in meta[n.microbatch,n.split].items))
    candidate=split_plan(ref,{source})
    def unexpected(*args,**kw):raise AssertionError('rebuilt storage metadata')
    monkeypatch.setattr(memory,'records',unexpected)
    import megatron.core.pipeline_parallel.dspp_memory_counts as counts
    monkeypatch.setattr(counts,'history_state_bytes',unexpected)
    counter.evaluate(candidate)
    assert len(calls)==1
    counter.evaluate(candidate)
    assert len(calls)==1
