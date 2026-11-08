"""Every legal packed Q/K pair is assigned exactly once, including CP3 tails."""
from collections import Counter

import pytest
import torch

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.kv_cache.dspp_packed_attention import DsppSequenceKVState
from megatron.core.kv_cache.dspp_cp_qo_attention import prepare_qo_metadata


@pytest.mark.parametrize('size', (2, 3))
@pytest.mark.parametrize('layout', ('paired', 'hybrid'))
@pytest.mark.parametrize('lengths', ((133, 67, 17, 1), (1,), (50, 16, 40, 26)))
def test_qo_exact_causal_pair_coverage(size, layout, lengths):
    capacity = 2*size*11
    plan = build_dspp_batch_plan(lengths, chunk_size=capacity)
    states = [DsppSequenceKVState(cp_layout=layout, cp_impl='query-out') for _ in range(size)]
    device = torch.device('cpu')
    for rank, state in enumerate(states):
        prepare_qo_metadata(state, plan.microbatches, size, rank, device)
    histories = [{} for _ in states]
    for meta in plan.microbatches:
        covered = Counter()
        for rank, state in enumerate(states):
            info, parts, _, _, _, transport = state._cp_prepared['qo',meta.microbatch_id,rank,device]
            positions = []
            for item, (runs, _, _) in zip(meta.items, info):
                previous = histories[rank].get(item.sequence_id, [])
                current = [r.position_begin+j for r in runs for j in range(r.local_end-r.local_begin)]
                positions.append(previous+current)
                histories[rank][item.sequence_id] = previous+current
            for i,qa,qb,ka,kb,causal in parts:
                item = meta.items[i]
                for q in range(qa,qb):
                    qpos = item.token_offset+q-meta.cu_seqlens_q[i]
                    for j,kpos in enumerate(positions[i][ka:kb]):
                        if not causal or j <= q-qa+(kb-ka)-(qb-qa):
                            covered[item.sequence_id,qpos,kpos] += 1
            if layout == 'hybrid' and any(i.has_history for i in meta.items):
                assert transport['width'] > 0
        expected = Counter((item.sequence_id,q,k)
            for item in meta.items
            for q in range(item.token_offset,item.token_offset+item.token_length)
            for k in range(q+1))
        assert covered == expected


@pytest.mark.parametrize('size',(2,3))
@pytest.mark.parametrize('layout',('paired','hybrid'))
@pytest.mark.parametrize('lengths',((800,224,600,424),(1500,548),(4608,768,768),
                                  (6401,2305,97,31),(4081,800,224,600,424),
                                  (2464,768,3520,3936,864,3328,4384,1)))
def test_qo_real_capacity_query_boundaries(size,layout,lengths):
    plan=build_dspp_batch_plan(lengths,chunk_size=2040)
    states=[DsppSequenceKVState(cp_layout=layout,cp_impl='query-out') for _ in range(size)]
    device=torch.device('cpu')
    for rank,state in enumerate(states):prepare_qo_metadata(state,plan.microbatches,size,rank,device)
    histories=[{} for _ in states]
    for meta in plan.microbatches:
        rank_parts=[]
        for rank,state in enumerate(states):
            info,parts,*_=state._cp_prepared['qo',meta.microbatch_id,rank,device]
            positions=[]
            for item,(runs,_,_) in zip(meta.items,info):
                current=[r.position_begin+j for r in runs for j in range(r.local_end-r.local_begin)]
                positions.append(histories[rank].get(item.sequence_id,[])+current)
                histories[rank][item.sequence_id]=positions[-1]
            rank_parts.append((positions,parts))
        for i,item in enumerate(meta.items):
            lo,hi=meta.cu_seqlens_q[i:i+2]
            samples={lo,hi-1}
            for _,parts in rank_parts:
                for j,qa,qb,*_ in parts:
                    if j==i:samples.update(x for x in (qa-1,qa,qa+1,qb-1,qb) if lo<=x<hi)
            for q in samples:
                keys=[]
                for positions,parts in rank_parts:
                    for j,qa,qb,ka,kb,causal in parts:
                        if j==i and qa<=q<qb:
                            stop=min(kb,ka+q-qa+(kb-ka)-(qb-qa)+1) if causal else kb
                            keys.extend(positions[i][ka:stop])
                assert sorted(keys)==list(range(item.token_offset+q-lo+1))
