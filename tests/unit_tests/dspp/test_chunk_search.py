"""Bounded search, cross-capacity costs, and shortlist/base reuse contracts."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from megatron.core.datasets.dspp_chunk_search import SearchOptions, ChunkSearch, sample_capacities, aggregate_features
from megatron.core.datasets.dspp_chunk_cost import ChunkCostModel


@pytest.mark.parametrize('spacing',['linear','log'])
@pytest.mark.parametrize('step',[1,64,257])
def test_large_domain_is_sampled_with_bounded_size_and_current(spacing,step):
    caps=sample_capacities(257,10**12,3058,SearchOptions(32,step,spacing))
    assert len(caps)<=32 and 3058 in caps and len(caps)==len(set(caps))
    assert min(caps)>=257 and max(caps)<=10**12


def test_top_two_plus_current_prepared_object_is_reused_and_rejection_does_not_expand():
    class Model:
        domain=(1,100)
        def coarse(self,hist,caps,mode):return [float(c) for c in caps]
    seen=[];objects={}
    def fine(c):
        seen.append(c)
        if c==1:raise ValueError('infeasible')
        objects[c]=SimpleNamespace(capacity=c,predicted_ms=100-c)
        return objects[c]
    p,metrics=ChunkSearch(Model(),SearchOptions(4,1,'linear')).select([1234],55,fine)
    assert seen==metrics['shortlist'] and len(seen)==3
    assert p is objects[55]


def test_fixed_mode_has_one_fine_and_no_coarse():
    class Model:
        domain=(1,100)
        def coarse(self,*args):raise AssertionError('fixed mode called coarse')
    calls=[]
    def fine(c):calls.append(c);return SimpleNamespace(capacity=c,predicted_ms=1)
    _,m=ChunkSearch(Model()).select([128],32,fine,fixed=37)
    assert calls==[37] and m['points']==0


def test_no_accepted_base_fails_without_a_fourth_candidate():
    class Model:
        domain=(1,100)
        def coarse(self,hist,caps,mode):return caps
    calls=[]
    def fine(c):calls.append(c);raise ValueError('infeasible')
    with pytest.raises(ValueError,match='no feasible base'):
        ChunkSearch(Model(),SearchOptions(10,1)).select([500],55,fine)
    assert len(calls)==3


def test_exact_moments_match_slice_enumeration_with_tails_and_repeats():
    lengths=[1024,1024,3058,8193]
    for c in [1,257,3058,8192]:
        x,_=aggregate_features(tuple(Counter(lengths).items()),[c])
        items=[(min(c,n-h),h) for n in lengths for h in range(0,n,c)]
        assert np.array_equal(x[0,2:],[sum(q for q,h in items),sum(q*(q+1)/2 for q,h in items),
            sum(q*h for q,h in items),sum(h for q,h in items),len(items)])


def test_unmeasured_capacity_and_dense_weight_query_use_model_coefficients():
    groups=[dict(stage=s,chunk=v,op=op,coefficients=[1,2,3,0,0,0,0],exact=[])
            for s in range(2) for v in range(2) for op in ('F','B_fused','D','H','W')]
    groups[0]['exact']=[[2048,[[2048,0]],17.]]
    model=ChunkCostModel(dict(schema='dspp-cross-capacity-v1',pp=2,domain=[100,10000],groups=groups))
    assert model.predict(0,0,'F',2048,((2048,0),))==17.
    assert model.predict(0,0,'F',3058,((1024,2000),))==1+2*3058+3*1024
    assert model.predict(0,0,'W',3058,((1024,2000),))==1+5*3058
    assert all(np.isfinite(model.coarse(((2048,17),),(3058,4097),'pipeline')))
