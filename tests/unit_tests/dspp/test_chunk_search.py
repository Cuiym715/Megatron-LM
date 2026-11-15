"""Bounded search, cross-capacity costs, and shortlist/base reuse contracts."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from megatron.core.datasets.dspp_chunk_search import SearchOptions, ChunkSearch, sample_capacities, aggregate_features
from megatron.core.datasets.dspp_chunk_cost import ChunkCostModel


@pytest.mark.parametrize('spacing',['linear','log'])
@pytest.mark.parametrize('step',[1,64,257])
def test_large_domain_samples_only_breakpoints_within_budget(spacing, step):
    lengths = (3058, 10**12 - 1)
    caps = sample_capacities(257, 10**12, lengths, SearchOptions(32, step, spacing))
    assert 2 <= len(caps) <= 32 and len(caps) == len(set(caps))
    assert min(caps) >= 257 and max(caps) <= 10**12
    assert all(c % step == 0 for c in caps)
    assert all(
        any((length - 1) // c < (length - 1) // (c - step) for length in lengths)
        for c in caps[1:-1]
    )


def test_slice_breakpoints_include_3200_without_reserving_4096():
    lengths = (1024, 3072, 3328, 3584, 3840, 6400)
    caps = sample_capacities(2048, 8192, lengths, SearchOptions())
    assert caps == (2048, 2176, 3072, 3200, 3328, 3584, 3840, 6400, 8192)


def test_only_top_two_reuse_packing_and_rejection_does_not_expand():
    packed = {}
    seen = []
    chosen = SimpleNamespace(capacity=3840, predicted_ms=1.)

    def pack(capacity):
        assert capacity not in packed
        plans = (SimpleNamespace(microbatches=(None, None)), SimpleNamespace(microbatches=(None,)))
        packed[capacity] = plans
        return plans

    class Model:
        domain = (2048, 8192)

        def coarse(self, histogram, capacities, mode, *, block_counts):
            assert capacities == tuple(packed)
            assert block_counts == [3] * len(capacities)
            return [{3200: 0., 3840: 1.}.get(c, 2.) for c in capacities]

    def fine(capacity, plans):
        seen.append(capacity)
        assert plans is packed[capacity]
        if capacity == 3200:
            raise ValueError('infeasible')
        return chosen

    prepared, metrics = ChunkSearch(Model()).select([6400, 3840], pack, fine)
    assert seen == metrics['shortlist'] == [3200, 3840]
    assert 4096 not in packed
    assert metrics['rejected'] == {3200: 'infeasible'}
    assert prepared is chosen


def test_fixed_mode_has_one_fine_and_no_coarse():
    class Model:
        domain=(1,100)
        def coarse(self,*args):raise AssertionError('fixed mode called coarse')
    calls=[]
    def fine(c, plans):
        assert plans is None
        calls.append(c)
        return SimpleNamespace(capacity=c, predicted_ms=1)
    _,m=ChunkSearch(Model()).select([128],None,fine,fixed=37)
    assert calls==[37] and m['points']==0


def test_no_accepted_base_fails_without_a_third_candidate():
    class Model:
        domain=(1,100)
        def coarse(self,hist,caps,mode,**kwargs):return caps
    calls=[]
    def fine(c, plans):calls.append(c);raise ValueError('infeasible')
    with pytest.raises(ValueError,match='no feasible base'):
        ChunkSearch(Model(),SearchOptions(10,1)).select([500],lambda c: (),fine)
    assert len(calls)==2


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
    assert model.coarse(((2048,17),),(3058,),'work',block_counts=[18])[0] > model.coarse(
        ((2048,17),),(3058,),'work',block_counts=[17],
    )[0]
