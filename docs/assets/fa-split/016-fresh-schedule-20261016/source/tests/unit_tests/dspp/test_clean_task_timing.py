"""Timing-only execution must not introduce annotation autograd or inner events."""
from types import SimpleNamespace
import json

import pytest
import torch

from megatron import profile_utils
from megatron.core import utils
from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.pipeline_parallel import dspp_timeline
from megatron.core.kv_cache.fa_split import prepare_backward, backward_dq, backward_dkv
from .test_fa_split_regions import make_inputs, packed_forward, SCALE


def test_annotations_off_preserve_gradients_without_wrapper_nodes(monkeypatch):
    monkeypatch.setattr(profile_utils, 'ANNOTATIONS_ENABLED', False)
    def forbidden(*a, **kw):
        raise AssertionError('annotation-only execution is still active')
    for cls in (profile_utils.WrapInputsFunction, profile_utils.WrapOutputsFunction,
                profile_utils.BackwardPushFunction, profile_utils.BackwardPopFunction):
        monkeypatch.setattr(cls, 'apply', forbidden)
    monkeypatch.setattr(torch.cuda, 'Event', forbidden)
    @profile_utils.annotate_forward_backward('test', 'test')
    def forward(x):
        return (x*x).sum()
    x=torch.tensor([2.,3.],requires_grad=True)
    with profile_utils.annotate_range('test'):
        loss=forward(x)
    with profile_utils.annotate_backward_range('test'):
        loss.backward()
    torch.testing.assert_close(x.grad,torch.tensor([4.,6.]))


def test_disabled_debug_sync_adds_no_autograd_node(monkeypatch):
    monkeypatch.setattr(utils, 'get_args', lambda:SimpleNamespace(kaimm_cuda_synchronize_level=0))
    def forbidden(*a, **kw):
        raise AssertionError('disabled sync inserted an autograd node')
    monkeypatch.setattr(utils.SyncAtBackwardFunction, 'apply', forbidden)
    x=torch.tensor([2.],requires_grad=True)
    assert utils.cuda_sync_and_record_at_backward(x,sync_level=2) is x


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('scope', ['bh', 'all'])
def test_boundary_events_preallocated_and_metadata_deferred(monkeypatch,tmp_path,scope):
    args=SimpleNamespace(dspp_task_timing_only=True,dspp_timeline_iteration=1,
                         curr_iteration=0,dspp_timeline_dir=str(tmp_path),dspp_task_timing_scope=scope,
                         dspp_timeline_iterations=[1,3])
    monkeypatch.setattr(dspp_timeline.dist,'barrier',lambda:None)
    monkeypatch.setattr(dspp_timeline.dist,'get_rank',lambda:0)
    recorder=dspp_timeline.DsppTimelineRecorder(args,stage=0,ordering={'entrance_order':[(0,0)]})
    meta=build_dspp_batch_plan([17],chunk_size=64).microbatches[0]
    x=torch.ones(64,device='cuda')
    def forbidden(*a, **kw):raise AssertionError('timing work leaked into task loop')
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda,'Event',forbidden)
        patch.setattr(torch.cuda,'synchronize',forbidden)
        patch.setattr(dspp_timeline,'task_signature',forbidden)
        patch.setattr(torch.cuda.nvtx,'range_push',forbidden)
        for kind in ('F0','B0','H0','W0'):
            with recorder.task(kind,(0,0),meta,chunk=0):x.add_(1)
        with recorder.communication('test'):pass
    assert not recorder.records
    assert len(recorder._boundary_records)==(2 if scope=='bh' else 4)
    recorder.finish()
    payload=json.loads(next(tmp_path.glob('iteration_*_rank_*.json')).read_text())
    assert [r['kind'] for r in payload['events']]==(['B0','H0'] if scope=='bh' else ['F0','B0','H0','W0'])
    assert payload['events'][0]['end_ms']<=payload['events'][1]['start_ms']
    assert not list(tmp_path.glob('*_costs.json'))
    summary=json.loads(next(tmp_path.glob('*_summary.json')).read_text())
    assert ('B/H only' if scope=='bh' else 'F/B/W/H boundaries') in summary['task_scope']
    assert 'bubble_ms' not in summary['stages']['0']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('hk',[16,8,4,1])
@pytest.mark.parametrize('qlens,klens',[((65,), (130,)), ((17,65,1,33),(17,128,65,98))])
def test_compact_region_matches_full_without_unselected_rows(hk,qlens,klens):
    cq=(0,*torch.tensor(qlens).cumsum(0).tolist())
    ck=(0,*torch.tensor(klens).cumsum(0).tolist())
    q,k,v,do=make_inputs(sum(qlens),sum(klens),16,hk)
    out,lse,_,_=packed_forward(q,k,v,torch.tensor(cq,device='cuda',dtype=torch.int32),
                               torch.tensor(ck,device='cuda',dtype=torch.int32),SCALE)
    state=prepare_backward(q,k,v,out,do,lse,cu_seqlens_q=cq,cu_seqlens_k=ck)
    for region in ('current','history'):
        full=backward_dkv(state,region)
        compact=backward_dkv(state,region,compact=True)
        slices=[(start+nk-nq,end) if region=='current' else (start,start+nk-nq)
                for start,end,nq,nk in zip(ck,ck[1:],qlens,klens)]
        for a,b in zip(compact,full):
            reference=torch.cat([b[start:end] for start,end in slices])
            torch.testing.assert_close(a,reference,atol=0,rtol=0)
            assert a.shape[0]==sum(end-start for start,end in slices)


@pytest.mark.parametrize('current,selected,enabled', [(0,[1,3],True),(1,[1,3],False),(2,[1,3],True)])
def test_iteration_list_and_disabled_no_events(monkeypatch,current,selected,enabled):
    args=SimpleNamespace(dspp_task_timing_only=True,dspp_timeline_iteration=2,
                         dspp_timeline_iterations=selected,curr_iteration=current,
                         dspp_timeline_dir='/unused',dspp_task_timing_scope='all')
    class Event:
        def record(self):pass
    monkeypatch.setattr(torch.cuda,'Event',lambda **kw:Event())
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(dspp_timeline.dist,'barrier',lambda:None)
    recorder=dspp_timeline.DsppTimelineRecorder(args,stage=0,ordering={'entrance_order':[(0,0)]})
    assert recorder.enabled is enabled
    assert len(recorder._boundary_pairs)==(8 if enabled else 0)
    if not enabled:
        def forbidden(*a,**kw):raise AssertionError('disabled sampling emitted event')
        monkeypatch.setattr(torch.cuda,'Event',forbidden)
        for kind in ('F0','B0','W0','H0'):
            with recorder.task(kind,(0,0),None,chunk=0):pass
        assert not recorder._boundary_records
