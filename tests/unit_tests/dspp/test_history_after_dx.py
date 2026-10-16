"""Complete block dx must be correct BEFORE any history kernel is executed."""
import copy
import json

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.kv_cache import dspp_packed_attention as packed
from megatron.core.kv_cache import fa_split
from .test_packed_split import Block, physical, unpack, run_blocks, errors

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def run_after_dx(model, source, upstream, plan, state, monkeypatch):
    inputs, outputs = [], []
    for meta in plan.microbatches:
        x = physical(source, meta)
        inputs.append(x)
        positions = torch.cat([
            torch.arange(i.token_offset, i.token_offset+i.token_length, device='cuda')
            for i in meta.items
        ])
        positions = F.pad(positions, (0, meta.padding_token_count))
        y = x
        for layer, block in enumerate(model):
            y = block(y, positions, state, meta, layer)
        outputs.append(y)
    actual_dkv = fa_split.backward_dkv
    in_b = False
    history_calls = []
    def dkv(s, region='all', **kwargs):
        if region == 'history':
            assert not in_b, 'history kernel ran before complete stage dx'
            if isinstance(s, fa_split.VarlenSplitBackwardState):
                history_calls.append(region)
        return actual_dkv(s, region, **kwargs)
    snapshots, records = {}, []
    with monkeypatch.context() as patch:
        patch.setattr(fa_split, 'backward_dkv', dkv)
        for index in reversed(range(len(plan.microbatches))):
            meta = plan.microbatches[index]
            state.begin_backward_task(meta.microbatch_id)
            in_b = True
            outputs[index].backward(physical(upstream, meta).detach())
            in_b = False
            # Snapshot full block/stage input gradient BEFORE H, not attention dQ.
            dx = inputs[index].grad
            assert dx is not None and torch.isfinite(dx).all()
            snapshots[index] = dx.clone()
            params = {n: p.grad.clone() for n, p in model.named_parameters()}
            expected = len(model) if any(i.token_offset for i in meta.items) else 0
            assert state.pending_history_count == expected
            for s, layer, layouts, shape in state._history_jobs:
                assert all(t.grad_fn is None for t in s.tensors())
                assert s.k.shape[1] == model[layer].hk
            with pytest.raises(RuntimeError, match='completed stage dx'):
                state.run_history_task(meta.microbatch_id)
            state.mark_dx_ready(meta.microbatch_id)
            with pytest.raises(RuntimeError, match='previous history'):
                state.begin_backward_task(meta.microbatch_id)
            with pytest.raises(RuntimeError, match='not completed'):
                state.finish_iteration()
            with pytest.raises(RuntimeError, match='unfinished history'):
                state.clear()
            before = len(history_calls)
            assert state.run_history_task(meta.microbatch_id) == expected
            assert len(history_calls)-before == expected
            # History changes only cached gradients of earlier chunks.
            torch.testing.assert_close(dx, snapshots[index], atol=0, rtol=0)
            for n, p in model.named_parameters():
                torch.testing.assert_close(p.grad, params[n], atol=0, rtol=0)
            assert not state.pending_history_count
            with pytest.raises(RuntimeError, match='completed stage dx'):
                state.run_history_task(meta.microbatch_id)
            records.append({'physical':meta.microbatch_id, 'history_layers':expected,
                            'dx_unchanged_by_history':True})
    state.finish_iteration()
    lengths = list(map(len, source))
    return (unpack(plan, [o.detach() for o in outputs], lengths),
            unpack(plan, [snapshots[i] for i in range(len(inputs))], lengths),
            {n:p.grad.clone() for n,p in model.named_parameters()}), records


@pytest.mark.parametrize('hk', [4,2,1])
@pytest.mark.parametrize('depth', [1,3])
@pytest.mark.parametrize('lengths', [(145,70,17,3), (17,3)])
@pytest.mark.parametrize('chunk_size', [63,64])
def test_complete_dx_before_history(depth, hk, lengths, chunk_size, monkeypatch):
    torch.manual_seed(301)
    model = nn.ModuleList([Block(4,hk) for _ in range(depth)]).cuda().to(torch.bfloat16)
    fused = copy.deepcopy(model)
    immediate = copy.deepcopy(model)
    plan = build_dspp_batch_plan(lengths, chunk_size=chunk_size)
    state = packed.DsppSequenceKVState(split_attention_backward=True, history_dkv_after_dx=True)
    results = []
    for iteration in range(2):
        gen = torch.Generator(device='cuda').manual_seed(730+iteration)
        source = [torch.randn(n,64,device='cuda',generator=gen).to(torch.bfloat16) for n in lengths]
        upstream = [torch.randn(n,64,device='cuda',generator=gen).to(torch.bfloat16)/32 for n in lengths]
        for m in (model, fused, immediate):
            m.zero_grad(set_to_none=True)
        a, records = run_after_dx(model,source,upstream,plan,state,monkeypatch)
        for name, reference_model, split in [('fused',fused,False),('immediate',immediate,True)]:
            b = run_blocks(reference_model,source,upstream,plan,
                           packed.DsppSequenceKVState(split_attention_backward=split))
            checks = {}
            for tensor_name,x,y in [('output',torch.cat(a[0]),torch.cat(b[0])),
                                    ('stage_dx_before_H',torch.cat(a[1]),torch.cat(b[1])),
                                    *[(n,g,b[2][n]) for n,g in a[2].items()]]:
                torch.testing.assert_close(x,y,atol=.004,rtol=.01)
                checks[tensor_name] = errors(x,y)
                assert checks[tensor_name]['relative_l2'] <= .02
            results.append({'iteration':iteration,'reference':name,'checks':checks})
    print('AFTER_DX_CHECK '+json.dumps({'depth':depth,'hk':hk,'lengths':lengths,'chunk_size':chunk_size,
                                      'tasks':records,'comparisons':results}))


def test_requires_task_owner_and_preserves_input_versions():
    with pytest.raises(ValueError, match='requires split'):
        packed.DsppSequenceKVState(history_dkv_after_dx=True)
    plan = build_dspp_batch_plan([129], chunk_size=64)
    state = packed.DsppSequenceKVState(split_attention_backward=True, history_dkv_after_dx=True)
    outputs=[]
    for meta in plan.microbatches:
        q,k,v = [torch.randn(64,1,4,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
                 for _ in range(3)]
        outputs.append(packed.dspp_packed_flash_attention(q,k,v,state=state,meta=meta,layer_idx=0))
    meta=plan.microbatches[-1]
    # Fail before preparing gradients or routing if invoked without the owner.
    with pytest.raises(RuntimeError, match='active physical task'):
        outputs[-1].sum().backward(retain_graph=True)
    state.begin_backward_task(meta.microbatch_id)
    outputs[-1].sum().backward()
    state.mark_dx_ready(meta.microbatch_id)
    with pytest.raises(RuntimeError, match='completed stage dx'):
        state.run_history_task(meta.microbatch_id+1)
    job=state._history_jobs[0][0]
    job.q.add_(1)
    with pytest.raises(RuntimeError, match='modified'):
        state.run_history_task(meta.microbatch_id)
    with pytest.raises(RuntimeError, match='not completed'):
        state.finish_iteration()


def test_cli_after_dx_requires_split(monkeypatch):
    import sys
    from megatron.arguments import parse_args, validate_args
    monkeypatch.setattr(sys, 'argv', ['test', '--dspp-history-dkv-after-dx',
        '--num-layers', '1', '--hidden-size', '512', '--num-attention-heads', '4',
        '--micro-batch-size', '1'])
    monkeypatch.setenv('WORLD_SIZE', '1')
    monkeypatch.setenv('RANK', '0')
    with pytest.raises(AssertionError, match='requires --dspp-split-attention-backward'):
        validate_args(parse_args())
