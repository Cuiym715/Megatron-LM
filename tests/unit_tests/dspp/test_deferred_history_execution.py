"""One GPU fixture: mixed fused/D paths, two pending Hs, three GQA layers."""
import copy
import json

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.kv_cache import dspp_packed_attention as packed
from .test_packed_split import Block, physical, run_blocks, unpack


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_three_layer_selective_deferred_h_matches_adjacent_and_fused(monkeypatch):
    original = packed._wrapped_flash_attn_varlen_backward
    def deterministic(*args, **kwargs):
        args = list(args)
        args[20] = True
        return original(*args, **kwargs)
    monkeypatch.setattr(packed, '_wrapped_flash_attn_varlen_backward', deterministic)
    torch.manual_seed(301)
    model = nn.ModuleList([Block(4,2) for _ in range(3)]).cuda().to(torch.bfloat16)
    lengths = (145,70,17,3)
    plan = build_dspp_batch_plan(lengths, chunk_size=64)
    gen = torch.Generator(device='cuda').manual_seed(730)
    source = [torch.randn(n,64,device='cuda',generator=gen).to(torch.bfloat16) for n in lengths]
    upstream = [torch.randn(n,64,device='cuda',generator=gen).to(torch.bfloat16)/32 for n in lengths]
    selected = {m.microbatch_id for m in plan.microbatches
                if any(i.has_history and i.is_last_segment for i in m.items)}
    assert len(selected) == 2

    def execute(m, delay):
        state = packed.DsppSequenceKVState(split_attention_backward=True, history_dkv_after_dx=True)
        state.configure_task_modes(plan.microbatches, selected)
        inputs, outputs, peaks = [], [], []
        for meta in plan.microbatches:
            x = physical(source, meta); inputs.append(x)
            positions = torch.cat([torch.arange(i.token_offset,i.token_offset+i.token_length,device='cuda') for i in meta.items])
            positions = F.pad(positions,(0,meta.padding_token_count))
            y = x
            for layer, block in enumerate(m):
                y = block(y,positions,state,meta,layer)
            outputs.append(y)
        dx = {}
        def history(task):
            before = {i: inputs[i].grad.clone() for i in dx}
            assert state.run_history_task(task) == 3
            for i in dx:
                torch.testing.assert_close(inputs[i].grad,before[i],atol=0,rtol=0)
        for index in reversed(range(len(plan.microbatches))):
            meta = plan.microbatches[index]
            task = meta.microbatch_id
            for dependency in sorted(state._required_history[task]-state._completed_history, reverse=True):
                history(dependency)
            state.check_history_consumers(task)
            if task in selected:
                state.begin_backward_task(task)
            outputs[index].backward(physical(upstream,meta).detach())
            dx[index] = inputs[index].grad.clone()
            if task in selected:
                state.mark_dx_ready(task)
                assert state.history_job_count(task) == 3
                peaks.append(len(state._history_records))
                if not delay:
                    history(task)
        assert not state._history_records
        state.finish_iteration()
        return (unpack(plan,[o.detach() for o in outputs],lengths),
                unpack(plan,[dx[i] for i in range(len(inputs))],lengths),
                {n:p.grad.clone() for n,p in m.named_parameters()}), max(peaks)

    adjacent, p1 = execute(copy.deepcopy(model), False)
    delayed, p2 = execute(copy.deepcopy(model), True)
    assert p1 == 1 and p2 == 2
    fused = run_blocks(copy.deepcopy(model),source,upstream,plan,packed.DsppSequenceKVState())
    maxima = {}
    for label, a, b, c in (
        ('outputs',torch.cat(delayed[0]),torch.cat(adjacent[0]),torch.cat(fused[0])),
        ('dx',torch.cat(delayed[1]),torch.cat(adjacent[1]),torch.cat(fused[1])),
        *((n,g,adjacent[2][n],fused[2][n]) for n,g in delayed[2].items())):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(a,c,atol=.004,rtol=.01)
        relative = (a.double()-c.double()).norm()/c.double().norm().clamp_min(1e-8)
        assert relative <= .02, label
        maxima[label] = dict(delayed_vs_adjacent=0., fused_max_abs=float((a-c).abs().max()),
                             fused_relative_l2=float(relative))
    print('P3_MULTILAYER '+json.dumps(dict(pending_tasks=p2, layers=3, hq=4, hk=2, checks=maxima)))
