"""CP2 fused/D/H against independent FP64 math, in one torchrun batch.

No production audit hooks or deterministic-FA override. H is explicitly
forbidden from issuing distributed calls, including on empty-query ranks.
"""
import argparse
from contextlib import contextmanager
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.datasets.dspp_cp_cases import CP_LAYOUT_CASES
from megatron.core.datasets.dspp_cp_layout import cp_token_ranges
from megatron.core.kv_cache.dspp_packed_attention import DsppSequenceKVState
from megatron.core.kv_cache.dspp_cp_attention import dspp_cp_flash_attention, prepare_cp_metadata


def shard(x, indices=None):
    if indices is not None:
        index = torch.tensor(indices, dtype=torch.long, device=x.device)
        valid = x.index_select(0, index)
        return torch.cat((valid, x.new_zeros(x.size(0)//dist.get_world_size()-len(indices), *x.shape[1:])))
    return torch.cat([x[a:b] for a, b in cp_token_ranges(x.size(0), dist.get_world_size(), dist.get_rank())])


def metrics(a, b):
    a, b = a.double(), b.double()
    delta = a-b
    norm = b.norm().item()
    return dict(relative_l2=(delta.norm()/norm).item() if norm > 1e-8 else None,
                reference_norm=norm,
                max_abs=delta.abs().max().item())


def check(a, b):
    result = metrics(a, b)
    torch.testing.assert_close(a.double(), b.double(), atol=.03, rtol=.03)
    if result['relative_l2'] is None:
        assert result['max_abs'] <= 3e-6, result
    else:
        assert result['relative_l2'] <= .02, result
    return result


@contextmanager
def no_history_communication():
    names = ('all_gather_into_tensor', 'reduce_scatter_tensor', 'all_reduce',
             'all_gather', 'all_to_all_single', 'isend', 'irecv', 'barrier')
    originals = {name: getattr(dist, name) for name in names}
    def forbidden(*args, **kwargs):
        raise AssertionError('H attempted communication')
    try:
        for name in names:
            setattr(dist, name, forbidden)
        yield
    finally:
        for name, original in originals.items():
            setattr(dist, name, original)


def math_forward(qkv, meta, history):
    outputs = []
    cursor = 0
    for item in meta.items:
        end = cursor + item.token_length
        q, k, v = (t[cursor:end, 0] for t in qkv)
        previous = history.setdefault(item.sequence_id, [])
        keys = torch.cat([p[0] for p in previous] + [k])
        values = torch.cat([p[1] for p in previous] + [v])
        ratio = q.size(1)//k.size(1)
        keys, values = (t.repeat_interleave(ratio, 1).transpose(0, 1)
                        for t in (keys, values))
        scores = q.transpose(0, 1) @ keys.transpose(1, 2) * q.size(-1)**-.5
        mask = (torch.arange(keys.size(1), device=q.device)[None, :]
                > item.token_offset + torch.arange(q.size(0), device=q.device)[:, None])
        out = scores.masked_fill(mask, -torch.inf).softmax(-1) @ values
        outputs.append(out.transpose(0, 1).unsqueeze(1))
        previous.append((k, v))
        cursor = end
    valid = torch.cat(outputs)
    return torch.cat((valid, valid.new_zeros(meta.chunk_size-cursor, *valid.shape[1:])))


def case(lengths, capacity, hk, mode, seed, layout='paired', packing_strategy='baseline', impl='key-value'):
    plan = build_dspp_batch_plan(lengths, chunk_size=capacity,packing_strategy=packing_strategy)
    selected = {m.microbatch_id for m in plan.microbatches if any(i.has_history for i in m.items)}
    state = DsppSequenceKVState(validate_runtime=True,
        split_attention_backward=mode != 'fused', history_dkv_after_dx=mode != 'fused',
        cp_layout=layout, cp_impl=impl)
    if mode != 'fused':
        state.configure_task_modes(plan.microbatches, selected)
    prepare_cp_metadata(state,plan.microbatches,dist.group.WORLD)
    generator = torch.Generator(device='cuda').manual_seed(seed)
    mathematical, inputs, gradients, expected, actual = [], [], [], [], []
    history = {}
    indices = []
    calls = dict(all_gather_into_tensor=0, reduce_scatter_tensor=0)
    payload_bytes = dict.fromkeys(calls,0)
    originals = {name: getattr(dist, name) for name in calls}
    def counted(name):
        def call(*args, **kwargs):
            calls[name] += 1
            payload_bytes[name] += args[1].numel()*args[1].element_size()//(dist.get_world_size() if name=='reduce_scatter_tensor' else 1)
            return originals[name](*args, **kwargs)
        return call
    for name in calls:
        setattr(dist, name, counted(name))
    for meta in plan.microbatches:
        full = tuple(torch.randn(capacity, 1, h, 128, device='cuda', dtype=torch.bfloat16,
                                 generator=generator) for h in (4, hk, hk))
        exact = tuple(t.double().requires_grad_() for t in full)
        index = (state.cp_ownership(meta, dist.get_world_size()).rank_indices[dist.get_rank()]
                 if layout == 'hybrid' else None)
        indices.append(index)
        local = tuple(shard(t, index).requires_grad_() for t in full)
        mathematical.append(exact)
        inputs.append(local)
        gradients.append(torch.randn(full[0].shape, device='cuda', dtype=torch.bfloat16,
                                     generator=generator))
        expected.append(math_forward(exact, meta, history))
        actual.append(dspp_cp_flash_attention(*local, state=state, meta=meta,
                       layer_idx=7, cp_group=dist.group.WORLD))
    for i in reversed(range(len(expected))):
        expected[i].backward(gradients[i].double(), retain_graph=i != 0)
    pending = []
    h_count = 0
    def run_h(task):
        nonlocal h_count
        before = tuple(t.grad.clone() for t in inputs[task])
        with no_history_communication():
            h_count += state.run_history_task(task)
        for tensor, saved in zip(inputs[task], before):
            torch.testing.assert_close(tensor.grad, saved, atol=0, rtol=0)
    observations = []
    for i in reversed(range(len(actual))):
        for task in tuple(pending):
            if task in state._required_history.get(i, ()):
                run_h(task)
                pending.remove(task)
        if i in selected and mode != 'fused':
            state.begin_backward_task(i)
        actual[i].backward(shard(gradients[i], indices[i]))
        if i in selected and mode != 'fused':
            state.mark_dx_ready(i)
            if mode == 'immediate':
                run_h(i)
            else:
                pending.append(i)
        observations.append(dict(task=i, quantity='output', **check(actual[i], shard(expected[i], indices[i]))))
        for name, local, exact in zip(('dq','dk','dv'), inputs[i], mathematical[i]):
            observations.append(dict(task=i, quantity=name, **check(local.grad, shard(exact.grad, indices[i]))))
    for task in pending:
        run_h(task)
    state.finish_iteration()
    expected_collective_tasks = len(actual)
    if layout == 'hybrid':
        expected_collective_tasks = sum(bool(state.cp_ownership(m, dist.get_world_size()).shared_items)
            or (impl == 'query-out' and any(item.segment_count>1 for item in m.items))
            for m in plan.microbatches)
    for name, original in originals.items():
        setattr(dist, name, original)
        expected = expected_collective_tasks * ((4 if name=='all_gather_into_tensor' else 2) if impl=='query-out' else 1)
        assert calls[name] == expected, (name, calls, expected)
    return dict(lengths=lengths, capacity=capacity, kv_heads=hk, mode=mode, layout=layout,
                tasks=len(actual), history_jobs=h_count, communication_calls=calls,
                communication_payload_bytes=payload_bytes,packing_strategy=packing_strategy,
                observations=observations)


def gemm_reference(path):
    reference = torch.load(path, weights_only=True, map_location='cpu')
    key = next(k for k in reference['linear'] if k.startswith('l0:p0:c0:')
               and k.endswith('self_attention.dense:input'))
    parts = key.split(':')
    name = ':'.join(parts[3:-1])
    weight = reference['params_before'][f'0:{name}.weight'].cuda().to(torch.bfloat16)
    x = reference['linear'][key].cuda().to(torch.bfloat16)
    exact = F.linear(x.double(), weight.double())
    full = F.linear(x, weight)
    local = F.linear(shard(x), weight)
    return dict(name=key, full_vs_fp64=metrics(full, exact),
                shard_vs_fp64=metrics(local, shard(exact)),
                shard_vs_full=metrics(local, shard(full)),
                original_vs_reproduced=metrics(full, reference['linear'][key[:-5]+'output'].cuda()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gemm-snapshot', type=Path)
    parser.add_argument('--small-only', action='store_true')
    parser.add_argument('--impl', choices=('key-value','query-out'), default='key-value')
    parser.add_argument('--modes', nargs='+', choices=('fused','immediate','deferred'))
    parser.add_argument('--layout', choices=('paired','hybrid'), default='paired')
    parser.add_argument('--packing-strategy',choices=('baseline','cp-aware'),default='baseline')
    parser.add_argument('--case',help='Run only one named case for a focused regression.')
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    assert dist.get_world_size() in (2,3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/f'rank{dist.get_rank()}.json'
    assert not path.exists(), path
    report = dict(rank=dist.get_rank(), started=True, criteria=dict(atol=.03, rtol=.03,
                  relative_l2=.02, zero_reference_norm=1e-8, zero_reference_atol=3e-6),
                  reference='independent-fp64', deterministic_fa=False, cp=dist.get_world_size(), impl=args.impl, cases=[])
    if args.gemm_snapshot:
        report['gemm'] = gemm_reference(args.gemm_snapshot)
        path.with_suffix('.gemm.json').write_text(json.dumps(report['gemm'], indent=2))
    # Preload the changed extension once before the matrix, not per case.
    if any(m != 'fused' for m in (args.modes or (('fused',) if args.impl == 'query-out' else ('fused','immediate','deferred')))):
        from megatron.core.kv_cache.fa_split_build import load_extension
        load_extension()
    small = (('boundary', (130,17,9,1), 64), ('multiple-history', (257,129,33,7), 128),
             ('empty-local-query', (1,), 64))
    cases = small if args.small_only else (*small, *CP_LAYOUT_CASES[:-1],
        ('cross-bin',(2464,768,3520,3936,864,3328,4384),2048))
    if args.case:
        cases=tuple(case for case in cases if case[0]==args.case)
        if not cases:
            raise ValueError(f'unknown case {args.case}')
    modes = args.modes or (('fused',) if args.impl == 'query-out' else ('fused','immediate','deferred'))
    for label, lengths, capacity in cases:
        divisor = 2*dist.get_world_size()
        capacity = max(divisor, capacity//divisor*divisor)
        for hk in (4,2,1):
            for mode in modes:
                row = case(lengths, capacity, hk, mode, 1234+len(report['cases']), args.layout,args.packing_strategy,args.impl)
                row['name'] = label
                report['cases'].append(row)
                print(f'rank={dist.get_rank()} {label} hkv={hk} {mode} passed', flush=True)
    report['status'] = 'passed'
    path.write_text(json.dumps(report, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
