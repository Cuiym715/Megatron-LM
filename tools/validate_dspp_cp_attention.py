"""Stage-1 numerical check: torchrun --nproc-per-node=2 tools/validate_dspp_cp_attention.py.

Uses default high-performance FA, no deterministic override. Each CP rank
independently computes the existing CP1 packed reference from identical data.
This checks the operator, not pipeline scheduling or an optimizer update.
"""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.datasets.dspp_cp_layout import cp_token_ranges
from megatron.core.kv_cache.dspp_packed_attention import (
    DsppSequenceKVState, dspp_packed_flash_attention,
)
from megatron.core.kv_cache.dspp_cp_attention import dspp_cp_flash_attention


def shard(tensor, size, rank):
    return torch.cat([tensor[a:b] for a, b in
                      cp_token_ranges(tensor.size(0), size, rank)], dim=0)


def compare(actual, expected):
    actual, expected = actual.float(), expected.float()
    error = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    torch.testing.assert_close(actual, expected, atol=.03, rtol=.03)
    if error.item() > .02:
        raise AssertionError(f'relative L2 {error.item()} > .02')
    return dict(relative_l2=error.item(), max_absolute=(actual - expected).abs().max().item())


def check_case(lengths, capacity, heads, kv_heads, seed):
    rank, size = dist.get_rank(), dist.get_world_size()
    plan = build_dspp_batch_plan(lengths, chunk_size=capacity)
    gen = torch.Generator(device='cuda').manual_seed(seed)
    reference, local, grads = [], [], []
    reference_state = DsppSequenceKVState(validate_runtime=True)
    cp_state = DsppSequenceKVState(validate_runtime=True)
    expected, actual = [], []
    for meta in plan.microbatches:
        qkv = tuple(torch.randn(capacity, 1, h, 128, generator=gen,
                                 device='cuda', dtype=torch.bfloat16,
                                 requires_grad=True) for h in (heads, kv_heads, kv_heads))
        local_qkv = tuple(shard(t.detach(), size, rank).requires_grad_() for t in qkv)
        reference.append(qkv)
        local.append(local_qkv)
        grads.append(torch.randn(qkv[0].shape, generator=gen, device='cuda',
                                 dtype=torch.bfloat16))  # Deliberately nonzero padding dO.
        expected.append(dspp_packed_flash_attention(
            *qkv, state=reference_state, meta=meta, layer_idx=7))
        actual.append(dspp_cp_flash_attention(
            *local_qkv, state=cp_state, meta=meta, layer_idx=7, cp_group=dist.group.WORLD))
    observations = []
    for i in reversed(range(len(plan.microbatches))):
        observations.append(dict(task=i, quantity='output', **compare(
            actual[i], shard(expected[i], size, rank))))
        expected[i].backward(grads[i])
        actual[i].backward(shard(grads[i], size, rank))
        for name, a, b in zip(('dq', 'dk', 'dv'), local[i], reference[i]):
            observations.append(dict(task=i, quantity=name, **compare(
                a.grad, shard(b.grad, size, rank))))
    reference_state.finish_iteration()
    cp_state.finish_iteration()
    return dict(lengths=lengths, capacity=capacity, heads=heads, kv_heads=kv_heads,
                tasks=len(reference), observations=observations)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import os
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    if dist.get_world_size() != 2:
        raise ValueError('this acceptance matrix requires exactly CP2')
    path = args.output_dir / f'rank{dist.get_rank()}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    results = []
    # History + packed tails, crossing boundaries, zero local Q, and repeated
    # independent iterations of the same layouts with fresh cache state.
    for repeat in range(2):
        for heads, kv_heads in ((4, 4), (4, 2), (4, 1)):
            for lengths, capacity in (([130, 17, 9, 1], 64), ([1], 64),
                                      ([65, 63, 16, 2], 64), ([257, 129, 33, 7], 128)):
                result = check_case(lengths, capacity, heads, kv_heads,
                                    100 + repeat * 1000 + len(results))
                results.append(result)
                print(f'rank={dist.get_rank()} case={len(results)} passed', flush=True)
    torch.cuda.synchronize()
    with path.open('x') as stream:
        json.dump(dict(status='passed', rank=dist.get_rank(), cp=dist.get_world_size(),
                       torch=str(torch.__version__), cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(), deterministic=False,
                       atol=.03, rtol=.03, relative_l2_limit=.02, cases=results), stream, indent=2)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
