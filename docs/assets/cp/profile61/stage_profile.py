"""Real DSPP stage replay on two CP ranks; no PP traffic or optimizer step.

Target PP roles are independent from the two-rank sampling topology. Both VPP
chunks are resident. Sampling topology is recorded separately from the target
model context, so these measurements never claim concurrent PP calibration.
"""
import argparse
from dataclasses import asdict
from functools import partial
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=int, choices=range(4), required=True)
parser.add_argument('--chunk', type=int, choices=(0, 1), required=True)
parser.add_argument('--backend', choices=('key-value', 'query-out'), required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--lengths', default='4608')
parser.add_argument('--warmup', type=int, default=2)
parser.add_argument('--repeats', type=int, default=4)
parser.add_argument('--verify', action='store_true')
parser.add_argument('--cpu-preflight', action='store_true')
cfg = parser.parse_args()
source = ROOT / 'source'
sys.path.insert(0, str(source))
import torch
import torch.distributed as dist
from megatron.arguments import parse_args, validate_args
from megatron.global_vars import set_global_variables
from megatron.core import parallel_state as mpu
from megatron.core.enums import ModelType
from megatron.core.zbpp_utils import WeightGradStore
from megatron.core.datasets.dspp_training import build_dspp_training_batch
from megatron.core.datasets.dspp_calibration import stage_context, SCHEMA, StageCalibration
from megatron.core.kv_cache.dspp_cp_attention import prepare_cp_metadata
from megatron.core.pipeline_parallel.schedules import _DsppVTaskBatch, forward_step, backward_step
from megatron.training import get_model
from megatron.initialize import _set_random_seed
from megatron import profile_utils
from pretrain_llama import model_provider, forward_step as llama_forward

rank = int(os.environ['RANK'])
assert int(os.environ['WORLD_SIZE']) == 2
if not cfg.cpu_preflight:
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
command = json.loads((ROOT / 'target-commands.json').read_text())['pp4cp2-kv']
entry = next(i for i, a in enumerate(command) if a.endswith('benchmark_entry61.py'))
sys.argv = [__file__] + command[entry + 1:]
sys.argv[sys.argv.index('--kaimm-context-parallel-impl') + 1] = cfg.backend
# Validate the target model contract with its target world size. Actual groups
# below are deliberately CP2/TP1/PP1, with only PP role getters overridden.
os.environ['WORLD_SIZE'] = '8'
args = parse_args()
validate_args(args)
os.environ['WORLD_SIZE'] = '2'
if cfg.cpu_preflight:
    from megatron.core.dspp_layout import physical_layer_range
    roles = {str(s): [list(physical_layer_range(61, 4, 'endpoint-3-2', s, c))
                     for c in range(2)] for s in range(4)}
    assert sorted(i for chunks in roles.values() for begin, end in chunks
                  for i in range(begin, end)) == list(range(61))
    assert args.context_parallel_size == 2 and args.virtual_pipeline_model_parallel_size == 2
    assert args.gradient_accumulation_fusion and args.hidden_size == 2048
    assert not torch.cuda.is_initialized()
    cfg.output.mkdir(parents=True, exist_ok=True)
    proof = dict(status='CPU-preflight-passed', target_pp=4, target_cp=2,
                 launch_world=2, layer_ranges=roles, CUDA_initialized=False,
                 GPU_execution_validated=False)
    (cfg.output/'cpu-preflight.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof))
    raise SystemExit(0)
set_global_variables(args)
profile_utils.ANNOTATIONS_ENABLED = False
dist.init_process_group('nccl')
mpu.initialize_model_parallel(context_parallel_size=2)
mpu.set_pipeline_model_parallel_world_size(4)
mpu.set_pipeline_model_parallel_rank(cfg.stage)
mpu.set_virtual_pipeline_model_parallel_world_size(2)
_set_random_seed(args.seed, False)
from zero_dropout import install
install()
from megatron.core.kv_cache import dspp_packed_attention as packed, fa_split_build
fa_split_build.NATIVE = Path('/workspace/experiments/dspp-cp-v1-20261106/source-current/megatron/core/kv_cache/fa_split_src')
fa_split_build.load_extension(arch=(8, 0))
native_backward = packed._wrapped_flash_attn_varlen_backward
def deterministic_backward(*a, **kw):
    if len(a) > 20:
        a = list(a)
        a[20] = True
    else:
        kw['deterministic'] = True
    return native_backward(*a, **kw)
packed._wrapped_flash_attn_varlen_backward = deterministic_backward
models = get_model(model_provider, ModelType.encoder_or_decoder)
mpu.set_virtual_pipeline_model_parallel_rank(cfg.chunk)
actual_layers = []
for model in models:
    model.train()
    actual_layers.append(sorted(int(m.layer_number) for m in model.modules()
                                if type(m).__name__ == 'ParallelTransformerLayer'))
contexts = {str(c): stage_context(args, cfg.stage, c) for c in range(2)}
assert all(actual_layers[c] == contexts[str(c)]['layers'] for c in range(2))
WeightGradStore.assert_supported()
lengths = [int(x) for x in cfg.lengths.split(',')]
tokens = torch.arange(len(lengths) * (max(lengths) + 1), device='cuda').reshape(
    len(lengths), max(lengths) + 1) % 511 + 1
batch = build_dspp_training_batch(tokens, [n + 1 for n in lengths],
    chunk_size=2048, max_sequence_length=32768)
metas = tuple(p.meta for p in batch.physical_microbatches)
fwd = partial(forward_step, forward_step_func=llama_forward,
              model=models[cfg.chunk], timers=None, enable_autocast=False)
bwd = partial(backward_step, model_type=ModelType.encoder_or_decoder, timers=None)
is_first = cfg.stage == 0 and cfg.chunk == 0
is_last = cfg.stage == 0 and cfg.chunk == 1
records, comparisons = [], []
cfg.output.mkdir(parents=True, exist_ok=True)
baseline_allocated = torch.cuda.memory_allocated()
reference = None

def reset_grads():
    for model in models:
        model.zero_grad_buffer()
        for p in model.parameters():
            p.grad = None

def capture(tensors):
    return [t.detach().float().clone() for t in tensors if t is not None]

def comparison(actual, expected):
    numerator = denominator = 0.0
    finite = True
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        a = a.detach().float()
        finite = finite and bool(torch.isfinite(a).all())
        numerator += (a - b).square().sum().item()
        denominator += b.square().sum().item()
    return dict(finite=finite, relative_l2=(numerator / max(denominator, 1e-30)) ** .5)

for split_mode in (False, True):
    args.dspp_split_attention_backward = split_mode
    args.dspp_history_dkv_after_dx = split_mode
    for repeat in range(cfg.warmup + cfg.repeats):
        reset_grads()
        WeightGradStore.assert_empty()
        task = _DsppVTaskBatch(batch, fwd, bwd, None, batch.valid_token_count,
                              is_last, cfg.chunk, split_mode, split_mode)
        prepare_cp_metadata(task.kv_state, metas, mpu.get_context_parallel_group())
        inputs, upstream, outputs, dxs, events = [], [], [], [], []
        generator = torch.Generator(device='cuda').manual_seed(9001)
        for meta in metas:
            indices = task.kv_state._cp_prepared[
                'input-index', meta.microbatch_id, rank, torch.device('cuda', torch.cuda.current_device())]
            full = torch.randn((2048, 1, 2048), generator=generator, device='cuda', dtype=torch.bfloat16)
            local = full.new_zeros((1024, 1, 2048))
            local[:indices.numel()].copy_(full.index_select(0, indices))
            inputs.append(None if is_first else local.detach().requires_grad_())
            grad = torch.randn((1024, 1, 2048), generator=generator, device='cuda', dtype=torch.bfloat16) / 1024
            upstream.append(grad)
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        def timed(kind, index, fn):
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            value = fn()
            end.record()
            events.append((kind, index, begin, end))
            return value
        for index in range(len(metas)):
            timed('F', index, lambda i=index: task.forward(i, inputs[i]))
            if cfg.verify and repeat == 0:
                outputs.append(task.outputs[index].detach().clone())
        for index in reversed(range(len(metas))):
            value = timed('D' if split_mode else 'B_fused', index,
                lambda i=index: task.backward(i, None if is_last else upstream[i], task_key=i))
            if cfg.verify and repeat == 0 and value is not None:
                dxs.append(value.detach().clone())
            if split_mode:
                timed('H', index, lambda i=index: task.history(i))
            timed('W', index, lambda i=index: task.weight(task_key=i))
        task.kv_state.finish_iteration()
        WeightGradStore.assert_empty()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        if repeat >= cfg.warmup:
            for kind, index, begin, end in events:
                duration = torch.tensor(begin.elapsed_time(end), dtype=torch.float64, device='cuda')
                dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                meta = metas[index]
                records.append(dict(context=contexts[str(cfg.chunk)], operation=kind,
                    stage=cfg.stage, chunk=cfg.chunk, kind=kind, duration_ms=duration.item(),
                    sequence_chunk_size=meta.chunk_size, items=[asdict(i) for i in meta.items],
                    logical_microbatch=0, physical_microbatch=meta.microbatch_id,
                    repeat=repeat-cfg.warmup, backward_mode='after_dx' if split_mode else 'fused'))
        if cfg.verify and repeat == 0:
            params = [p.main_grad for p in models[cfg.chunk].parameters() if p.requires_grad]
            if not split_mode:
                reference = (capture(outputs), capture(dxs), capture(params))
            else:
                comparisons = [comparison(a, b) for a, b in zip((outputs, dxs, params), reference)]
                assert all(c['finite'] and c['relative_l2'] <= .02 for c in comparisons), comparisons
                reference = None
        del task, events, outputs, dxs, inputs, upstream
        gc.collect()
        if rank == 0:
            print(json.dumps(dict(stage=cfg.stage, chunk=cfg.chunk, split=split_mode,
                                  repeat=repeat, allocated_peak=peak)), flush=True)

assert all(torch.isfinite(p).all() and torch.isfinite(p.main_grad).all()
           for p in models[cfg.chunk].parameters())
result = dict(status='complete', stage=cfg.stage, chunk=cfg.chunk, cp_rank=rank,
    backend=cfg.backend, layers=actual_layers, contexts=contexts, records=records,
    verify=comparisons, baseline_allocated_bytes=baseline_allocated,
    last_replay_allocated_peak=peak, optimizer_state='not allocated; not a memory-profile budget',
    sampling=dict(world_size=2, cp=2, target_pp=4, pp_communication=False,
                  scope='isolated real stage, two VPP chunks resident; synthetic stage inputs',
                  cp_time_reduction='max', default_fa=False, deterministic_fa=True))
(cfg.output / f'rank{rank}.json').write_text(json.dumps(result, indent=2) + '\n')
if rank == 0:
    profile = StageCalibration(records).to_dict()
    profile['sampling'] = result['sampling']
    (cfg.output / 'compute.json').write_text(json.dumps(profile, indent=2) + '\n')
dist.barrier()
dist.destroy_process_group()
