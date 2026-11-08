#!/usr/bin/env python3
"""Paired P1 training diagnostics: legacy -> template/A, or clean memory run.

CPU snapshot copies intentionally synchronize; numerical mode is not a timing
benchmark. memory mode omits tensor hooks and reports whole-step CUDA peaks.
Uses the existing split-training tolerances, fixed before execution.
"""
import json
import os
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
import torch.distributed as dist
from megatron import get_args
import megatron.training as training
from megatron.core.pipeline_parallel import schedules
from megatron.model import llama_model

OUTPUT = Path(os.environ['DSPP_PLAN_CHECK_OUTPUT'])
SNAPSHOTS = Path(os.environ['DSPP_PLAN_CHECK_SNAPSHOTS'])
MODE = os.environ['DSPP_PLAN_CHECK_MODE']
DETERMINISTIC_FA = os.environ.get('DSPP_PLAN_CHECK_DETERMINISTIC_FA', '0') == '1'
CP_REFERENCE = os.environ.get('DSPP_PLAN_CHECK_CP_REFERENCE', '0') == '1'
REPORT_FAILURES = os.environ.get('DSPP_PLAN_CHECK_REPORT_FAILURES', '0') == '1'
ATTENTION_TRACE = os.environ.get('DSPP_PLAN_CHECK_ATTENTION_TRACE', '0') == '1'
CP_NUMERICS = os.environ.get('DSPP_PLAN_CHECK_CP_NUMERICS', '0') == '1'
ALIGN_STATE = os.environ.get('DSPP_PLAN_CHECK_ALIGN_STATE', '0') == '1'
LINEAR_PROBE = os.environ.get('DSPP_PLAN_CHECK_LINEAR_PROBE', '')
EXPORT_REFERENCE = os.environ.get('DSPP_PLAN_CHECK_EXPORT_REFERENCE', '')
comparison_failed = False
PLAN_ROOT = os.environ.get('DSPP_PLAN_CHECK_PLAN_ROOT')
assert MODE in {'reference', 'compare', 'memory'}
OUTPUT.mkdir(parents=True, exist_ok=True)
SNAPSHOTS.mkdir(parents=True, exist_ok=True)
original_train = training.train_step
original_init = schedules._DsppVTaskBatch.__init__
original_forward = schedules._DsppVTaskBatch.forward
original_backward = schedules._DsppVTaskBatch.backward
original_logits = llama_model.parallel_lm_logits
active = None
current_task = None

if PLAN_ROOT:
    # Exercise the public prebuilt-plan input instead of the runtime builder.
    from megatron.core.pipeline_parallel.dspp_plan import DsppSchedulePlan
    original_dispatch = training.get_forward_backward_func

    def dispatch(*args, **kwargs):
        execute = original_dispatch(*args, **kwargs)
        assert execute is schedules.pipelining_with_dspp_stage_local_v

        def execute_plan(*a, **kw):
            iteration = get_args().curr_iteration + 1
            payload = json.loads((Path(PLAN_ROOT) / f'iteration{iteration}.json').read_text())
            return execute(*a, **kw, dspp_plan=DsppSchedulePlan.from_dict(payload))

        return execute_plan

    training.get_forward_backward_func = dispatch

if DETERMINISTIC_FA:
    # Diagnostic only: select the existing fused FA deterministic reduction.
    # Production attention and kernel sources are untouched. Apply identically
    # to the paired reference/candidate; never use this run as default-path timing.
    from megatron.core.kv_cache import dspp_packed_attention as packed
    original_fa_backward = packed._wrapped_flash_attn_varlen_backward
    assert original_fa_backward is not None, 'requires the baseline FA wrapped API'

    def deterministic_fa_backward(*args, **kwargs):
        if len(args) > 20:
            args = list(args)
            assert isinstance(args[20], bool), 'unexpected FA backward ABI'
            args[20] = True
        else:
            kwargs['deterministic'] = True
        return original_fa_backward(*args, **kwargs)

    packed._wrapped_flash_attn_varlen_backward = deterministic_fa_backward


def snapshot(value):
    return value.detach().float().cpu().clone()


if ATTENTION_TRACE or LINEAR_PROBE:
    # Numerical diagnosis only: capture the first divergence before/after FA.
    # No changes to FA mode, GEMMs, gradients or the production task loop.
    from megatron.model import transformer
    from megatron.core.kv_cache import dspp_cp_attention

    def traced_attention(original):
        def call(*qkv, **kwargs):
            result = original(*qkv, **kwargs)
            if (active is not None and current_task is not None and (ATTENTION_TRACE
                    or (current_task == 'l0:p0:c0' and kwargs['layer_idx'] == 1))):
                prefix = f'{current_task}:layer{kwargs["layer_idx"]}'
                for name, tensor in zip(('q', 'k', 'v', 'out'), (*qkv[:3], result)):
                    active['attention'][f'{prefix}:{name}'] = snapshot(tensor)
            return result
        return call

    transformer.dspp_packed_flash_attention = traced_attention(transformer.dspp_packed_flash_attention)
    dspp_cp_attention.dspp_cp_flash_attention = traced_attention(dspp_cp_attention.dspp_cp_flash_attention)


def tensor_map(value, prefix=''):
    if value is None:
        return {}
    if isinstance(value, torch.Tensor):
        return {prefix: snapshot(value)}
    return {k: t for i, v in enumerate(value) for k, t in tensor_map(v, f'{prefix}/{i}').items()}


def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if active is not None:
        self._plan_check_logical = active['batch_count'] // 2
        active['batch_count'] += 1


def forward(self, split, tensor):
    global current_task
    current_task = f'l{self._plan_check_logical}:p{split}:c{self.chunk}'
    try:
        from megatron.core import parallel_state
        if (parallel_state.get_context_parallel_world_size() > 1
                and self.kv_state.cp_layout == 'hybrid'):
            meta = self.training_batch.physical_microbatches[split].meta
            active['cp_indices'][current_task] = self.kv_state.cp_ownership(
                meta, parallel_state.get_context_parallel_world_size()).rank_indices[
                    parallel_state.get_context_parallel_rank()]
        if LINEAR_PROBE and current_task == 'l0:p0:c0':
            from dataclasses import asdict
            active['probe_meta'][current_task] = asdict(self.training_batch.physical_microbatches[split].meta)
        result = original_forward(self, split, tensor)
        active['valid_masks'][current_task] = snapshot(
            self.training_batch.physical_microbatches[split].loss_mask)
        active['outputs'].update(tensor_map(self.outputs[split], current_task))
        return result
    finally:
        current_task = None


def backward(self, split, grad, *args, **kwargs):
    result = original_backward(self, split, grad, *args, **kwargs)
    key = f'l{self._plan_check_logical}:p{split}:c{self.chunk}'
    active['dx'].update(tensor_map(result, key))
    return result


def logits(*args, **kwargs):
    result = original_logits(*args, **kwargs)
    if active is not None and current_task is not None:
        active['logits'][current_task] = snapshot(result)
    return result


def compare(actual, reference, exact=False, category=''):
    assert actual.keys() == reference.keys(), 'task/tensor set mismatch'
    maximum = 0.0
    max_relative_l2 = 0.0
    failures = []
    details = []
    for name, a in actual.items():
        b = reference[name]
        task = ':'.join(name.split(':')[:3]).split('/')[0]
        indices = active['cp_indices'].get(task)
        if CP_REFERENCE and a.shape != b.shape:
            # Reference uses the same PP/VPP layout with CP1. Only sequence
            # axes may be sharded; parameter gradients must stay replicated.
            cp = get_args().context_parallel_size
            dims = [i for i, (x, y) in enumerate(zip(a.shape, b.shape)) if x != y]
            assert category in {'outputs', 'dx', 'logits', 'attention', 'linear'} and len(dims) == 1, (category, name)
            dim = dims[0]
            assert a.ndim == b.ndim and a.shape[dim] * cp == b.shape[dim], name
            from megatron.core.context_parallel.dattention import slice_cp
            from megatron.core import parallel_state
            if indices is not None:
                valid = b.index_select(dim, torch.tensor(indices, dtype=torch.long))
                shape = list(b.shape)
                shape[dim] //= cp
                b = b.new_zeros(shape)
                b.narrow(dim, 0, len(indices)).copy_(valid)
            else:
                b = slice_cp(b, dim, cp, parallel_state.get_context_parallel_rank())
        if CP_NUMERICS and category in {'outputs', 'dx', 'logits', 'attention', 'linear'}:
            mask = active['valid_masks'][task]
            if a.ndim >= 2 and a.shape[0] == 1 and a.shape[1] == mask.size(1):
                a, b = a[:, mask[0].bool()], b[:, mask[0].bool()]
            elif a.ndim >= 2:
                if indices is not None:
                    valid_rows = torch.arange(a.size(0)) < len(indices)
                elif CP_REFERENCE:
                    from megatron.core.context_parallel.dattention import slice_cp
                    from megatron.core import parallel_state
                    valid_rows = slice_cp(mask, 1, get_args().context_parallel_size,
                        parallel_state.get_context_parallel_rank())[0].bool()
                else:
                    valid_rows = mask[0].bool()
                a, b = a[valid_rows], b[valid_rows]
        if a.numel() == 0:
            assert b.numel() == 0
            continue
        assert torch.isfinite(a).all(), name
        norm = b.double().norm().item()
        diff = a.double() - b.double()
        relative_l2 = diff.norm().item() / max(norm, 1e-12)
        max_relative_l2 = max(max_relative_l2, relative_l2)
        forward_numeric = CP_NUMERICS and category in {'outputs', 'logits', 'attention', 'linear'}
        atol, rtol = ((.03, .03) if forward_numeric else (.004, .01))
        mismatch = ~torch.isclose(a, b, atol=0 if exact else atol, rtol=0 if exact else rtol)
        if ATTENTION_TRACE and category in {'attention', 'linear'}:
            details.append(dict(name=name, different_elements=(a != b).sum().item(),
                                max_abs=diff.abs().max().item(), relative_l2=relative_l2))
        if REPORT_FAILURES:
            if mismatch.any() or (norm > 1e-8 and not exact and relative_l2 > .02):
                item = dict(name=name, mismatched_elements=mismatch.sum().item(),
                            elements=a.numel(), max_abs=diff.abs().max().item(),
                            relative_l2=relative_l2)
                if category == 'logits' and not CP_NUMERICS:
                    mask = active['valid_masks'][name]
                    if a.shape[0] != mask.shape[1]:
                        from megatron.core.context_parallel.dattention import slice_cp
                        from megatron.core import parallel_state
                        mask = slice_cp(mask, 1, get_args().context_parallel_size,
                                        parallel_state.get_context_parallel_rank())
                    mask = mask.T.unsqueeze(-1).bool()
                    item['valid_token_mismatches'] = (mismatch & mask).sum().item()
                failures.append(item)
        else:
            torch.testing.assert_close(a, b, atol=0 if exact else atol, rtol=0 if exact else rtol,
                                       msg=lambda m: f'{category}/{name}: {m}')
            if norm > 1e-8 and not exact:
                assert relative_l2 <= .02, name
        maximum = max(maximum, diff.abs().max().item())
    result = dict(tensors=len(actual), max_abs=maximum, max_relative_l2=max_relative_l2,
                  passed=not failures, failures=failures)
    if details:
        result['details'] = details
    return result


def parameters(model):
    return {f'{i}:{n}': snapshot(p) for i, m in enumerate(model) for n, p in m.named_parameters()}


def train_step(*args, **kwargs):
    global active, comparison_failed
    model = kwargs.get('model', args[3] if len(args) > 3 else None)
    optimizer = kwargs.get('optimizer', args[4] if len(args) > 4 else None)
    config = get_args()
    iteration, rank = config.curr_iteration + 1, dist.get_rank()
    reference_rank = rank
    if CP_REFERENCE:
        from megatron.core import parallel_state
        reference_rank = parallel_state.get_pipeline_model_parallel_rank()
    path = SNAPSHOTS / f'iteration{iteration}-rank{reference_rank}.pt'
    optimizer_path = SNAPSHOTS / f'optimizer-iteration{iteration}-rank{reference_rank}.pt'
    record_path = OUTPUT / f'iteration{iteration}-rank{rank}.json'
    assert not record_path.exists(), 'refusing to overwrite validation report'
    if MODE == 'memory':
        torch.cuda.synchronize()
        allocated_start = torch.cuda.memory_allocated()
        reserved_start = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        result = original_train(*args, **kwargs)
        torch.cuda.synchronize()
        record = dict(allocated_start=allocated_start, reserved_start=reserved_start,
                      allocated_peak=torch.cuda.max_memory_allocated(),
                      reserved_peak=torch.cuda.max_memory_reserved(),
                      allocated_end=torch.cuda.memory_allocated(),
                      reserved_end=torch.cuda.memory_reserved())
    else:
        active = dict(batch_count=0, outputs={}, dx={}, logits={}, valid_masks={}, attention={}, linear={}, cp_indices={}, probes={}, probe_meta={})
        reference = torch.load(path, map_location='cpu', weights_only=True) if MODE == 'compare' else None
        if ALIGN_STATE:
            if MODE == 'reference':
                assert not optimizer_path.exists(), optimizer_path
                torch.save(optimizer.state_dict(), optimizer_path)
            else:
                optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu', weights_only=True))
                with torch.no_grad():
                    for i, chunk in enumerate(model):
                        for name, parameter in chunk.named_parameters():
                            parameter.copy_(reference['params_before'][f'{i}:{name}'])
        if EXPORT_REFERENCE:
            export = Path(EXPORT_REFERENCE)
            export.mkdir(parents=True, exist_ok=True)
            torch.save(optimizer.state_dict(), export/f'optimizer-iteration{iteration}-rank{rank}.pt')
        values = dict(params_before=parameters(model))
        original_step = optimizer.step
        def step(*a, **kw):
            values['gradients'] = {}
            for i, m in enumerate(model):
                for n, p in m.named_parameters():
                    if p.requires_grad:
                        grad = getattr(p, 'main_grad', None)
                        if grad is None:
                            grad = p.grad
                        assert grad is not None, (i, n)
                        values['gradients'][f'{i}:{n}'] = snapshot(grad)
            return original_step(*a, **kw)
        optimizer.step = step
        handles = []
        if ATTENTION_TRACE or LINEAR_PROBE:
            from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
            def capture_linear(name):
                def hook(module, inputs, output):
                    if active is not None and current_task is not None:
                        prefix = f'{current_task}:{name}'
                        if ATTENTION_TRACE:
                            active['linear'][f'{prefix}:input'] = snapshot(inputs[0])
                            active['linear'][f'{prefix}:output'] = snapshot(output[0])
                        if LINEAR_PROBE and current_task == 'l0:p0:c0':
                            active['probes'][prefix] = dict(input=snapshot(inputs[0]),output=snapshot(output[0]),weight=snapshot(module.weight))
                return hook
            for chunk in model:
                for name, module in chunk.named_modules():
                    if (isinstance(module, (ColumnParallelLinear, RowParallelLinear))
                            and (not LINEAR_PROBE or name.endswith(tuple(LINEAR_PROBE.split(','))))):
                        handles.append(module.register_forward_hook(capture_linear(name)))
        try:
            result = original_train(*args, **kwargs)
        finally:
            optimizer.step = original_step
            for handle in handles:
                handle.remove()
        values.update(params_after=parameters(model), outputs=active['outputs'],
                      dx=active['dx'], logits=active['logits'],
                      loss={k: torch.as_tensor(v).float().cpu() for k, v in result[0].items()})
        if ATTENTION_TRACE:
            values['attention'] = active['attention']
            values['linear'] = active['linear']
        assert values['outputs'] and values['gradients']
        if rank == 0:
            assert values['logits'] and values['loss']
        if LINEAR_PROBE:
            torch.save(dict(tensors=active['probes'],attention=active['attention'],meta=active['probe_meta'],indices=active['cp_indices']),
                       OUTPUT/f'probe-iteration{iteration}-rank{rank}.pt')
        if EXPORT_REFERENCE:
            torch.save(values, Path(EXPORT_REFERENCE)/f'iteration{iteration}-rank{rank}.pt')
        if MODE == 'reference':
            assert not path.exists(), 'refusing to overwrite reference snapshot'
            torch.save(values, path)
            record = dict(reference=True)
        else:
            record = {k: compare(v, reference[k], exact=(k == 'params_before' and (iteration == 1 or ALIGN_STATE)), category=k)
                      for k, v in values.items()}
            comparison_failed |= any(not item['passed'] for item in record.values())
        active = None
    plan = getattr(config, '_dspp_last_schedule_plan', None)
    record.update(rank=rank, iteration=iteration, mode=MODE, schedule=config.dspp_schedule,
                  diagnostic_deterministic_fa=DETERMINISTIC_FA,
                  cp_reference=CP_REFERENCE,
                  report_failures=REPORT_FAILURES,
                  attention_trace=ATTENTION_TRACE,
                  cp_numerics=CP_NUMERICS,
                  aligned_optimizer_and_weights=ALIGN_STATE,
                  plan_input='external' if PLAN_ROOT else 'runtime',
                  plan=plan, metrics=getattr(config, '_dspp_last_iteration_metrics', {}),
                  criteria=dict(atol=.004, rtol=.01, relative_l2_max=.02,
                    forward_atol=.03 if CP_NUMERICS else .004,
                    forward_rtol=.03 if CP_NUMERICS else .01,
                    valid_tokens_only=CP_NUMERICS))
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    return result


training.train_step = train_step
if MODE != 'memory':
    schedules._DsppVTaskBatch.__init__ = init
    schedules._DsppVTaskBatch.forward = forward
    schedules._DsppVTaskBatch.backward = backward
    llama_model.parallel_lm_logits = logits
if __name__ == '__main__':
    if '--dspp-split-attention-backward' in sys.argv:
        # Prepare once, before PP traffic. Never compile on a first D deadline.
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
        from megatron.core.kv_cache.fa_split_build import load_extension
        load_extension()
    runpy.run_path(str(REPO / 'pretrain_llama.py'), run_name='__main__')
    if REPORT_FAILURES and comparison_failed:
        raise SystemExit('numerical diagnostic completed with failed comparisons; see JSON reports')
