#!/usr/bin/env python3
"""Isolated diagnostic, not production B/H timing or a training speedup test.

Compare eager submission with CUDA-graph replay of the SAME operations, then
capture a separate profiler trace. Only the outer call has timing events.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from flash_attn.flash_attn_interface import (
    _wrapped_flash_attn_varlen_forward, _wrapped_flash_attn_varlen_backward,
)
from megatron.core.kv_cache.fa_split import prepare_backward, backward_dq, backward_dkv, backward_dq_current
from megatron.core.kv_cache.fa_split_build import load_extension


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=3)
    p.add_argument('--combined-probe', action='store_true', help='Build a diagnostic-only combined-current kernel')
    p.add_argument('--q-heads', type=int, default=4)
    p.add_argument('--kv-heads', type=int, default=2)
    p.add_argument('--production-current', action='store_true', help='Diagnose the integrated combined B path only')
    args = p.parse_args()
    if args.q_heads < 1 or args.kv_heads < 1 or args.q_heads % args.kv_heads:
        p.error('Q heads must be a positive multiple of KV heads')
    if args.production_current and args.combined_probe:
        p.error('Select production-current or diagnostic combined-probe')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(1234)
    q, k, v, dout = [torch.randn(n, h, 128, device='cuda', dtype=torch.bfloat16)
                     for n, h in ((2048, args.q_heads), (6144, args.kv_heads),
                                  (6144, args.kv_heads), (2048, args.q_heads))]
    cq = torch.tensor([0, 2048], device='cuda', dtype=torch.int32)
    ck = torch.tensor([0, 6144], device='cuda', dtype=torch.int32)
    scale = 128 ** -.5
    out, lse, _, rng = _wrapped_flash_attn_varlen_forward(
        q, k, v, cq, ck, 2048, 6144, 0., scale, True, -1, -1, 0., None, False,
        None, None, None, False)

    def prepare():
        return prepare_backward(q, k, v, out, dout, lse, softmax_scale=scale,
                                cu_seqlens_q=(0, 2048), cu_seqlens_k=(0, 6144))

    def fused():
        grads = [torch.empty_like(t) for t in (q, k, v)]
        _wrapped_flash_attn_varlen_backward(
            dout, q, k, v, out, lse, *grads, cq, ck, 2048, 6144, 0., scale,
            True, -1, -1, 0., None, False, rng, False)
        return tuple(grads)

    def split_b():
        s = prepare()
        return backward_dq(s), *backward_dkv(s, 'current', compact=True)

    ext = load_extension()
    def native_b():
        delta = ext.prepare(q, k, v, out, dout, lse, scale)
        return (ext.dq(q, k, v, out, dout, lse, delta, scale),
                *ext.dkv(q, k, v, out, dout, lse, delta, scale, 4096, 6144, True))

    state = prepare()
    calls = {'fused': fused, 'split_B': split_b, 'native_B': native_b,
             'prepare': prepare, 'dq': lambda: backward_dq(state),
             'current': lambda: backward_dkv(state, 'current', compact=True),
             'H': lambda: backward_dkv(state, 'history', compact=True)}
    check_names = ['split_B', 'native_B']
    if args.production_current:
        def production_b():
            return backward_dq_current(prepare())
        def production_native_b():
            delta = ext.prepare(q, k, v, out, dout, lse, scale)
            return ext.dq_current(q, k, v, out, dout, lse, delta, scale)
        calls = {'fused':fused, 'combined_B':production_b,
                 'combined_native_B':production_native_b,
                 'H':lambda:backward_dkv(state,'history',compact=True)}
        check_names = ['combined_B', 'combined_native_B']
    if args.combined_probe:
        from torch.utils.cpp_extension import load
        from megatron.core.kv_cache.fa_split_build import source_root
        here = Path(__file__).resolve().parent
        major, minor = torch.cuda.get_device_capability()
        probe = load(name='dspp_cost_probe_cuda',
                     sources=[str(here/'dspp_cost_probe_bindings.cpp'), str(here/'dspp_cost_probe.cu')],
                     extra_include_paths=[str(Path(ext.__file__).parent),
                        str(source_root()/'csrc/flash_attn/src'), str(source_root()/'csrc/cutlass/include')],
                     extra_cflags=['-O3'], extra_cuda_cflags=['-O3','-std=c++17','--use_fast_math',
                        '--expt-relaxed-constexpr','--expt-extended-lambda',
                        f'-DDSPP_FA_SPLIT_ARCH={major}{minor}',
                        f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'])
        def combined():
            delta = ext.prepare(q, k, v, out, dout, lse, scale)
            return probe.combined(q, k, v, out, dout, lse, delta, scale)
        calls['combined_native_B'] = combined
        check_names.append('combined_native_B')
    reference = fused()
    correctness = {}
    for name in check_names:
        result = calls[name]()
        correctness[name] = []
        for actual, expected in zip(result, (reference[0], reference[1][4096:], reference[2][4096:])):
            torch.testing.assert_close(actual, expected, atol=.004, rtol=.01)
            correctness[name].append(float((actual.float()-expected.float()).norm()/expected.float().norm()))
    for actual, expected in zip(calls['H'](), (reference[1][:4096], reference[2][:4096])):
        torch.testing.assert_close(actual, expected, atol=.004, rtol=.01)
    for _ in range(5):
        for fn in calls.values(): fn()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record(); end.record(); end.synchronize()

    def measure(fn):
        start.record(); result = fn(); end.record(); end.synchronize()
        return start.elapsed_time(end)*1000, result

    eager = {n: [] for n in calls}
    names = list(calls)
    for i in range(args.samples):
        for name in names[i:]+names[:i]:
            us, _ = measure(calls[name]); eager[name].append(us)
    graphs = {}
    # Separate stream for capture; no CPU submission gaps within graph replay.
    # Calls using state created on the original stream must be captured there.
    for name, fn in calls.items():
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=torch.cuda.current_stream()):
            result = fn()
        for _ in range(3): g.replay()
        torch.cuda.synchronize()
        samples = [measure(g.replay)[0] for _ in range(args.samples)]
        graphs[name] = samples
        # result remains live until all replay work is complete.
        del result, g
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for name, fn in calls.items():
            with torch.profiler.record_function('COST/' + name): fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(args.output/'execution.pt.trace.json'))
    stats = lambda d: {n: {'median_us': statistics.median(v), 'samples_us': v} for n,v in d.items()}
    (args.output/'timing.json').write_text(json.dumps({
        'scope': f'Attention-only, Q2048/K6144/Hq{args.q_heads}/Hkv{args.kv_heads}/d128/BF16; NOT complete B.',
        'method': f'{args.samples} outer-boundary samples; profiler separate. Graph is an execution-cost diagnostic, not production training.',
        'correctness_relative_l2': correctness, 'eager': stats(eager), 'graph': stats(graphs),
    }, indent=2)+'\n')
    print((args.output/'timing.json').read_text())


if __name__ == '__main__':
    # Both retained states and graph capture use this same non-default stream.
    with torch.cuda.stream(torch.cuda.Stream()):
        main()
