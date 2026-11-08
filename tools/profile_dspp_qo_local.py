"""Single-GPU Q/O local attention timings; never a CP stage calibration.

Production metadata and FA kernels, with Q/O/LSE materialized before timing.
Excludes CP/PP communication, projections/MLP/W and KV-owner gradient routing.
An optional frozen v1 module supplies its actual old interval partition.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.datasets.dspp_calibration import hardware_identity
from megatron.core.kv_cache import dspp_cp_qo_attention as current
from megatron.core.kv_cache import dspp_packed_attention as packed
from megatron.core.kv_cache.dspp_cp_attention import _prepared
from megatron.core.kv_cache.fa_split import (
    prepare_partial_backward, backward_dq, backward_dq_current, backward_dkv_range,
)
from megatron.core.kv_cache.fa_split_build import load_extension


def forward(q, k, v, cq, ck, causal):
    return packed._wrapped_flash_attn_varlen_forward(
        q, k, v, cq, ck, q.size(0), k.size(0), 0., q.size(-1)**-.5,
        causal=causal, window_size_left=-1, window_size_right=-1, softcap=0.,
        alibi_slopes=None, return_softmax=False, block_table=None,
        leftpad_k=None, seqused_k=None, zero_tensors=False)


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(repeats)]
    for begin, end in events:
        begin.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    values = [a.elapsed_time(b) for a, b in events]
    return dict(median_ms=statistics.median(values), samples_ms=values)


def operations(q, kv, global_out, global_lse, do, info, parts, cu, old):
    # All communication inputs and contiguous part views are setup-only.
    calls = []
    for n, (i, qa, qb, ka, kb, causal) in enumerate(parts):
        qi, ki, vi = q[qa:qb].contiguous(), kv[2*i][ka:kb].contiguous(), kv[2*i+1][ka:kb].contiguous()
        oi, li, _, rng = forward(qi, ki, vi, cu[n, :2], cu[n, 2:], causal)
        calls.append((i, qa, qb, ka, kb, causal, qi, ki, vi,
                      global_out[qa:qb].contiguous(), do[qa:qb].contiguous(),
                      global_lse[qa:qb].T.contiguous(), cu[n, :2], cu[n, 2:], rng))

    def f():
        out = torch.zeros_like(q, dtype=torch.float32)
        lse = q.new_full(q.shape[:2], -torch.inf, dtype=torch.float32)
        for i, qa, qb, ka, kb, causal, qi, ki, vi, _, _, _, cq, ck, _ in calls:
            oi, li, _, _ = forward(qi, ki, vi, cq, ck, causal)
            li = li.T
            if old:
                merged = torch.logaddexp(lse[qa:qb], li)
                a, b = torch.exp(lse[qa:qb]-merged), torch.exp(li-merged)
                out[qa:qb] = out[qa:qb]*a[..., None]+oi.float()*b[..., None]
                lse[qa:qb] = merged
            else:
                out[qa:qb], lse[qa:qb] = oi, li
        weight = torch.where(torch.isfinite(lse), torch.exp(lse-global_lse), 0.)
        return (out*weight[..., None]).to(q.dtype)

    def b():
        dq = torch.zeros_like(q, dtype=torch.float32)
        grads = [torch.zeros_like(t, dtype=torch.float32) for t in kv]
        for i, qa, qb, ka, kb, causal, qi, ki, vi, oi, doi, li, cq, ck, rng in calls:
            dqi, dki, dvi = torch.empty_like(qi), torch.empty_like(ki), torch.empty_like(vi)
            packed._wrapped_flash_attn_varlen_backward(
                doi, qi, ki, vi, oi, li, dqi, dki, dvi, cq, ck,
                qb-qa, kb-ka, 0., q.size(-1)**-.5, causal, -1, -1, 0., None, False, rng, False)
            dq[qa:qb].add_(dqi)
            grads[2*i][ka:kb].add_(dki)
            grads[2*i+1][ka:kb].add_(dvi)
        return dq, grads

    result = dict(F_local=f, B_local=b)
    if not old and any(h for _, _, h in info):
        def prepare(call):
            i, qa, qb, ka, kb, causal, qi, ki, vi, oi, doi, li, _, _, _ = call
            return prepare_partial_backward(qi, ki, vi, oi, doi, li, causal=causal)

        def d():
            dq = torch.zeros_like(q, dtype=torch.float32)
            grads = [torch.zeros_like(t[h:], dtype=torch.float32)
                     for t, h in zip(kv, (h for _, _, h in info for _ in (0, 1)))]
            saved = []
            for call in calls:
                i, qa, qb, _, kb, *_ = call
                h = info[i][2]
                state = prepare(call)
                if kb <= h:
                    dqi = backward_dq(state)
                else:
                    dqi, dki, dvi = backward_dq_current(state, current_begin=h)
                    grads[2*i][:kb-h].add_(dki)
                    grads[2*i+1][:kb-h].add_(dvi)
                dq[qa:qb].add_(dqi)
                saved.append(state)
            return dq, grads, saved

        # H consumes already-prepared D state, exactly as the runtime does.
        saved = [(call, prepare(call)) for call in calls if info[call[0]][2]]

        def h():
            grads = [torch.zeros_like(t[:n], dtype=torch.float32)
                     for t, n in zip(kv, (h for _, _, h in info for _ in (0, 1)))]
            for call, state in saved:
                i = call[0]
                n = info[i][2]
                dk, dv = backward_dkv_range(state, 0, n)
                grads[2*i].add_(dk)
                grads[2*i+1].add_(dv)
            return grads

        result.update(D_local=d, H_local=h)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initial-source', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=15)
    p.add_argument('--smoke-only', action='store_true', help='One history shape; not timing evidence.')
    cfg = p.parse_args()
    assert not cfg.output.exists(), cfg.output
    cfg.output.mkdir(parents=True)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.15)
    torch.manual_seed(20261105)
    load_extension()
    name = 'megatron.core.kv_cache._qo_profile_initial'
    spec = importlib.util.spec_from_file_location(name, cfg.initial_source/'megatron/core/kv_cache/dspp_cp_qo_attention.py')
    initial = importlib.util.module_from_spec(spec)
    sys.modules[name] = initial
    spec.loader.exec_module(initial)
    device = torch.device('cuda', 0)
    rows = []
    scope = dict(schema='dspp-qo-local-compute-v1', hardware=hardware_identity(),
        initial=initial.IMPLEMENTATION, candidate=current.IMPLEMENTATION,
        cp_sizes=[2, 3], chunk_size=2040, query_heads=8, kv_heads=[8, 1], head_dim=128,
        layout='hybrid', dtype='bfloat16', warmup=cfg.warmup, repeats=cfg.repeats,
        excludes=['CP/PP communication', 'complete-stage F/B/D/H/W', 'Q/K/V communication packing',
                  'owner gradient routing', 'KV assembly', 'projection/MLP', 'metadata preparation'],
        online_B_profile_compatible=False, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    scope['performance_evidence'] = not cfg.smoke_only
    (cfg.output/'scope.json').write_text(json.dumps(scope, indent=2))
    def exclusive():
        if cfg.smoke_only:
            return
        raw = subprocess.check_output(['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        if len(raw.strip().splitlines()) != 1:
            scope.update(status='stopped-gpu-sharing', performance_evidence=False, rows=len(rows))
            (cfg.output/'scope.json').write_text(json.dumps(scope, indent=2))
            raise RuntimeError('GPU sharing detected outside timed windows; discard this profile.')
    for hk in (8, 1):
        if cfg.smoke_only and hk != 8:
            continue
        for label, lengths in (('long', (32768,)), ('mixed', (6401, 2305, 97, 31)),
                               ('short', (800, 224, 600, 424))):
            if cfg.smoke_only and label != 'long':
                continue
            plan = build_dspp_batch_plan(lengths, chunk_size=2040)
            bank = {s: tuple(torch.randn(n, hk, 128, device=device, dtype=torch.bfloat16)
                             for _ in (0, 1)) for s, n in enumerate(lengths)}
            for cp in (2, 3):
                if cfg.smoke_only and cp != 2:
                    continue
                for rank in range(cp):
                    if cfg.smoke_only and rank != 0:
                        continue
                    states = {v: packed.DsppSequenceKVState(cp_layout='hybrid', cp_impl='query-out')
                              for v in ('initial', 'candidate')}
                    for version, module in (('initial', initial), ('candidate', current)):
                        module.prepare_qo_metadata(states[version], plan.microbatches, cp, rank, device)
                    history = {}
                    for index, meta in enumerate(plan.microbatches):
                        state = states['candidate']
                        runs = _prepared(state, meta, cp, rank, device)[0]
                        kv = []
                        for i, item in enumerate(meta.items):
                            positions = history.setdefault(item.sequence_id, [])
                            positions.extend(j for run in runs if run.item_index == i
                                             for j in range(run.position_begin, run.position_begin+run.local_end-run.local_begin))
                            ids = torch.tensor(positions, device=device, dtype=torch.long)
                            kv.extend(t.index_select(0, ids) for t in bank[item.sequence_id])
                        if label == 'long' and index not in (0, 1, 8, len(plan.microbatches)-1):
                            continue
                        if cfg.smoke_only and index != 1:
                            continue
                        q = torch.randn(2040, 8, 128, device=device, dtype=torch.bfloat16)
                        do = torch.randn_like(q)
                        out = torch.zeros_like(q)
                        lse = q.new_full(q.shape[:2], -torch.inf, dtype=torch.float32)
                        for i, item in enumerate(meta.items):
                            a, b = meta.cu_seqlens_q[i:i+2]
                            k, v = (t[:item.token_offset+item.token_length].contiguous() for t in bank[item.sequence_id])
                            cq = torch.tensor([0, b-a], device=device, dtype=torch.int32)
                            ck = torch.tensor([0, k.size(0)], device=device, dtype=torch.int32)
                            oi, li, _, _ = forward(q[a:b], k, v, cq, ck, True)
                            out[a:b], lse[a:b] = oi, li.T
                        variants = {}
                        for version in ('initial', 'candidate'):
                            info, parts, cu, *_ = states[version]._cp_prepared['qo', meta.microbatch_id, rank, device]
                            variants[version] = (parts, operations(q, kv, out, lse, do, info, parts, cu, version=='initial'))
                        timings = {v: {} for v in variants}
                        exclusive()
                        # ABBA windows; no source/profile mutation between windows.
                        for version in ('initial', 'candidate', 'candidate', 'initial'):
                            parts, ops = variants[version]
                            for op in ('F_local', 'B_local'):
                                timings[version].setdefault(op, []).append(measure(ops[op], cfg.warmup, cfg.repeats))
                        for op in ('D_local', 'H_local'):
                            if op in variants['candidate'][1]:
                                timings['candidate'][op] = [measure(variants['candidate'][1][op], cfg.warmup, cfg.repeats)]
                        exclusive()
                        row = dict(case=label, lengths=lengths, physical=index, cp=cp, rank=rank, kv_heads=hk,
                                   valid_tokens=meta.valid_token_count, parts={v: len(x[0]) for v, x in variants.items()}, timings=timings)
                        rows.append(row)
                        with (cfg.output/'rows.jsonl').open('a') as f:
                            f.write(json.dumps(row)+'\n')
                        print(f'{label} hk={hk} cp={cp} rank={rank} physical={index} parts={row["parts"]}', flush=True)
    torch.cuda.synchronize()
    scope.update(rows=len(rows), allocated_peak=torch.cuda.max_memory_allocated(),
                 reserved_peak=torch.cuda.max_memory_reserved(), status='complete')
    (cfg.output/'scope.json').write_text(json.dumps(scope, indent=2))
    groups = {}
    for row in rows:
        groups.setdefault((row['case'], row['cp'], row['kv_heads'], row['physical']), []).append(row)
    summary = []
    for (label, cp, hk, physical), owners in groups.items():
        values = {}
        for version in ('initial', 'candidate'):
            ops = set(op for row in owners for op in row['timings'][version])
            values[version] = {op: max(statistics.mean(w['median_ms'] for w in row['timings'][version][op])
                for row in owners if op in row['timings'][version]) for op in sorted(ops)}
        summary.append(dict(case=label, cp=cp, kv_heads=hk, physical=physical,
            max_owner_ms=values, fa_calls_max={v: max(row['parts'][v] for row in owners)
                                             for v in ('initial', 'candidate')}))
    (cfg.output/'summary.json').write_text(json.dumps(dict(scope=scope, groups=summary,
        aggregation='Mean ABBA window medians per owner, then maximum owner; selected shapes only, not end-to-end.'), indent=2))
    print(json.dumps(scope), flush=True)


if __name__ == '__main__':
    main()
