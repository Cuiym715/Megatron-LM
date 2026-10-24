"""Evaluate InfiniPipe's unchanged quadratic predictor on recorded DSPP tasks.

Read-only author sources are executed by selecting their original AST methods,
avoiding model/CUDA imports and constructor side effects. This measures transfer
to DSPP, not InfiniPipe end-to-end training accuracy. B includes the matched W.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median
import time
from types import SimpleNamespace

import numpy as np


def author_class(path, name, methods, globals_):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    assert {n.name for n in cls.body} == set(methods)
    tree.body = [cls]
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), globals_)
    return globals_[name]


def shape(row):
    return (row['sequence_chunk_size'], tuple((i['token_length'], i['token_offset']) for i in row['items']))


def load_rows(directory):
    rows = []
    for path in sorted(directory.glob('*_rank_*.json')):
        data = json.loads(path.read_text())
        if data.get('backward_mode') != 'fused':
            raise ValueError(f'expected fused timing: {path}')
        events = data['events']
        weights = {(r['chunk'], r['logical_microbatch'], r['physical_microbatch']): r for r in events if r['kind'][0] == 'W'}
        for row in events:
            if row['kind'][0] not in 'FB':
                continue
            row = dict(row, source_file=str(path), operation=row['kind'][0])
            if row['operation'] == 'B':
                w = weights.pop((row['chunk'], row['logical_microbatch'], row['physical_microbatch']))
                assert shape(row) == shape(w)
                row['duration_ms'] += w['duration_ms']
            rows.append(row)
        if weights:
            raise ValueError('unmatched W tasks')
    return rows


def statistics(rows):
    if not rows:
        return dict(samples=0)
    valid = [r for r in rows if np.isfinite(r['predicted_ms']) and r['predicted_ms'] > 0]
    rel = sorted(abs(r['predicted_ms'] - r['duration_ms']) / r['duration_ms'] for r in rows)
    return dict(samples=len(rows), positive_predictions=len(valid),
        median_actual_ms=median(r['duration_ms'] for r in rows),
        median_absolute_ms=median(abs(r['predicted_ms']-r['duration_ms']) for r in rows),
        median_relative=median(rel) if rel else None,
        p95_relative=rel[max(0, int(np.ceil(.95*len(rel)))-1)] if rel else None,
        total_bias=(sum(r['predicted_ms'] for r in rows)/sum(r['duration_ms'] for r in rows)-1))


def native_archive(directory, Cost):
    """Recompute archived validation predictions and report absolute errors.

    No fitting-set R² is used as a substitute for validation accuracy. Preserve
    every archived run, including the first short 768-token calibration.
    """
    runs = []
    for path in sorted(directory.glob('*comp/layer_*/result_*.json')):
        data = json.loads(path.read_text())
        model = Cost.__new__(Cost)
        model.cluster_size = model.pp_deg = model.n_layer = 1
        for k, value in data.items():
            if k.startswith(('fwd_', 'bwd_')):
                setattr(model, k, value)
        predictions, max_difference = [], 0.
        for category in ('multi_seqs', 'single_seq_ctx', 'multi_seqs_ctx'):
            for direction, op in (('fwd', 'F'), ('bwd', 'B')):
                for key, sample in data[f'{category}_{direction}']['computation time'].items():
                    coordinates = ast.literal_eval(key)
                    if category == 'multi_seqs':
                        count, length = coordinates; offset = 0
                    elif category == 'single_seq_ctx':
                        offset, length = coordinates; count = 1
                    else:
                        offset, count, length = coordinates
                    meta = SimpleNamespace(offset=offset, cu_seqlens_q_cpu=np.arange(count+1)*length)
                    f, b, _ = model.cost_comp(meta, 0)
                    predicted = f if op == 'F' else b
                    max_difference = max(max_difference, abs(predicted-sample['estimated']))
                    predictions.append(dict(operation=op, category=category, shape=coordinates,
                        predicted_ms=predicted*1000, duration_ms=sample['profiled']*1000))
        if max_difference > 1e-7:
            raise ValueError(f'current author model disagrees with archived predictions: {path}')
        runs.append(dict(source_file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            max_prediction_reproduction_difference_seconds=max_difference,
            anchor_lengths=sorted(map(int, data['single_seq_fwd']['computation time'])),
            summary={op:{category: statistics([r for r in predictions if r['operation']==op and r['category']==category])
                         for category in ('multi_seqs', 'single_seq_ctx', 'multi_seqs_ctx')} for op in 'FB'},
            predictions=predictions))
    return dict(scope='archived native InfiniPipe 1/3-layer profiler validation, not full 24-layer stage accuracy', runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--infinipipe', type=Path, required=True)
    parser.add_argument('--train', type=Path, nargs='+', required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fit', choices=('polyfit', 'nonnegative'), default='polyfit',
                        help='polyfit is original; nonnegative is a pre-existing local modification')
    args = parser.parse_args()
    comp = args.infinipipe/'infiniPipe/profiler/comp_profiler.py'
    cost = args.infinipipe/'infiniPipe/solver/costmodel.py'
    driver = args.infinipipe/'infiniPipe/do_profile.py'
    env = dict(np=np, SeqChunkMetaData=object)
    Cost = author_class(cost, 'CostModel', (
        'cost_single_sequence_fwd_no_beta', 'cost_single_sequence_bwd_no_beta',
        'cost_multiple_sequence_fwd_no_beta', 'cost_multiple_sequence_bwd_no_beta', 'cost_comp'), env)
    Profiler = author_class(comp, 'CompProfiler', ('fit_quadratic_curv',), dict(np=np, r2_score=np.corrcoef))
    fitter = Profiler.__new__(Profiler)
    fitter.draw_picture = lambda *a: None
    fit_tree = ast.parse(driver.read_text())
    fit_tree.body = [n for n in fit_tree.body if isinstance(n, ast.FunctionDef) and n.name=='_fit_nonnegative_quadratic']
    fit_env = dict(np=np)
    exec(compile(ast.fix_missing_locations(fit_tree), str(driver), 'exec'), fit_env)
    train = [r for directory in args.train for r in load_rows(directory)]
    test = load_rows(args.test)
    models, coefficients = {}, []
    group = lambda r: (r['stage'], r['chunk'], r['operation'])
    for key in sorted({group(r) for r in train}):
        anchors = defaultdict(list)
        for row in train:
            if group(row) == key and len(row['items']) == 1 and row['items'][0]['token_offset'] == 0:
                anchors[row['valid_tokens']].append(row['duration_ms']/1000)
        if len(anchors) < 3:
            raise ValueError(f'fewer than 3 single/no-history anchors for {key}')
        lengths = sorted(anchors)
        # Same polyfit as author profiler; average repeats before fitting so
        # common lengths do not receive an accidental frequency weight.
        times = [sum(anchors[q])/len(anchors[q]) for q in lengths]
        a, b, beta, r2 = fitter.fit_quadratic_curv(np.array(lengths)/1000, np.array(times), 'comparison')
        if args.fit == 'nonnegative':
            a, b, beta = fit_env['_fit_nonnegative_quadratic'](np.array(lengths)/1000, np.array(times))
        model = Cost.__new__(Cost)
        # Fit the actual local stage directly: no ideal division by GPU count.
        model.cluster_size = model.pp_deg = model.n_layer = 1
        model.fwd_alpha_1 = model.bwd_alpha_1 = a
        model.fwd_alpha_2 = model.bwd_alpha_2 = b
        model.fwd_beta = model.bwd_beta = beta
        models[key] = model
        coefficients.append(dict(stage=key[0], chunk=key[1], operation=key[2],
            single_no_history_lengths=lengths, samples=sum(map(len, anchors.values())),
            alpha_1=a, alpha_2=b, beta=beta, polyfit_in_sample_correlation_squared=r2))
    train_shapes = {(*group(r), shape(r)) for r in train}
    predictions, calls = [], []
    for row in test:
        items = row['items']
        if any(i['token_offset'] for i in items[1:]):
            raise ValueError('InfiniPipe supports history in the first item only')
        cu = np.concatenate(([0], np.cumsum([i['token_length'] for i in items])))
        meta = SimpleNamespace(offset=items[0]['token_offset'], cu_seqlens_q_cpu=cu)
        model = models[group(row)]
        f, b, _ = model.cost_comp(meta, ckpt_layer=0)
        predicted = (f if row['operation'] == 'F' else b)*1000
        category = ('packed-history' if len(items)>1 else 'single-history') if meta.offset else ('packed-no-history' if len(items)>1 else 'single-no-history')
        predictions.append(dict(stage=row['stage'], chunk=row['chunk'], operation=row['operation'],
            shape=shape(row), category=category, predicted_ms=predicted,
            duration_ms=row['duration_ms'], seen_shape=(*group(row), shape(row)) in train_shapes,
            source_file=row['source_file']))
        calls.append((model, meta))
    summaries = {}
    for op in 'FB':
        subset = [r for r in predictions if r['operation'] == op]
        summaries[op] = dict(all=statistics(subset), unseen_shapes=statistics([r for r in subset if not r['seen_shape']]),
            categories={c: statistics([r for r in subset if r['category']==c]) for c in sorted({r['category'] for r in subset})},
            stages={f'{s}/{c}': statistics([r for r in subset if (r['stage'],r['chunk'])==(s,c)]) for s,c in sorted({(r['stage'],r['chunk']) for r in subset})})
    for model, meta in calls[:100]:
        model.cost_comp(meta, 0)
    durations = []
    for _ in range(5):
        start = time.perf_counter()
        for model, meta in calls:
            model.cost_comp(meta, 0)
        durations.append((time.perf_counter()-start)*1e6/len(calls))
    result = dict(scope='InfiniPipe formula refit and evaluated on DSPP full-stage eager CUDA events; not InfiniPipe native accuracy or end-to-end speed',
        training=[str(p) for p in args.train], testing=str(args.test), train_samples=len(train), test_samples=len(test),
        backward_boundary='matched B + W; no communication or optimizer',
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (cost,comp,driver)},
        fitting_method=args.fit,
        fitting_method_origin=('original CompProfiler.fit_quadratic_curv' if args.fit=='polyfit'
                               else 'pre-existing local do_profile.py modification, not upstream algorithm'),
        coefficients=coefficients, summaries=summaries,
        prediction_us_per_call_median=median(durations), coefficients_frozen_before_test=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    (args.output/'predictions.json').write_text(json.dumps(predictions, indent=2))
    (args.output/'native-archive.json').write_text(json.dumps(native_archive(
        args.infinipipe/'infiniPipe/profiler/logs', Cost), indent=2))
    print(json.dumps(dict(summaries=summaries, prediction_us_per_call_median=median(durations)), indent=2))


if __name__ == '__main__':
    main()
