"""Paired CPU comparison on frozen DSPP timelines; never changes runtime costs.

Train/test seeds, stage boundaries and task identities are shared. The anchor
track uses identical single/no-history measurements for both implementations.
The full-profile track also fits InfiniPipe's three-term formula to all shapes;
that fitting procedure is an explicit comparison adaptation, not author code.
Old timelines remain evaluation data, not versioned production calibrations.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
from statistics import median
import time
from types import SimpleNamespace

import numpy as np

from evaluate_dspp_infinipipe_cost import author_class, shape, statistics
from megatron.core.datasets.dspp_execution_cost import (
    DsppExecutionCostModel, meta_from_timeline_record,
)


def load_raw(directories):
    rows, files = [], {}
    for directory in directories:
        for path in sorted(directory.glob('*_rank_*.json')):
            payload = json.loads(path.read_text())
            if payload.get('backward_mode') != 'fused':
                raise ValueError(f'not fused: {path}')
            files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in payload['events']:
                if row['kind'][0] in 'FBW':
                    row = dict(row, source_file=str(path), operation=row['kind'][0])
                    meta_from_timeline_record(row).validate()
                    if not np.isfinite(row['duration_ms']) or row['duration_ms'] <= 0:
                        raise ValueError('nonpositive/nonfinite observation')
                    rows.append(row)
    return rows, files


def task_key(row):
    return (row['source_file'], row['stage'], row['chunk'],
            row['logical_microbatch'], row['physical_microbatch'])


def complete_tasks(rows):
    grouped = defaultdict(dict)
    for row in rows:
        tasks = grouped[task_key(row)]
        assert row['operation'] not in tasks, 'duplicate task'
        tasks[row['operation']] = row
    result = []
    for tasks in grouped.values():
        assert set(tasks) == set('FBW'), 'incomplete F/B/W'
        assert shape(tasks['F']) == shape(tasks['B']) == shape(tasks['W'])
        f, b, w = (tasks[k] for k in 'FBW')
        result.extend((dict(f, components=[f]), dict(b,
            duration_ms=b['duration_ms']+w['duration_ms'], components=[b, w])))
    return result


def group(row):
    return row['stage'], row['chunk'], row['operation']


def anchor(row):
    return len(row['items']) == 1 and row['items'][0]['token_offset'] == 0


def dense_w(row):
    if row['operation'] != 'W':
        return row
    c = row['sequence_chunk_size']
    result = dict(row, items=[dict(sequence_id=0, segment_id=0,
        segment_count=1, token_offset=0, token_length=c, sequence_length=c)])
    result.pop('signature', None)
    return result


def fit_existing(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[group(row)].append(dense_w(row))
    return {key: DsppExecutionCostModel.from_records(values, extended_features=True)
            for key, values in groups.items()}


def formula_features(row):
    q = np.array([i['token_length'] for i in row['items']], dtype=float)/1000
    histories = [i['token_offset'] for i in row['items']]
    if any(histories[1:]):
        raise ValueError('author formula requires history only in first item')
    return np.array([sum(q*q)+2*q[0]*histories[0]/1000, sum(q), 1.])


def nonnegative_fit(x, y):
    """Exact active-subset NNLS for three columns, used only in adaptation."""
    best, loss = np.zeros(3), float('inf')
    for count in range(1, 4):
        for subset in itertools.combinations(range(3), count):
            values = np.linalg.lstsq(x[:, subset], y, rcond=None)[0]
            if np.any(values < -1e-12):
                continue
            coef = np.zeros(3)
            coef[list(subset)] = np.maximum(values, 0)
            candidate = float(np.sum((x@coef-y)**2))
            if candidate < loss:
                best, loss = coef, candidate
    return best


def fit_formula(rows, Cost, fit_mode, local_fit):
    groups = defaultdict(lambda: defaultdict(list))
    for row in complete_tasks(rows):
        groups[group(row)][shape(row)].append(row)
    models, fitted = {}, []
    for key, shapes in sorted(groups.items()):
        representatives = []
        for values in shapes.values():
            representatives.append(dict(values[0], duration_ms=(
                sum(v['duration_ms'] for v in values)/len(values)
                if fit_mode != 'full-formula-nnls' else median(v['duration_ms'] for v in values))))
        x = np.array([formula_features(r) for r in representatives])
        y = np.array([r['duration_ms']/1000 for r in representatives])
        if fit_mode == 'author-polyfit':
            assert all(anchor(r) for r in representatives) and len(representatives) >= 3
            coef = np.polyfit(x[:, 1], y, 2)
        elif fit_mode == 'local-anchor-nnls':
            coef = local_fit(x[:, 1], y)
        else:
            coef = nonnegative_fit(x, y)
        model = Cost.__new__(Cost)
        model.cluster_size = model.pp_deg = model.n_layer = 1
        for direction in ('fwd', 'bwd'):
            for name, value in zip(('alpha_1', 'alpha_2', 'beta'), coef):
                setattr(model, f'{direction}_{name}', float(value))
        models[key] = model
        fitted.append(dict(group=key, unique_shapes=len(shapes),
                           samples=sum(map(len, shapes.values())), coefficients=list(coef)))
    return models, fitted


def query_meta(row):
    return SimpleNamespace(offset=row['items'][0]['token_offset'],
        cu_seqlens_q_cpu=np.concatenate(([0], np.cumsum([i['token_length'] for i in row['items']]))))


def summarize(rows):
    result = {}
    for op in ('F', 'B'):
        subset = [r for r in rows if r['operation'] == op]
        result[op] = dict(all=statistics(subset),
            seen=statistics([r for r in subset if r['seen_shape']]),
            unseen=statistics([r for r in subset if not r['seen_shape']]),
            by_route={s: statistics([r for r in subset if r['source']==s])
                      for s in sorted({r['source'] for r in subset})},
            by_stage={f'{s}/{c}': statistics([r for r in subset if (r['stage'],r['chunk'])==(s,c)])
                      for s,c in sorted({(r['stage'],r['chunk']) for r in subset})})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--infinipipe', required=True, type=Path)
    parser.add_argument('--train', required=True, nargs='+', type=Path)
    parser.add_argument('--test', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    train, train_files = load_raw(args.train)
    test, test_files = load_raw([args.test])
    assert not set(train_files).intersection(test_files), 'train/test file leakage'
    test_tasks = complete_tasks(test)
    cost_path = args.infinipipe/'infiniPipe/solver/costmodel.py'
    driver = args.infinipipe/'infiniPipe/do_profile.py'
    Cost = author_class(cost_path, 'CostModel', (
        'cost_single_sequence_fwd_no_beta', 'cost_single_sequence_bwd_no_beta',
        'cost_multiple_sequence_fwd_no_beta', 'cost_multiple_sequence_bwd_no_beta', 'cost_comp'),
        dict(np=np, SeqChunkMetaData=object))
    tree = ast.parse(driver.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='_fit_nonnegative_quadratic']
    env = dict(np=np)
    exec(compile(ast.fix_missing_locations(tree), str(driver), 'exec'), env)
    results, predictions, component_predictions = {}, {}, {}
    for track in ('matched-anchors', 'full-profile'):
        fitting = [r for r in train if anchor(r)] if track=='matched-anchors' else train
        existing = fit_existing(fitting)
        seen = {(*group(r), shape(r)) for r in complete_tasks(fitting)}
        methods = {'existing': existing}
        coefficients = {}
        modes = ('author-polyfit', 'local-anchor-nnls') if track=='matched-anchors' else ('full-formula-nnls',)
        for mode in modes:
            methods[mode], coefficients[mode] = fit_formula(fitting, Cost, mode, env['_fit_nonnegative_quadratic'])
        track_results = {}
        for method, models in methods.items():
            output, components = [], []
            calls = []
            for row in test_tasks:
                if method == 'existing':
                    parts = []
                    for part in row['components']:
                        meta = meta_from_timeline_record(dense_w(part))
                        model = models[group(part)]
                        p = model.predict_detailed(part['stage'], part['operation'], meta)
                        parts.append(p)
                        components.append(dict(part, predicted_ms=p.duration_ms, source=p.source))
                    value = sum(p.duration_ms for p in parts)
                    source = '+'.join(f'{part["operation"]}:{p.source}' for part,p in zip(row['components'],parts))
                    # Query cost includes the production median and p95 calls,
                    # plus W normalization, but excludes meta parsing shared by runtime.
                    original_metas = [(part, meta_from_timeline_record(part)) for part in row['components']]
                    calls.append(original_metas)
                else:
                    meta = query_meta(row)
                    f, b, _ = models[group(row)].cost_comp(meta, 0)
                    value, source = (f if row['operation']=='F' else b)*1000, method
                    # One author call computes both directions, as its API does.
                    calls.append((models[group(row)], meta))
                    assert np.isclose(value/1000, formula_features(row) @ np.array([
                        models[group(row)].fwd_alpha_1, models[group(row)].fwd_alpha_2, models[group(row)].fwd_beta]))
                output.append(dict(stage=row['stage'], chunk=row['chunk'], operation=row['operation'],
                    shape=shape(row), duration_ms=row['duration_ms'], predicted_ms=value,
                    source=source, seen_shape=(*group(row), shape(row)) in seen,
                    source_file=row['source_file'], logical_microbatch=row['logical_microbatch'],
                    physical_microbatch=row['physical_microbatch']))
            def predict_all():
                for call in calls:
                    if method != 'existing':
                        call[0].cost_comp(call[1], 0)
                    else:
                        for part, meta in call:
                            if part['operation']=='W':
                                from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
                                meta = build_dspp_batch_plan([meta.chunk_size], chunk_size=meta.chunk_size).microbatches[0]
                            for percentile in (.5, .95):
                                models[group(part)].predict_detailed(part['stage'], part['operation'], meta, percentile=percentile)
            predict_all()
            times = []
            for _ in range(3):
                start = time.perf_counter()
                predict_all()
                times.append((time.perf_counter()-start)*1000)
            entry = dict(summary=summarize(output), query_batch_ms_median=median(times),
                query_us_per_test_task=median(times)*1000/len(output))
            if method == 'existing':
                entry['component_routes'] = {op:{s:statistics([r for r in components if r['operation']==op and r['source']==s])
                    for s in sorted({r['source'] for r in components if r['operation']==op})} for op in 'FBW'}
                # The v2 production wrapper rejects all non-exact regions;
                # these counts describe numerical route coverage, not accepted profiles.
                entry['all_components_exact_tasks'] = sum(all(p.endswith('exact-profile') for p in r['source'].split('+')) for r in output)
                component_predictions[track] = components
            track_results[method] = entry
            predictions[f'{track}/{method}'] = output
        results[track] = dict(train_raw_events=len(fitting), train_complete_tasks=len(complete_tasks(fitting)),
                            methods=track_results, coefficients=coefficients)
    source_paths = [Path(__file__), cost_path, driver,
                    Path('megatron/core/datasets/dspp_execution_cost.py'),
                    Path('megatron/core/datasets/dspp_calibration.py')]
    report = dict(scope='paired offline numerical accuracy, fixed capacity 2048, fused B; not runtime acceptance',
        boundary='F and matched B+W, complete physical/virtual stage CUDA events; no P2P/optimizer',
        error='abs(prediction-actual)/actual; median and nearest-rank P95 over all observations',
        train_files=train_files, test_files=test_files, test_tasks=len(test_tasks),
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        results=results)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (args.output/'predictions.json').write_text(json.dumps(predictions, indent=2, allow_nan=False))
    (args.output/'component-predictions.json').write_text(json.dumps(component_predictions, indent=2, allow_nan=False))
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
