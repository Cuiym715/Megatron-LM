#!/usr/bin/env python3
"""Offline native cost export from measured PP8 and isolated stage sources. NumPy/CPU only."""
import argparse
from collections import defaultdict
import hashlib
import gzip
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path

import numpy as np


def features(c, shape):
    return [1., c, sum(q for q,h in shape), sum(q*(q+1)/2 for q,h in shape),
            sum(q*h for q,h in shape), sum(h for q,h in shape), len(shape)]


def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    scale = np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1.)
    weight = np.median(y)/y
    a, b = x/scale*weight[:,None], y*weight
    gram = a.T@a/len(y)+1e-6*np.eye(x.shape[1])
    target = a.T@b/len(y)
    best, objective = np.zeros(x.shape[1]), 0.
    for n in range(1, x.shape[1]+1):
        for active in combinations(range(x.shape[1]), n):
            value = np.linalg.solve(gram[np.ix_(active, active)], target[list(active)])
            if min(value) < -1e-10:
                continue
            beta = np.zeros(x.shape[1]); beta[list(active)] = np.maximum(value, 0.)
            loss = beta@gram@beta-2*target@beta
            if loss < objective:
                best, objective = beta, loss
    return best/scale


NATIVE = 'native-default-bf16-v1'


def read_json(path):
    blob = path.read_bytes()
    return json.loads(gzip.decompress(blob) if path.suffix == '.gz' else blob)


def signature(value, op):
    cap, items = value.split('|')
    c = int(cap[1:])
    shape = tuple((int(q), int(k)-int(q)) for q,k in
                  (x.split(':') for x in items.split(';')))
    return c, ((c, 0),) if op == 'W' else shape


def profile_rows(payload):
    """Read old raw samples or compact medians without refitting old models."""
    if 'records' in payload:
        for row in payload['records']:
            yield (row['stage'], row['chunk'], row['operation']), row['context'], row['signature'], row['duration_ms']
    else:
        for group in payload['groups']:
            key = group['stage'], group['virtual_chunk'], group['op']
            for row in group['model']['exact']:
                yield key, group['context'], row['signature'], row['median_ms']


def context_identity(context):
    value = deepcopy(context)
    # Older snapshots recorded a source digest here; later snapshots did not.
    # Keep the original context in provenance; this is not an execution mode.
    value['hardware']['kernels'].pop('split_sources', None)
    if value['model'].get('context_parallel_size', 1) == 1:
        value['model'].pop('dspp_cp_layout', None)  # CP layout has no effect in CP1.
    return value


def role(context):
    value = context_identity(context)
    return (len(value['layers']), value['embedding'], value['output_head'],
            value['model'], value['hardware'], value['dtype'], value['execution'], value['boundary'])


def collect(entries, root, sources, provenance):
    samples, contexts = defaultdict(lambda: defaultdict(list)), {}
    for entry in entries:
        path = root / entry['path']
        payload = read_json(path)
        sources[entry['path']] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Legacy files need an explicit source declaration in the manifest.
        # Never infer native execution from an absent arithmetic_modes field.
        modes = payload.get('arithmetic_modes', {})
        if entry['execution'] != NATIVE or any(
                'canonical' in value or 'deterministic' in value
                for value in modes.values()):
            raise ValueError('native model cannot ingest diagnostic profiles: '+str(path))
        original_contexts = {}
        for key, context, sig, value in profile_rows(payload):
            if key[2] not in entry['operations']:
                continue
            if key[:2] in contexts and context_identity(contexts[key[:2]]) != context_identity(context):
                raise ValueError('incompatible stage contexts: '+str(path))
            contexts[key[:2]] = context
            original_contexts[str(key[:2])] = context
            samples[key][signature(sig, key[2])].append(value)
        provenance.append(dict(**entry, contexts=original_contexts))
    return ({key: {point: float(np.median(values)) for point, values in rows.items()}
             for key, rows in samples.items()}, contexts)


def calibrated_transfer(points, local):
    common = sorted(points.keys() & local.keys())
    if not common:
        return {}, None
    scale = float(np.median([points[p]/local[p] for p in common]))
    return ({p: value*scale for p,value in local.items() if p not in points},
            dict(anchor_points=len(common), scale=scale,
                 anchor_relative_error_p95=float(np.percentile(
                     [abs(local[p]*scale-points[p])/points[p] for p in common], 95)),
                 scope='role-matched isolated PP2 transfer; PP8 cross-capacity validation pending'))


def capacity_holdout(points, local):
    """Withhold target capacity from both anchors and local training data."""
    capacities = sorted({c for c,s in points})
    if len(capacities) < 2:
        return None
    errors = []
    for capacity in capacities:
        target_train = {p:y for p,y in points.items() if p[0] != capacity}
        local_train = {p:y for p,y in local.items() if p[0] != capacity}
        transfer, _ = calibrated_transfer(target_train, local_train)
        train = {**transfer, **target_train}
        beta = fit([features(c,s) for c,s in train], list(train.values()))
        errors.extend(abs(np.dot(features(c,s), beta)-y)/y
                      for (c,s),y in points.items() if c == capacity)
    return list(map(float, errors))


def export(manifest_path):
    manifest = read_json(manifest_path)
    root = manifest_path.parent
    sources, provenance = {}, []
    groups, contexts = collect(manifest['compute'], root, sources, provenance)
    local, local_contexts = collect(manifest.get('local_compute', []), root, sources, provenance)
    # Local PP2 measurements inform the fitted curve only. They never enter
    # the PP8 exact table. Calibrate each stage/op using overlapping signatures.
    transferred, transfer_reports, local_pools = {}, {}, {}
    for key, points in groups.items():
        candidates = [values for local_key, values in local.items()
                      if local_key[2] == key[2] and
                      role(local_contexts[local_key[:2]]) == role(contexts[key[:2]])]
        merged = defaultdict(list)
        for candidate in candidates:
            for point, value in candidate.items():
                if manifest['domain'][0] <= point[0] <= manifest['domain'][1]:
                    merged[point].append(value)
        pooled = {p: float(np.median(v)) for p,v in merged.items()}
        local_pools[key] = pooled
        transferred[key], transfer_reports[key] = calibrated_transfer(points, pooled)
    fitted, diagnostics = [], []
    for (stage, chunk, op), points in sorted(groups.items()):
        training = {**transferred.get((stage, chunk, op), {}), **points}
        rows = [(c,s,y) for (c,s),y in sorted(training.items())]
        x = [features(c,s) for c,s,y in rows]; y = np.array([y for c,s,y in rows])
        beta = fit(x, y)
        error = np.abs(np.array(x)@beta-y)/y
        exact = [(c,s,y) for (c,s),y in sorted(points.items())]
        fitted.append(dict(stage=stage, chunk=chunk, op=op, coefficients=beta.tolist(),
            measured_capacities=sorted({c for c,s in points}), exact=exact,
            transferred_points=len(transferred.get((stage, chunk, op), {}))))
        target_capacities = sorted({c for c,s in points})
        held = capacity_holdout(points, local_pools.get((stage, chunk, op), {}))
        diagnostics.append(dict(stage=stage, chunk=chunk, op=op, points=len(rows),
            exact_points=len(exact), measured_capacities=target_capacities,
            transfer=transfer_reports.get((stage, chunk, op)),
            capacities=sorted({c for c,s,y in rows}), train_p50=float(np.median(error)),
            train_p95=float(np.percentile(error,95)), capacity_holdout_errors=held))
    memories = []
    for entry in manifest['memory']:
        path = root / entry['path']
        memories.append(read_json(path))
        sources[entry['path']] = hashlib.sha256(path.read_bytes()).hexdigest()
    template = memories[-1]
    rows = []
    scalable = ('activation_bytes','weight_forward_bytes','weight_backward_bytes','weight_bytes','capture_bytes')
    for stage in range(len(template['stages'])):
        row = {}
        for key in memories[0]['stages'][stage]:
            selected = [m for m, entry in zip(memories, manifest['memory'])
                        if 'fields' not in entry or key in entry['fields']]
            values = [m['stages'][stage][key] for m in selected]
            if key in scalable:
                row[key] = [max(v[j]/m['layout']['chunk_size'] for m,v in zip(selected,values)) for j in range(2)]
            elif isinstance(values[0], list):
                row[key] = [max(v[j] for v in values) for j in range(2)]
            else:
                row[key] = min(values) if key == 'physical_bytes' else max(values)
        rows.append(row)
    p2p_path = root / manifest['p2p']
    p2p = json.loads(p2p_path.read_text())
    sources[manifest['p2p']] = hashlib.sha256(p2p_path.read_bytes()).hexdigest()
    model = dict(schema='dspp-cross-capacity-v1', pp=len(rows), domain=manifest['domain'],
        execution=NATIVE, anchor_execution=NATIVE,
        status='CPU implementation; GPU prediction/memory/numerics pending',
        groups=fitted, contexts=list(contexts.values()), layout=template['layout'],
        provenance=provenance, memory_sources=manifest['memory'], limitations=manifest['limitations'],
        memory=dict(schema=template['schema'], accounting=template['accounting'],
            dynamic_growth=template['dynamic_growth'], device_name=template['device_name'],
            scaled_fields=scalable, stages=rows,
            scope='upper observed bytes/token envelope, unvalidated between capacities and for new D/H'),
        p2p=p2p, source_sha256=sources)
    return model, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model, diagnostics = export(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, separators=(',',':'))+'\n')
    args.output.with_suffix('.fit.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    print(json.dumps(dict(bytes=args.output.stat().st_size, domain=model['domain'], groups=len(model['groups']))))


if __name__ == '__main__':
    main()
