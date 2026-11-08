#!/usr/bin/env python3
"""Offline compact export from same-execution PP8 snapshots. NumPy/CPU only."""
import argparse
from collections import defaultdict
import hashlib
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


def export(directory):
    paths = sorted(directory.glob('*-compact.json'))
    groups, contexts, sources, capacities = defaultdict(dict), {}, {}, set()
    modes = None
    for path in paths:
        blob = path.read_bytes(); payload = json.loads(blob)
        sources[path.name] = hashlib.sha256(blob).hexdigest()
        fused = payload['arithmetic_modes']['fused']
        if modes is None:
            modes = fused
        if fused != modes:
            raise ValueError('cannot combine different fused execution modes')
        for group in payload['groups']:
            key = group['stage'], group['virtual_chunk'], group['op']
            ctxkey = key[:2]
            if ctxkey in contexts and contexts[ctxkey] != group['context']:
                raise ValueError('cannot combine different stage/model/hardware contexts')
            contexts[ctxkey] = group['context']
            for row in group['model']['exact']:
                cap, items = row['signature'].split('|')
                c = int(cap[1:]); capacities.add(c)
                shape = tuple((int(q), int(k)-int(q)) for q,k in (x.split(':') for x in items.split(';')))
                if key[2] == 'W':
                    shape = ((c, 0),)
                groups[key][c,shape] = row['median_ms']
    fitted, diagnostics = [], []
    for (stage, chunk, op), points in sorted(groups.items()):
        rows = [(c,s,y) for (c,s),y in sorted(points.items())]
        x = [features(c,s) for c,s,y in rows]; y = np.array([y for c,s,y in rows])
        beta = fit(x, y)
        error = np.abs(np.array(x)@beta-y)/y
        exact = [rows[i] for i in np.linspace(0,len(rows)-1,min(32,len(rows)),dtype=int)]
        fitted.append(dict(stage=stage, chunk=chunk, op=op, coefficients=beta.tolist(),
            measured_capacities=sorted({c for c,s,y in rows}), exact=exact))
        held = []
        if len({c for c,s,y in rows}) >= 3:
            for capacity in sorted({c for c,s,y in rows})[1:-1]:
                train = [i for i,r in enumerate(rows) if r[0]!=capacity]
                test = [i for i,r in enumerate(rows) if r[0]==capacity]
                prediction = np.array(x)[test]@fit(np.array(x)[train], y[train])
                held.extend((np.abs(prediction-y[test])/y[test]).tolist())
        diagnostics.append(dict(stage=stage, chunk=chunk, op=op, points=len(rows),
            capacities=sorted({c for c,s,y in rows}), train_p50=float(np.median(error)),
            train_p95=float(np.percentile(error,95)), capacity_holdout_errors=held))
    memories = []
    for path in sorted(directory.glob('*-fused-memory.json')):
        memories.append(json.loads(path.read_text()))
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    mixed = next(directory.glob('*-mixed-confirmed-memory.json'))
    memories.append(json.loads(mixed.read_text()))
    sources[mixed.name] = hashlib.sha256(mixed.read_bytes()).hexdigest()
    template = memories[-1]
    rows = []
    scalable = ('activation_bytes','weight_forward_bytes','weight_backward_bytes','weight_bytes','capture_bytes')
    for stage in range(len(template['stages'])):
        row = {}
        for key in template['stages'][stage]:
            values = [m['stages'][stage][key] for m in memories]
            if key in scalable:
                row[key] = [max(v[j]/m['layout']['chunk_size'] for m,v in zip(memories,values)) for j in range(2)]
            elif isinstance(values[0], list):
                row[key] = [max(v[j] for v in values) for j in range(2)]
            else:
                row[key] = min(values) if key == 'physical_bytes' else max(values)
        rows.append(row)
    p2p_path = next(directory.glob('*-p2p-combined.json'))
    p2p = json.loads(p2p_path.read_text())
    sources[p2p_path.name] = hashlib.sha256(p2p_path.read_bytes()).hexdigest()
    model = dict(schema='dspp-cross-capacity-v1', pp=len(rows), domain=[min(capacities),max(capacities)],
        execution='canonical-query-v2', anchor_execution=modes,
        status='CPU implementation; GPU prediction/memory/numerics pending',
        groups=fitted, contexts=list(contexts.values()), layout=template['layout'],
        memory=dict(schema=template['schema'], accounting=template['accounting'],
            dynamic_growth=template['dynamic_growth'], device_name=template['device_name'],
            scaled_fields=scalable, stages=rows,
            scope='upper observed bytes/token envelope, unvalidated between capacities and for new D/H'),
        p2p=p2p, source_sha256=sources)
    return model, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model, diagnostics = export(args.profiles_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, separators=(',',':'))+'\n')
    args.output.with_suffix('.fit.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    print(json.dumps(dict(bytes=args.output.stat().st_size, domain=model['domain'], groups=len(model['groups']))))


if __name__ == '__main__':
    main()
