"""Build an opt-in v3 profile from frozen storage-owner audits.

Run in the training environment. Inputs are read-only; output must be new.
Resident, workspace, physical reserve, communication policy and coverage limits
are inherited. This reclassifies measured F/W storage, not allocator residuals.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from megatron.core.pipeline_parallel.dspp_memory_lifetimes import ACCOUNTING, stage_attention_sizes

MIB = 1 << 20


def attention_storage(storage, heads):
    return any((len(shape) == 3 and shape[1] in heads and shape[2] == 128)
               or (len(shape) == 2 and shape[0] == heads[0] and dtype == 'torch.float32')
               for _, shape, _, dtype in storage['views'])


def partition(forward, weight, heads, capacity):
    """Exclusive classes by storage identity, including all aliases of a view."""
    captured = {k for k, s in forward.items() if attention_storage(s, heads)}
    # Long assembled KV is charged per key token. Short assembly can remain
    # in capture as a conservative overlap; no deduction based on a slice size.
    capture = {k for k in captured if any(
        (len(v[1]) == 3 and v[1][0] <= capacity) or len(v[1]) == 2
        for v in forward[k]['views'])}
    f = {k for k in forward if k not in weight and k not in captured
         and 'KV-pending-H' not in forward[k]['owners']}
    wf = (weight.keys() & forward.keys()) - captured
    wb = weight.keys() - forward.keys()
    return dict(activation=sum(forward[k]['bytes'] for k in f),
                weight_forward=sum(weight[k]['bytes'] for k in wf),
                weight_backward=sum(weight[k]['bytes'] for k in wb),
                capture=sum(forward[k]['bytes'] for k in capture))


def audit_rows(data):
    baseline = set(data['boundaries'][0]['storages'])
    previous_f, previous_w, forwards, weights = set(), set(), {}, {}
    for boundary in data['boundaries']:
        rows = boundary['storages']
        dynamic = {k for k, s in rows.items() if k not in baseline
                   and s['device'].startswith('cuda')
                   and not any(o.startswith('resident-') for o in s['owners'])}
        f = {k for k in dynamic if 'F-activations' in rows[k]['owners']}
        w = {k for k in dynamic if 'W-closures' in rows[k]['owners']}
        tag = boundary['label'][1:]
        if boundary['label'].startswith('F'):
            forwards[tag] = {k: rows[k] for k in f - previous_f}
        if boundary['label'][0] in 'BD':
            weights.setdefault(tag, {}).update({k: rows[k] for k in w - previous_w})
        previous_f, previous_w = f, w
    if not forwards or forwards.keys() != weights.keys():
        raise ValueError('incomplete F/backward audit')
    return forwards, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-profile', type=Path, required=True)
    parser.add_argument('--owners', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state-margin', type=float, default=.15)
    cfg = parser.parse_args()
    if not math.isfinite(cfg.state_margin) or cfg.state_margin < .15:
        parser.error('state margin must be at least 15%')
    if cfg.output.exists():
        parser.error('output must be a new directory')
    base = json.loads(cfg.base_profile.read_text())
    layout = base['layout']
    args = SimpleNamespace(**layout['model'])
    if layout['dtype'] != 'torch.bfloat16' or layout['vpp'] != 2:
        raise ValueError('requires BF16 VPP2 audit')
    hq = args.num_attention_heads
    hk = args.num_query_groups if args.group_query_attention else hq
    profile = deepcopy(base)
    profile.update(schema='dspp-counted-memory-v3', accounting=ACCOUNTING,
                   source='storage-owner F/W partition; analytical capture/KV/pending floors; '
                          'inherited workspace/resident/reserve/coverage; see evidence.json')
    evidence, provenance = [], []
    for rank, row in enumerate(profile['stages']):
        maxima = {k: [0, 0] for k in ('activation', 'weight_forward', 'weight_backward', 'capture')}
        details = []
        paths = sorted(cfg.owners.glob(f'*_stage_{rank}.json'))
        if not paths:
            raise ValueError(f'missing stage {rank} audit')
        for path in paths:
            raw = path.read_bytes()
            provenance.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
            data = json.loads(raw)
            if data['stage'] != rank:
                raise ValueError('audit stage mismatch')
            forwards, weights = audit_rows(data)
            for tag, forward in forwards.items():
                c = int(tag[0])
                values = partition(forward, weights[tag], (hq, hk), layout['chunk_size'])
                for k, value in values.items():
                    maxima[k][c] = max(maxima[k][c], value)
                details.append(dict(iteration=data['iteration'], task=tag, **values))
        for c in range(2):
            capture, kv = stage_attention_sizes(args, layout['pp'], rank, c, layout['chunk_size'])
            maxima['capture'][c] = max(maxima['capture'][c], capture)
            for key in ('attention_bytes_per_key_token', 'pending_bytes_per_token'):
                values = row.setdefault(key, [0, 0])
                values[c] = max(values[c], math.ceil(kv * (1 + cfg.state_margin)))
        for key, values in maxima.items():
            row[key + '_bytes'] = [max(MIB, math.ceil(v * (1 + cfg.state_margin) / MIB) * MIB) for v in values]
        row['weight_bytes'] = [a + b for a, b in zip(row['weight_forward_bytes'], row['weight_backward_bytes'])]
        evidence.append(dict(stage=rank, maxima_bytes=maxima, tasks=details))
    profile['calibration'] = dict(state_margin=cfg.state_margin, inputs=provenance,
        base_profile_sha256=hashlib.sha256(cfg.base_profile.read_bytes()).hexdigest(),
        note='KV legacy coefficient is unused in v3; workspace and communication remain conservative')
    cfg.output.mkdir(parents=True)
    (cfg.output/'memory.json').write_text(json.dumps(profile, indent=2) + '\n')
    (cfg.output/'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps([dict(stage=r, **{k:v for k,v in s.items() if k.endswith('_bytes')})
                      for r,s in enumerate(profile['stages'])]))


if __name__ == '__main__':
    main()
