#!/usr/bin/env python3
"""Diagnostic GPU replay of first-step Apex Adam using frozen PP3 captures."""
import gc
import json
import argparse
from pathlib import Path
import torch
from apex.optimizers import FusedAdam

torch.set_num_threads(1)
parser = argparse.ArgumentParser(description='Replay first-step PP3 diagnostic Adam updates')
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root


def replay(base, gradients):
    params, groups = {}, []
    for group in base['optimizer_groups']:
        copied = {k: v for k, v in group.items() if k != 'params'}
        copied['params'] = []
        for name in group['params']:
            p = torch.nn.Parameter(base['before_inner'][name]['master'].cuda())
            p.grad = gradients['before_inner'][name]['grad'].cuda()
            params[name] = p
            copied['params'].append(p)
        groups.append(copied)
    optimizer = FusedAdam(groups)
    for name, p in params.items():
        optimizer.state[p] = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                              for k, v in base['before_inner'][name]['state'].items()}
    optimizer.step()
    out = {n: dict(master=p.detach().cpu().clone(),
                   model=p.detach().bfloat16().cpu().clone(),
                   state={k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                          for k, v in optimizer.state[p].items()})
           for n, p in params.items()}
    del optimizer, params, groups
    torch.cuda.empty_cache()
    return out


def compare(a, b):
    assert a.keys() == b.keys()
    counts = dict(master=0, model=0, exp_avg=0, exp_avg_sq=0)
    maximum = dict.fromkeys(counts, 0.)
    for name in a:
        for item in counts:
            x = a[name][item] if item in ('master', 'model') else a[name]['state'][item]
            y = b[name][item] if item in ('master', 'model') else b[name]['state'][item]
            d = (x.float() - y.float()).abs()
            counts[item] += int(torch.count_nonzero(d))
            maximum[item] = max(maximum[item], float(d.max()))
    return dict(different=counts, max_abs=maximum)


reports = []
for rank in (0, 1, 2):
    a = torch.load(root / f'legacy/iteration1-rank{rank}.pt', weights_only=False)
    b = torch.load(root / f'permuted/iteration1-rank{rank}.pt', weights_only=False)
    reference = replay(a, a)
    control = replay(a, a)
    candidate = replay(a, b)
    def final_expected(data):
        return {n: {**r, 'model': data['final_optimizer'][n]['model']}
                for n, r in data['after_inner'].items()}

    row = dict(rank=rank, identical_gradient_control=compare(reference, control),
               reference_matches_training=compare(reference, final_expected(a)),
               changed_gradient_matches_candidate_training=compare(candidate, final_expected(b)),
               changed_gradient_effect=compare(reference, candidate))
    # after_inner.model precedes the model-copy stage; compare it against the
    # final BF16 copy, while master/m/v remain those immediately after Adam.
    for label, replayed, original in (
        ('reference_final_copy', reference, a), ('candidate_final_copy', candidate, b)):
        row[label] = sum(int(torch.count_nonzero(replayed[n]['model'] != r['model']))
                         for n, r in original['final_optimizer'].items())
    reports.append(row)
    (root / 'optimizer-replay.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(row), flush=True)
    del a, b, reference, control, candidate
    gc.collect()
