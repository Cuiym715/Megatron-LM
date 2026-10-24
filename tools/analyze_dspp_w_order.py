#!/usr/bin/env python3
"""CPU analysis of fixed PP3 W/Adam diagnostic captures; no training changes."""
import gc
import json
import argparse
from pathlib import Path
import torch

torch.set_num_threads(1)
parser = argparse.ArgumentParser(description='Analyze PP3 diagnose_dspp_w_order.py captures')
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--include-frozen', action='store_true')
parser.add_argument('--verify-reduce', action='store_true')
parser.add_argument('--baseline-snapshots', type=Path)
args = parser.parse_args()
root = args.root


def comparison(a, b):
    assert a.keys() == b.keys(), (a.keys() - b.keys(), b.keys() - a.keys())
    maximum, count, witness, failures = 0., 0, None, 0
    for name in a:
        x, y = a[name].float(), b[name].float()
        d = (x - y).abs()
        n = int(torch.count_nonzero(d))
        count += n
        error = float(d.max()) if d.numel() else 0.
        if error > maximum:
            maximum = error
            idx = int(d.flatten().argmax())
            witness = dict(name=name, index=idx, reference=float(x.flatten()[idx]),
                           candidate=float(y.flatten()[idx]))
        failures += int(torch.count_nonzero(d > .004 + .01 * x.abs()))
    return dict(max_abs=maximum, different=count, witness=witness,
                mixed_tolerance_failures=failures)


def field(data, name, item):
    return {n: r[item] for n, r in data[name].items()}


if args.verify_reduce:
    assert args.baseline_snapshots is not None, '--baseline-snapshots is required'
    reports = []
    for rank in (0, 1, 2):
        a = torch.load(root / f'legacy-reduce/iteration1-rank{rank}.pt', weights_only=False)
        b = torch.load(root / f'permuted-reduce/iteration1-rank{rank}.pt', weights_only=False)
        old = torch.load(args.baseline_snapshots / f'iteration1-rank{rank}.pt', weights_only=True)
        report = dict(rank=rank, closures=len(a['closures']))
        for label, data in (('reference', a), ('candidate', b)):
            for name, replay in data['kernel_replay'].items():
                assert torch.equal(replay, data['before_reduce_gradients'][name]), (label, name)
            report[label + '_kernel_replay_exact_before_reduce'] = True
        for key in a['closures']:
            x, y = a['closures'][key], b['closures'][key]
            assert all(torch.equal(u, v) for u, v in zip(x['operands'], y['operands']))
            assert torch.equal(x['contribution'], y['contribution'])
        deferred_names = set(a['kernel_replay'])
        for name in a['before_reduce_gradients']:
            if name not in deferred_names:
                assert torch.equal(a['before_reduce_gradients'][name], b['before_reduce_gradients'][name]), name
        report['all_B_direct_gradients_exact_between_orders'] = True
        # The existing V pipeline ties input/output embedding grads by local add/copy.
        if rank == 0:
            input_name = '0:module.module.language_model.embedding.word_embeddings.weight'
            output_name = '1:module.module.word_embeddings.weight'
            for label, data in (('reference', a), ('candidate', b)):
                tied = data['before_reduce_gradients'][input_name].clone()
                tied.add_(data['kernel_replay'][output_name])
                assert torch.equal(tied, data['raw_gradients'][input_name])
                assert torch.equal(tied, data['raw_gradients'][output_name])
            report['embedding_tie_add_copy_exact'] = True
        for name, actual in a['raw_gradients'].items():
            assert torch.equal(actual.float(), old['gradients'][name]), name
        for name, actual in a['final_optimizer'].items():
            assert torch.equal(actual['model'].float(), old['params_after'][name]), name
        report['legacy_gradient_and_update_match_original_reference_without_W_hooks'] = True
        # Supplementary first-step run must match the first full three-step capture.
        for label, data in (('legacy', a), ('permuted', b)):
            full = torch.load(root / f'{label}/iteration1-rank{rank}.pt', weights_only=False)
            for name, value in data['raw_gradients'].items():
                assert torch.equal(value, full['raw_gradients'][name]), (label, name)
            report[label + '_supplemental_matches_full_run'] = True
            del full
        reports.append(report)
        (root / 'reduce-replay.json').write_text(json.dumps(reports, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        del a, b, old
        gc.collect()
    raise SystemExit(0)

rows = []
for frozen in ((False, True) if args.include_frozen else (False,)):
    suffix = '-frozen' if frozen else ''
    for iteration in (1, 2, 3):
        for rank in (0, 1, 2):
            a = torch.load(root / f'legacy{suffix}/iteration{iteration}-rank{rank}.pt',
                           map_location='cpu', weights_only=False)
            b = torch.load(root / f'permuted{suffix}/iteration{iteration}-rank{rank}.pt',
                           map_location='cpu', weights_only=False)
            row = dict(frozen=frozen, iteration=iteration, rank=rank,
                       grad_norm=[a['grad_norm'], b['grad_norm']])
            for category in ('outputs', 'dx', 'logits', 'loss', 'raw_gradients'):
                row[category] = comparison(a[category], b[category])
            for stage in ('initial_optimizer', 'before_inner', 'after_inner', 'final_optimizer'):
                for item in ('master', 'model'):
                    row[f'{stage}/{item}'] = comparison(field(a, stage, item), field(b, stage, item))
            row['clipped_gradients'] = comparison(field(a, 'before_inner', 'grad'),
                                                   field(b, 'before_inner', 'grad'))
            if iteration == 1:
                assert a['backward_order'] == b['backward_order']
                assert a['closures'].keys() == b['closures'].keys()
                for identity, record in a['closures'].items():
                    candidate = b['closures'][identity]
                    assert record['executed'] == candidate['executed'] == 1
                    assert all(record['unchanged']) and all(candidate['unchanged'])
                    assert all(torch.equal(x, y) for x, y in
                               zip(record['operands'], candidate['operands'])), identity
                    assert torch.equal(record['contribution'], candidate['contribution']), identity
                row['closures'] = dict(count=len(a['closures']), operands_and_contributions_exact=True,
                                       order_changed=a['execution_order'] != b['execution_order'])
                # Post-reduction tied embeddings also contain B-direct grads.
                # --verify-reduce audits those exact add/copy operations.
                row['reference_replay_vs_post_reduce_unadjusted'] = comparison(
                    a['kernel_replay'], {n: a['raw_gradients'][n] for n in a['kernel_replay']})
                row['candidate_replay_vs_post_reduce_unadjusted'] = comparison(
                    b['kernel_replay'], {n: b['raw_gradients'][n] for n in b['kernel_replay']})
                row['replay_order_difference'] = comparison(a['kernel_replay'], b['kernel_replay'])
                independent = {}
                for name in a['kernel_replay']:
                    shape = a['kernel_replay'][name].shape
                    acc64 = torch.zeros(shape, dtype=torch.float64)
                    sums = []
                    for data in (a, b):
                        acc32 = torch.zeros(shape)
                        for identity in data['execution_order']:
                            record = data['closures'][identity]
                            if record['name'] == name:
                                contribution = record['contribution']
                                acc32.add_(contribution)
                                if data is a:
                                    acc64.add_(contribution.double())
                        sums.append(acc32)
                    independent[name] = dict(
                        fp32_order_difference=float((sums[0] - sums[1]).abs().max()),
                        reference_fp32_vs_fp64=float((sums[0].double() - acc64).abs().max()),
                        candidate_fp32_vs_fp64=float((sums[1].double() - acc64).abs().max()))
                row['independent_contribution_sums'] = independent
            if not frozen and iteration == 1 and row['final_optimizer/model']['different']:
                w = row['final_optimizer/model']['witness']
                name, idx = w['name'], w['index']
                trace = dict(name=name, index=idx)
                for label, data in (('reference', a), ('candidate', b)):
                    def at(t):
                        return None if t is None else float(t.flatten()[idx])
                    before, after = data['before_inner'][name], data['after_inner'][name]
                    trace[label] = dict(raw_gradient=at(data['raw_gradients'][name]),
                                        clipped_gradient=at(before['grad']),
                                        master_before=at(before['master']), master_after=at(after['master']),
                                        bf16_before=at(before['model']),
                                        bf16_after=at(data['final_optimizer'][name]['model']),
                                        m=at(after['state']['exp_avg']),
                                        v=at(after['state']['exp_avg_sq']))
                row['first_update_coordinate'] = trace
            rows.append(row)
            (root / 'analysis.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(json.dumps({k: row[k] for k in ('frozen', 'iteration', 'rank',
                                                  'raw_gradients', 'outputs', 'logits',
                                                  'final_optimizer/model')}), flush=True)
            del a, b
            gc.collect()
