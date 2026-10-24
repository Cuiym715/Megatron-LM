"""Audit saved paired runs and compute effective-token wall throughput."""
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
METHODS = ('legacy', 'template', 'a')
SEEDS = (11, 19, 29)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def key(node):
    return tuple(node[k] for k in ('stage', 'kind', 'chunk', 'microbatch', 'split'))

def order(plan, stage, kinds, chunk=None):
    return [key(n) for n in plan['schedules'][stage]
            if n['kind'] in kinds and (chunk is None or n['chunk'] == chunk)]

def audit(candidate, reference):
    assert candidate['binding'] == reference['binding'], 'input/layout/profile mismatch'
    assert candidate['split_counts'] == reference['split_counts']
    assert candidate['task_map'] == reference['task_map']
    assert candidate['messages'] == reference['messages'], 'FIFO/wire changed'
    moved, moved_real_frontiers = 0, 0
    real = {(g, s) for g, s, _, _ in reference['task_map']}
    for stage in range(3):
        a, b = candidate['schedules'][stage], reference['schedules'][stage]
        assert sorted(map(key, a)) == sorted(map(key, b)), 'task set mismatch'
        assert len({key(n) for n in a}) == len(a), 'duplicate task'
        assert order(candidate, stage, 'FB') == order(reference, stage, 'FB')
        for chunk in (0, 1):
            assert order(candidate, stage, 'W', chunk) == order(reference, stage, 'W', chunk)
            assert [k[3:] for k in order(candidate, stage, 'W', chunk)] == [
                k[3:] for k in order(candidate, stage, 'B', chunk)]
        positions = {key(n): i for i, n in enumerate(b)}
        def frontiers(sequence):
            count, result = 0, {}
            for n in sequence:
                if n['kind'] in 'FB':
                    count += 1
                elif n['kind'] == 'W' and (n['microbatch'], n['split']) in real:
                    result[key(n)] = count
            return result
        before, after = frontiers(b), frontiers(a)
        assert before.keys() == after.keys()
        moved_real_frontiers += sum(before[k] != after[k] for k in before)
        for i, n in enumerate(a):
            if n['kind'] == 'W':
                moved += positions[key(n)] != i
                bk = (stage, 'B', n['chunk'], n['microbatch'], n['split'])
                assert next(j for j, x in enumerate(a) if key(x) == bk) < i
    return moved, moved_real_frontiers

def main(partial=False):
    rows, pairs = [], []
    for repeat in (1, 2):
        for seed in SEEDS:
            paths = {m: ROOT / f'bench/repeat{repeat}/seed{seed}/{m}' for m in METHODS}
            if partial and any(not (p / 'exit.json').exists() for p in paths.values()):
                continue
            groups = {}
            lengths = [int(x) for x in (ROOT / f'data/seed{seed}/lengths.txt').read_text().split()]
            for method, path in paths.items():
                assert json.loads((path / 'exit.json').read_text())['code'] == 0
                ranks = [json.loads((path / f'rank{r}.json').read_text()) for r in range(3)]
                groups[method] = ranks
                for r, report in enumerate(ranks):
                    assert report['rank'] == r and report['warmup_iterations'] == 4
                    assert [x['iteration'] for x in report['records']] == list(range(5, 17))
                    assert not any(x['skipped'] for x in report['records'])
                metrics = [r['metrics'] for r in ranks[0]['records']]
                for iteration, metric in zip(range(5, 17), metrics):
                    assert metric['valid_tokens'] == sum(lengths[(iteration-1)*16:iteration*16]) - 16
                for i in range(12):
                    hashes = {digest(r['records'][i]['plan']) for r in ranks}
                    assert len(hashes) == 1, 'cross-rank plan mismatch'
                tokens = sum(m['valid_tokens'] for m in metrics)
                seconds = max(r['seconds'] for r in ranks)
                text = (path / 'train.log').read_text()
                assert 'number of skipped iterations:   0' in text
                assert 'number of nan iterations:   0' in text
                assert 'Traceback (most recent call last)' not in text
                row = dict(repeat=repeat, seed=seed, method=method, valid_tokens=tokens,
                           seconds=seconds, effective_tokens_per_second=tokens/seconds,
                           seconds_per_iteration=seconds/12,
                           allocated_peak_mib=[r['allocated_peak']/2**20 for r in ranks],
                           reserved_peak_mib=[r['reserved_peak']/2**20 for r in ranks])
                rows.append(row)
            assert len({r['valid_tokens'] for r in rows[-3:]}) == 1
            moved, moved_frontiers, accepted, predicted, fallbacks = [], [], 0, [], collections.Counter()
            predicted_reference, predicted_candidate = 0.0, 0.0
            for i in range(12):
                template = groups['template'][0]['records'][i]['plan']
                candidate = groups['a'][0]['records'][i]['plan']
                assert template['algorithm'] == 'template' and not template['fallback_reason']
                assert candidate['algorithm'] == 'A'
                predicted_reference += candidate['reference_ms']
                predicted_candidate += candidate['predicted_ms']
                changed, changed_frontiers = audit(candidate, template)
                moved.append(changed)
                moved_frontiers.append(changed_frontiers)
                if candidate['fallback_reason']:
                    fallbacks[candidate['fallback_reason']] += 1
                else:
                    assert moved[-1] > 0 and moved_frontiers[-1] > 0 and candidate['predicted_ms'] < candidate['reference_ms']
                    accepted += 1
                    predicted.append(candidate['reference_ms']/candidate['predicted_ms']-1)
            by_method = {r['method']: r for r in rows[-3:]}
            pair = dict(repeat=repeat, seed=seed, accepted_iterations=accepted,
                        fallbacks=dict(fallbacks), moved_W_nodes=moved,
                        moved_real_W_FB_frontiers=moved_frontiers,
                        predicted_reference_ms=predicted_reference,
                        predicted_candidate_ms=predicted_candidate,
                        prediction_speedup_mean=statistics.mean(predicted) if predicted else 0)
            for reference in ('legacy', 'template'):
                pair[f'a_vs_{reference}_throughput_change_percent'] = 100 * (
                    by_method['a']['effective_tokens_per_second'] /
                    by_method[reference]['effective_tokens_per_second'] - 1)
                pair[f'a_vs_{reference}_allocated_peak_change_percent_by_rank'] = [
                    100*(a/b-1) for a, b in zip(by_method['a']['allocated_peak_mib'],
                                               by_method[reference]['allocated_peak_mib'])]
            pairs.append(pair)
    totals = {}
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        if selected:
            seconds = sum(r['seconds'] for r in selected)
            tokens = sum(r['valid_tokens'] for r in selected)
            totals[method] = dict(runs=len(selected), valid_tokens=tokens, seconds=seconds,
                                 effective_tokens_per_second=tokens/seconds,
                                 allocated_peak_mib=[max(r['allocated_peak_mib'][i] for r in selected)
                                                     for i in range(3)],
                                 reserved_peak_mib=[max(r['reserved_peak_mib'][i] for r in selected)
                                                    for i in range(3)])
    comparisons = {}
    if totals:
        for reference in ('legacy', 'template'):
            values = [p[f'a_vs_{reference}_throughput_change_percent'] for p in pairs]
            comparisons[reference] = dict(
                pooled_throughput_change_percent=100*(totals['a']['effective_tokens_per_second']/
                                                      totals[reference]['effective_tokens_per_second']-1),
                paired_mean_percent=statistics.mean(values),
                paired_median_percent=statistics.median(values),
                paired_min_percent=min(values), paired_max_percent=max(values),
                paired_wins=sum(v > 0 for v in values))
    failures = []
    for path in sorted((ROOT/'failures').glob('*/exit.json')):
        failed = json.loads(path.read_text())
        assert failed['code'] != 0
        failures.append(dict(path=str(path.parent), **failed,
                             classification='legacy stalled after iteration12; stack dumped then manually terminated; root cause unresolved; excluded from completed-run throughput'))
    result = dict(partial=partial, runs=rows, pairs=pairs, totals=totals,
                  comparisons=comparisons, excluded_failed_attempts=failures,
                  scope='Throughput of completed runs; failure remains a separate reliability result, not a passed stability test.')
    if not partial:
        assert len(rows) == 18
        for seed in SEEDS:
            for method in ('template', 'a'):
                repeated = [json.loads((ROOT/f'bench/repeat{repeat}/seed{seed}/{method}/rank0.json').read_text())
                            for repeat in (1, 2)]
                assert [r['plan'] for r in repeated[0]['records']] == [r['plan'] for r in repeated[1]['records']], 'plan changed between repeated same-input runs'
        result['audit'] = dict(paired_iterations=72, measured_rank_iterations=648,
                               cross_rank_plans_agree=True, tokens_match_raw_documents=True,
                               fixed_FB=True, fixed_W_per_chunk=True, W_complete=True,
                               messages_match_template=True, repeated_plans_identical=True,
                               no_skipped_or_nan_iterations=True)
        (ROOT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(totals=totals, comparisons=comparisons,
                         accepted=sum(p['accepted_iterations'] for p in pairs),
                         paired_runs=len(pairs)), indent=2))

if __name__ == '__main__':
    main('--partial' in sys.argv)
