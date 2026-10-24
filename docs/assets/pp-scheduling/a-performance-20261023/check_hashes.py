"""Compare every recorded tensor byte hash, without changing tolerances."""
import collections
import json
from pathlib import Path
import statistics
from analyze import audit, digest

root = Path(__file__).resolve().parent
records = {}
for method in ('legacy', 'template', 'a'):
    path = root / f'diagnostic/seed11/{method}'
    assert json.loads((path/'exit.json').read_text())['code'] == 0
    records[method] = {(i, r): json.loads((path/f'iteration{i}-rank{r}.json').read_text())
                       for i in range(1, 7) for r in range(3)}
    assert all(d['deterministic_fa'] and d['skipped'] == 0 for d in records[method].values())

comparisons = {}
for method in ('template', 'a'):
    totals, mismatches = collections.Counter(), []
    for identity, actual in records[method].items():
        expected = records['legacy'][identity]
        assert actual['values'].keys() == expected['values'].keys()
        for category, values in actual['values'].items():
            reference = expected['values'][category]
            assert values.keys() == reference.keys(), (identity, category)
            totals[category] += len(values)
            for key, value in values.items():
                if value != reference[key]:
                    mismatches.append(dict(iteration=identity[0], rank=identity[1],
                                           category=category, tensor=key,
                                           actual=value, reference=reference[key]))
    timings = [d['build_runtime_plan_seconds'] for (i, r), d in records[method].items() if i in (5, 6)]
    comparisons[method] = dict(bitwise_equal=not mismatches, rank_step_reports=18,
                               tensor_comparisons=dict(totals),
                               mismatched_tensors=len(mismatches), first_mismatches=mismatches[:30],
                               runtime_plan_diagnostic_seconds=dict(median=statistics.median(timings),
                                                                    minimum=min(timings), maximum=max(timings)))
accepted = sum(not d['plan']['fallback_reason'] for (i, r), d in records['a'].items() if r == 0)
frontiers = []
for iteration in range(1, 7):
    for method in ('template', 'a'):
        assert len({digest(records[method][iteration, rank]['plan']) for rank in range(3)}) == 1
    frontiers.append(audit(records['a'][iteration, 0]['plan'], records['template'][iteration, 0]['plan'])[1])
assert accepted == 6 and all(n > 0 for n in frontiers)
report = dict(comparisons=comparisons, a_accepted_steps=accepted,
              moved_real_W_FB_frontiers=frontiers, cross_rank_plans_agree=True,
              scope='6 optimizer steps, seed11, deterministic FA; all stage output/dx, logits/loss, full gradients and before/after model parameter byte hashes; no tensor subsampling',
              limitation='Numerical run adds CPU copies/hashing and deterministic FA. Its timing/memory is not performance evidence. Plan CPU duration excludes subsequent validation/hash/all-gather/publication, and is an indicative diagnostic only. Default FA numerical gate remains separate.')
(root/'numeric-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
