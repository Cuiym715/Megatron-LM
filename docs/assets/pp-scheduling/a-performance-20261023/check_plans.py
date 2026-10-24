"""Production plan/dependency validation plus existing finite-credit CPU replay."""
import ast
import importlib
import json
from pathlib import Path
import sys
import types

repo = Path('/home/cuiym/src/Megatron-LM-kwai')
root = Path(__file__).resolve().parent
pkg = types.ModuleType('performance_cpu_check')
pkg.__path__ = [str(repo / 'megatron/core/pipeline_parallel')]
sys.modules[pkg.__name__] = pkg
module = importlib.import_module(pkg.__name__ + '.dspp_plan')
test = repo / 'tests/unit_tests/dspp/test_schedule_plan.py'
function = next(n for n in ast.parse(test.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'simulate_protocol')
namespace = vars(module).copy()
exec(compile(ast.Module(body=[function], type_ignores=[]), str(test), 'exec'), namespace)
rows = []
for repeat in (1, 2):
    for seed in (11, 19, 29):
        base = root / f'bench/repeat{repeat}/seed{seed}'
        plans = {}
        for method in ('template', 'a'):
            records = [json.loads((base / method / f'rank{r}.json').read_text()) for r in range(3)]
            plans[method] = []
            for index in range(12):
                decoded = [module.DsppSchedulePlan.from_dict(r['records'][index]['plan']) for r in records]
                assert len({p.plan_hash for p in decoded}) == 1
                plans[method].append(decoded[0])
        for index, (reference, candidate) in enumerate(zip(plans['template'], plans['a'])):
            module.validate_plan(candidate, reference, binding=reference.binding)
            assert module.preserves_retention_envelope(candidate, reference)
            for method, plan in [('template', reference), ('a', candidate)]:
                for lookahead, depth in [(0, 1), (1, 1), (2, 2), (4, 1)]:
                    namespace['simulate_protocol'](plan, lookahead, depth)
                rows.append(dict(repeat=repeat, seed=seed, iteration=index+5, method=method,
                                 plan_hash=plan.plan_hash, task_counts=list(map(len, plan.schedules)),
                                 fallback=plan.fallback_reason))
assert len(rows) == 144
report = dict(plans=len(rows), finite_credit_replays=4*len(rows), checks=rows,
              limitation='CPU atomic rendezvous detects logical wait cycles; it does not simulate NCCL/CUDA stream liveness or certify real owner memory.')
(root / 'plan-checks.json').write_text(json.dumps(report, indent=2)+'\n')
print('144 saved plans validated; 576 finite-credit CPU replays drained; cross-rank hashes and W retention envelope valid.')
