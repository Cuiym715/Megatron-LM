"""Numerical fixture: real PP stage D, unrelated tasks, then H before consumer.

Uses the normal executor and its memory admission. No production scheduling
hook. Environment/arguments are those of validate_dspp_plan_training.py.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def install():
    from megatron.core.pipeline_parallel import schedules
    from megatron.core.pipeline_parallel.dspp_plan import node_key, split_plan, validate_plan
    from megatron.core.datasets.dspp_profile_snapshot import freeze_runtime_profiles
    original = schedules.build_runtime_plan
    prepared = False

    def build(args, *a, **kw):
        nonlocal prepared
        if not prepared:
            freeze_runtime_profiles(args)
            prepared = True
        kw.update(explicit_history=True, optimize=False)
        reference, _ = original(args, *a, **kw)
        sources = {edge[:2] for edge in reference.history_dependencies}
        selected = [node_key(n) for seq in reference.schedules for n in seq
                    if n.kind == 'B' and (n.microbatch,n.split) in sources]
        adjacent = split_plan(reference, selected)
        output, delayed = [], 0
        for seq in adjacent.schedules:
            out = [n for n in seq if n.kind != 'H']
            for h in [n for n in seq if n.kind == 'H']:
                targets = {edge[2:] for edge in reference.history_dependencies
                           if edge[:2] == (h.microbatch,h.split)}
                index = min(i for i,n in enumerate(out) if n.kind in ('B','D')
                            and n.chunk == h.chunk and (n.microbatch,n.split) in targets)
                d = next(i for i,n in enumerate(out) if n.kind == 'D'
                         and (n.chunk,n.microbatch,n.split) == (h.chunk,h.microbatch,h.split))
                delayed += int(index > d+1)
                out.insert(index,h)
            output.append(tuple(replace(n,slot=i) for i,n in enumerate(out)))
        plan = replace(adjacent,schedules=tuple(output))
        validate_plan(plan,reference)
        args._dspp_memory_plan = args._dspp_counted_memory.check(plan)
        root = Path(os.environ['DSPP_PLAN_CHECK_OUTPUT'])/'plans'
        root.mkdir(exist_ok=True)
        import torch.distributed as dist
        if dist.get_rank() == 0:
            (root/f'iteration{args.curr_iteration+1}.json').write_text(json.dumps(dict(
                delayed=delayed,selected=len(selected),plan=plan.to_dict()),indent=2))
        return reference,plan

    schedules.build_runtime_plan = build


if __name__ == '__main__':
    install()
    runpy.run_path(str(REPO/'tools/validate_dspp_plan_training.py'),run_name='__main__')
