#!/usr/bin/env python3
"""Read-only 027 evidence cross-check. Writes only the explicitly named report.

Production modules never import the research archive. This validation adapter
replays frozen orders and runs P1 A against exactly the archived cost/owners.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megatron.core.pipeline_parallel.dspp_plan import (
    PlanBinding, PlanOwner, node_key, plan_configuration_a, receive_orders,
    replay_plan, template_plan, validate_plan, preserves_retention_envelope,
)
from megatron.core.pipeline_parallel.slice_v import build_slice_v_schedule
from dataclasses import replace


def check(path):
    raw = json.loads(path.read_text())
    fused = raw['fused']
    counts = raw['ordering']['schedule_group_sizes_with_padding']
    real_counts = raw['ordering']['schedule_group_sizes']
    entrance = iter(raw['ordering']['entrance_order'])
    mapping = [(g, s, *next(entrance)) for g, c in enumerate(real_counts) for s in range(c)]
    tasks = {(g, s): (l, p) for g, s, l, p in mapping}
    schedules, _ = build_slice_v_schedule(raw['pp'], len(counts), counts)
    plan = template_plan(schedules, counts, mapping, PlanBinding(0, hashlib.sha256(path.read_bytes()).hexdigest(), '027'))
    names = {}
    for seq in schedules:
        for n in seq:
            l, p = tasks[(n.microbatch, n.split)]
            names[node_key(n)] = f's{n.stage}:c{n.chunk}:{n.kind}:l{l}:p{p}'
    inverse = {name: key for key, name in names.items()}
    assert all([names[node_key(n)] for n in seq] == fused['baseline_resource_orders'][str(r)]
               for r, seq in enumerate(schedules))
    costs = {key: fused['nodes'][name]['cost_ms'] for key, name in names.items()}
    comm = {('message', a, b, *msg): 1.0337119698524475
            for (a, b), lane in receive_orders(schedules).items() for msg in lane}
    tokens, owners = {}, []
    # Exact unsplit 025/027 owner contract; H births/users map back to fused B.
    for key, name in names.items():
        meta = fused['nodes'][name]
        if meta['kind'] != 'F':
            continue
        q = sum(z['token_length'] for z in meta['items'])
        tokens[key] = q
        b = inverse[name.replace(':F:', ':B:')]
        w = inverse[name.replace(':F:', ':W:')]
        owners.append(PlanOwner(meta['rank'], 32 * q * 4096, (key,), (w,)))
        for item in meta['items']:
            future = {other for other, othername in names.items()
                      if othername in fused['nodes'] and other[1] == 'F'
                      and other[0] == key[0] and other[2] == key[2]
                      and fused['nodes'][othername]['logical'] == meta['logical']
                      and any(z['sequence_id'] == item['sequence_id'] and z['segment_id'] > item['segment_id']
                              for z in fused['nodes'][othername]['items'])}
            writers = tuple((r, 'B', c, g, s) for r, _, c, g, s in future)
            owners.append(PlanOwner(meta['rank'], item['token_length'] * 8192,
                                    (key,), tuple({b} | future)))
            if writers:
                owners.append(PlanOwner(meta['rank'], item['token_length'] * 16384, writers, (w,)))
    frozen_path = path.parent.parent / f'result-{path.stem.removeprefix("input-")}-fixed-FB-r0.json'
    frozen = json.loads(frozen_path.read_text())
    ref = replay_plan(plan, costs, comm, owners, tokens)
    assert abs(ref['makespan_ms'] - frozen['original_ms']) < 1e-6
    frozen_plan = replace(plan, algorithm='A', schedules=tuple(
        tuple(replace(next(n for n in schedules[r] if node_key(n) == inverse[name]), slot=i)
              for i, name in enumerate(order)) for r, order in enumerate(frozen['resource_orders'])))
    validate_plan(frozen_plan, plan)
    replay = replay_plan(frozen_plan, costs, comm, owners, tokens)
    assert abs(replay['makespan_ms'] - frozen['makespan_ms']) < 1e-6
    assert list(replay['owner_peaks']) == frozen['arena_peak_bytes']
    candidate = plan_configuration_a(plan, costs, comm, owners=owners, tokens=tokens)
    validate_plan(candidate, plan)
    actual = replay_plan(candidate, costs, comm, owners, tokens)
    # Numeric tie IDs in production use stage-local template order; 027 uses
    # serialized research-node order. These frozen cases must match exactly;
    # this does not establish equivalence for arbitrary inputs with tied dates.
    assert not candidate.fallback_reason, candidate.fallback_reason
    assert candidate.schedules == frozen_plan.schedules, path.name
    assert abs(actual['makespan_ms'] - replay['makespan_ms']) < 1e-6, path.name
    assert actual['owner_peaks'] == replay['owner_peaks'], path.name
    assert preserves_retention_envelope(candidate, plan), path.name
    return dict(case=path.stem, pp=raw['pp'], input_sha256=plan.binding.input_hash,
                original_ms=ref['makespan_ms'], frozen_a_ms=replay['makespan_ms'],
                p1_a_ms=actual['makespan_ms'], difference_ms=actual['makespan_ms'] - replay['makespan_ms'],
                owner_peaks=actual['owner_peaks'], frozen_owner_peaks=replay['owner_peaks'],
                identical_orders=candidate.schedules == frozen_plan.schedules,
                retention_envelope=preserves_retention_envelope(candidate, plan),
                fallback_reason=candidate.fallback_reason)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'refusing to overwrite existing validation evidence'
    reports = [check(path) for path in sorted((args.archive / 'inputs').glob('input-*.json'))]
    assert len(reports) == 8 and {r['pp'] for r in reports} == {4, 8}, 'expected all eight 027 PP4/8 cases'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))
