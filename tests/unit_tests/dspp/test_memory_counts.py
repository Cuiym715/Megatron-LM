"""Counted capacity admission, independent of CUDA and predicted milliseconds."""
from copy import deepcopy
from dataclasses import replace
import json
import random
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.datasets.dspp_execution_cost import DsppExecutionCostModel
from megatron.core.pipeline_parallel.dspp_memory_counts import CountedMemory, memory_layout, history_state_bytes
from megatron.core.pipeline_parallel.dspp_plan import (
    PlanBinding, node_key, order_owner_peaks, plan_configuration_a, receive_orders,
    template_plan, validate_plan,
)
from megatron.core.pipeline_parallel.dspp_plan_runtime import build_runtime_plan
from megatron.core.pipeline_parallel.slice_v import build_slice_v_schedule
from megatron.core.pipeline_parallel.schedules import _pad_dspp_v_split_counts


def case(pp=3):
    args = SimpleNamespace(hidden_size=8, dspp_schedule='a', dspp_recv_lookahead=2,
                           dspp_send_inflight_depth=2)
    batch = build_dspp_batch_plan([193, 129, 33, 17], chunk_size=64)
    groups = [tuple((l, p) for p in range(len(batch.microbatches))) for l in range(2)]
    counts = _pad_dspp_v_split_counts([len(g) for g in groups], pp)
    seqs, _ = build_slice_v_schedule(pp, len(counts), counts)
    mapping = tuple((g, s, *task) for g, group in enumerate(groups) for s, task in enumerate(group))
    reference = replace(template_plan(seqs, counts, mapping, PlanBinding(1, 'input', 'layout')),
                        wire_bytes=64 * 8 * 2, wire_shape=(64, 1, 8), wire_dtype='torch.bfloat16')
    meta = {(g, s): batch.microbatches[p] for g, s, l, p in mapping}
    profile = dict(schema='dspp-counted-memory-v1', source='unit-size fixture',
        device_name='test', layout=memory_layout(args, pp, torch.bfloat16, 64),
        dynamic_growth=.05, limits=dict(max_tasks=100, max_logical_batches=2,
                                       max_key_tokens=1000, max_items=16),
        stages=[dict(resident_bytes=1000, workspace_bytes=30, physical_bytes=1000000,
            reserve_bytes=1000, activation_bytes=[100, 130], weight_bytes=[45, 55],
            kv_bytes_per_token=[3, 3], attention_bytes_per_key_token=[1, 1]) for _ in range(pp)])
    return args, batch, groups, reference, meta, profile


@pytest.mark.parametrize('pp', [3, 4, 8])
def test_local_order_peak_matches_independent_active_sum_and_a(pp):
    args, _, _, ref, meta, profile = case(pp)
    memory = CountedMemory(profile, args, ref, meta, torch.bfloat16, 64)
    positions = {node_key(n): i for seq in ref.schedules for i, n in enumerate(seq)}
    expected = []
    for r, seq in enumerate(ref.schedules):
        expected.append(max(sum(o.bytes for o in memory.owners if o.stage == r
            and min(positions[k] for k in o.born) <= i <= max(positions[k] for k in o.users))
            for i in range(len(seq))))
    assert order_owner_peaks(ref, memory.owners) == tuple(expected)
    rng = random.Random(7)
    compute = {node_key(n): rng.uniform(.2, 4) for seq in ref.schedules for n in seq}
    comm = {('message', a, b, *msg): .13 for (a, b), lane in receive_orders(ref.schedules).items() for msg in lane}
    candidate = plan_configuration_a(ref, compute, comm, owners=memory.owners,
        capacities=memory.capacities, order_based_capacity=True)
    validate_plan(candidate, ref)
    report = memory.check(candidate)
    assert all(x <= b * 1.05 for x, b in zip(report['dynamic_peak_bytes'], memory.baseline_dynamic))


def test_w_starts_charging_at_f_and_shared_kv_has_one_record_per_logical_chunk():
    args, _, _, ref, meta, profile = case()
    memory = CountedMemory(profile, args, ref, meta, torch.bfloat16, 64)
    nreal = len(meta) * 2 * 3
    assert len(memory.owners) == nreal * 2 + 2 * 2 * 3
    for activation, weight in zip(memory.owners[:2*nreal:2], memory.owners[1:2*nreal:2]):
        assert weight.born == activation.born
        assert weight.born[0][1] == 'F' and weight.users[0][1] == 'W'
    delayed = replace(ref, algorithm='A', schedules=tuple(tuple(
        replace(n, slot=i) for i, n in enumerate([n for n in seq if n.kind != 'W'] +
                                                [n for n in seq if n.kind == 'W'])) for seq in ref.schedules))
    validate_plan(delayed, ref)
    assert any(a > b for a, b in zip(order_owner_peaks(delayed, memory.owners),
                                    order_owner_peaks(ref, memory.owners)))
    # Make W retention dominate fixed communication reserve for this check.
    larger = deepcopy(profile)
    for row in larger['stages']:
        row['weight_bytes'] = [10000, 10000]
    memory = CountedMemory(larger, args, ref, meta, torch.bfloat16, 64)
    with pytest.raises(ValueError, match='capacity'):
        memory.check(delayed)


@pytest.mark.parametrize('mutation, error', [
    (lambda p: p['layout'].update(chunk_size=128), 'layout mismatch'),
    (lambda p: p['limits'].update(max_key_tokens=64), 'coverage'),
    (lambda p: p.update(dynamic_growth=float('nan')), 'growth'),
    (lambda p: p['stages'][0].update(weight_bytes=[-1, 3]), 'integer bytes'),
    (lambda p: p['stages'][0].update(physical_bytes=2001), 'capacity'),
])
def test_rejects_stale_uncovered_or_insufficient_profiles(mutation, error):
    args, _, _, ref, meta, profile = case()
    mutation(profile)
    with pytest.raises(ValueError, match=error):
        CountedMemory(profile, args, ref, meta, torch.bfloat16, 64)


def test_runtime_passes_real_owners_capacities_and_binds_profile(tmp_path, monkeypatch):
    args, batch, groups, ref, meta, profile = case()
    path = tmp_path/'memory.json'
    path.write_text(json.dumps(profile))
    args.dspp_memory_profile = str(path)
    records = [dict(stage=r, kind=op, duration_ms=1., sequence_chunk_size=64,
                    items=[vars(i) for i in m.items])
               for r in range(3) for op in 'FBW' for m in batch.microbatches]
    cost = tmp_path/'cost.json'
    DsppExecutionCostModel.from_records(records).save(cost)
    args.dspp_execution_profile = str(cost)
    import megatron.core.pipeline_parallel.dspp_plan_runtime as runtime
    original = runtime.plan_configuration_a
    calls = []
    def observed(*a, **kw):
        calls.append(kw)
        return original(*a, **kw)
    monkeypatch.setattr(runtime, 'plan_configuration_a', observed)
    reference, candidate = build_runtime_plan(args, ref.schedules, ref.split_counts,
        groups, [SimpleNamespace(plan=batch)] * 2, torch.bfloat16, (64, 1, 8), 64)
    assert calls and calls[0]['owners'] and calls[0]['order_based_capacity']
    assert candidate.binding.capacity_version == args._dspp_counted_memory.version
    assert reference.binding.capacity_version == candidate.binding.capacity_version
    assert args._dspp_memory_plan == args._dspp_counted_memory.check(candidate)
    profile['stages'][0]['weight_bytes'][0] += 1
    path.write_text(json.dumps(profile))
    newer, _ = build_runtime_plan(args, ref.schedules, ref.split_counts,
        groups, [SimpleNamespace(plan=batch)] * 2, torch.bfloat16, (64, 1, 8), 64)
    assert newer.binding.capacity_version != reference.binding.capacity_version


def test_packed_key_coverage_and_unimplemented_split_are_explicit():
    args, _, _, ref, meta, profile = case()
    longest = max(i.sequence_length for m in meta.values() for i in m.items)
    packed_keys = max(sum(m.key_lengths) for m in meta.values())
    assert packed_keys > longest
    profile['limits']['max_key_tokens'] = longest
    with pytest.raises(ValueError, match='coverage'):
        CountedMemory(profile, args, ref, meta, torch.bfloat16, 64)
    profile['limits']['max_key_tokens'] = packed_keys
    CountedMemory(profile, args, ref, meta, torch.bfloat16, 64)
    args.dspp_split_attention_backward = True
    with pytest.raises(ValueError, match='fused B'):
        CountedMemory(profile, args, ref, meta, torch.bfloat16, 64)


def split_case(pp=3):
    args, batch, groups, ref, meta, profile = case(pp)
    args.dspp_split_attention_backward = args.dspp_history_dkv_after_dx = True
    args.num_layers, args.hidden_size = pp * 2, 512
    args.num_attention_heads, args.kv_channels = 4, 128
    args.num_query_groups, args.group_query_attention = 1, False
    profile.update(schema='dspp-counted-memory-v2', split_state='fa2-after-dx-v1',
                   layout=memory_layout(args, pp, torch.bfloat16, 64))
    for row in profile['stages']:
        row.update(physical_bytes=1 << 40, activation_bytes=[16 << 20] * 2,
                   weight_bytes=[8 << 20] * 2)
    return args, batch, groups, ref, meta, profile


@pytest.mark.parametrize('pp', [3, 4, 8])
def test_split_sources_last_until_h_and_pending_cache_last_user(pp):
    args, _, _, ref, meta, profile = split_case(pp)
    memory = CountedMemory(profile, args, ref, meta, torch.bfloat16, 64)
    plan, owners, sizes = memory.records(ref)
    assert sizes['history_source'] > 0 and sizes['history_created'] > 0
    positions = {node_key(n): i for seq in plan.schedules for i, n in enumerate(seq)}
    expected = tuple(max(sum(o.bytes for o in owners if o.stage == r
        and min(positions[k] for k in o.born) <= i <= max(positions[k] for k in o.users))
        for i in range(len(seq))) for r, seq in enumerate(plan.schedules))
    assert order_owner_peaks(plan, owners) == expected
    for owner in owners:
        if len(owner.users) == 1 and owner.users[0][1] == 'H':
            assert owner.born[0][1] in ('F', 'D')
    assert memory.check(plan)['owner_peak_bytes'] == list(expected)
    # Two logical batches permit independent H jobs to coexist. Move each H
    # across the other batch's D only when all consumer constraints still fit.
    found = False
    for r, seq in enumerate(plan.schedules):
        for i, n in enumerate(seq[:-2]):
            if n.kind != 'H':
                continue
            for j in range(i + 2, len(seq)):
                if seq[j].kind != 'D' or (seq[j].microbatch == n.microbatch):
                    continue
                out = list(seq); out.insert(j, out.pop(i))
                candidate = replace(plan, schedules=tuple(tuple(out) if rank == r else s
                                                          for rank, s in enumerate(plan.schedules)))
                try:
                    _, delayed, _ = memory.records(candidate)
                except ValueError:
                    continue
                ds = {node_key(x): p for s in candidate.schedules for p, x in enumerate(s)}
                h_records = [o for o in delayed if o.stage == r and o.born[0][1] == 'D']
                assert max(sum(ds[o.born[0]] <= p <= ds[o.users[0]] for o in h_records)
                           for p in range(len(out))) >= 2
                found = True
                break
            if found:
                break
        if found:
            break
    if not found:
        # PP8's template serializes these small groups. Build a local accounting
        # fixture with all F ready, round-robin backward chains, then all W.
        # This is a memory-interface test, not a runnable global PP schedule.
        seq = plan.schedules[0]
        chains = {}
        for n in seq:
            if n.kind in ('B', 'D'):
                chains.setdefault((n.microbatch, n.chunk), []).append(n)
        hs = {node_key(n): n for n in seq if n.kind == 'H'}
        out = [n for n in seq if n.kind == 'F']
        while any(chains.values()):
            pending = []
            for chain in chains.values():
                if chain:
                    n = chain.pop(0); out.append(n)
                    if n.kind == 'D':
                        pending.append(hs[(n.stage, 'H', n.chunk, n.microbatch, n.split)])
            out.extend(pending)
        out.extend(n for n in seq if n.kind == 'W')
        candidate = replace(plan, schedules=(tuple(out), *plan.schedules[1:]))
        _, delayed, _ = memory.records(candidate)
        ds = {node_key(x): i for i, x in enumerate(out)}
        h_records = [o for o in delayed if o.stage == 0 and o.born[0][1] == 'D']
        assert max(sum(ds[o.born[0]] <= i <= ds[o.users[0]] for o in h_records)
                   for i in range(len(out))) >= 2


def test_split_rejects_missing_duplicate_and_late_history():
    args, _, _, ref, meta, profile = split_case()
    memory = CountedMemory(profile, args, ref, meta, torch.bfloat16, 64)
    plan, _, _ = memory.records(ref)
    seq = list(plan.schedules[0]); i = next(i for i, n in enumerate(seq) if n.kind == 'H')
    for out, message in ((seq[:i] + seq[i+1:], 'missing H'),
                         (seq[:i] + [seq[i]] + seq[i:], 'duplicate'),
                         (seq[:i] + seq[i+1:] + [seq[i]], 'consumer')):
        bad = replace(plan, schedules=(tuple(out), *plan.schedules[1:]))
        with pytest.raises(ValueError, match=message):
            memory.check(bad)


def test_split_size_uses_gqa_and_per_sequence_delta_rounding():
    args, batch, _, _, _, _ = split_case()
    packed = next(m for m in batch.microbatches if len(m.items) > 1 and any(i.has_history for i in m.items))
    source, created, scratch = history_state_bytes(args, 3, 1, 0, packed)
    q, k, c, h = packed.valid_token_count, sum(packed.key_lengths), 64, 4
    assert created == c*h*128*2 + (q + sum((i.token_length+127)//128*128 for i in packed.items))*h*4
    args.group_query_attention = True
    gs, gc, gw = history_state_bytes(args, 3, 1, 0, packed)
    assert gs < source and gw < scratch and gc == created
    empty = next(m for m in batch.microbatches if not any(i.has_history for i in m.items))
    assert history_state_bytes(args, 3, 1, 0, empty) == (0, 0, 0)


def test_split_template_runtime_binds_new_profile(tmp_path):
    args, batch, groups, ref, _, profile = split_case()
    args.dspp_schedule = 'template'
    p = tmp_path / 'split.json'; p.write_text(json.dumps(profile))
    args.dspp_memory_profile = str(p)
    reference, _ = build_runtime_plan(args, ref.schedules, ref.split_counts, groups,
        [SimpleNamespace(plan=batch)] * 2, torch.bfloat16, (64, 1, 512), 64)
    assert reference.binding.capacity_version == args._dspp_counted_memory.version
    assert args._dspp_memory_plan['category_allowance_bytes']['history_source'] > 0


def test_cache_and_history_sources_are_not_released_by_w():
    args, _, _, ref, meta, profile = split_case()
    memory = CountedMemory(profile, args, ref, meta, torch.bfloat16, 64)
    plan, _, _ = memory.records(ref)
    seq = list(plan.schedules[0])
    h = next(n for n in seq if n.kind == 'H')
    wkey = (h.stage, 'W', h.chunk, h.microbatch, h.split)
    w = next(n for n in seq if node_key(n) == wkey)
    seq.remove(w); seq.insert(seq.index(h), w)
    candidate = replace(plan, schedules=(tuple(seq), *plan.schedules[1:]))
    _, owners, _ = memory.records(candidate)
    assert any(node_key(h) in o.users and wkey in o.users for o in owners)
    # H-created state remains charged across W.
    assert any(o.born[0][1] == 'D' and o.users == (node_key(h),) for o in owners)


def test_split_retention_can_reject_an_insufficient_budget():
    args, _, _, ref, meta, profile = split_case()
    # With no allowed growth, extra F->H and D->H reservations cannot be
    # hidden by declaring D equivalent to B and discarding its retained inputs.
    profile['dynamic_growth'] = 0
    with pytest.raises(ValueError, match='capacity'):
        CountedMemory(profile, args, ref, meta, torch.bfloat16, 64)
