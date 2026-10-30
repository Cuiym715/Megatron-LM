"""Frozen state-size allowances for the existing planner's byte counters.

No tensor traversal, allocator tracing, CUDA queries or timing prediction.
Coefficients are explicit per-model/per-stage calibration inputs, not inferred
from an allocator residual. Split-state allowances follow fa_split.py shapes;
explicit D/H orders are accounting inputs, not a new execution path.
"""
from dataclasses import replace
from copy import deepcopy
import json
import math
from pathlib import Path
import torch

from .dspp_plan import PlanOwner, digest, node_key, order_owner_peaks
from .dspp_progress import dspp_communication_buffer_count
from .dspp_memory_lifetimes import (
    ACCOUNTING, CompiledStorage, stage_attention_sizes, records as storage_records,
)


def memory_layout(args, pp, dtype, chunk_size):
    names = (
        'num_layers', 'hidden_size', 'ffn_hidden_size', 'num_attention_heads',
        'num_query_groups', 'group_query_attention', 'kv_channels', 'padded_vocab_size', 'dspp_v_layer_layout',
        'normalization', 'norm_epsilon', 'swiglu', 'position_embedding_type',
        'rotary_percent', 'rotary_base', 'hidden_dropout', 'attention_dropout',
        'gradient_accumulation_fusion', 'tensor_model_parallel_size',
        'context_parallel_size', 'sequence_parallel', 'transformer_impl',
        'untie_embeddings_and_output_weights', 'recompute_granularity',
        'recompute_method', 'recompute_num_layers', 'use_flash_attn',
        'rms_norm', 'use_fast_rms_norm', 'layernorm_epsilon',
        'use_rotary_position_embeddings', 'disable_bias_linear',
        'bias_gelu_fusion', 'bias_dropout_fusion', 'fp32_residual_connection',
    )
    return dict(pp=pp, vpp=2, dtype=str(dtype), chunk_size=chunk_size,
                model={k: getattr(args, k, None) for k in names},
                mode=('after-dx-eager-v1' if getattr(args, 'dspp_history_dkv_after_dx', False)
                      else 'fused-B-eager-v1'), torch=torch.__version__, cuda=torch.version.cuda)


def history_state_bytes(args, pp, stage, chunk, meta):
    """Conservative bytes retained by all layers of one physical H task.

    F sources: Q (allow its whole QKV backing), O, assembled K/V, outer LSE.
    D additions: physical dO backing, per-sequence contiguous LSE and delta
    rounded to 128 query rows, exactly as prepare_backward allocates them.
    No GQA expansion of retained K/V; H scratch does use query-head partials.
    v1/v2 source allowances deliberately overlap F/W accounting. v3 uses a
    shared capture owner and only consumes the created/scratch values here.
    """
    if not any(i.has_history for i in meta.items):
        return 0, 0, 0
    hq = args.num_attention_heads
    hk = args.num_query_groups if getattr(args, 'group_query_attention', False) else hq
    dim = args.kv_channels or args.hidden_size // hq
    if dim != 128 or not 0 < hk <= hq or hq % hk:
        raise ValueError('unsupported counted split attention shape')
    slots = args.num_layers + (getattr(args, 'dspp_v_layer_layout', 'balanced') != 'balanced')
    if slots % (2 * pp):
        raise ValueError('invalid counted split layer layout')
    width = slots // (2 * pp)
    logical = stage if chunk == 0 else 2 * pp - stage - 1
    layers = max(0, min(args.num_layers, (logical + 1) * width) - logical * width)
    q, k, c = meta.valid_token_count, sum(meta.key_lengths), meta.chunk_size
    source = (c * (2 * hq + 2 * hk) * dim * 2 + 2 * k * hk * dim * 2 + q * hq * 4)
    delta_rows = sum((i.token_length + 127) // 128 * 128 for i in meta.items)
    created = c * hq * dim * 2 + (q + delta_rows) * hq * 4
    # One layer at a time: query-head partials, compact dKV, concatenation,
    # routing and pending-add copies. Add to the existing workspace allowance.
    scratch = 2 * k * hq * dim * 2 + 8 * k * hk * dim * 2
    return source * layers, created * layers, scratch


class CountedMemory:
    """Build accounting records once, then reuse ready/last-user counters."""

    def __init__(self, profile, args, reference, slot_meta, dtype, chunk_size):
        self.profile = profile
        pp = len(reference.schedules)
        self.v3 = profile.get('schema') == 'dspp-counted-memory-v3'
        if profile.get('schema') not in ('dspp-counted-memory-v1', 'dspp-counted-memory-v2', 'dspp-counted-memory-v3'):
            raise ValueError('unsupported counted memory profile')
        if profile.get('layout') != memory_layout(args, pp, dtype, chunk_size):
            raise ValueError('counted memory layout mismatch')
        self.after_dx = bool(getattr(args, 'dspp_history_dkv_after_dx', False))
        split = bool(getattr(args, 'dspp_split_attention_backward', False))
        if (getattr(args, 'dspp_auto_chunk_size', False) or (split and not self.after_dx)
                or (self.after_dx and not split)):
            raise ValueError('counted memory requires fixed capacity and fused B or after-dx')
        if split and (profile.get('schema') not in ('dspp-counted-memory-v2', 'dspp-counted-memory-v3')
                      or profile.get('split_state') != 'fa2-after-dx-v1'
                      or str(dtype) != 'torch.bfloat16'):
            raise ValueError('missing versioned BF16 split-state allowance')
        self.args, self.reference, self.slot_meta = args, reference, slot_meta
        self.stages = profile['stages']
        if len(self.stages) != pp or not profile.get('source') or not profile.get('device_name'):
            raise ValueError('incomplete counted memory calibration')
        growth = profile['dynamic_growth']
        if (not isinstance(growth, (int, float)) or not math.isfinite(growth)
                or not 0 <= growth <= .05):
            raise ValueError('counted dynamic growth must be in [0, .05]')
        scalar = ('resident_bytes', 'workspace_bytes', 'physical_bytes', 'reserve_bytes')
        vectors = ('activation_bytes', 'weight_bytes', 'kv_bytes_per_token',
                   'attention_bytes_per_key_token')
        if self.v3:
            if profile.get('accounting') != ACCOUNTING or str(dtype) != 'torch.bfloat16':
                raise ValueError('v3 requires versioned BF16 storage calibration')
            vectors += ('weight_forward_bytes', 'weight_backward_bytes', 'capture_bytes',
                        'pending_bytes_per_token')
        for stage in self.stages:
            values = [stage[name] for name in scalar]
            for name in vectors:
                if len(stage[name]) != 2:
                    raise ValueError('counted memory needs two virtual chunks')
                values.extend(stage[name])
            if any(type(v) is not int or v < 0 for v in values):
                raise ValueError('counted memory sizes must be nonnegative integer bytes')
            if stage['physical_bytes'] <= stage['reserve_bytes'] + stage['resident_bytes']:
                raise ValueError('counted physical memory has no dynamic capacity')
            if any(v <= 0 for name in ('activation_bytes', 'weight_bytes') for v in stage[name]):
                raise ValueError('missing activation/W allowance')
        if self.v3:
            for r, stage in enumerate(self.stages):
                for c in range(2):
                    capture, kv = stage_attention_sizes(args, pp, r, c, chunk_size)
                    if (stage['capture_bytes'][c] < capture
                            or stage['attention_bytes_per_key_token'][c] < kv
                            or stage['pending_bytes_per_token'][c] < kv
                            or stage['weight_forward_bytes'][c] + stage['weight_backward_bytes'][c]
                            < stage['weight_bytes'][c]):
                        raise ValueError('v3 storage allowance below required floor')
        limits = profile['limits']
        for name in ('max_tasks', 'max_logical_batches', 'max_key_tokens', 'max_items'):
            if type(limits[name]) is not int or limits[name] <= 0:
                raise ValueError('invalid counted memory coverage limits')
        logical = {l for _, _, l, _ in reference.task_map}
        if (len(slot_meta) > limits['max_tasks'] or len(logical) > limits['max_logical_batches']
                or any(m.chunk_size != chunk_size or sum(m.key_lengths) > limits['max_key_tokens']
                       or len(m.items) > limits['max_items'] for m in slot_meta.values())):
            raise ValueError('input outside counted memory calibration coverage')
        self.version = digest(profile)
        self.mapping = {(g, s): l for g, s, l, _ in reference.task_map}
        accounting, self.owners, self.category_sizes = self.records(reference)
        same_baseline = reference.schema_version == 2 or not self.after_dx
        if same_baseline:
            baseline_owners = self.owners
        else:
            _, baseline_owners, _ = self.records(reference, expand_after_dx=False)
        self.fixed = []
        for r, seq in enumerate(reference.schedules):
            neighbors = int(r > 0) + int(r < pp - 1)
            pools = dspp_communication_buffer_count(neighbor_count=neighbors,
                recv_lookahead=getattr(args, 'dspp_recv_lookahead', 2),
                send_inflight_depth=getattr(args, 'dspp_send_inflight_depth', 2))
            # ready_sends is not bounded by inflight depth. Reserve all
            # possible published F/B payloads, including padding/local V turns.
            queued = sum(n.kind in 'FB' for n in seq)
            scratch = max((history_state_bytes(args, pp, r, n.chunk,
                slot_meta[n.microbatch, n.split])[2] for n in seq
                if self.after_dx and not self.v3 and n.kind == 'F' and (n.microbatch, n.split) in slot_meta), default=0)
            self.fixed.append(self.stages[r]['workspace_bytes'] + scratch + (pools + queued) * reference.wire_bytes)
        reference_peaks = order_owner_peaks(reference, baseline_owners)
        self.baseline_dynamic = tuple(p + f for p, f in zip(reference_peaks, self.fixed))
        self.capacities = tuple(min(
            s['physical_bytes'] - s['reserve_bytes'] - s['resident_bytes'] - f,
            math.floor(base * (1 + growth)) - f)
            for s, f, base in zip(self.stages, self.fixed, self.baseline_dynamic))
        self._check_cache = {}
        self._compiled_storage = None
        self._compile_eligible = (
            self.v3 and (reference.schema_version == 2 or not self.after_dx)
            and all(n.kind in ('F', 'B', 'W') for seq in reference.schedules for n in seq))
        # These exact records/peaks were just built to establish capacity.
        # Reuse them for fallback admission instead of building them again.
        peaks = reference_peaks if same_baseline else order_owner_peaks(accounting, self.owners)
        self._store_check(reference, peaks, len(self.owners), self.category_sizes)

    def records(self, plan, *, expand_after_dx=True):
        """Reuse PlanOwner counters for legacy adjacent H and explicit D/H."""
        explicit = any(n.kind in ('D', 'H') for seq in plan.schedules for n in seq)
        if len(plan.schedules) != len(self.reference.schedules) or any(
                n.stage != r for r, seq in enumerate(plan.schedules) for n in seq):
            raise ValueError('invalid memory stage placement')
        if explicit and not self.after_dx:
            raise ValueError('D/H accounting requires split-state calibration')
        seqs = []
        for seq in plan.schedules:
            out = []
            for n in seq:
                meta = self.slot_meta.get((n.microbatch, n.split))
                if (self.after_dx and expand_after_dx and not explicit and plan.schema_version == 1 and n.kind == 'B'
                        and meta and any(i.has_history for i in meta.items)):
                    out.extend((replace(n, kind='D'), replace(n, kind='H')))
                else:
                    out.append(n)
            seqs.append(tuple(out))
        accounting = replace(plan, schedules=tuple(seqs))
        keys = [node_key(n) for seq in seqs for n in seq]
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate memory task')
        actual = set(keys)
        canonical = {(k[0], 'B' if k[1] == 'D' else k[1], *k[2:])
                     for k in keys if k[1] != 'H'}
        expected = {node_key(n) for seq in self.reference.schedules for n in seq}
        if canonical != expected or len(canonical) != sum(k[1] != 'H' for k in keys):
            raise ValueError('incomplete memory task set')
        pos = {node_key(n): i for seq in seqs for i, n in enumerate(seq)}
        if self.v3:
            return storage_records(self, accounting, pos, actual)
        owners, grouped = [], {}
        sizes = dict(activation=0, weight=0, shared_kv=0, history_source=0, history_created=0)
        expected_h, chains = set(), {}
        for seq in seqs:
            for n in seq:
                if n.kind != 'F' or (n.microbatch, n.split) not in self.slot_meta:
                    continue
                meta = self.slot_meta[n.microbatch, n.split]
                row, c, f = self.stages[n.stage], n.chunk, node_key(n)
                b, w = (n.stage, 'B', c, n.microbatch, n.split), (n.stage, 'W', c, n.microbatch, n.split)
                d, h = (n.stage, 'D', c, n.microbatch, n.split), (n.stage, 'H', c, n.microbatch, n.split)
                backward = d if d in actual else b
                if not pos[f] < pos[backward] < pos[w]:
                    raise ValueError('invalid F/backward/W memory order')
                activation = row['activation_bytes'][c] + row['attention_bytes_per_key_token'][c] * sum(meta.key_lengths)
                # W inputs already exist in F; reserve them there, not at B.
                owners.extend((PlanOwner(n.stage, activation, (f,), (backward,)),
                               PlanOwner(n.stage, row['weight_bytes'][c], (f,), (w,))))
                sizes['activation'] += activation
                sizes['weight'] += row['weight_bytes'][c]
                source, created = 0, 0
                if d in actual:
                    source, created, _ = history_state_bytes(self.args, len(seqs), n.stage, c, meta)
                    if source:
                        expected_h.add(h)
                        if h not in actual or pos[h] <= pos[d]:
                            raise ValueError('missing H or H before D')
                        owners.extend((PlanOwner(n.stage, source, (f,), (h,)),
                                       PlanOwner(n.stage, created, (d,), (h,))))
                        sizes['history_source'] += source
                        sizes['history_created'] += created
                key = (n.stage, c, self.mapping[n.microbatch, n.split])
                group = grouped.setdefault(key, dict(born=[], users=[], tokens=0))
                group['born'].append(f)
                group['users'].append(w)
                if source:
                    group['users'].append(h)
                group['tokens'] += meta.valid_token_count
                for item in meta.items:
                    chain = chains.setdefault((*key, item.sequence_id), {})
                    if item.segment_id in chain:
                        raise ValueError('duplicate memory segment')
                    chain[item.segment_id] = (pos[backward], pos[h] if source else -1)
        if {k for k in actual if k[1] == 'H'} != expected_h:
            raise ValueError('orphan or history-free H')
        for chain in chains.values():
            latest_h = -1
            for segment in sorted(chain, reverse=True):
                backward, history = chain[segment]
                if latest_h >= backward:
                    raise ValueError('history consumer precedes required H')
                latest_h = max(latest_h, history)
        for (r, c, _), group in grouped.items():
            # One allowance for the shared cache and pending gradients of a
            # whole logical batch. Reserving all its tokens at first F and
            # holding through last W/H overcounts but avoids view/last-B guesses.
            size = group['tokens'] * self.stages[r]['kv_bytes_per_token'][c]
            owners.append(PlanOwner(r, size, tuple(group['born']), tuple(group['users'])))
            sizes['shared_kv'] += size
        return accounting, tuple(owners), sizes

    @classmethod
    def load(cls, path, *args):
        return cls(json.loads(Path(path).read_text()), *args)

    def check(self, plan):
        # Coefficients/metadata belong to this frozen instance. Multiple
        # callers audit the identical template/candidate at construction and
        # commit; reuse that count, without caching the live GPU capacity check.
        key = (plan.schema_version, plan.schedules, plan.task_map, plan.split_counts,
               plan.history_dependencies, tuple(self.capacities))
        if key in self._check_cache:
            return deepcopy(self._check_cache[key])
        # The common v3 path has an unsplit reference. Legacy implicit D/H
        # expansion retains its original audit. No new budget or allowance.
        if self._compile_eligible:
            if self._compiled_storage is None:
                self._compiled_storage = CompiledStorage(self)
            peaks, count, sizes = self._compiled_storage.evaluate(plan)
            return self._store_check(plan, peaks, count, sizes)
        accounting, owners, sizes = self.records(plan)
        peaks = order_owner_peaks(accounting, owners)
        return self._store_check(plan, peaks, len(owners), sizes)

    def _store_check(self, plan, peaks, owner_count, sizes):
        key = (plan.schema_version, plan.schedules, plan.task_map, plan.split_counts,
               plan.history_dependencies, tuple(self.capacities))
        if any(p > cap for p, cap in zip(peaks, self.capacities)):
            raise ValueError('counted memory capacity exceeded')
        result = dict(schema='dspp-counted-memory-report-v1', version=self.version,
            scope='frozen coefficient allowances; not a general allocator guarantee',
            baseline_dynamic_bytes=list(self.baseline_dynamic),
            owner_peak_bytes=list(peaks), fixed_bytes=self.fixed,
            dynamic_peak_bytes=[p + f for p, f in zip(peaks, self.fixed)],
            total_peak_bytes=[s['resident_bytes'] + p + f for s, p, f in zip(self.stages, peaks, self.fixed)],
            capacity_bytes=list(self.capacities), owner_records=owner_count,
            category_allowance_bytes=sizes)
        if len(self._check_cache) >= 8:
            self._check_cache.pop(next(iter(self._check_cache)))
        self._check_cache[key] = deepcopy(result)
        return result
