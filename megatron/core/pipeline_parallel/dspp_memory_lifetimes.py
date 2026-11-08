"""Task-local storage lifetimes for separately calibrated v3 memory profiles.

No tensor inspection at planning time. Capture and deferred GEMM coefficients
are disjoint storage classes established by the offline ownership audit.
"""
from collections import defaultdict

from .dspp_plan import PlanOwner, node_key


ACCOUNTING = 'task-storage-lifetimes-v1'

_CATEGORIES = ('activation', 'weight_forward', 'weight_backward', 'attention_capture',
               'attention_kv', 'history_created', 'history_scratch',
               'pending_accumulation', 'pending_backing')


class CompiledStorage:
    """Input-local byte tables; candidate audits allocate no PlanOwner objects.

    Each stage has dense F/B/D/H/W IDs. Sizes and history producer sets are
    compiled once, independent of candidate order. Only a changed stage's
    event array is evaluated; cached stages retain their exact peak/counts.
    The object belongs to one CountedMemory instance, never another input.
    """
    def __init__(self, memory):
        from .dspp_memory_counts import history_state_bytes

        self.stages, self.caches = [], []
        self.after_dx = memory.after_dx
        for stage, seq in enumerate(memory.reference.schedules):
            keys, rows, segments, metadata = {}, [], {}, []
            for n in seq:
                if n.kind != 'F':
                    continue
                index = len(rows)
                for op, kind in enumerate(('F', 'B', 'D', 'H', 'W')):
                    keys[kind, n.chunk, n.microbatch, n.split] = 5 * index + op
                meta = memory.slot_meta.get((n.microbatch, n.split))
                metadata.append(meta)
                if meta is None:
                    rows.append(None)
                    continue
                row, c = memory.stages[stage], n.chunk
                _, created, scratch = history_state_bytes(
                    memory.args, len(memory.reference.schedules), stage, c, meta)
                unit = row['pending_bytes_per_token'][c]
                counts = _token_counts(memory,meta)
                key_tokens = counts[0]
                rows.append((row['activation_bytes'][c], row['weight_forward_bytes'][c],
                             row['weight_backward_bytes'][c], row['capture_bytes'][c],
                             row['attention_bytes_per_key_token'][c] * key_tokens,
                             created, scratch, key_tokens * unit,
                             counts[1] * unit,
                             any(i.has_history for i in meta.items)))
                identity = (c, memory.mapping[n.microbatch, n.split])
                for item_index,item in enumerate(meta.items):
                    key = (*identity, item.sequence_id, item.segment_id)
                    if key in segments:
                        raise ValueError('duplicate memory segment')
                    segments[key] = (index, counts[2][item_index] * unit)
            writers = defaultdict(set)
            for n in seq:
                if n.kind != 'F':
                    continue
                index = keys['F', n.chunk, n.microbatch, n.split] // 5
                meta = metadata[index]
                if meta is None:
                    continue
                identity = (n.chunk, memory.mapping[n.microbatch, n.split])
                for item in meta.items:
                    for segment in range(item.segment_id):
                        key = (*identity, item.sequence_id, segment)
                        if key not in segments:
                            raise ValueError('missing history memory segment')
                        writers[key].add(index)
            targets = tuple((*segments[key], tuple(sorted(producers)))
                            for key, producers in writers.items())
            self.stages.append((keys, tuple(rows), targets))
            self.caches.append({})

    def evaluate(self, plan):
        if len(plan.schedules) != len(self.stages):
            raise ValueError('invalid memory stage placement')
        peaks, totals, count = [], [0] * len(_CATEGORIES), 0
        for stage, seq in enumerate(plan.schedules):
            keys, rows, targets = self.stages[stage]
            try:
                order = tuple(keys[n.kind, n.chunk, n.microbatch, n.split] for n in seq)
            except KeyError as exc:
                raise ValueError('incomplete memory task set') from exc
            if any(n.stage != stage for n in seq):
                raise ValueError('invalid memory stage placement')
            if not self.after_dx and any(n.kind in ('D', 'H') for n in seq):
                raise ValueError('D/H accounting requires split-state calibration')
            cache = self.caches[stage]
            result = cache.get(order)
            if result is None:
                result = self._stage(order, rows, targets)
                if len(cache) >= 8:
                    cache.pop(next(iter(cache)))
                cache[order] = result
            peak, owners, sizes = result
            peaks.append(peak)
            count += owners
            for i, size in enumerate(sizes):
                totals[i] += size
        return tuple(peaks), count, {k: v for k, v in zip(_CATEGORIES, totals) if v}

    @staticmethod
    def _stage(order, rows, targets):
        positions = [-1] * (5 * len(rows))
        for position, index in enumerate(order):
            if positions[index] >= 0:
                raise ValueError('duplicate memory task')
            positions[index] = position
        events, sizes, count = [0] * (len(order) + 1), [0] * len(_CATEGORIES), 0
        backward, writer, selected = [], [], []

        def add(size, begin, finish, category):
            nonlocal count
            if size:
                events[begin] += size
                events[finish + 1] -= size
                sizes[category] += size
                count += 1

        for i, row in enumerate(rows):
            f, b, d, h, w = positions[5*i:5*i+5]
            if f < 0 or w < 0 or (b >= 0) == (d >= 0):
                raise ValueError('incomplete memory task set')
            split = d >= 0
            back = d if split else b
            backward.append(back)
            writer.append(h if split else b)
            selected.append(split)
            if row is None:
                if h >= 0:
                    raise ValueError('orphan or history-free H')
                continue
            if not f < back < w:
                raise ValueError('invalid F/backward/W memory order')
            if split and (h <= d or not row[9]):
                raise ValueError('missing H, history-free H or H before D')
            if not split and h >= 0:
                raise ValueError('orphan or history-free H')
            add(row[0], f, back, 0)
            add(row[1], f, w, 1)
            add(row[2], back, w, 2)
            add(row[3], f, max(w, h), 3)
            add(row[4], f, max(back, h), 4)
            if split:
                add(row[5], d, h, 5)
                add(row[6], d, d, 6)
                add(row[6], h, h, 6)

        # Select first/second writers by position without sorting producer sets.
        # A first writer's backing remains live through its last target consumer.
        last_user = [-1] * len(rows)
        for target, size, producers in targets:
            end = backward[target]
            first = second = len(order)
            first_id = -1
            for producer in producers:
                start = writer[producer]
                if start >= end:
                    raise ValueError('history consumer precedes required H/backward')
                if start < first:
                    second, first, first_id = first, start, producer
                elif start < second:
                    second = start
            last_user[first_id] = max(last_user[first_id], end)
            if second < len(order):
                add(size, second, end, 7)
        for i, end in enumerate(last_user):
            if end >= 0:
                add(rows[i][8 if selected[i] else 7], writer[i], end, 8)
        live = peak = 0
        for delta in events:
            live += delta
            peak = max(peak, live)
        return peak, count, tuple(sizes)



def attention_token_counts(args, meta):
    """Per-rank maxima for Q/O owners; native KV keeps global-history counts.

    Non-final segments occupy complete physical chunks, so their local KV
    histories are evenly sharded. Current packed tails/shorts use actual owners.
    Called once per input metadata in CountedMemory, not per search candidate.
    """
    cp = getattr(args,'context_parallel_size',1)
    if cp < 2 or getattr(args,'kaimm_context_parallel_impl','key-value') != 'query-out':
        return sum(meta.key_lengths),sum(i.token_offset for i in meta.items),tuple(i.token_length for i in meta.items)
    from megatron.core.datasets.dspp_cp_layout import hybrid_cp_ownership, cp_token_ranges
    rows = (hybrid_cp_ownership(meta,cp).rank_indices
            if getattr(args,'dspp_cp_layout','paired') == 'hybrid' else
            tuple(tuple(i for a,b in cp_token_ranges(meta.chunk_size,cp,r) for i in range(a,b)) for r in range(cp)))
    offsets = meta.cu_seqlens_q
    from bisect import bisect_left
    rows = tuple(tuple(sorted(row)) for row in rows)
    item_max = tuple(max(bisect_left(row,offsets[i+1])-bisect_left(row,offsets[i]) for row in rows)
                     for i in range(len(meta.items)))
    current = max(bisect_left(row,meta.valid_token_count) for row in rows)
    history = sum((i.token_offset+cp-1)//cp for i in meta.items)
    return current+history,history,item_max


def _token_counts(memory,meta):
    cache = getattr(memory,'local_attention_counts',None)
    return cache[id(meta)] if cache is not None else attention_token_counts(memory.args,meta)

def stage_attention_sizes(args, pp, stage, chunk, capacity):
    from megatron.core.dspp_layout import physical_layer_range
    hq = args.num_attention_heads
    hk = args.num_query_groups if getattr(args, 'group_query_attention', False) else hq
    dim = args.kv_channels or args.hidden_size // hq
    if dim != 128 or not 0 < hk <= hq or hq % hk:
        raise ValueError('unsupported v3 attention shape')
    begin, end = physical_layer_range(
        args.num_layers, pp, getattr(args, 'dspp_v_layer_layout', 'balanced'),
        stage, chunk)
    layers = end - begin
    # Current Q/K/V, O and outer LSE; also covers a shared unsplit QKV backing
    # plus O. Calibration takes the maximum with observed backing sizes.
    # Assembled K/V is charged separately from the original cached K/V.
    if getattr(args,'kaimm_context_parallel_impl','key-value') == 'query-out':
        capacity //= getattr(args,'context_parallel_size',1)
    capture = capacity * ((2*hq + 2*hk)*dim*2 + hq*4) * layers
    kv_per_token = 2 * hk * dim * 2 * layers  # BF16 dK/dV and assembled K/V
    return capture, kv_per_token


def records(memory, plan, positions, actual):
    from .dspp_memory_counts import history_state_bytes

    owners, sizes = [], defaultdict(int)
    tasks, segments = {}, {}

    def add(stage, size, born, users, category):
        users = tuple(sorted(set(users), key=positions.__getitem__))
        if any(positions[u] < positions[born] for u in users):
            raise ValueError('v3 storage consumer precedes producer')
        if size:
            owners.append(PlanOwner(stage, size, (born,), users))
            sizes[category] += size

    expected_h = set()
    for stage, seq in enumerate(plan.schedules):
        for n in seq:
            if n.kind != 'F' or (n.microbatch, n.split) not in memory.slot_meta:
                continue
            meta = memory.slot_meta[n.microbatch, n.split]
            c, f = n.chunk, node_key(n)
            b = (stage, 'B', c, n.microbatch, n.split)
            d, h, w = [(stage, kind, c, n.microbatch, n.split) for kind in ('D', 'H', 'W')]
            backward = d if d in actual else b
            if not positions[f] < positions[backward] < positions[w]:
                raise ValueError('invalid F/backward/W memory order')
            row = memory.stages[stage]
            selected = d in actual
            if selected:
                if h not in actual or positions[h] <= positions[d] or not any(i.has_history for i in meta.items):
                    raise ValueError('missing H, history-free H or H before D')
                expected_h.add(h)
            capture_users = [backward, w] + ([h] if selected else [])
            add(stage, row['activation_bytes'][c], f, [backward], 'activation')
            add(stage, row['weight_forward_bytes'][c], f, [w], 'weight_forward')
            add(stage, row['weight_backward_bytes'][c], backward, [w], 'weight_backward')
            # A single capture allocation serves F/B, W and delayed H. A
            # selected H extends its lifetime instead of adding another copy.
            add(stage, row['capture_bytes'][c], f, capture_users, 'attention_capture')
            add(stage, row['attention_bytes_per_key_token'][c] * _token_counts(memory,meta)[0],
                f, [backward] + ([h] if selected else []), 'attention_kv')
            if selected:
                _, created, scratch = history_state_bytes(memory.args, len(plan.schedules), stage, c, meta)
                add(stage, created, d, [h], 'history_created')
                # Scratch is task-local, not a reservation for every possible H.
                add(stage, scratch, d, [d], 'history_scratch')
                add(stage, scratch, h, [h], 'history_scratch')
            identity = (stage, c, memory.mapping[n.microbatch, n.split])
            task = dict(meta=meta, backward=backward, writer=h if selected else backward,
                        selected=selected, identity=identity)
            tasks[f] = task
            for item_index,item in enumerate(meta.items):
                key = (*identity, item.sequence_id, item.segment_id)
                if key in segments:
                    raise ValueError('duplicate memory segment')
                segments[key] = (f, item, _token_counts(memory,meta)[2][item_index])
    if {k for k in actual if k[1] == 'H'} != expected_h:
        raise ValueError('orphan or history-free H')

    # Pending's first write stores a view of the complete dKV allocation.
    # Charge that backing once per writer, shared by all first-write targets.
    # Later additions allocate target-sized tensors; their transient overlap
    # is covered by the separately calibrated workspace allowance.
    writers = defaultdict(set)
    for task in tasks.values():
        for item in task['meta'].items:
            for segment in range(item.segment_id):
                key = (*task['identity'], item.sequence_id, segment)
                if key not in segments:
                    raise ValueError('missing history memory segment')
                target_f, _, _ = segments[key]
                consumer = tasks[target_f]['backward']
                if positions[task['writer']] >= positions[consumer]:
                    raise ValueError('history consumer precedes required H/backward')
                writers[key].add(task['writer'])
    first_users = defaultdict(set)
    source_tasks = {t['writer']: t for t in tasks.values()}
    for key, producer_set in writers.items():
        producers = sorted(producer_set, key=positions.__getitem__)
        target_f, item, local_length = segments[key]
        consumer = tasks[target_f]['backward']
        first_users[producers[0]].add(consumer)
        if len(producers) > 1:
            stage, c = key[:2]
            unit = memory.stages[stage]['pending_bytes_per_token'][c]
            add(stage, local_length * unit, producers[1], [consumer], 'pending_accumulation')
    for writer, users in first_users.items():
        task = source_tasks[writer]
        meta = task['meta']
        # H returns compact history; fused B returns current + history.
        counts = _token_counts(memory,meta)
        length = counts[1] if task['selected'] else counts[0]
        unit = memory.stages[writer[0]]['pending_bytes_per_token'][writer[2]]
        add(writer[0], length * unit, writer, users, 'pending_backing')
    return plan, tuple(owners), dict(sizes)
