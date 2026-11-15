"""Bounded CPU chunk search. Cost models are loaded/fitted outside this module."""
from collections import Counter
from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class SearchOptions:
    points: int = 64
    step: int = 64
    spacing: str = 'log'
    score: str = 'pipeline'

    def __post_init__(self):
        if self.points < 2 or self.step < 1:
            raise ValueError('chunk search needs points>=2 and step>=1')
        if self.spacing not in ('linear', 'log') or self.score not in ('work', 'pipeline'):
            raise ValueError('unknown chunk search spacing/score')


def sample_capacities(lower, upper, lengths, options):
    """Find slice-count changes on the capacity grid, within the point budget."""
    if lower > upper:
        return ()
    step = options.step
    first = (lower + step - 1) // step
    last = upper // step
    if first > last:
        return (lower,) if lower == upper else (lower, upper)

    prefixes = tuple(length - 1 for length in lengths if length > first * step)
    candidates = {first, last}
    index = first
    while index < last and len(candidates) <= options.points:
        next_index = last
        for prefix in prefixes:
            # ceil(length / capacity) - 1 is constant up to this boundary.
            cuts = prefix // (index * step)
            if cuts:
                next_index = min(next_index, prefix // (cuts * step) + 1)
        index = next_index
        candidates.add(index)

    if len(candidates) > options.points:
        # When there are many changes, sample across the domain and move each
        # interior point to the first grid value with the same slice counts.
        spacing = np.geomspace if options.spacing == 'log' else np.linspace
        candidates = {first, last}
        for anchor in np.rint(spacing(first, last, options.points)[1:-1]):
            capacity = int(anchor) * step
            start = first
            for prefix in prefixes:
                slices = prefix // capacity + 1
                start = max(start, prefix // (slices * step) + 1)
            candidates.add(start)

    return tuple(index * step for index in sorted(candidates))


def aggregate_features(histogram, capacities, block_counts=None):
    """Seven-feature totals, using packed block counts when available.

    Features: tasks, sum(capacity), query tokens, causal query pairs,
    query*history pairs, sum(history lengths), item count. No packing/graph.
    """
    lengths = np.asarray([p[0] for p in histogram], dtype=np.float64)[None, :]
    weight = np.asarray([p[1] for p in histogram], dtype=np.float64)[None, :]
    c = np.asarray(capacities, dtype=np.float64)[:, None]
    n = np.floor(lengths/c)
    tail = lengths-n*c
    full = (n*weight).sum(axis=1)
    residual = ((tail>0)*weight).sum(axis=1)
    cs = c[:, 0]
    if block_counts is None:
        continuation = (((tail>0)&(n>0))*weight).sum(axis=1)
        remainder = (tail*weight).sum(axis=1)
        bins = np.maximum(continuation, np.ceil(remainder/cs))
        for k in range(1, 5):
            bins = np.maximum(bins, np.ceil(((tail*(k+1)>c)*weight).sum(axis=1)/k))
        tasks = full+bins
    else:
        tasks = np.asarray(block_counts, dtype=np.float64)
    query = np.full(len(cs), float((lengths*weight).sum()))
    causal = ((n*c*(c+1)/2+tail*(tail+1)/2)*weight).sum(axis=1)
    history = ((c*c*n*(n-1)/2+tail*n*c)*weight).sum(axis=1)
    history_lengths = ((c*n*(n-1)/2+(tail>0)*n*c)*weight).sum(axis=1)
    x = np.column_stack((tasks, tasks*cs, query, causal, history, history_lengths, full+residual))
    w = np.column_stack((tasks, tasks*cs, tasks*cs, tasks*cs*(cs+1)/2,
                         np.zeros(len(cs)), np.zeros(len(cs)), tasks))
    return x, w


class ChunkSearch:
    def __init__(self, model, options=SearchOptions()):
        self.model, self.options = model, options

    def select(self, lengths, pack, fine, *, fixed=None):
        """Return the already prepared winner; never invoke B for candidates."""
        started = time.perf_counter()
        if fixed is not None:
            prepared = fine(fixed, None)
            return prepared, dict(shortlist=[fixed], points=0, coarse_ms=0.,
                fine_ms=(time.perf_counter()-started)*1000, selected=fixed)
        histogram = Counter(lengths)
        capacities = sample_capacities(*self.model.domain, histogram, self.options)
        if not capacities:
            raise ValueError('empty chunk search domain')
        packed = [pack(capacity) for capacity in capacities]
        block_counts = [sum(len(plan.microbatches) for plan in plans) for plans in packed]
        scores = self.model.coarse(
            tuple(histogram.items()), capacities, self.options.score, block_counts=block_counts,
        )
        order = sorted(range(len(capacities)), key=lambda i: (scores[i], capacities[i]))
        finalists = [(capacities[i], packed[i]) for i in order[:2]]
        del packed
        shortlist = [capacity for capacity, _ in finalists]
        coarse_ms = (time.perf_counter()-started)*1000
        started = time.perf_counter()
        accepted, rejected = [], {}
        for capacity, plans in finalists:
            try:
                accepted.append(fine(capacity, plans))
            except ValueError as exc:
                rejected[capacity] = str(exc)
        if not accepted:
            raise ValueError('no feasible base in shortlist: '+str(rejected))
        best = min(accepted, key=lambda p: (p.predicted_ms, p.capacity))
        return best, dict(selected=best.capacity, shortlist=shortlist, points=len(capacities),
            coarse_ms=coarse_ms, fine_ms=(time.perf_counter()-started)*1000, rejected=rejected,
            base_ms={p.capacity:p.predicted_ms for p in accepted})
