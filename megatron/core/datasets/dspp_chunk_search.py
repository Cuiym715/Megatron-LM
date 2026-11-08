"""Bounded CPU chunk search. Cost models are loaded/fitted outside this module."""
from collections import Counter
from dataclasses import dataclass
import math
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


def sample_capacities(lower, upper, current, options):
    """Sample grid ordinals directly; never enumerate the integer domain."""
    first = (lower + options.step - 1) // options.step * options.step
    count = max(0, (upper - first) // options.step + 1)
    reserve = int(lower <= current <= upper)
    n = min(count, options.points - reserve)
    if n:
        if options.spacing == 'log':
            values = np.geomspace(first, first + (count-1)*options.step, n)
            indices = np.rint((values-first)/options.step).astype(np.int64)
        else:
            indices = np.rint(np.linspace(0, count-1, n)).astype(np.int64)
        result = set(map(int, first + indices*options.step))
    else:
        result = set()
    if reserve:
        result.add(current)
    return tuple(sorted(result))


def aggregate_features(histogram, capacities):
    """Seven-feature totals, using exact slice moments and estimated tail bins.

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
    continuation = (((tail>0)&(n>0))*weight).sum(axis=1)
    remainder = (tail*weight).sum(axis=1)
    cs = c[:, 0]
    bins = np.maximum(continuation, np.ceil(remainder/cs))
    for k in range(1, 5):
        bins = np.maximum(bins, np.ceil(((tail*(k+1)>c)*weight).sum(axis=1)/k))
    tasks = full+bins
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

    def select(self, lengths, current, fine, *, fixed=None):
        """Return the already prepared winner; never invoke B for candidates."""
        started = time.perf_counter()
        if fixed is not None:
            prepared = fine(fixed)
            return prepared, dict(shortlist=[fixed], points=0, coarse_ms=0.,
                fine_ms=(time.perf_counter()-started)*1000, selected=fixed)
        histogram = tuple(Counter(lengths).items())
        capacities = sample_capacities(*self.model.domain, current, self.options)
        if not capacities:
            raise ValueError('empty chunk search domain')
        scores = self.model.coarse(histogram, capacities, self.options.score)
        order = sorted(range(len(capacities)), key=lambda i: (scores[i], capacities[i]))
        shortlist = [capacities[i] for i in order[:2]]
        if current in capacities and current not in shortlist:
            shortlist.append(current)
        coarse_ms = (time.perf_counter()-started)*1000
        started = time.perf_counter()
        accepted, rejected = [], {}
        for capacity in shortlist:
            try:
                accepted.append(fine(capacity))
            except ValueError as exc:
                rejected[capacity] = str(exc)
        if not accepted:
            raise ValueError('no feasible base in shortlist: '+str(rejected))
        best = min(accepted, key=lambda p: (p.predicted_ms, p.capacity != current, p.capacity))
        return best, dict(selected=best.capacity, shortlist=shortlist, points=len(capacities),
            coarse_ms=coarse_ms, fine_ms=(time.perf_counter()-started)*1000, rejected=rejected,
            base_ms={p.capacity:p.predicted_ms for p in accepted})
