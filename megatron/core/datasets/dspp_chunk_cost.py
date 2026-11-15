"""Compact cross-capacity predictions; startup load, no runtime fitting/I/O."""
import json
from pathlib import Path

import numpy as np

from .dspp_chunk_search import aggregate_features


def shape_features(capacity, shape):
    return (1., capacity, sum(q for q,h in shape), sum(q*(q+1)/2 for q,h in shape),
            sum(q*h for q,h in shape), sum(h for q,h in shape), len(shape))


class ChunkCostModel:
    def __init__(self, payload):
        if payload['schema'] != 'dspp-cross-capacity-v1':
            raise ValueError('unsupported chunk cost model')
        self.payload = payload
        self.domain = tuple(payload['domain'])
        self.pp = payload['pp']
        self.coefficients = {}
        self.exact = {}
        self.work = np.zeros((self.pp, 7))
        self.weight = np.zeros((self.pp, 7))
        for group in payload['groups']:
            key = group['stage'], group['chunk'], group['op']
            beta = tuple(group['coefficients'])
            self.coefficients[key] = beta
            for row in group.get('exact', ()):
                c, shape, value = row
                self.exact[key+(c, tuple(map(tuple, shape)))] = value
            if key[2] in ('F', 'B_fused'):
                self.work[key[0]] += beta
            elif key[2] == 'W':
                self.weight[key[0]] += beta
        for s in range(self.pp):
            for v in range(2):
                for op in ('F', 'B_fused', 'W', 'D', 'H'):
                    if (s, v, op) not in self.coefficients:
                        raise ValueError(f'missing stage/op model: {(s,v,op)}')

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def predict(self, stage, chunk, op, capacity, shape):
        if op == 'W':
            shape = ((capacity, 0),)
        key = stage, chunk, op
        exact = self.exact.get(key+(capacity, shape))
        if exact is not None:
            return exact
        x = shape_features(capacity, shape)
        return max(1e-9, sum(a*b for a,b in zip(self.coefficients[key], x)))

    def coarse(self, histogram, capacities, mode, *, block_counts=None):
        scores = []
        for start in range(0, len(capacities), 256):
            stop = start + 256
            counts = None if block_counts is None else block_counts[start:stop]
            x, w = aggregate_features(histogram, capacities[start:stop], counts)
            stages = (np.einsum('ij,sj->is',x,self.work,optimize=False)
                      + np.einsum('ij,sj->is',w,self.weight,optimize=False))
            # Configurable proxy, not a makespan oracle or lower bound.
            value = (stages.sum(axis=1) if mode == 'work' else
                     stages.max(axis=1)*(1+2*(self.pp-1)/np.maximum(x[:, 0], 1)))
            scores.extend(map(float, value))
        return scores
