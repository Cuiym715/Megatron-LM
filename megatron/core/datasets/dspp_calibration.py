"""Versioned, stage-local calibration shared by A/B/C.

Legacy execution profiles remain readable by their original API. This format
deliberately requires a complete context and never borrows another stage's data.
Statistical errors are diagnostics, not memory bounds or acceptance policies.
"""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from statistics import median

from .dspp_execution_cost import DsppExecutionCostModel, meta_from_timeline_record
from .dspp_batch_plan import build_dspp_batch_plan


SCHEMA = 'dspp-stage-calibration-v2'
ESTIMATOR = 'extended-nonnegative-ridge-v1'
OPERATIONS = ('F', 'B_fused', 'D', 'H', 'W')
# Fixed policy, included in both profile and runtime plan identities. These are
# empirical validation thresholds, not certified prediction error bounds.
LEGACY_INTERPOLATION_POLICY = dict(version='single-item-query-local-v1', max_span=512,
    max_gap=256, min_repeats=3, max_relative_error=.05, max_relative_noise=.05)
NOISE_RELAXED_INTERPOLATION_POLICY = dict(LEGACY_INTERPOLATION_POLICY,
    version='single-item-query-local-v2', max_relative_noise=None)
INTERPOLATION_POLICY = dict(NOISE_RELAXED_INTERPOLATION_POLICY,
    version='single-item-query-local-v3', max_relative_error=None,
    acceptance='nominal-makespan-v1')
SUPPORTED_INTERPOLATION_POLICIES = (
    LEGACY_INTERPOLATION_POLICY, NOISE_RELAXED_INTERPOLATION_POLICY, INTERPOLATION_POLICY)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def operation(kind, backward_mode):
    """Split-immediate is neither fused B nor stage D followed by H."""
    if kind.startswith('B'):
        if backward_mode == 'after_dx':
            return 'D'
        if backward_mode in ('fused', 'selective'):
            return 'B_fused'
        raise ValueError('split_immediate needs its own calibration path')
    if kind[:1] not in ('F', 'D', 'H', 'W'):
        raise ValueError(f'unknown calibration operation {kind}')
    return kind[0]


@lru_cache(maxsize=1)
def hardware_identity():
    import torch
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    versions = {}
    for package in ('flash-attn', 'transformer-engine', 'apex'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'absent'
    kernel_root = Path(__file__).resolve().parents[1] / 'kv_cache'
    versions['split_sources'] = fingerprint({
        str(path.relative_to(kernel_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((kernel_root / 'fa_split_src').glob('*')) if path.is_file()
    })
    return dict(gpu=p.name, capability=[p.major, p.minor], total_memory=p.total_memory,
                torch=str(torch.__version__), cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), kernels=versions)


def stage_context(args, stage, chunk, *, hardware=None):
    pp = int(args.pipeline_model_parallel_size)
    balanced = getattr(args, 'dspp_v_layer_layout', 'balanced') == 'balanced'
    slots = int(args.num_layers) + (0 if balanced else 1)
    count = slots // (2 * pp)
    logical = stage if chunk == 0 else 2 * pp - stage - 1
    layers = list(range(logical * count + 1, min(args.num_layers, (logical + 1) * count) + 1))
    # num_query_groups is ignored when GQA is disabled. Bind the switch too,
    # so MHA and GQA/MQA cannot share a complete-stage timing identity.
    names = ('hidden_size', 'ffn_hidden_size', 'num_attention_heads', 'num_query_groups',
             'group_query_attention',
             'kv_channels', 'padded_vocab_size', 'normalization', 'norm_epsilon',
             'swiglu', 'position_embedding_type', 'rotary_percent', 'rotary_base',
             'attention_dropout', 'hidden_dropout', 'gradient_accumulation_fusion',
             'tensor_model_parallel_size', 'context_parallel_size',
             'sequence_parallel', 'untie_embeddings_and_output_weights',
             'recompute_granularity', 'recompute_method', 'recompute_num_layers')
    return dict(hardware=hardware if hardware is not None else hardware_identity(),
                model={name: getattr(args, name, None) for name in names},
                dtype=str(getattr(args, 'params_dtype', None)), pp=pp, vpp=2,
                stage=int(stage), virtual_chunk=int(chunk), layers=layers,
                embedding=(stage == 0 and chunk == 0),
                output_head=(stage == 0 and chunk == 1),
                execution='eager', overlap='compute-p2p', boundary='complete-stage-v1')


@dataclass(frozen=True)
class Prediction:
    duration_ms: object
    source: str
    error_ms: object
    confidence: str
    extrapolated: bool
    out_of_range: tuple = ()


def _query_regions(rows, policy):
    """Validate a local line using an interior shape excluded from its fit.

    Only q varies: capacity, history, packing and final/non-final path stay
    fixed. No history interpolation, multivariate interpolation or extrapolation.
    Build once when loading the frozen profile, never during GPU execution.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if len(row['items']) != 1:
            continue
        item = row['items'][0]
        key = (int(row['sequence_chunk_size']), int(item['token_offset']),
               item['segment_id'] == item['segment_count'] - 1)
        groups[key][int(item['token_length'])].append(float(row['duration_ms']))
    regions = defaultdict(list)
    for key, shapes in groups.items():
        xs = sorted(shapes)
        for lo, mid, hi in zip(xs, xs[1:], xs[2:]):
            if (hi - lo > policy['max_span'] or max(mid-lo, hi-mid) > policy['max_gap']
                    or min(len(shapes[x]) for x in (lo, mid, hi)) < policy['min_repeats']):
                continue
            ys = [median(shapes[x]) for x in (lo, mid, hi)]
            noises = [max(abs(y-center) for y in shapes[x])
                      for x, center in zip((lo, mid, hi), ys)]
            estimated = ys[0] + (ys[2]-ys[0]) * (mid-lo)/(hi-lo)
            residual = abs(estimated-ys[1])
            noise_limit = policy['max_relative_noise']
            error_limit = policy['max_relative_error']
            if ((error_limit is not None and residual/ys[1] > error_limit)
                    or (noise_limit is not None and any(n/y > noise_limit for n,y in zip(noises,ys)))):
                continue
            # Retain the held-out residual, repeated-sample spread and a 1%
            # floor as diagnostics; v1/v2 also use these in acceptance replays.
            error = residual + max(noises) + .01 * max(ys)
            regions[key].append(dict(lo=lo, mid=mid, hi=hi, left_ms=ys[0],
                right_ms=ys[2], error_ms=error, validation_relative=residual/ys[1],
                repeats=[len(shapes[x]) for x in (lo, mid, hi)]))
    return dict(regions)


@lru_cache(maxsize=32)
def _dense_w_meta(capacity):
    return build_dspp_batch_plan([capacity], chunk_size=capacity).microbatches[0]


class StageCalibration:
    def __init__(self, records, *, interpolation_policy=None):
        policy = INTERPOLATION_POLICY if interpolation_policy is None else interpolation_policy
        if policy not in SUPPORTED_INTERPOLATION_POLICIES:
            raise ValueError('stage interpolation policy version mismatch')
        self.interpolation_policy = dict(policy)
        self.records = tuple(records)
        self.groups = defaultdict(list)
        for row in self.records:
            context, op = row['context'], row['operation']
            required = {'hardware', 'model', 'dtype', 'pp', 'vpp', 'stage', 'virtual_chunk',
                        'layers', 'embedding', 'output_head', 'execution', 'overlap', 'boundary'}
            if not required.issubset(context):
                raise ValueError('incomplete stage calibration identity')
            if op not in OPERATIONS or row['stage'] != context['stage']:
                raise ValueError('invalid stage/operation in calibration')
            if int(row['chunk']) != context['virtual_chunk']:
                raise ValueError('virtual stage mismatch in calibration')
            if not math.isfinite(float(row['duration_ms'])) or row['duration_ms'] <= 0:
                raise ValueError('calibration duration must be finite and positive')
            meta_from_timeline_record(row).validate()
            self.groups[(canonical(context), op)].append(dict(row, kind=op))
        self.models = {}
        self.coverage = {}
        self.query_regions = {}
        for key, rows in self.groups.items():
            histories = [i['token_offset'] for r in rows for i in r['items']]
            self.coverage[key] = (
                {int(r['sequence_chunk_size']) for r in rows},
                min(histories), max(histories), {len(r['items']) for r in rows})
            fitting = rows
            if key[1] == 'W':
                # Deferred GEMMs execute dense capacity, including padded rows.
                # q/history/packing must not create fictitious W shape costs.
                fitting = []
                for row in rows:
                    c = row['sequence_chunk_size']
                    dense = dict(row, items=[dict(sequence_id=0, segment_id=0,
                        segment_count=1, token_offset=0, token_length=c, sequence_length=c)])
                    dense.pop('signature', None)
                    fitting.append(dense)
            self.models[key] = DsppExecutionCostModel.from_records(fitting, extended_features=True)
            self.query_regions[key] = _query_regions(rows, self.interpolation_policy) if key[1] != 'W' else {}

    @property
    def version(self):
        return fingerprint(self.to_dict())

    def to_dict(self):
        return dict(schema=SCHEMA, records=list(self.records),
                    estimator=ESTIMATOR,
                    interpolation_policy=self.interpolation_policy.copy(),
                    error_semantics='empirical residuals, not hard upper bounds')

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, value):
        if value.get('schema') != SCHEMA:
            raise ValueError('not a stage calibration v2 profile')
        if value.get('estimator') != ESTIMATOR:
            raise ValueError('stage calibration estimator version mismatch')
        # Frozen profiles keep their admission policy. New profiles constructed
        # from the raw records use v3; archives need no in-place conversion.
        policy = value.get('interpolation_policy', LEGACY_INTERPOLATION_POLICY)
        if policy not in SUPPORTED_INTERPOLATION_POLICIES:
            raise ValueError('stage interpolation policy version mismatch')
        return cls(value['records'], interpolation_policy=policy)

    def predict_detailed(self, context, op, meta):
        key = (canonical(context), op)
        if key not in self.models:
            # None, never H=0. Consumers must keep an unsplit fallback.
            return Prediction(None, 'missing-stage-path', None, 'none', True, ('context/path',))
        shape = (_dense_w_meta(meta.chunk_size)
                 if op == 'W' else meta)
        p = self.models[key].predict_detailed(context['stage'], op, shape)
        p95 = self.models[key].predict_detailed(context['stage'], op, shape, percentile=.95)
        outside = []
        capacities, min_history, max_history, packs = self.coverage[key]
        if meta.chunk_size not in capacities:
            outside.append('capacity-unmeasured')
        if op != 'W' and any(i.token_offset < min_history or i.token_offset > max_history for i in meta.items):
            outside.append('history')
        if op != 'W' and len(meta.items) not in packs:
            outside.append('packing')
        if p.source != 'exact-profile' and not outside and len(meta.items) == 1 and op != 'W':
            item = meta.items[0]
            region_key = (meta.chunk_size, item.token_offset, item.is_last_segment)
            matches = [r for r in self.query_regions[key].get(region_key, ())
                       if r['lo'] < item.token_length < r['hi']]
            if matches:
                region = min(matches, key=lambda r: (r['error_ms'], r['hi']-r['lo'], r['lo']))
                fraction = (item.token_length-region['lo'])/(region['hi']-region['lo'])
                value = region['left_ms'] + fraction*(region['right_ms']-region['left_ms'])
                return Prediction(value, 'validated-query-interpolation', region['error_ms'],
                                  'medium', False)
        # Marginal feature bounds alone still do not admit other predictions.
        if p.source != 'exact-profile':
            outside.append('kernel-region-unverified')
        return Prediction(p.duration_ms, p.source, max(p.error_ms, abs(p95.duration_ms - p.duration_ms)),
                          'low' if outside else p.confidence,
                          bool(outside) or p.extrapolated, tuple(outside))


def timeline_records(payload):
    """Reject old unbound timelines rather than relabel them as exact P2 data."""
    if payload.get('calibration_schema') != SCHEMA:
        raise ValueError('timeline has no versioned stage/kernel identity')
    for row in payload['events']:
        if row.get('calibration_operation') is None:
            continue
        yield dict(row, operation=row['calibration_operation'],
                   context=payload['calibration_contexts'][str(row['chunk'])])


def holdout_report(records):
    """Whole capacity/history/packing folds; never split repeated shapes."""
    folds = []
    def history_band(row):
        h = max(i['token_offset'] for i in row['items'])
        c = row['sequence_chunk_size']
        return 'none' if h == 0 else 'short' if h <= c else 'long'
    dimensions = {
        'capacity': lambda r: r['sequence_chunk_size'],
        'history': history_band,
        'packing': lambda r: 'packed' if len(r['items']) > 1 else 'single',
    }
    for dimension, classify in dimensions.items():
        for value in sorted({classify(r) for r in records}, key=str):
            train = [r for r in records if classify(r) != value]
            test = [r for r in records if classify(r) == value]
            model = StageCalibration(train)
            results = defaultdict(list)
            for row in test:
                p = model.predict_detailed(row['context'], row['operation'], meta_from_timeline_record(row))
                actual = row['duration_ms']
                results[row['operation']].append(dict(
                    absolute_ms=None if p.duration_ms is None else abs(p.duration_ms - actual),
                    relative=None if p.duration_ms is None else abs(p.duration_ms - actual) / actual,
                    covered=not p.extrapolated, source=p.source, out_of_range=p.out_of_range))
            summary = {}
            for op, predictions in results.items():
                absolute = [p['absolute_ms'] for p in predictions if p['absolute_ms'] is not None]
                relative = sorted(p['relative'] for p in predictions if p['relative'] is not None)
                summary[op] = dict(samples=len(predictions), available=len(absolute),
                    covered_fraction=sum(p['covered'] for p in predictions)/len(predictions),
                    median_absolute_ms=median(absolute) if absolute else None,
                    median_relative=median(relative) if relative else None,
                    p95_relative=relative[math.ceil(.95*len(relative))-1] if relative else None)
            folds.append(dict(dimension=dimension, held_out=value, train=len(train),
                              test=len(test), operations=dict(results), summary=summary))
    return dict(folds=folds, ranking_and_split_decisions='requires frozen candidate plans; not inferred from per-op errors')


@lru_cache(maxsize=8)
def load_runtime_calibration(path, version):
    # version is the frozen file digest, so rewriting a path cannot reuse the
    # old lru entry. Legacy files keep their explicitly legacy API semantics.
    contents = Path(path).read_bytes()
    payload = json.loads(contents)
    if fingerprint_bytes(contents) != version:
        raise ValueError('execution profile changed during plan construction')
    if payload.get('schema') == SCHEMA:
        return StageCalibration.from_dict(payload)
    return DsppExecutionCostModel.from_json(path)


def fingerprint_bytes(value):
    return hashlib.sha256(value).hexdigest()
