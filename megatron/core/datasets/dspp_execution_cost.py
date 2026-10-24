"""Profile-backed execution cost model for DSPP planning.

The model is deliberately metadata-only at prediction time.  It consumes
CUDA-event timeline records collected on real hardware, fits one small model
per ``(pipeline stage, F/B/W)`` task kind, and falls back to the analytical
metadata work proxy when a profile does not cover a shape.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from .dspp_batch_plan import DsppMicrobatchMeta


def _op_kind(kind: str) -> str:
    value = str(kind)
    if value in {"B_fused", "D", "H"}:
        return value
    return value[0] if value and value[0] in "FBW" else value


def _features(meta: DsppMicrobatchMeta, extended: bool = False) -> Tuple[float, ...]:
    token_work = 0.0
    attention_work = 0.0
    for item in meta.items:
        q = float(item.token_length)
        k = float(item.token_offset + item.token_length)
        token_work += q
        attention_work += q * k
    base = (
        1.0,
        token_work,
        attention_work,
        float(len(meta.items)),
        float(meta.padding_token_count),
    )
    return base + (float(meta.chunk_size), sum(
        float(i.token_length * i.token_offset) for i in meta.items
    )) if extended else base


def _signature(meta: DsppMicrobatchMeta) -> str:
    items = ";".join(
        f"{item.token_length}:{item.token_offset + item.token_length}"
        for item in meta.items
    )
    return f"c{meta.chunk_size}|{items}"


@dataclass(frozen=True)
class _FittedTaskModel:
    coefficients: Tuple[float, ...]
    residual_median_ms: float
    residual_p95_ms: float
    samples: int
    feature_min: Tuple[float, ...] = ()
    feature_max: Tuple[float, ...] = ()
    duration_median_ms: float = 0.0


@dataclass(frozen=True)
class _ProfilePoint:
    features: Tuple[float, ...]
    median_ms: float
    p95_ms: float
    samples: int


@dataclass(frozen=True)
class _StageOpCalibration:
    """Cheap affine correction from the PP-free adapter to real DSPP tasks."""

    scale: float
    overhead_ms: float
    residual_median_ms: float
    residual_p95_ms: float
    samples: int


@dataclass(frozen=True)
class DsppCostPrediction:
    duration_ms: float
    source: str
    error_ms: float
    confidence: str
    extrapolated: bool


@dataclass(frozen=True)
class DsppExecutionCostModel:
    """Execution-time predictor with exact-shape and regression paths."""

    exact_median_ms: Mapping[Tuple[int, str, str], float]
    exact_p95_ms: Mapping[Tuple[int, str, str], float]
    fitted: Mapping[Tuple[int, str], _FittedTaskModel]
    points: Mapping[Tuple[int, str], Tuple[_ProfilePoint, ...]] = None
    w_capacity_median_ms: Mapping[Tuple[int, int], float] = None
    w_capacity_p95_ms: Mapping[Tuple[int, int], float] = None
    source: str = "profile"
    stage_op_calibrations: Mapping[Tuple[int, str], _StageOpCalibration] = None
    extended_features: bool = False

    def __post_init__(self):
        if self.points is None:
            object.__setattr__(self, "points", {})
        if self.w_capacity_median_ms is None:
            object.__setattr__(self, "w_capacity_median_ms", {})
        if self.w_capacity_p95_ms is None:
            object.__setattr__(self, "w_capacity_p95_ms", {})
        if self.stage_op_calibrations is None:
            object.__setattr__(self, "stage_op_calibrations", {})

    @staticmethod
    def empty() -> "DsppExecutionCostModel":
        return DsppExecutionCostModel({}, {}, {}, {}, {}, {}, source="metadata-fallback")

    @classmethod
    def from_compute_profile(cls, payload: Mapping[str, Any]) -> "DsppExecutionCostModel":
        """Load the PP-free physical-shape calibration format.

        The local calibration represents one balanced logical stage.  Its
        stage-0 model is reused by other stages at prediction time.  F/B rows
        retain query/history shape; deferred W is indexed by dense capacity
        because its GEMM shape is independent of packing metadata.
        """

        converted = []
        w_median: Dict[Tuple[int, int], float] = {}
        w_p95: Dict[Tuple[int, int], float] = {}
        for record in payload.get("records", ()):
            op = _op_kind(str(record.get("operation", "")))
            capacity = int(record["capacity"])
            stage = int(record.get("stage", 0))
            if op == "W":
                w_median[(stage, capacity)] = float(record["median_ms"])
                w_p95[(stage, capacity)] = float(record.get("p95_ms", record["median_ms"]))
                continue
            query = tuple(int(value) for value in record.get("query_lengths", ()))
            history = tuple(int(value) for value in record.get("history_lengths", ()))
            if op not in {"F", "B"} or not query or len(query) != len(history):
                continue
            items = [
                {
                    "sequence_id": index,
                    "segment_id": 0,
                    "token_offset": h,
                    "token_length": q,
                    "sequence_length": h + q,
                    "segment_count": 1,
                }
                for index, (q, h) in enumerate(zip(query, history))
            ]
            converted.append({
                "stage": stage,
                "kind": op,
                "duration_ms": float(record["median_ms"]),
                "p95_ms": float(record.get("p95_ms", record["median_ms"])),
                "sequence_chunk_size": capacity,
                "items": items,
            })
        base = cls.from_records(converted)
        return cls(
            base.exact_median_ms,
            base.exact_p95_ms,
            base.fitted,
            base.points,
            w_median,
            w_p95,
            source="compute-profile",
        )

    @classmethod
    def from_records(
        cls, records: Sequence[Mapping[str, Any]], *, ridge: float = 1e-6,
        extended_features: bool = False,
    ) -> "DsppExecutionCostModel":
        grouped: Dict[Tuple[int, str], list[Tuple[Tuple[float, ...], float, str]]]
        grouped = defaultdict(list)
        exact: Dict[Tuple[int, str, str], list[float]] = defaultdict(list)
        for record in records:
            if "duration_ms" not in record or "items" not in record:
                continue
            stage = int(record["stage"])
            op = _op_kind(str(record.get("calibration_operation") or record.get("kind", "")))
            if op not in {"F", "B", "W", "B_fused", "D", "H"}:
                continue
            items = tuple(record.get("items", ()))
            if not items:
                continue
            meta = meta_from_timeline_record(record, items)
            features = _features(meta, extended_features)
            duration = float(record["duration_ms"])
            signature = str(record.get("signature", _signature(meta)))
            grouped[(stage, op)].append((features, duration, signature))
            exact[(stage, op, signature)].append(duration)

        fitted: Dict[Tuple[int, str], _FittedTaskModel] = {}
        profile_points: Dict[Tuple[int, str], Tuple[_ProfilePoint, ...]] = {}
        for key, samples in grouped.items():
            x = torch.tensor([sample[0] for sample in samples], dtype=torch.float64)
            y = torch.tensor([sample[1] for sample in samples], dtype=torch.float64)
            if len(samples) >= len(samples[0][0]):
                # Center/scale the non-intercept features before solving.  The
                # raw token and q*k columns are badly conditioned for real
                # traces, and an unscaled normal equation can produce absurd
                # non-negative coefficients despite small residuals.
                features = x[:, 1:]
                scale = torch.std(features, dim=0, unbiased=False)
                scale = torch.where(scale > 1.0, scale, torch.ones_like(scale))
                x_scaled = x.clone()
                x_scaled[:, 1:] = features / scale
                gram = x_scaled.T @ x_scaled
                gram = gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
                beta_scaled = torch.linalg.solve(gram, x_scaled.T @ y)
                if extended_features:
                    # Clipping an unconstrained solution is not nonnegative
                    # least squares: correlated capacity/padding/history terms
                    # can cancel before clipping and explode afterwards. P2
                    # uses projected gradient on the small normalized system.
                    beta_scaled = torch.zeros(x.shape[1], dtype=x.dtype)
                    beta_scaled[0] = torch.mean(y)
                    step = 1.0 / max(float(torch.linalg.eigvalsh(gram)[-1]), 1e-12)
                    target = x_scaled.T @ y
                    for _ in range(512):
                        updated = torch.clamp(beta_scaled - step * (gram @ beta_scaled - target), min=0.0)
                        if torch.max(torch.abs(updated - beta_scaled)) < 1e-9:
                            break
                        beta_scaled = updated
                beta = beta_scaled.clone()
                beta[1:] = beta_scaled[1:] / scale
                beta = torch.clamp(beta, min=0.0)
            else:
                beta = torch.zeros(x.shape[1], dtype=x.dtype)
                beta[0] = torch.median(y)
            prediction = x @ beta
            residuals = torch.abs(y - prediction).tolist()
            fitted[key] = _FittedTaskModel(
                coefficients=tuple(float(value) for value in beta.tolist()),
                residual_median_ms=float(median(residuals)) if residuals else 0.0,
                residual_p95_ms=_percentile(residuals, 0.95),
                samples=len(samples),
                feature_min=tuple(float(v) for v in torch.min(x, dim=0).values.tolist()),
                feature_max=tuple(float(v) for v in torch.max(x, dim=0).values.tolist()),
                duration_median_ms=float(torch.median(y)),
            )
            by_features: Dict[Tuple[float, ...], list[float]] = defaultdict(list)
            for features, duration, _signature_value in samples:
                by_features[features].append(duration)
            profile_points[key] = tuple(
                _ProfilePoint(
                    features=features,
                    median_ms=_percentile(values, 0.5),
                    p95_ms=_percentile(values, 0.95),
                    samples=len(values),
                )
                for features, values in sorted(by_features.items())
            )
        return cls(
            exact_median_ms={key: _percentile(values, 0.5) for key, values in exact.items()},
            exact_p95_ms={key: _percentile(values, 0.95) for key, values in exact.items()},
            fitted=fitted,
            points=profile_points,
            extended_features=extended_features,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DsppExecutionCostModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and payload.get('schema') == 'dspp-stage-calibration-v2':
            raise ValueError('stage calibration v2 requires the context-aware StageCalibration API')
        records = payload.get("records") if isinstance(payload, Mapping) else None
        if records and "query_lengths" in records[0] and "median_ms" in records[0]:
            return cls.from_compute_profile(payload)
        if "records" in payload:
            return cls.from_records(payload["records"])
        if "events" in payload:
            return cls.from_records(payload["events"])
        if "cost_model" in payload:
            return cls.from_dict(payload["cost_model"])
        if "fitted" in payload or "exact" in payload:
            return cls.from_dict(payload)
        # Preserve compatibility with the old signature -> duration JSON.
        return cls.empty()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DsppExecutionCostModel":
        exact_median = {}
        exact_p95 = {}
        for row in payload.get("exact", []):
            key = (int(row["stage"]), str(row["op"]), str(row["signature"]))
            exact_median[key] = float(row["median_ms"])
            exact_p95[key] = float(row.get("p95_ms", row["median_ms"]))
        fitted = {}
        for row in payload.get("fitted", []):
            key = (int(row["stage"]), str(row["op"]))
            fitted[key] = _FittedTaskModel(
                coefficients=tuple(float(x) for x in row["coefficients"]),
                residual_median_ms=float(row.get("residual_median_ms", 0.0)),
                residual_p95_ms=float(row.get("residual_p95_ms", 0.0)),
                samples=int(row.get("samples", 0)),
                feature_min=tuple(float(x) for x in row.get("feature_min", ())),
                feature_max=tuple(float(x) for x in row.get("feature_max", ())),
                duration_median_ms=float(row.get("duration_median_ms", 0.0)),
            )
        points = {}
        for row in payload.get("points", []):
            key = (int(row["stage"]), str(row["op"]))
            points.setdefault(key, []).append(_ProfilePoint(
                features=tuple(float(x) for x in row["features"]),
                median_ms=float(row["median_ms"]),
                p95_ms=float(row.get("p95_ms", row["median_ms"])),
                samples=int(row.get("samples", 1)),
            ))
        w_median = {
            (int(row["stage"]), int(row["capacity"])): float(row["median_ms"])
            for row in payload.get("w_capacity", [])
        }
        w_p95 = {
            (int(row["stage"]), int(row["capacity"])): float(
                row.get("p95_ms", row["median_ms"])
            )
            for row in payload.get("w_capacity", [])
        }
        calibrations = {}
        for row in payload.get("stage_op_calibrations", []):
            key = (int(row["stage"]), _op_kind(str(row["op"])))
            calibrations[key] = _StageOpCalibration(
                scale=max(0.0, float(row.get("scale", 1.0))),
                overhead_ms=max(0.0, float(row.get("overhead_ms", 0.0))),
                residual_median_ms=max(
                    0.0, float(row.get("residual_median_ms", 0.0))
                ),
                residual_p95_ms=max(
                    0.0, float(row.get("residual_p95_ms", 0.0))
                ),
                samples=max(0, int(row.get("samples", 0))),
            )
        return cls(
            exact_median,
            exact_p95,
            fitted,
            {key: tuple(value) for key, value in points.items()},
            w_median,
            w_p95,
            stage_op_calibrations=calibrations,
            source=str(payload.get("source", "profile")),
            extended_features=bool(payload.get("extended_features", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        exact_keys = sorted(self.exact_median_ms)
        return {
            # Keep the wire version compatible with existing saved profiles;
            # new optional fields are ignored by older readers.
            "version": 1,
            "source": self.source,
            "extended_features": self.extended_features,
            "exact": [
                {
                    "stage": stage,
                    "op": op,
                    "signature": signature,
                    "median_ms": self.exact_median_ms[key],
                    "p95_ms": self.exact_p95_ms.get(key, self.exact_median_ms[key]),
                }
                for key in exact_keys
                for stage, op, signature in [key]
            ],
            "fitted": [
                {
                    "stage": stage,
                    "op": op,
                    "coefficients": list(model.coefficients),
                    "residual_median_ms": model.residual_median_ms,
                    "residual_p95_ms": model.residual_p95_ms,
                    "samples": model.samples,
                    "feature_min": list(model.feature_min),
                    "feature_max": list(model.feature_max),
                    "duration_median_ms": model.duration_median_ms,
                }
                for (stage, op), model in sorted(self.fitted.items())
            ],
            "points": [
                {
                    "stage": stage,
                    "op": op,
                    "features": list(point.features),
                    "median_ms": point.median_ms,
                    "p95_ms": point.p95_ms,
                    "samples": point.samples,
                }
                for (stage, op), values in sorted(self.points.items())
                for point in values
            ],
            "stage_op_calibrations": [
                {
                    "stage": stage,
                    "op": op,
                    "scale": calibration.scale,
                    "overhead_ms": calibration.overhead_ms,
                    "residual_median_ms": calibration.residual_median_ms,
                    "residual_p95_ms": calibration.residual_p95_ms,
                    "samples": calibration.samples,
                }
                for (stage, op), calibration in sorted(
                    self.stage_op_calibrations.items()
                )
            ],
            "w_capacity": [
                {
                    "stage": stage,
                    "capacity": capacity,
                    "median_ms": value,
                    "p95_ms": self.w_capacity_p95_ms.get((stage, capacity), value),
                }
                for (stage, capacity), value in sorted(
                    self.w_capacity_median_ms.items()
                )
            ],
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def predict(
        self,
        stage: int,
        op: str,
        meta: DsppMicrobatchMeta,
        *,
        percentile: float = 0.5,
        fallback: Optional[float] = None,
    ) -> Tuple[float, str, float]:
        prediction = self.predict_detailed(
            stage, op, meta, percentile=percentile, fallback=fallback
        )
        return prediction.duration_ms, prediction.source, prediction.error_ms

    def predict_detailed(
        self,
        stage: int,
        op: str,
        meta: DsppMicrobatchMeta,
        *,
        percentile: float = 0.5,
        fallback: Optional[float] = None,
    ) -> DsppCostPrediction:
        op = _op_kind(op)
        prediction = self._predict_base_detailed(
            stage, op, meta, percentile=percentile, fallback=fallback
        )
        calibration = self.stage_op_calibrations.get((int(stage), op))
        if calibration is None:
            # A stage-aware calibration is preferred.  For a PP-free profile
            # reused across stages, stage 0 is the deliberate fallback.
            calibration = self.stage_op_calibrations.get((0, op))
        if calibration is None or prediction.source == "metadata-fallback":
            return prediction
        residual = (
            calibration.residual_p95_ms
            if percentile >= 0.95
            else calibration.residual_median_ms
        )
        return DsppCostPrediction(
            duration_ms=max(
                0.0,
                calibration.scale * prediction.duration_ms
                + calibration.overhead_ms,
            ),
            source=prediction.source,
            error_ms=max(0.0, calibration.scale * prediction.error_ms + residual),
            confidence=prediction.confidence,
            extrapolated=prediction.extrapolated,
        )

    def _predict_base_detailed(
        self,
        stage: int,
        op: str,
        meta: DsppMicrobatchMeta,
        *,
        percentile: float = 0.5,
        fallback: Optional[float] = None,
    ) -> DsppCostPrediction:
        op = _op_kind(op)
        # A PP-free calibration measures one balanced logical stage. Reuse
        # stage 0 only when the requested stage has no dedicated profile.
        profile_stage = int(stage)
        if (
            (profile_stage, op) not in self.fitted
            and not any(key[0] == profile_stage and key[1] == op for key in self.exact_median_ms)
            and ((0, op) in self.fitted or any(key[0] == 0 and key[1] == op for key in self.exact_median_ms))
        ):
            profile_stage = 0
        if op == "W":
            w_stage = int(stage)
            w_key = (w_stage, meta.chunk_size)
            if w_key not in self.w_capacity_median_ms:
                w_key = (0, meta.chunk_size)
            if w_key in self.w_capacity_median_ms:
                value = (
                    self.w_capacity_p95_ms.get(w_key, self.w_capacity_median_ms[w_key])
                    if percentile >= 0.95 else self.w_capacity_median_ms[w_key]
                )
                return DsppCostPrediction(
                    float(value), "exact-capacity-profile", 0.0, "high", False
                )
        key = (profile_stage, op, _signature(meta))
        if key in self.exact_median_ms:
            value = self.exact_p95_ms[key] if percentile >= 0.95 else self.exact_median_ms[key]
            return DsppCostPrediction(float(value), "exact-profile", 0.0, "high", False)
        fitted = self.fitted.get((profile_stage, op))
        features = _features(meta, self.extended_features)
        in_bounds = _features_in_bounds(features, fitted)
        points = self.points.get((profile_stage, op), ())
        if points and in_bounds:
            value, local_error = _interpolate_profile_points(
                features, points, fitted, percentile
            )
            relative_error = local_error / max(value, 1e-6)
            confidence = "high" if relative_error <= 0.15 else (
                "medium" if relative_error <= 0.35 else "low"
            )
            return DsppCostPrediction(
                max(0.0, value), "interpolated-profile", local_error,
                confidence, False,
            )
        if fitted is not None:
            value = sum(a * b for a, b in zip(fitted.coefficients, features))
            error = fitted.residual_p95_ms if percentile >= 0.95 else fitted.residual_median_ms
            relative_error = error / max(value, 1e-6)
            confidence = "low" if not in_bounds else (
                "medium" if relative_error <= 0.35 else "low"
            )
            source = "fitted-profile" if in_bounds else "fitted-extrapolated"
            return DsppCostPrediction(
                max(0.0, value), source, float(error), confidence, not in_bounds
            )
        if op in {'D', 'H', 'B_fused'} and fallback is None:
            raise ValueError(f'missing {op} execution cost; explicit fallback required')
        return DsppCostPrediction(
            float(fallback or 0.0), "metadata-fallback", 0.0, "none", True
        )

    def with_stage_op_calibration(
        self, records: Sequence[Mapping[str, Any]]
    ) -> "DsppExecutionCostModel":
        """Fit a small stage/op correction from real DSPP CUDA-event tasks.

        The PP-free profile retains the shape curve across chunk capacities.
        A real pipeline calibration only learns a non-negative affine mapping
        per stage and F/B/W kind.  The intercept is intentionally per task: it
        lets the planner charge candidates that create more physical tasks
        without adding an ad-hoc microbatch-count term.
        """

        grouped: Dict[Tuple[int, str], list[Tuple[float, float]]] = defaultdict(list)
        for record in records:
            if "duration_ms" not in record or "stage" not in record:
                continue
            op = _op_kind(str(record.get("kind", record.get("operation", ""))))
            if op not in "FBW" or not record.get("items"):
                continue
            try:
                meta = meta_from_timeline_record(record)
            except (KeyError, TypeError, ValueError):
                continue
            prediction = self._predict_base_detailed(
                int(record["stage"]), op, meta, fallback=None
            )
            if prediction.source == "metadata-fallback" or prediction.duration_ms <= 0:
                continue
            grouped[(int(record["stage"]), op)].append(
                (float(prediction.duration_ms), float(record["duration_ms"]))
            )

        calibrations: Dict[Tuple[int, str], _StageOpCalibration] = {}
        for key, samples in grouped.items():
            scale, overhead = _fit_nonnegative_affine(samples)
            residuals = [
                abs(actual - (scale * predicted + overhead))
                for predicted, actual in samples
            ]
            calibrations[key] = _StageOpCalibration(
                scale=scale,
                overhead_ms=overhead,
                residual_median_ms=_percentile(residuals, 0.5),
                residual_p95_ms=_percentile(residuals, 0.95),
                samples=len(samples),
            )
        return DsppExecutionCostModel(
            exact_median_ms=self.exact_median_ms,
            exact_p95_ms=self.exact_p95_ms,
            fitted=self.fitted,
            points=self.points,
            w_capacity_median_ms=self.w_capacity_median_ms,
            w_capacity_p95_ms=self.w_capacity_p95_ms,
            source=(
                f"{self.source}+stage-op-calibration"
                if calibrations else self.source
            ),
            stage_op_calibrations=calibrations,
            extended_features=self.extended_features,
        )

    def predict_task_workload(
        self,
        meta: DsppMicrobatchMeta,
        *,
        pipeline_size: int,
        fallback: float,
        percentile: float = 0.5,
    ) -> Tuple[float, str, float]:
        values = []
        sources = []
        errors = []
        for stage in range(pipeline_size):
            total = 0.0
            for op in "FBW":
                value, source, error = self.predict(stage, op, meta, percentile=percentile)
                if source == "metadata-fallback":
                    total = None
                    break
                total += value
                sources.append(source)
                errors.append(error)
            if total is not None:
                values.append(total)
        if not values:
            return float(fallback), "metadata-fallback", 0.0
        source = "exact-profile" if all(item == "exact-profile" for item in sources) else "fitted-profile"
        return max(values), source, max(errors, default=0.0)


def meta_from_timeline_record(
    record: Mapping[str, Any],
    items: Optional[Sequence[Mapping[str, Any]]] = None,
) -> DsppMicrobatchMeta:
    from .dspp_batch_plan import DsppSegment

    if items is None:
        items = tuple(record.get("items", ()))
    if not items:
        raise ValueError("timeline task record contains no items")
    segments = tuple(
        DsppSegment(
            sequence_id=int(item["sequence_id"]),
            segment_id=int(item["segment_id"]),
            token_offset=int(item["token_offset"]),
            token_length=int(item["token_length"]),
            sequence_length=int(item["sequence_length"]),
            segment_count=int(item["segment_count"]),
            estimated_flops=0.0,
        )
        for item in items
    )
    sequence_chunk_size = record.get("sequence_chunk_size")
    if sequence_chunk_size is None:
        # Older timeline files predate the explicit field.  Their ``chunk``
        # value is the VPP chunk id, while the sequence chunk is encoded in
        # signatures such as ``c3584|...``.
        signature = str(record.get("signature", ""))
        prefix = signature.split("|", 1)[0]
        if prefix.startswith("c") and prefix[1:].isdigit():
            sequence_chunk_size = int(prefix[1:])
        else:
            sequence_chunk_size = max(item.token_length for item in segments)
    return DsppMicrobatchMeta(
        microbatch_id=int(record.get("physical_microbatch", 0)),
        chunk_size=int(sequence_chunk_size),
        items=segments,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _fit_nonnegative_affine(
    samples: Sequence[Tuple[float, float]],
) -> Tuple[float, float]:
    """Fit ``actual ~= scale * base + overhead`` with non-negative terms."""

    if not samples:
        return 1.0, 0.0
    # Solve the two-parameter least-squares problem explicitly.  This keeps
    # calibration CPU-only and avoids pulling a tensor dependency into the
    # profile fitting path.
    n = float(len(samples))
    sx = sum(float(x) for x, _ in samples)
    sy = sum(float(y) for _, y in samples)
    sxx = sum(float(x) * float(x) for x, _ in samples)
    sxy = sum(float(x) * float(y) for x, y in samples)
    denominator = n * sxx - sx * sx
    if denominator <= 1.0e-12:
        return 1.0, max(0.0, sy / n - sx / n)
    scale = (n * sxy - sx * sy) / denominator
    overhead = (sy - scale * sx) / n
    # The planner must never turn a positive compute estimate into a negative
    # cost.  Refit the intercept after clipping the slope so the correction is
    # still the least-squares affine mapping under the non-negative constraint.
    scale = max(0.0, float(scale))
    overhead = max(0.0, float(overhead))
    return scale, overhead


def _features_in_bounds(
    features: Sequence[float], fitted: Optional[_FittedTaskModel]
) -> bool:
    if fitted is None or not fitted.feature_min or not fitted.feature_max:
        return False
    return all(
        lower <= value <= upper
        for value, lower, upper in zip(
            features, fitted.feature_min, fitted.feature_max, strict=True
        )
    )


def _interpolate_profile_points(
    features: Sequence[float],
    points: Sequence[_ProfilePoint],
    fitted: Optional[_FittedTaskModel],
    percentile: float,
) -> Tuple[float, float]:
    if fitted is None:
        scale = [1.0] * len(features)
    else:
        scale = [
            max(1.0, upper - lower)
            for lower, upper in zip(
                fitted.feature_min, fitted.feature_max, strict=True
            )
        ]
    distances = []
    for point in points:
        distance = sum(
            ((value - sample) / width) ** 2
            for value, sample, width in zip(
                features, point.features, scale, strict=True
            )
        ) ** 0.5
        distances.append((distance, point))
    nearest = sorted(distances, key=lambda item: item[0])[: min(4, len(distances))]
    weights = [1.0 / max(distance, 1e-6) for distance, _point in nearest]
    values = [
        point.p95_ms if percentile >= 0.95 else point.median_ms
        for _distance, point in nearest
    ]
    total = sum(weights)
    prediction = sum(weight * value for weight, value in zip(weights, values)) / total
    local_error = sum(
        weight * abs(value - prediction)
        for weight, value in zip(weights, values)
    ) / total
    if fitted is not None:
        residual = (
            fitted.residual_p95_ms if percentile >= 0.95
            else fitted.residual_median_ms
        )
        local_error = max(local_error, residual)
    return float(prediction), float(local_error)


@lru_cache(maxsize=16)
def load_execution_cost_model(path: Optional[str]) -> DsppExecutionCostModel:
    if not path:
        return DsppExecutionCostModel.empty()
    try:
        return DsppExecutionCostModel.from_json(path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return DsppExecutionCostModel.empty()
