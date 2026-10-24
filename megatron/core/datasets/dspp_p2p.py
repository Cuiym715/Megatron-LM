"""Small, independent P2P cost model used by the DSPP metadata planner.

The model intentionally does not assume that communication is serialized with
compute.  It predicts the duration of one directed communication-lane event;
the weighted DAG decides whether that event overlaps a compute node.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    position = (len(ordered) - 1) * percentile
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    frac = position - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


@dataclass(frozen=True)
class DsppP2PPoint:
    bytes: int
    direction: str
    overlap: str
    median_ms: float
    p95_ms: float


@dataclass(frozen=True)
class DsppP2PPrediction:
    duration_ms: float
    source: str
    confidence: str
    extrapolated: bool
    out_of_range: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DsppP2PCostModel:
    """Piecewise-linear interpolation over measured directed P2P points."""

    points: Tuple[DsppP2PPoint, ...] = ()
    alpha_ms: Mapping[Tuple[str, str], float] = None
    beta_bytes_per_ms: Mapping[Tuple[str, str], float] = None

    def __post_init__(self):
        if self.alpha_ms is None:
            object.__setattr__(self, "alpha_ms", {})
        if self.beta_bytes_per_ms is None:
            object.__setattr__(self, "beta_bytes_per_ms", {})

    @classmethod
    def empty(cls) -> "DsppP2PCostModel":
        return cls()

    @classmethod
    def from_records(cls, records: Sequence[Mapping[str, Any]]) -> "DsppP2PCostModel":
        grouped = {}
        for row in records:
            if "bytes" not in row:
                continue
            key = (
                int(row["bytes"]),
                str(row.get("direction", "unknown")),
                str(row.get("overlap", "isolated")),
            )
            duration = row.get("duration_ms", row.get("median_ms"))
            if duration is None:
                continue
            if key[0] < 0 or not math.isfinite(float(duration)) or float(duration) <= 0:
                raise ValueError('invalid P2P calibration sample')
            grouped.setdefault(key, []).append(float(duration))
        points = tuple(
            DsppP2PPoint(
                bytes=size,
                direction=direction,
                overlap=overlap,
                median_ms=_percentile(values, 0.5),
                p95_ms=_percentile(values, 0.95),
            )
            for (size, direction, overlap), values in sorted(grouped.items())
        )
        return cls(points=points)

    @classmethod
    def from_json(cls, path: str | Path) -> "DsppP2PCostModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("records", payload.get("points", []))
        return cls.from_records(payload if isinstance(payload, list) else [])

    def to_dict(self):
        return {
            "version": 1,
            "points": [
                {
                    "bytes": p.bytes,
                    "direction": p.direction,
                    "overlap": p.overlap,
                    "median_ms": p.median_ms,
                    "p95_ms": p.p95_ms,
                }
                for p in self.points
            ],
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def predict(
        self,
        payload_bytes: int,
        *,
        direction: str = "unknown",
        overlap: str = "isolated",
        percentile: float = 0.5,
        fallback_bandwidth_bytes_per_ms: float = 25.0e6,
        fallback_latency_ms: float = 0.02,
    ) -> Tuple[float, str]:
        size = max(0, int(payload_bytes))
        candidates = [
            p for p in self.points
            if p.direction == direction and p.overlap == overlap
        ]
        # Fall back to direction-only points before using the analytical model.
        if not candidates:
            candidates = [p for p in self.points if p.direction == direction]
        if candidates:
            candidates.sort(key=lambda p: p.bytes)
            if size <= candidates[0].bytes:
                point = candidates[0]
                value = point.p95_ms if percentile >= 0.95 else point.median_ms
                return float(value), "profile-clamped"
            if size >= candidates[-1].bytes:
                point = candidates[-1]
                value = point.p95_ms if percentile >= 0.95 else point.median_ms
                return float(value), "profile-clamped"
            for left, right in zip(candidates, candidates[1:]):
                if left.bytes <= size <= right.bytes:
                    ratio = (size - left.bytes) / float(right.bytes - left.bytes)
                    lv = left.p95_ms if percentile >= 0.95 else left.median_ms
                    rv = right.p95_ms if percentile >= 0.95 else right.median_ms
                    return float(lv + ratio * (rv - lv)), "profile-interpolated"
        return float(fallback_latency_ms + size / max(1.0, fallback_bandwidth_bytes_per_ms)), "analytical-fallback"

    def predict_detailed(self, payload_bytes, *, direction, overlap, percentile=0.5):
        """Strict overlap/size coverage; preserve legacy predict() separately."""
        candidates = sorted((p for p in self.points if p.direction == direction
                             and p.overlap == overlap), key=lambda p: p.bytes)
        size = int(payload_bytes)
        if size < 0:
            raise ValueError('negative P2P payload size')
        if candidates and candidates[0].bytes <= size <= candidates[-1].bytes:
            value, _ = self.predict(size, direction=direction, overlap=overlap, percentile=percentile)
            exact = any(p.bytes == size for p in candidates)
            return DsppP2PPrediction(value, 'exact-profile' if exact else 'interpolated-profile',
                                     'high' if exact else 'medium', False)
        value, _ = DsppP2PCostModel.empty().predict(size)
        return DsppP2PPrediction(value, 'analytical-fallback', 'low', True,
                                 ('bytes',) if candidates else ('direction/overlap',))


@lru_cache(maxsize=16)
def load_p2p_cost_model(path: Optional[str]) -> DsppP2PCostModel:
    if not path:
        return DsppP2PCostModel.empty()
    try:
        return DsppP2PCostModel.from_json(path)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return DsppP2PCostModel.empty()
