"""Deterministic DSPP slicing and residual-packing plans.

This module deliberately contains no pipeline-schedule logic.  It converts a
logical batch of independent documents into fixed-capacity physical
microbatches while retaining enough metadata to reconstruct causal attention
and gradients across sequence segments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, pstdev
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


CostFn = Callable[[int, int, int], float]


@dataclass(frozen=True)
class DsppSegment:
    """A contiguous query-token interval from one logical sequence."""

    sequence_id: int
    segment_id: int
    token_offset: int
    token_length: int
    sequence_length: int
    segment_count: int
    estimated_flops: float

    @property
    def has_history(self) -> bool:
        return self.token_offset > 0

    @property
    def is_last_segment(self) -> bool:
        return self.segment_id + 1 == self.segment_count

    @property
    def is_tail(self) -> bool:
        if not self.has_history or not self.is_last_segment:
            return False
        chunk_size = self.token_offset // self.segment_id
        return self.token_length < chunk_size

    @property
    def is_short_sequence(self) -> bool:
        return self.segment_count == 1


@dataclass(frozen=True)
class DsppMicrobatchMeta:
    """Attention metadata for one fixed-capacity physical microbatch."""

    microbatch_id: int
    chunk_size: int
    items: Tuple[DsppSegment, ...]

    @property
    def valid_token_count(self) -> int:
        return sum(item.token_length for item in self.items)

    @property
    def padding_token_count(self) -> int:
        return self.chunk_size - self.valid_token_count

    @property
    def sequence_ids(self) -> Tuple[int, ...]:
        return tuple(item.sequence_id for item in self.items)

    @property
    def query_lengths(self) -> Tuple[int, ...]:
        return tuple(item.token_length for item in self.items)

    @property
    def key_lengths(self) -> Tuple[int, ...]:
        return tuple(item.token_offset + item.token_length for item in self.items)

    @staticmethod
    def _cumulative(lengths: Iterable[int]) -> Tuple[int, ...]:
        result = [0]
        for length in lengths:
            result.append(result[-1] + int(length))
        return tuple(result)

    @property
    def cu_seqlens_q(self) -> Tuple[int, ...]:
        return self._cumulative(self.query_lengths)

    @property
    def cu_seqlens_k(self) -> Tuple[int, ...]:
        return self._cumulative(self.key_lengths)

    @property
    def max_seqlen_q(self) -> int:
        return max(self.query_lengths, default=0)

    @property
    def max_seqlen_k(self) -> int:
        return max(self.key_lengths, default=0)

    def validate(self) -> None:
        if self.microbatch_id < 0:
            raise ValueError("microbatch_id must be non-negative")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not self.items:
            raise ValueError("a DSPP physical microbatch cannot be empty")
        if self.valid_token_count > self.chunk_size:
            raise ValueError(
                f"microbatch {self.microbatch_id} has {self.valid_token_count} tokens "
                f"but chunk_size is {self.chunk_size}"
            )
        continuation_items = [item for item in self.items if item.has_history]
        if len(self.items) > 1 and len(continuation_items) > 1:
            raise ValueError(
                "a residual pack may contain at most one continuation segment"
            )
        if continuation_items and self.items[0] is not continuation_items[0]:
            raise ValueError(
                "the continuation segment must be the first item in a residual pack"
            )
        if len(self.items) > 1:
            for item in self.items[1:]:
                if item.segment_count != 1 or item.token_offset != 0:
                    raise ValueError(
                        "only independent short sequences may follow a continuation"
                    )


@dataclass(frozen=True)
class DsppPackedBatch:
    """Materialized tokens and labels for one physical microbatch."""

    meta: DsppMicrobatchMeta
    tokens: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor
    loss_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "DsppPackedBatch":
        return DsppPackedBatch(
            meta=self.meta,
            tokens=self.tokens.to(device),
            labels=self.labels.to(device),
            position_ids=self.position_ids.to(device),
            loss_mask=self.loss_mask.to(device),
        )


@dataclass(frozen=True)
class DsppBatchPlan:
    """A complete logical-batch to physical-microbatch mapping."""

    chunk_size: int
    sequence_lengths: Tuple[int, ...]
    microbatches: Tuple[DsppMicrobatchMeta, ...]
    packing_stats: Dict[str, float]

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not self.sequence_lengths:
            raise ValueError("sequence_lengths cannot be empty")
        if any(length <= 0 for length in self.sequence_lengths):
            raise ValueError(
                "every logical sequence must contain at least one training token"
            )

        occurrences: Dict[int, List[DsppSegment]] = {
            sequence_id: [] for sequence_id in range(len(self.sequence_lengths))
        }
        for expected_id, microbatch in enumerate(self.microbatches):
            if microbatch.microbatch_id != expected_id:
                raise ValueError(
                    "microbatch ids must be dense and match physical order"
                )
            microbatch.validate()
            for item in microbatch.items:
                if item.sequence_id not in occurrences:
                    raise ValueError(f"unknown sequence id {item.sequence_id}")
                occurrences[item.sequence_id].append(item)

        for sequence_id, expected_length in enumerate(self.sequence_lengths):
            segments = occurrences[sequence_id]
            if not segments:
                raise ValueError(f"sequence {sequence_id} has no physical segment")
            expected_segment_count = math.ceil(expected_length / self.chunk_size)
            if len(segments) != expected_segment_count:
                raise ValueError(
                    f"sequence {sequence_id} has {len(segments)} segments; "
                    f"expected {expected_segment_count}"
                )
            for segment_id, segment in enumerate(segments):
                expected_offset = segment_id * self.chunk_size
                expected_segment_length = min(
                    self.chunk_size, expected_length - expected_offset
                )
                if (
                    segment.segment_id != segment_id
                    or segment.token_offset != expected_offset
                    or segment.token_length != expected_segment_length
                    or segment.sequence_length != expected_length
                    or segment.segment_count != expected_segment_count
                ):
                    raise ValueError(
                        f"sequence {sequence_id} has a broken segment chain at {segment_id}: "
                        f"{segment}"
                    )

        planned_tokens = sum(
            microbatch.valid_token_count for microbatch in self.microbatches
        )
        if planned_tokens != sum(self.sequence_lengths):
            raise ValueError(
                f"plan covers {planned_tokens} tokens but logical batch contains "
                f"{sum(self.sequence_lengths)}"
            )

    def materialize(
        self,
        token_sequences: Sequence[Sequence[int] | torch.Tensor],
        *,
        pad_token_id: int = 0,
        device: Optional[torch.device | str] = None,
    ) -> Tuple[DsppPackedBatch, ...]:
        """Create `[1, chunk_size]` model inputs from token-plus-label sequences.

        Each input sequence must contain one more token than the corresponding
        `sequence_lengths` entry: token `i` predicts token `i + 1`.
        """

        self.validate()
        if len(token_sequences) != len(self.sequence_lengths):
            raise ValueError("token_sequences length does not match sequence_lengths")

        normalized: List[torch.Tensor] = []
        for sequence_id, (raw_tokens, training_length) in enumerate(
            zip(token_sequences, self.sequence_lengths, strict=True)
        ):
            tokens = torch.as_tensor(
                raw_tokens, dtype=torch.long, device=device or "cpu"
            ).flatten()
            if tokens.numel() != training_length + 1:
                raise ValueError(
                    f"sequence {sequence_id} has {tokens.numel()} tokens; "
                    f"expected {training_length + 1}"
                )
            normalized.append(tokens)

        batches: List[DsppPackedBatch] = []
        for microbatch in self.microbatches:
            batch_device = normalized[0].device
            tokens = torch.full(
                (1, self.chunk_size),
                pad_token_id,
                dtype=torch.long,
                device=batch_device,
            )
            labels = torch.full(
                (1, self.chunk_size),
                pad_token_id,
                dtype=torch.long,
                device=batch_device,
            )
            position_ids = torch.zeros(
                (1, self.chunk_size), dtype=torch.long, device=batch_device
            )
            loss_mask = torch.zeros(
                (1, self.chunk_size), dtype=torch.float32, device=batch_device
            )
            physical_offset = 0
            for item in microbatch.items:
                source = normalized[item.sequence_id]
                begin = item.token_offset
                end = begin + item.token_length
                physical_end = physical_offset + item.token_length
                tokens[0, physical_offset:physical_end] = source[begin:end]
                labels[0, physical_offset:physical_end] = source[begin + 1 : end + 1]
                position_ids[0, physical_offset:physical_end] = torch.arange(
                    begin, end, device=batch_device
                )
                loss_mask[0, physical_offset:physical_end] = 1.0
                physical_offset = physical_end
            batches.append(
                DsppPackedBatch(
                    meta=microbatch,
                    tokens=tokens,
                    labels=labels,
                    position_ids=position_ids,
                    loss_mask=loss_mask,
                )
            )
        return tuple(batches)


@dataclass
class _ResidualBin:
    index: int
    items: List[DsppSegment]
    total_tokens: int = 0
    total_flops: float = 0.0

    def add(self, item: DsppSegment) -> None:
        self.items.append(item)
        self.total_tokens += item.token_length
        self.total_flops += item.estimated_flops


def _default_cost(sequence_id: int, prefix_start: int, prefix_end: int) -> float:
    del sequence_id
    return float(prefix_end * prefix_end - prefix_start * prefix_start)


def _pack_residuals(
    residuals: Sequence[DsppSegment], chunk_size: int
) -> List[_ResidualBin]:
    if not residuals:
        return []
    tails = [item for item in residuals if item.has_history]
    shorts = [item for item in residuals if not item.has_history]
    lower_bound = max(
        len(tails),
        math.ceil(sum(item.token_length for item in residuals) / float(chunk_size)),
    )
    for bin_count in range(max(1, lower_bound), len(residuals) + 1):
        bins = [_ResidualBin(index=index, items=[]) for index in range(bin_count)]
        for index, item in enumerate(
            sorted(tails, key=lambda entry: entry.sequence_id)
        ):
            bins[index].add(item)
        success = True
        for item in sorted(
            shorts,
            key=lambda entry: (
                -entry.token_length,
                -entry.estimated_flops,
                entry.sequence_id,
            ),
        ):
            feasible = [
                packed
                for packed in bins
                if packed.total_tokens + item.token_length <= chunk_size
            ]
            if not feasible:
                success = False
                break
            selected = min(
                feasible,
                key=lambda packed: (
                    packed.total_flops,
                    chunk_size - packed.total_tokens - item.token_length,
                    packed.index,
                ),
            )
            selected.add(item)
        if success:
            return [packed for packed in bins if packed.items]
    raise RuntimeError("failed to pack residual sequence segments")


def build_dspp_batch_plan(
    sequence_lengths: Sequence[int],
    chunk_size: int,
    *,
    cost_fn: Optional[CostFn] = None,
    validate: bool = True,
) -> DsppBatchPlan:
    """Build the simulator-compatible full-chunk plus residual-pack layout."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    normalized_lengths = tuple(int(length) for length in sequence_lengths)
    if not normalized_lengths or any(length <= 0 for length in normalized_lengths):
        raise ValueError("sequence_lengths must contain positive integers")
    estimate = cost_fn or _default_cost

    full_microbatches: List[Tuple[DsppSegment, ...]] = []
    residuals: List[DsppSegment] = []
    for sequence_id, sequence_length in enumerate(normalized_lengths):
        segment_count = math.ceil(sequence_length / chunk_size)
        full_chunks, tail_length = divmod(sequence_length, chunk_size)
        for segment_id in range(full_chunks):
            offset = segment_id * chunk_size
            full_microbatches.append(
                (
                    DsppSegment(
                        sequence_id=sequence_id,
                        segment_id=segment_id,
                        token_offset=offset,
                        token_length=chunk_size,
                        sequence_length=sequence_length,
                        segment_count=segment_count,
                        estimated_flops=float(
                            estimate(sequence_id, offset, offset + chunk_size)
                        ),
                    ),
                )
            )
        if tail_length:
            offset = full_chunks * chunk_size
            residuals.append(
                DsppSegment(
                    sequence_id=sequence_id,
                    segment_id=full_chunks,
                    token_offset=offset,
                    token_length=tail_length,
                    sequence_length=sequence_length,
                    segment_count=segment_count,
                    estimated_flops=float(
                        estimate(sequence_id, offset, sequence_length)
                    ),
                )
            )

    residual_bins = _pack_residuals(residuals, chunk_size)
    item_groups = full_microbatches + [tuple(packed.items) for packed in residual_bins]
    microbatches = tuple(
        DsppMicrobatchMeta(
            microbatch_id=microbatch_id,
            chunk_size=chunk_size,
            items=items,
        )
        for microbatch_id, items in enumerate(item_groups)
    )
    flops = [sum(item.estimated_flops for item in items) for items in item_groups]
    utilizations = [
        sum(item.token_length for item in items) / chunk_size for items in item_groups
    ]
    mean_flops = mean(flops) if flops else 0.0
    plan = DsppBatchPlan(
        chunk_size=chunk_size,
        sequence_lengths=normalized_lengths,
        microbatches=microbatches,
        packing_stats={
            "microbatch_count": float(len(microbatches)),
            "mean_flops": float(mean_flops),
            "flops_range": float(max(flops) - min(flops)) if flops else 0.0,
            "flops_cv": float(pstdev(flops) / mean_flops) if mean_flops else 0.0,
            "mean_token_utilization": (
                float(mean(utilizations)) if utilizations else 0.0
            ),
        },
    )
    if validate:
        plan.validate()
    return plan
