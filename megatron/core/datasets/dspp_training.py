"""Training-data adapter for DSPP logical batches.

The dataloader produces a padded matrix of independent documents plus their
explicit lengths.  This adapter turns one such logical Megatron microbatch
into the fixed-shape physical microbatches consumed by DSPP attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from .dspp_batch_plan import (
    CostFn,
    DsppBatchPlan,
    DsppPackedBatch,
    build_dspp_batch_plan,
)


@dataclass
class DsppTrainingBatch:
    """One logical dataloader batch expanded into physical microbatches."""

    plan: DsppBatchPlan
    physical_microbatches: Tuple[DsppPackedBatch, ...]
    _source_sequences: Optional[Tuple[torch.Tensor, ...]] = None
    _pad_token_id: int = 0
    _pack_residuals: bool = True
    _packing_strategy: str = "baseline"

    @property
    def loss_mask(self) -> torch.Tensor:
        """Return a `[1, physical_microbatches * chunk_size]` loss mask."""

        return torch.cat(
            [batch.loss_mask for batch in self.physical_microbatches], dim=1
        )

    @property
    def valid_token_count(self) -> int:
        return sum(self.plan.sequence_lengths)

    def validate(self) -> None:
        self.plan.validate()
        if len(self.physical_microbatches) != len(self.plan.microbatches):
            raise ValueError("physical microbatch count does not match DSPP plan")
        for expected, physical in zip(
            self.plan.microbatches, self.physical_microbatches, strict=True
        ):
            if physical.meta != expected:
                raise ValueError(
                    "physical microbatch metadata does not match DSPP plan"
                )
            expected_shape = (1, self.plan.chunk_size)
            for name in ("tokens", "labels", "position_ids", "loss_mask"):
                tensor = getattr(physical, name)
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(
                        f"{name} has shape {tuple(tensor.shape)}; expected {expected_shape}"
                    )

    def rechunk_(self, chunk_size: int) -> None:
        """Rebuild this iteration-owned batch with a different physical shape."""

        if chunk_size == self.plan.chunk_size:
            return
        if self._source_sequences is None:
            raise RuntimeError(
                "DSPP batch cannot be rechunked because source sequences were not retained"
            )
        plan = build_dspp_batch_plan(
            self.plan.sequence_lengths,
            chunk_size=chunk_size,
            pack_residuals=self._pack_residuals,
            packing_strategy=self._packing_strategy,
            validate=False,
        )
        self.plan = plan
        self.physical_microbatches = plan.materialize(
            self._source_sequences,
            pad_token_id=self._pad_token_id,
            device=self._source_sequences[0].device,
        )

    def release_rechunk_source_(self) -> None:
        """Release source views once the iteration's chunk size is final."""

        self._source_sequences = None


def build_dspp_training_batch(
    tokens_with_labels: torch.Tensor,
    sequence_lengths: Sequence[int] | torch.Tensor,
    *,
    chunk_size: int,
    pad_token_id: int = 0,
    max_sequence_length: Optional[int] = None,
    cost_fn: Optional[CostFn] = None,
    pack_residuals: bool = True,
    packing_strategy: str = "baseline",
    validate: bool = False,
    retain_rechunk_source: bool = False,
) -> DsppTrainingBatch:
    """Build a DSPP training batch without inferring length from token values.

    ``sequence_lengths`` contains raw document lengths, including the final
    label token.  Explicit lengths are required because a pad-token value may
    also occur legitimately inside a document.
    """

    if tokens_with_labels.ndim != 2:
        raise ValueError("tokens_with_labels must have shape [batch, sequence]")
    lengths = torch.as_tensor(sequence_lengths, dtype=torch.long, device="cpu")
    if lengths.ndim != 1 or lengths.numel() != tokens_with_labels.size(0):
        raise ValueError("sequence_lengths must contain one entry per batch row")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    row_capacity = tokens_with_labels.size(1)
    raw_sequences = []
    training_lengths = []
    for row, raw_length_tensor in zip(tokens_with_labels, lengths, strict=True):
        raw_length = int(raw_length_tensor.item())
        if max_sequence_length is not None:
            raw_length = min(raw_length, int(max_sequence_length) + 1)
        if raw_length > row_capacity:
            raise ValueError(
                f"sequence length {raw_length} exceeds padded row capacity {row_capacity}"
            )
        if raw_length < 2:
            # A document without a token/label pair contributes no LM target.
            continue
        raw_sequences.append(row[:raw_length])
        training_lengths.append(raw_length - 1)

    if not raw_sequences:
        raise ValueError("DSPP batch contains no sequence with a training target")

    plan = build_dspp_batch_plan(
        training_lengths,
        chunk_size=chunk_size,
        cost_fn=cost_fn,
        pack_residuals=pack_residuals,
        packing_strategy=packing_strategy,
        validate=False,
    )
    physical = plan.materialize(
        raw_sequences,
        pad_token_id=pad_token_id,
        device=tokens_with_labels.device,
    )
    result = DsppTrainingBatch(
        plan=plan,
        physical_microbatches=physical,
        _source_sequences=(tuple(raw_sequences) if retain_rechunk_source else None),
        _pad_token_id=pad_token_id,
        _pack_residuals=pack_residuals,
        _packing_strategy=packing_strategy,
    )
    if validate:
        result.validate()
    return result
