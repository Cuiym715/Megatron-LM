from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets.dspp_training import build_dspp_training_batch
from megatron.data import data_samplers


def test_training_adapter_uses_explicit_lengths_and_keeps_internal_pad_value():
    tokens = torch.tensor(
        [
            [7, 0, 8, 9, 10, 11, 99],
            [20, 21, 22, 99, 99, 99, 99],
        ]
    )
    batch = build_dspp_training_batch(
        tokens,
        [6, 3],
        chunk_size=4,
        pad_token_id=0,
        validate=True,
    )

    assert batch.plan.sequence_lengths == (5, 2)
    assert batch.valid_token_count == 7
    assert int(batch.loss_mask.sum()) == 7
    reconstructed = {0: [], 1: []}
    for physical in batch.physical_microbatches:
        cursor = 0
        for item in physical.meta.items:
            end = cursor + item.token_length
            reconstructed[item.sequence_id].extend(
                physical.tokens[0, cursor:end].tolist()
            )
            cursor = end
    assert reconstructed[0] == [7, 0, 8, 9, 10]
    assert reconstructed[1] == [20, 21]


def test_training_adapter_filters_rows_without_a_training_target():
    tokens = torch.tensor([[4, 0, 0], [5, 6, 7]])
    batch = build_dspp_training_batch(tokens, [1, 3], chunk_size=2)

    assert batch.plan.sequence_lengths == (2,)
    assert batch.valid_token_count == 2


def test_training_adapter_rejects_invalid_length_metadata():
    tokens = torch.ones((2, 4), dtype=torch.long)

    with pytest.raises(ValueError, match="one entry per batch row"):
        build_dspp_training_batch(tokens, [4], chunk_size=2)
    with pytest.raises(ValueError, match="exceeds padded row capacity"):
        build_dspp_training_batch(tokens, [4, 5], chunk_size=2)


def test_dspp_collate_emits_exact_raw_lengths(monkeypatch):
    args = SimpleNamespace(
        variable_seq_pad_token_id=0,
        seq_length=8,
        variable_seq_debug_num_batches=0,
        dspp=True,
    )
    monkeypatch.setattr(data_samplers, "get_args", lambda: args)
    result = data_samplers.variable_seq_collate(
        [{"text": [1, 0, 2, 3]}, {"text": [4, 5]}]
    )

    assert result["text"].tolist() == [[1, 0, 2, 3], [4, 5, 0, 0]]
    assert result["lengths"].tolist() == [4, 2]
