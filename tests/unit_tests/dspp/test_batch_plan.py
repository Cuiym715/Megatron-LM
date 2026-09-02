from dataclasses import replace

import pytest
import torch

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan


def _signature(plan):
    return [
        [
            (
                item.sequence_id,
                item.segment_id,
                item.token_offset,
                item.token_length,
            )
            for item in microbatch.items
        ]
        for microbatch in plan.microbatches
    ]


def test_build_plan_slices_and_packs_variable_sequences():
    plan = build_dspp_batch_plan([18, 8, 3, 2], chunk_size=8)

    assert _signature(plan) == [
        [(0, 0, 0, 8)],
        [(0, 1, 8, 8)],
        [(1, 0, 0, 8)],
        [(0, 2, 16, 2), (2, 0, 0, 3), (3, 0, 0, 2)],
    ]
    residual = plan.microbatches[-1]
    assert residual.valid_token_count == 7
    assert residual.padding_token_count == 1
    assert residual.cu_seqlens_q == (0, 2, 5, 7)
    assert residual.cu_seqlens_k == (0, 18, 21, 23)
    assert residual.items[0].is_tail
    assert all(item.is_short_sequence for item in residual.items[1:])
    plan.validate()


def test_plan_is_deterministic_and_balances_residual_cost():
    lengths = [11, 10, 5, 4, 3, 2]

    first = build_dspp_batch_plan(lengths, chunk_size=8)
    second = build_dspp_batch_plan(lengths, chunk_size=8)

    assert _signature(first) == _signature(second)
    residuals = [
        mb for mb in first.microbatches if any(i.token_length < 8 for i in mb.items)
    ]
    assert len(residuals) == 3
    assert all(sum(item.has_history for item in mb.items) <= 1 for mb in residuals)
    assert all(mb.valid_token_count <= 8 for mb in residuals)


@pytest.mark.parametrize(
    "lengths",
    [
        [1],
        [1, 2, 3],
        [8],
        [9],
        [17, 15, 7, 1],
    ],
)
def test_plan_edge_case_matrix(lengths):
    plan = build_dspp_batch_plan(lengths, chunk_size=8)

    plan.validate()
    assert sum(mb.valid_token_count for mb in plan.microbatches) == sum(lengths)
    assert all(mb.valid_token_count <= 8 for mb in plan.microbatches)


def test_materialize_preserves_tokens_labels_positions_and_loss_count():
    lengths = [18, 8, 3, 2]
    sequences = [
        torch.arange(100 * seq_id, 100 * seq_id + length + 1)
        for seq_id, length in enumerate(lengths)
    ]
    plan = build_dspp_batch_plan(lengths, chunk_size=8)

    batches = plan.materialize(sequences, pad_token_id=999)

    seen = {sequence_id: [] for sequence_id in range(len(lengths))}
    total_loss_tokens = 0
    for batch in batches:
        cursor = 0
        for item in batch.meta.items:
            end = cursor + item.token_length
            source = sequences[item.sequence_id]
            assert torch.equal(
                batch.tokens[0, cursor:end],
                source[item.token_offset : item.token_offset + item.token_length],
            )
            assert torch.equal(
                batch.labels[0, cursor:end],
                source[
                    item.token_offset + 1 : item.token_offset + item.token_length + 1
                ],
            )
            assert torch.equal(
                batch.position_ids[0, cursor:end],
                torch.arange(item.token_offset, item.token_offset + item.token_length),
            )
            seen[item.sequence_id].extend(batch.tokens[0, cursor:end].tolist())
            cursor = end
        assert torch.all(batch.loss_mask[0, :cursor] == 1)
        assert torch.all(batch.loss_mask[0, cursor:] == 0)
        assert torch.all(batch.tokens[0, cursor:] == 999)
        total_loss_tokens += int(batch.loss_mask.sum())

    assert total_loss_tokens == sum(lengths)
    for sequence_id, length in enumerate(lengths):
        assert seen[sequence_id] == sequences[sequence_id][:-1].tolist()


def test_plan_validator_rejects_broken_segment_chain():
    plan = build_dspp_batch_plan([10, 3], chunk_size=8)
    broken_item = replace(plan.microbatches[-1].items[0], token_offset=7)
    broken_microbatch = replace(
        plan.microbatches[-1],
        items=(broken_item,) + plan.microbatches[-1].items[1:],
    )
    broken_plan = replace(
        plan,
        microbatches=plan.microbatches[:-1] + (broken_microbatch,),
    )

    with pytest.raises(ValueError, match="broken segment chain"):
        broken_plan.validate()


@pytest.mark.parametrize("lengths", [[], [0], [3, -1]])
def test_plan_rejects_invalid_lengths(lengths):
    with pytest.raises(ValueError):
        build_dspp_batch_plan(lengths, chunk_size=8)
