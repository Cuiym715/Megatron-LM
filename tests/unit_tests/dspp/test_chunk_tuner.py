import pytest
import torch

from megatron.core.datasets.dspp_chunk_tuner import (
    estimate_dspp_chunk_layout,
    select_dspp_chunk_size,
    select_dspp_structural_chunk_size,
    update_activation_bytes_per_live_token,
)
from megatron.core.datasets.dspp_chunk_selector import (
    load_profiled_chunk_bank,
    select_per_iteration_chunks,
)
from megatron.core.datasets.dspp_training import build_dspp_training_batch


SATURATED_LENGTH_GROUPS = (
    (768, 512, 192, 128),
    (768, 512, 192, 128),
)


def test_layout_estimate_uses_real_slice_v_activation_liveness():
    layout = estimate_dspp_chunk_layout(
        SATURATED_LENGTH_GROUPS,
        chunk_size=256,
        pipeline_size=3,
    )

    assert layout.max_segment_count == 3
    assert layout.physical_microbatches == 14
    assert layout.peak_live_slots == 10
    assert layout.peak_live_tokens == 2560
    assert layout.physical_token_capacity == 3584


def test_first_iteration_uses_exact_safe_bootstrap_even_below_default_minimum():
    selection = select_dspp_chunk_size(
        ((96, 64), (80, 32)),
        pipeline_size=3,
        bootstrap_chunk_size=32,
        min_chunk_size=256,
        max_chunk_size=96,
        alignment=256,
        activation_bytes_per_live_token=None,
        activation_budget_bytes=10**9,
        memory_safety_factor=1.2,
    )

    assert selection.chunk_size == 32
    assert selection.reason == "bootstrap-no-activation-sample"


def test_selection_uses_largest_memory_feasible_chunk():
    selections = {}
    for chunk_size in (256, 512, 768):
        layout = estimate_dspp_chunk_layout(
            SATURATED_LENGTH_GROUPS,
            chunk_size=chunk_size,
            pipeline_size=3,
        )
        selections[chunk_size] = (
            layout.max_segment_count,
            layout.physical_token_capacity,
        )

    assert selections == {
        256: (3, 3584),
        512: (2, 4096),
        768: (1, 4608),
    }
    selection = select_dspp_chunk_size(
        SATURATED_LENGTH_GROUPS,
        pipeline_size=3,
        bootstrap_chunk_size=256,
        min_chunk_size=256,
        max_chunk_size=768,
        alignment=256,
        activation_bytes_per_live_token=1000.0,
        activation_budget_bytes=10**9,
        memory_safety_factor=1.2,
    )

    assert selection.chunk_size == 768
    assert selection.layout.max_segment_count == 1

    constrained = select_dspp_chunk_size(
        SATURATED_LENGTH_GROUPS,
        pipeline_size=3,
        bootstrap_chunk_size=256,
        min_chunk_size=256,
        max_chunk_size=768,
        alignment=256,
        activation_bytes_per_live_token=1000.0,
        activation_budget_bytes=5_000_000,
        memory_safety_factor=1.2,
    )
    assert constrained.chunk_size == 512


def test_longest_length_uses_alignment_without_power_of_two_rounding():
    selection = select_dspp_chunk_size(
        ((700,),),
        pipeline_size=1,
        bootstrap_chunk_size=256,
        min_chunk_size=256,
        max_chunk_size=1024,
        alignment=256,
        activation_bytes_per_live_token=1000.0,
        activation_budget_bytes=10**9,
        memory_safety_factor=1.2,
    )

    assert selection.chunk_size == 768
    assert selection.layout.max_segment_count == 1


def test_selection_reports_when_even_smallest_candidate_exceeds_memory():
    selection = select_dspp_chunk_size(
        SATURATED_LENGTH_GROUPS,
        pipeline_size=3,
        bootstrap_chunk_size=256,
        min_chunk_size=256,
        max_chunk_size=768,
        alignment=256,
        activation_bytes_per_live_token=1000.0,
        activation_budget_bytes=100_000,
        memory_safety_factor=1.2,
    )

    assert selection.chunk_size == 0
    assert selection.reason == "no-memory-feasible-chunk"


def test_activation_memory_model_keeps_high_water_mark():
    first = update_activation_bytes_per_live_token(
        None,
        baseline_allocated_bytes=100,
        peak_allocated_bytes=1100,
        peak_live_tokens=10,
    )
    second = update_activation_bytes_per_live_token(
        first,
        baseline_allocated_bytes=100,
        peak_allocated_bytes=2100,
        peak_live_tokens=10,
        smoothing=0.25,
    )

    assert first == 100.0
    assert second == 200.0


def test_selection_reserves_bounded_communication_window():
    selection = select_dspp_chunk_size(
        SATURATED_LENGTH_GROUPS,
        pipeline_size=3,
        bootstrap_chunk_size=256,
        min_chunk_size=256,
        max_chunk_size=768,
        alignment=256,
        activation_bytes_per_live_token=1000.0,
        activation_budget_bytes=10**9,
        memory_safety_factor=1.2,
        communication_bytes_per_chunk_token=4096,
    )

    assert selection.chunk_size == 768
    assert selection.estimated_communication_bytes == 768 * 4096
    assert selection.estimated_peak_dynamic_bytes == (
        selection.estimated_peak_activation_bytes
        + selection.estimated_communication_bytes
    )


def test_structural_selector_stays_inside_saturation_window():
    selection = select_dspp_structural_chunk_size(
        SATURATED_LENGTH_GROUPS,
        pipeline_size=3,
        bootstrap_chunk_size=256,
        min_chunk_size=256,
        max_chunk_size=768,
        alignment=256,
        activation_bytes_per_live_token=1000.0,
        activation_budget_bytes=10**9,
        memory_safety_factor=1.2,
        saturation_chunk_size=512,
    )
    assert selection.chunk_size >= 512
    assert selection.reason == "structure-aware-saturation-window"
    assert selection.layout.chunk_size == selection.chunk_size


def test_training_batch_rechunk_preserves_tokens_labels_and_loss_mask():
    raw = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15, 0],
            [20, 21, 22, 23, 0, 0, 0],
        ]
    )
    batch = build_dspp_training_batch(
        raw,
        [6, 4],
        chunk_size=4,
        retain_rechunk_source=True,
        validate=True,
    )

    batch.rechunk_(2)
    batch.validate()
    assert batch.plan.chunk_size == 2
    assert batch.plan.sequence_lengths == (5, 3)
    assert batch.valid_token_count == 8
    assert int(batch.loss_mask.sum()) == 8
    for physical in batch.physical_microbatches:
        physical_begin = 0
        for item in physical.meta.items:
            begin = item.token_offset
            end = begin + item.token_length
            physical_end = physical_begin + item.token_length
            expected = raw[item.sequence_id, begin:end]
            assert torch.equal(
                physical.tokens[0, physical_begin:physical_end], expected
            )
            assert torch.equal(
                physical.labels[0, physical_begin:physical_end],
                raw[item.sequence_id, begin + 1 : end + 1],
            )
            physical_begin = physical_end

    batch.release_rechunk_source_()
    with pytest.raises(RuntimeError, match="source sequences were not retained"):
        batch.rechunk_(3)


def test_per_iteration_selector_evaluates_only_menu_and_records_overhead():
    decisions = select_per_iteration_chunks(
        (
            ((768, 512, 192, 128),),
            ((2048, 1024, 512, 256),),
        ),
        candidate_chunks=(256, 512),
        pipeline_size=1,
    )
    assert len(decisions) == 2
    assert all(decision.selected_chunk_size in (256, 512) for decision in decisions)
    assert all(decision.decision_ms >= 0.0 for decision in decisions)
    assert all(len(decision.candidates) == 2 for decision in decisions)


def test_profile_bank_loads_rejected_measured_capacities(tmp_path):
    path = tmp_path / "menu.json"
    path.write_text(
        '{"candidates":[512],"scores":['
        '{"capacity":256,"retained":false},'
        '{"capacity":512,"retained":true}]}'
    )

    bank = load_profiled_chunk_bank(path)

    assert tuple(entry.capacity for entry in bank) == (256, 512)
    assert not bank[0].robust_plateau
    assert bank[1].robust_plateau


def test_branch_and_bound_matches_exhaustive_without_fixed_candidate_count():
    trace = (
        ((768, 512, 192, 128), (1024, 384, 256, 128)),
        ((2048, 1024, 512, 256), (1536, 768, 384, 192)),
    )
    kwargs = dict(
        candidate_chunks=(128, 256, 384, 512, 768),
        pipeline_size=3,
    )

    bounded = select_per_iteration_chunks(
        trace, selection_strategy="branch-and-bound", **kwargs
    )
    exhaustive = select_per_iteration_chunks(
        trace, selection_strategy="exhaustive", **kwargs
    )

    assert [item.selected_chunk_size for item in bounded] == [
        item.selected_chunk_size for item in exhaustive
    ]
    assert all(
        sum(candidate.fully_planned for candidate in decision.candidates)
        <= len(kwargs["candidate_chunks"])
        for decision in bounded
    )
