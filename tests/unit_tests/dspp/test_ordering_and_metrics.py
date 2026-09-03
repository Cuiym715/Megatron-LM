import json

import torch

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.datasets.dspp_ordering import (
    build_dspp_ordering_plan,
    render_dspp_microbatch_report,
    task_signature,
)
from megatron.core.datasets.dspp_training import DsppTrainingBatch
from megatron.core.pipeline_parallel.dspp_metrics import summarize_dspp_metrics
from megatron.core.pipeline_parallel.dspp_timeline import (
    build_profile_costs,
    render_timeline_svg,
    summarize_timeline,
)
from megatron.core.pipeline_parallel.slice_v import build_slice_v_schedule


def _batch(lengths, chunk_size=100):
    plan = build_dspp_batch_plan(lengths, chunk_size=chunk_size, validate=True)
    sequences = [torch.arange(length + 1) for length in lengths]
    return DsppTrainingBatch(plan, plan.materialize(sequences))


def _long_sequence_ids(batch, order):
    result = []
    for logical_id, physical_id in order:
        meta = batch[logical_id].physical_microbatches[physical_id].meta
        long_items = [item for item in meta.items if not item.is_short_sequence]
        if long_items:
            result.append((logical_id, long_items[0].sequence_id))
    return result


def test_ordering_keeps_long_chains_atomic_and_steady_longest_first():
    batches = [_batch([500, 300, 90, 80, 70, 60, 50]), _batch([40, 30])]
    plan = build_dspp_ordering_plan(batches, pp_degree=2)
    assert plan.warmup_task_count == plan.warmup_count

    sequence_order = _long_sequence_ids(batches, plan.entrance_order)
    # Warmup shortage uses the shortest chain (300), then steady state uses
    # the remaining longest chain (500).
    assert sequence_order[:3] == [(0, 1)] * 3
    assert sequence_order[3:8] == [(0, 0)] * 5

    group_by_task = {
        task: group_id
        for group_id, group in enumerate(plan.schedule_groups)
        for task in group
    }
    for logical_id, batch in enumerate(batches):
        for sequence_id, length in enumerate(batch.plan.sequence_lengths):
            if length <= batch.plan.chunk_size:
                continue
            tasks = [
                (logical_id, physical_id)
                for physical_id, physical in enumerate(batch.physical_microbatches)
                if any(
                    item.sequence_id == sequence_id and not item.is_short_sequence
                    for item in physical.meta.items
                )
            ]
            assert len({group_by_task[task] for task in tasks}) == 1
            positions = [plan.entrance_order.index(task) for task in tasks]
            assert positions == sorted(positions)


def test_ordering_uses_sequence_chains_as_slice_v_microbatches():
    batches = [
        _batch([128, 768, 192, 768], chunk_size=256),
        _batch([512, 128, 128, 512], chunk_size=256),
    ]

    plan = build_dspp_ordering_plan(batches, pp_degree=3)
    split_counts = [len(group) for group in plan.schedule_groups]

    assert [len(batch.physical_microbatches) for batch in batches] == [8, 5]
    assert split_counts == [1, 1, 1, 2, 2, 3, 3]
    assert max(split_counts) == 3
    assert plan.warmup_count == 7
    assert len(plan.entrance_order) == 13
    assert len(set(plan.entrance_order)) == 13

    stage_schedules, _ = build_slice_v_schedule(3, len(split_counts), split_counts)
    forward_counts = []
    for stage_schedule in stage_schedules:
        first_b1 = next(
            index
            for index, node in enumerate(stage_schedule)
            if node.kind == "B" and node.chunk == 1
        )
        before_b1 = stage_schedule[:first_b1]
        forward_counts.append(
            sum(node.kind == "F" for node in before_b1)
        )
    assert forward_counts == [10, 10, 10]


def test_ordering_uses_saved_machine_profile(tmp_path):
    batch = _batch([90, 80, 70], chunk_size=100)
    signatures = [task_signature(item.meta) for item in batch.physical_microbatches]
    profile = tmp_path / "costs.json"
    profile.write_text(
        json.dumps({"costs": {signature: index + 1 for index, signature in enumerate(signatures)}})
    )
    plan = build_dspp_ordering_plan(
        [batch], pp_degree=1, profile_path=str(profile)
    )
    assert plan.cost_source == "profile"
    report = render_dspp_microbatch_report(
        [batch], plan, schedule_split_counts=[len(plan.entrance_order)]
    )
    assert "Constructed physical microbatches" in report
    assert "Final pipeline entrance order" in report
    assert "L0/S" in report


def test_timeline_artifacts_and_metric_summary_are_pure_and_compact():
    records = [
        {
            "kind": "F0", "stage": 0, "logical_microbatch": 0,
            "physical_microbatch": 0, "signature": "a", "start_ms": 1.0,
            "end_ms": 3.0, "duration_ms": 2.0,
        },
        {
            "kind": "B0", "stage": 0, "logical_microbatch": 0,
            "physical_microbatch": 0, "signature": "a", "start_ms": 5.0,
            "end_ms": 8.0, "duration_ms": 3.0,
        },
        {
            "kind": "F0", "stage": 1, "logical_microbatch": 0,
            "physical_microbatch": 0, "signature": "a", "start_ms": 2.0,
            "end_ms": 6.0, "duration_ms": 4.0,
        },
    ]
    timeline = summarize_timeline(records)
    assert timeline["critical_span_ms"] == 7.0
    assert timeline["stages"]["0"]["bubble_ms"] == 2.0
    assert timeline["task_duration_ms_by_stage"]["0"]["B0"]["median"] == 3.0
    assert build_profile_costs(records) == {"a": 5.0}
    svg = render_timeline_svg(records, "DSPP test")
    assert "DSPP test" in svg
    assert "One compute lane per stage" in svg

    metrics = summarize_dspp_metrics(
        [
            {
                "iteration_ms": 10.0, "valid_tokens": 80,
                "physical_token_capacity": 100, "schedule_padding_slots": 1,
                "schedule_slots": 10,
            },
            {
                "iteration_ms": 20.0, "valid_tokens": 90,
                "physical_token_capacity": 100, "schedule_padding_slots": 0,
                "schedule_slots": 10,
            },
        ]
    )
    assert metrics["iteration_median_ms"] == 15.0
    assert metrics["packing_utilization"] == 0.85
    assert metrics["schedule_padding_ratio"] == 0.05
