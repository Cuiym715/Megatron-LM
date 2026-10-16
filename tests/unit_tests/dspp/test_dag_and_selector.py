import json

import pytest

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.datasets.dspp_chunk_selector import (
    load_dspp_length_trace,
    select_global_fixed_chunk,
)
from megatron.core.datasets.dspp_dag import plan_dspp_iteration
from megatron.core.datasets.dspp_execution_cost import DsppExecutionCostModel
from megatron.core.datasets.dspp_p2p import DsppP2PCostModel


def test_p2p_profile_interpolates_directed_points():
    model = DsppP2PCostModel.from_records([
        {"bytes": 100, "duration_ms": 1.0, "direction": "0->1", "overlap": "compute"},
        {"bytes": 200, "duration_ms": 2.0, "direction": "0->1", "overlap": "compute"},
    ])
    value, source = model.predict(150, direction="0->1", overlap="compute")
    assert source == "profile-interpolated"
    assert value == 1.5


def test_weighted_dag_keeps_compute_and_p2p_as_separate_lanes():
    plan = plan_dspp_iteration(
        ((768, 512, 192, 128), (768, 512, 192, 128)),
        chunk_size=256,
        pipeline_size=3,
        hidden_size=8,
        dtype_bytes=2,
    )
    assert plan.predicted_makespan_ms > 0
    assert any(node.kind == "P2P" for node in plan.nodes)
    assert len(plan.stage_compute_ms) == 3
    assert plan.predicted_p95_makespan_ms >= plan.predicted_makespan_ms
    assert plan.p2p_source == "ignored"
    assert all(node.duration_ms == 0 for node in plan.nodes if node.kind == "P2P")


def test_weighted_dag_analytical_p2p_assigns_nonzero_payload_cost():
    plan = plan_dspp_iteration(
        ((768, 512, 192, 128), (768, 512, 192, 128)),
        chunk_size=256,
        pipeline_size=3,
        hidden_size=8,
        dtype_bytes=2,
        communication_cost_mode="analytical",
    )
    p2p = [node for node in plan.nodes if node.kind == "P2P"]
    assert p2p
    assert all(node.duration_ms > 0 for node in p2p)


def test_fixed_selector_reports_oracle_regret():
    result = select_global_fixed_chunk(
        [
            ((768, 512, 192, 128),),
            ((512, 384, 256, 128),),
        ],
        candidate_chunks=(256, 512),
        pipeline_size=3,
    )
    assert result.selected_chunk_size in (256, 512)
    assert len(result.oracle_choices) == 2
    assert result.regret >= 0.0


def test_global_selector_rejects_uncovered_profile_shapes(tmp_path):
    plan = build_dspp_batch_plan([8], chunk_size=8)
    records = []
    meta = plan.microbatches[0]
    items = [item.__dict__ for item in meta.items]
    for stage in range(3):
        for kind, duration in (("F0", 1.0), ("B0", 2.0), ("W0", 0.5)):
            records.append({
                "stage": stage,
                "kind": kind,
                "chunk": 8,
                "signature": "c8|8:8",
                "items": items,
                "duration_ms": duration,
            })
    profile = tmp_path / "profile.json"
    DsppExecutionCostModel.from_records(records).save(profile)
    with pytest.raises(ValueError, match="no chunk candidate"):
        select_global_fixed_chunk(
            [((16,),)],
            candidate_chunks=(16,),
            pipeline_size=3,
            execution_profile_path=str(profile),
            require_profile_coverage=True,
        )


def test_text_length_trace_matches_training_grouping(tmp_path):
    path = tmp_path / "lengths.txt"
    path.write_text("9\n5\n7\n3\n11\n9\n5\n3\n", encoding="utf-8")
    trace = load_dspp_length_trace(
        path,
        sequence_group_size=2,
        groups_per_iteration=2,
        raw_lengths_include_label=True,
    )
    assert trace == (((8, 4), (6, 2)), ((10, 8), (4, 2)))
