import json

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.datasets.dspp_execution_cost import (
    DsppExecutionCostModel,
    meta_from_timeline_record,
)


def _records():
    plan = build_dspp_batch_plan([8, 4], chunk_size=8)
    rows = []
    for stage in (0, 1):
        for physical_id, meta in enumerate(plan.microbatches):
            items = [
                {
                    "sequence_id": item.sequence_id,
                    "segment_id": item.segment_id,
                    "segment_count": item.segment_count,
                    "token_offset": item.token_offset,
                    "token_length": item.token_length,
                    "sequence_length": item.sequence_length,
                }
                for item in meta.items
            ]
            signature = "c8|" + ";".join(
                f"{item['token_length']}:{item['token_offset'] + item['token_length']}"
                for item in items
            )
            for kind, scale in (("F0", 1.0), ("B0", 2.0), ("W0", 0.5)):
                rows.append(
                    {
                        "stage": stage,
                        "kind": kind,
                        "chunk": 8,
                        "physical_microbatch": physical_id,
                        "signature": signature,
                        "items": items,
                        "duration_ms": scale * (1.0 + meta.valid_token_count / 8.0),
                    }
                )
    return rows


def test_execution_model_fits_and_roundtrips(tmp_path):
    model = DsppExecutionCostModel.from_records(_records())
    assert model.fitted
    plan = build_dspp_batch_plan([8, 4], chunk_size=8)
    value, source, error = model.predict(0, "F", plan.microbatches[0])
    assert value > 0
    assert source in {"exact-profile", "fitted-profile"}
    assert error >= 0

    path = tmp_path / "model.json"
    model.save(path)
    restored = DsppExecutionCostModel.from_json(path)
    assert restored.to_dict()["version"] == 1


def test_execution_model_task_workload_falls_back_without_coverage():
    model = DsppExecutionCostModel.empty()
    plan = build_dspp_batch_plan([8], chunk_size=8)
    value, source, error = model.predict_task_workload(
        plan.microbatches[0], pipeline_size=2, fallback=123.0
    )
    assert value == 123.0
    assert source == "metadata-fallback"
    assert error == 0.0


def test_timeline_vpp_chunk_is_not_sequence_chunk_size():
    record = _records()[0]
    record["chunk"] = 0
    meta = meta_from_timeline_record(record)
    assert meta.chunk_size == 8
    assert meta.padding_token_count >= 0


def test_compute_profile_reuses_balanced_stage_and_indexes_w_by_capacity(tmp_path):
    payload = {
        "metadata": {"stage_count": 1},
        "records": [
            {
                "stage": 0, "operation": "F", "capacity": 8,
                "query_lengths": [8], "history_lengths": [0],
                "median_ms": 2.0, "p95_ms": 2.5,
            },
            {
                "stage": 0, "operation": "B", "capacity": 8,
                "query_lengths": [8], "history_lengths": [0],
                "median_ms": 4.0, "p95_ms": 4.5,
            },
            {
                "stage": 0, "operation": "W", "capacity": 8,
                "query_lengths": [8], "history_lengths": [0],
                "median_ms": 1.0, "p95_ms": 1.5,
            },
        ],
    }
    path = tmp_path / "compute_profile.json"
    path.write_text(json.dumps(payload))
    model = DsppExecutionCostModel.from_json(path)
    meta = build_dspp_batch_plan([5, 3], chunk_size=8).microbatches[0]
    w = model.predict_detailed(2, "W", meta)
    assert w.duration_ms == 1.0
    assert w.source == "exact-capacity-profile"
    # F/B may interpolate because the packed runtime shape differs, but the
    # homogeneous stage must use the measured model rather than metadata-only.
    f = model.predict_detailed(2, "F", meta)
    assert f.source != "metadata-fallback"


def test_stage_op_calibration_preserves_shape_profile_and_w_capacity():
    base = DsppExecutionCostModel.from_records(_records())
    timeline = _records()
    calibrated = base.with_stage_op_calibration(timeline)
    plan = build_dspp_batch_plan([8, 4], chunk_size=8)
    value = calibrated.predict_detailed(0, "F", plan.microbatches[0]).duration_ms
    assert value > 0
    assert calibrated.stage_op_calibrations
