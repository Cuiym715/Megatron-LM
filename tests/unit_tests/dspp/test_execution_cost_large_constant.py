import math

from megatron.core.datasets.dspp_execution_cost import DsppExecutionCostModel


def test_nonnegative_fit_with_repeated_large_constant_features():
    row = dict(stage=0, kind='F', sequence_chunk_size=2048, duration_ms=2.5,
        items=[dict(sequence_id=0,segment_id=15,segment_count=16,
            token_offset=30720,token_length=2048,sequence_length=32768)])
    model = DsppExecutionCostModel.from_records([row]*32, extended_features=True)
    fitted = model.fitted[(0,'F')]
    assert all(math.isfinite(v) and v>=0 for v in fitted.coefficients)
    assert fitted.residual_p95_ms < 1e-8
