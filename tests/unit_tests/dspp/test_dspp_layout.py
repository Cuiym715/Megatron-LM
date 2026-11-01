import pytest

from megatron.core.dspp_layout import (
    logical_layer_ranges,
    physical_layer_count,
    physical_layer_range,
)


def test_endpoint_3_2_covers_pp8_61_exactly():
    ranges = logical_layer_ranges(61, 8, 'endpoint-3-2')
    assert [end - begin for begin, end in ranges] == [3] + [4] * 14 + [2]
    assert ranges[0] == (0, 3)
    assert ranges[-1] == (59, 61)
    assert [layer for begin, end in ranges for layer in range(begin, end)] == list(range(61))


def test_endpoint_3_2_physical_v_mapping_keeps_both_endpoint_chunks():
    assert physical_layer_range(61, 8, 'endpoint-3-2', 0, 0) == (0, 3)
    assert physical_layer_range(61, 8, 'endpoint-3-2', 0, 1) == (59, 61)
    assert physical_layer_count(61, 8, 'endpoint-3-2', 0) == 5
    assert [physical_layer_count(61, 8, 'endpoint-3-2', stage) for stage in range(1, 8)] == [8] * 7


@pytest.mark.parametrize('num_layers,pipeline_size', [(60, 8), (61, 4), (63, 8)])
def test_endpoint_3_2_rejects_other_shapes(num_layers, pipeline_size):
    with pytest.raises(ValueError):
        logical_layer_ranges(num_layers, pipeline_size, 'endpoint-3-2')


def test_existing_layouts_are_unchanged():
    assert [end - begin for begin, end in logical_layer_ranges(63, 8, 'legacy-output-slot')] == [4] * 15 + [3]
    assert [end - begin for begin, end in logical_layer_ranges(64, 8, 'balanced')] == [4] * 16
