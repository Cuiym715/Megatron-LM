from megatron.core.pipeline_parallel.dspp_progress import (
    dspp_communication_buffer_count,
)


def test_receive_lookahead_is_reserved_per_neighbor():
    assert dspp_communication_buffer_count(
        neighbor_count=2, recv_lookahead=0, send_inflight_depth=1
    ) == 4
    assert dspp_communication_buffer_count(
        neighbor_count=2, recv_lookahead=1, send_inflight_depth=1
    ) == 6
    assert dspp_communication_buffer_count(
        neighbor_count=2, recv_lookahead=4, send_inflight_depth=4
    ) == 18
