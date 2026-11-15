"""P2P buffer accounting used by memory planning."""

def dspp_communication_buffer_count(
    *, neighbor_count: int, recv_lookahead: int, send_inflight_depth: int
) -> int:
    """Conservative per-rank P2P payload reserve for adaptive chunking."""
    if neighbor_count < 0 or recv_lookahead < 0 or send_inflight_depth <= 0:
        raise ValueError(
            "DSPP neighbor count/lookahead must be non-negative and send depth positive"
        )
    # Reserve every outgoing in-flight send, one receive at the current
    # dependency, and every future preposted receive per neighbor.
    return neighbor_count * (send_inflight_depth + 1 + recv_lookahead)
