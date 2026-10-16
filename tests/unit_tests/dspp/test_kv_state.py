import torch

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.kv_cache.dspp_packed_attention import DsppSequenceKVState


def test_all_independent_pack_reuses_kv_and_grad_tensors():
    plan = build_dspp_batch_plan([3, 2], chunk_size=8)
    assert len(plan.microbatches) == 1
    meta = plan.microbatches[0]
    assert len(meta.items) == 2

    state = DsppSequenceKVState(validate_runtime=True)
    current_key = torch.randn(5, 2, 16)
    current_value = torch.randn_like(current_key)
    assembled_key, assembled_value, layouts = state.build_key_value(
        0, meta, current_key, current_value
    )

    assert assembled_key is current_key
    assert assembled_value is current_value
    assembled_key_grad = torch.randn_like(assembled_key)
    assembled_value_grad = torch.randn_like(assembled_value)
    current_key_grad, current_value_grad = state.distribute_key_value_grads(
        0,
        layouts,
        assembled_key_grad,
        assembled_value_grad,
        current_key.shape,
    )
    assert current_key_grad is assembled_key_grad
    assert current_value_grad is assembled_value_grad
    state.finish_iteration()


def test_first_long_segment_still_waits_for_history_gradients():
    plan = build_dspp_batch_plan([10], chunk_size=8)
    first_meta, final_meta = plan.microbatches
    state = DsppSequenceKVState(validate_runtime=True)

    first_key = torch.randn(8, 2, 16)
    first_value = torch.randn_like(first_key)
    assembled_first_key, assembled_first_value, first_layouts = state.build_key_value(
        0, first_meta, first_key, first_value
    )
    # The first long segment may be represented by a full-range view, but it
    # must not be mistaken for an all-final pack during backward routing.
    assert assembled_first_key.data_ptr() == first_key.data_ptr()
    assert assembled_first_value.data_ptr() == first_value.data_ptr()

    final_key = torch.randn(2, 2, 16)
    final_value = torch.randn_like(final_key)
    assembled_final_key, assembled_final_value, final_layouts = state.build_key_value(
        0, final_meta, final_key, final_value
    )
    assert assembled_final_key.size(0) == 10
    assert assembled_final_value.size(0) == 10
    assert assembled_final_key is not final_key
    assert assembled_final_value is not final_value

    final_key_grad = torch.randn_like(assembled_final_key)
    final_value_grad = torch.randn_like(assembled_final_value)
    current_final_key_grad, current_final_value_grad = state.distribute_key_value_grads(
        0,
        final_layouts,
        final_key_grad,
        final_value_grad,
        final_key.shape,
    )
    torch.testing.assert_close(current_final_key_grad, final_key_grad[8:])
    torch.testing.assert_close(current_final_value_grad, final_value_grad[8:])

    first_local_key_grad = torch.randn_like(first_key)
    first_local_value_grad = torch.randn_like(first_value)
    current_first_key_grad, current_first_value_grad = state.distribute_key_value_grads(
        0,
        first_layouts,
        first_local_key_grad,
        first_local_value_grad,
        first_key.shape,
    )
    torch.testing.assert_close(
        current_first_key_grad, first_local_key_grad + final_key_grad[:8]
    )
    torch.testing.assert_close(
        current_first_value_grad, first_local_value_grad + final_value_grad[:8]
    )
    state.finish_iteration()
