from types import SimpleNamespace

import pytest

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.datasets.dspp_theoretical_cost import TheoreticalCostModel


def config(cp=2, impl='query-out', speed=1.):
    return SimpleNamespace(context_parallel_size=cp, kaimm_context_parallel_impl=impl,
        dspp_cp_layout='hybrid', dspp_assumed_tflops=speed, dspp_assumed_bandwidth_gbps=speed,
        num_attention_heads=4, kv_channels=128, num_query_groups=4, group_query_attention=False,
        hidden_size=512, ffn_hidden_size=1024, padded_vocab_size=32000, num_layers=4,
        pipeline_model_parallel_size=2, dspp_v_layer_layout='balanced')


@pytest.mark.parametrize('cp', (2, 3))
@pytest.mark.parametrize('impl', ('key-value', 'query-out'))
def test_owner_pair_conservation(cp, impl):
    plan = build_dspp_batch_plan([145, 17, 5], chunk_size=72)
    model = TheoreticalCostModel(config(cp, impl), {(0, i): m for i, m in enumerate(plan.microbatches)})
    for meta in plan.microbatches:
        pairs, _ = model.rows[id(meta)]
        assert sum(h+c for h, c in pairs) == sum(
            i.token_length*i.token_offset+i.token_length*(i.token_length+1)//2 for i in meta.items)


def test_split_recomputation_and_zero_h_communication():
    plan = build_dspp_batch_plan([145], chunk_size=72)
    metadata = {(0, i): m for i, m in enumerate(plan.microbatches)}
    model = TheoreticalCostModel(config(), metadata)
    twice = TheoreticalCostModel(config(speed=2.), metadata)
    context = dict(stage=1, virtual_chunk=0)
    meta = plan.microbatches[1]
    b, d, h = (model.predict_detailed(context, op, meta) for op in ('B_fused', 'D', 'H'))
    assert d.flops < b.flops < d.flops+h.flops
    assert b.communication_bytes == d.communication_bytes > 0
    assert h.communication_bytes == 0
    for op in ('F', 'B_fused', 'D', 'H', 'W'):
        assert twice.predict_detailed(context, op, meta).duration_ms == pytest.approx(
            model.predict_detailed(context, op, meta).duration_ms/2)


def test_tiny_tail_still_communicates_and_private_shorts_do_not():
    plan = build_dspp_batch_plan([145], chunk_size=72)
    model = TheoreticalCostModel(config(3), {(0, i): m for i, m in enumerate(plan.microbatches)})
    context = dict(stage=1, virtual_chunk=0)
    tail = plan.microbatches[-1]
    assert tail.valid_token_count == 1
    assert model.predict_detailed(context, 'F', tail).communication_bytes > 0
    short = build_dspp_batch_plan([17, 5, 20], chunk_size=72).microbatches[0]
    model = TheoreticalCostModel(config(3), {(0, 0): short})
    assert model.predict_detailed(context, 'F', short).communication_bytes == 0
    assert model.p2p_ms(1048576) == 1048576/1e6
