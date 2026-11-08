"""Q/O memory counts actual owner maxima, not a blanket division by CP."""
from types import SimpleNamespace

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.pipeline_parallel.dspp_memory_lifetimes import attention_token_counts,stage_attention_sizes
from megatron.core.pipeline_parallel.dspp_memory_counts import history_state_bytes


def args(impl='query-out'):
    return SimpleNamespace(context_parallel_size=3,kaimm_context_parallel_impl=impl,
        dspp_cp_layout='hybrid',num_attention_heads=4,num_query_groups=2,
        group_query_attention=True,kv_channels=128,hidden_size=512,
        num_layers=6,dspp_v_layer_layout='balanced')


def test_private_sequences_use_maximum_owner_not_average():
    meta=build_dspp_batch_plan([17,5,20],chunk_size=66).microbatches[0]
    keys,history,items=attention_token_counts(args(),meta)
    assert keys==20 and history==0 and sorted(items)==[5,17,20]
    assert attention_token_counts(args('key-value'),meta)[:2]==(42,0)


def test_history_shards_and_full_query_h_state():
    plan=build_dspp_batch_plan([133],chunk_size=66)
    middle=next(m for m in plan.microbatches if m.items[0].segment_id==1)
    tail=next(m for m in plan.microbatches if m.items[0].segment_id==2)
    assert attention_token_counts(args(),middle)==(44,22,(22,))
    assert attention_token_counts(args(),tail)==(45,44,(1,))
    native=stage_attention_sizes(args('key-value'),3,0,0,66)
    qo=stage_attention_sizes(args(),3,0,0,66)
    assert qo[0]*3==native[0] and qo[1]==native[1]
    _,created,_=history_state_bytes(args(),3,0,0,middle)
    # One layer: retained global Q/O/dO, plus FP32 LSE/delta. Do not divide by CP.
    assert created >= 3*66*4*128*2
