"""CP-QO partial-KV derivatives use the FULL attention O/LSE."""
import pytest
import torch

from megatron.core.kv_cache.fa_split import (
    prepare_partial_backward, backward_dq, backward_dq_current, backward_dkv_range,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('kv_heads',(4,2,1))
@pytest.mark.parametrize('local_k',(9,47))
def test_noncausal_partial_global_normalizer(kv_heads,local_k):
    torch.manual_seed(321+kv_heads+local_k)
    q=torch.randn(31,4,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(97,kv_heads,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k);do=torch.randn_like(q)
    ratio=4//kv_heads
    q64=q.double().transpose(0,1)
    k64=k.double().repeat_interleave(ratio,1).transpose(0,1)
    v64=v.double().repeat_interleave(ratio,1).transpose(0,1)
    scores=q64@k64.transpose(1,2)*128**-.5
    probability=scores.softmax(-1)
    out64=probability@v64
    out=out64.transpose(0,1).to(q.dtype).contiguous()
    lse=scores.logsumexp(-1).float().contiguous()
    start=17
    ki=k[start:start+local_k].contiguous();vi=v[start:start+local_k].contiguous()
    state=prepare_partial_backward(q,ki,vi,out,do,lse)
    from megatron.core.kv_cache.fa_split_build import load_extension
    configured=load_extension().dispatch_counters()
    dq=backward_dq(state)
    dk,dv=backward_dkv_range(state,0,local_k)
    joint=backward_dq_current(state,current_begin=0)
    do64=do.double().transpose(0,1)
    p=probability[:,:,start:start+local_k]
    ds=p*(do64@v64[:,start:start+local_k].transpose(1,2)-(do64*out64).sum(-1,keepdim=True))
    expected_q=(ds@k64[:,start:start+local_k]*128**-.5).transpose(0,1)
    expected_k=(ds.transpose(1,2)@q64*128**-.5).transpose(0,1).reshape(local_k,kv_heads,ratio,128).sum(2)
    expected_v=(p.transpose(1,2)@do64).transpose(0,1).reshape(local_k,kv_heads,ratio,128).sum(2)
    for actual,expected in zip((dq,dk,dv),(expected_q,expected_k,expected_v)):
        torch.testing.assert_close(actual.double(),expected,atol=.03,rtol=.03)
        assert (actual.double()-expected).norm()/expected.norm() < .02
    assert load_extension().dispatch_counters()==configured
    for actual,expected in zip(joint,(dq,dk,dv)):
        torch.testing.assert_close(actual,expected,atol=.004,rtol=.01)
