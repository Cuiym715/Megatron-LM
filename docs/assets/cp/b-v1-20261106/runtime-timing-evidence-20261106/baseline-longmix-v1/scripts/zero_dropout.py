"""Exact p=0 residual path; preserve scripted path for configured dropout."""

def install():
    import megatron.model.transformer as module
    native_train=module.bias_dropout_add_fused_train
    native_eval=module.bias_dropout_add_fused_inference
    stats=dict(zero_train=0,zero_eval=0,nonzero_train=0,nonzero_eval=0)
    def train(x,bias,residual,prob):
        if prob==0.0:
            stats['zero_train']+=1
            return residual+(x if bias is None else x+bias)
        stats['nonzero_train']+=1
        return native_train(x,bias,residual,prob)
    def evaluation(x,bias,residual,prob):
        if prob==0.0:
            stats['zero_eval']+=1
            return residual+(x if bias is None else x+bias)
        stats['nonzero_eval']+=1
        return native_eval(x,bias,residual,prob)
    module.bias_dropout_add_fused_train=train
    module.bias_dropout_add_fused_inference=evaluation
    return stats,native_train,train


def probe():
    import torch
    import torch.nn.functional as F
    stats,native,safe=install()
    x=torch.ones((4096,1,2048),device='cuda',dtype=torch.bfloat16,requires_grad=True)
    residual=torch.zeros_like(x,requires_grad=True)
    for _ in range(5):
        warm=native(x,None,residual,.1)
    del warm
    before=torch.cuda.get_rng_state();counts=[]
    for _ in range(64):
        y=native(x,None,residual,0.0)
        counts.append(int((y!=1).sum().item()));del y
    native_rng_changed=not torch.equal(before,torch.cuda.get_rng_state())
    before=torch.cuda.get_rng_state();y=F.dropout(x,p=0.0,training=True)+residual
    eager_bad=int((y!=1).sum().item());del y
    eager_rng_changed=not torch.equal(before,torch.cuda.get_rng_state())
    before=torch.cuda.get_rng_state();y=safe(x,None,residual,0.0)
    safe_bad=int((y!=1).sum().item());del y
    safe_rng_changed=not torch.equal(before,torch.cuda.get_rng_state())
    assert eager_bad==safe_bad==0 and not safe_rng_changed
    # Nonzero dropout must delegate exactly, with an identical RNG state.
    state=torch.cuda.get_rng_state();a=native(x,None,residual,.1)
    torch.cuda.set_rng_state(state);b=safe(x,None,residual,.1)
    assert torch.equal(a,b)
    result=dict(elements_tested=64*x.numel(),native_zero_prob_nonones=sum(counts),native_per_call=counts,
        native_zero_prob_rng_changed=native_rng_changed,eager_zero_prob_nonones=eager_bad,
        eager_zero_prob_rng_changed=eager_rng_changed,safe_nonones=safe_bad,safe_rng_changed=safe_rng_changed,
        nonzero_dropout_delegation_exact=True,torch_version=torch.__version__,cuda_version=torch.version.cuda,
        native_inlined_graph=str(native.inlined_graph))
    del x,residual,a,b
    torch.cuda.empty_cache()
    return result,stats
