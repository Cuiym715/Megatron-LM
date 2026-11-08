"""Opt-in CP1 adapter for canonical query tiles at arbitrary chunk boundaries.

Native D/H remains native. The four old aligned capacities retain their query
partition; other capacities may split a tile at a physical chunk boundary.
GPU numerical/performance validation of those new boundaries is still pending.
"""
import inspect
import textwrap

import torch


def query_tiles(length, tile=2048):
    return ((lo,min(length,lo+tile)) for lo in reversed(range(0,length,tile)))


def adapt_backward_source(source):
    source = textwrap.dedent(source).removeprefix('@staticmethod\n')
    for anchor in ('key_grad = torch.empty_like(key)', 'value_grad = torch.empty_like(value)',
                   '_wrapped_flash_attn_varlen_backward(', '    if ctx.valid_tokens == ctx.query_physical_shape[0]:\n'):
        if source.count(anchor) != 1:
            raise ValueError('canonical attention adapter needs updated native backward: '+anchor)
    source = source.replace('key_grad = torch.empty_like(key)','key_grad = torch.empty_like(key, dtype=torch.float32)')
    source = source.replace('value_grad = torch.empty_like(value)','value_grad = torch.empty_like(value, dtype=torch.float32)')
    source = source.replace('_wrapped_flash_attn_varlen_backward(','_adaptive_canonical(ctx,')
    anchor = '    if ctx.valid_tokens == ctx.query_physical_shape[0]:\n'
    return source.replace(anchor,'    current_key_grad = current_key_grad.to(key.dtype)\n'
        '    current_value_grad = current_value_grad.to(value.dtype)\n'+anchor)


def install():
    from . import dspp_packed_attention as packed
    from . import fa_split
    import megatron.model.transformer as transformer
    cls = packed._DsppPackedFlashAttention
    if getattr(cls, '_adaptive_chunk_installed', False):
        return
    original = cls.backward
    native_fa = packed._wrapped_flash_attn_varlen_backward
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    def deterministic(*args, **kwargs):
        args = list(args); args[20] = True
        return native_fa(*args, **kwargs)
    packed._wrapped_flash_attn_varlen_backward = deterministic
    native_current = fa_split.backward_dq_current
    def current(state, **kwargs):
        return native_current(state, deterministic=True)
    fa_split.backward_dq_current = current

    state_cls = packed.DsppSequenceKVState
    native_route = state_cls._route_key_value_grads
    def accumulate(target, key, grad):
        value = grad.float()
        previous = target.get(key)
        target[key] = value.clone() if previous is None else previous+value
    def route(self, layer_idx, layouts, key_grad, value_grad, current_shape, **kwargs):
        result = native_route(self,layer_idx,layouts,key_grad,value_grad,current_shape,**kwargs)
        return None if result is None else (result[0].to(key_grad.dtype),result[1].to(value_grad.dtype))
    state_cls._accumulate = staticmethod(accumulate)
    state_cls._route_key_value_grads = route

    def canonical(ctx, *args):
        dy,q,k,v,o,lse,dq,dk,dv,cuq,cuk,maxq,maxk,dropout,scale,causal,wl,wr,softcap,alibi,det,rng,zero = args
        meta = ctx.meta
        dk.zero_(); dv.zero_()
        offset = 0
        for layout in ctx.layouts:
            entry = ctx.state._entries.get((ctx.layer_idx,layout.item.sequence_id))
            regions = [(segment,length,False) for segment,length in layout.history]
            regions.append((layout.item.segment_id,layout.item.token_length,True))
            for segment,length,is_current in regions:
                if entry is not None and segment in entry.pending_key_grads:
                    pk = entry.pending_key_grads[segment]; pv = entry.pending_value_grads[segment]
                    dk[offset:offset+length].copy_(pk); dv[offset:offset+length].copy_(pv)
                    if is_current:
                        entry.pending_key_grads[segment] = torch.zeros_like(pk)
                        entry.pending_value_grads[segment] = torch.zeros_like(pv)
                    else:
                        del entry.pending_key_grads[segment]; del entry.pending_value_grads[segment]
                offset += length
        for (qa,qb),(ka,kb) in zip(zip(meta.cu_seqlens_q,meta.cu_seqlens_q[1:]),
                                 zip(meta.cu_seqlens_k,meta.cu_seqlens_k[1:])):
            history = kb-ka-(qb-qa)
            for lo,hi in query_tiles(qb-qa):
                q0,q1,k1 = qa+lo,qa+hi,ka+history+hi
                qs,ks,vs = q[q0:q1].contiguous(),k[ka:k1].contiguous(),v[ka:k1].contiguous()
                local_dq,local_dk,local_dv = torch.empty_like(qs),torch.empty_like(ks),torch.empty_like(vs)
                local_q = torch.tensor([0,hi-lo],device=q.device,dtype=torch.int32)
                local_k = torch.tensor([0,k1-ka],device=q.device,dtype=torch.int32)
                native_fa(dy[q0:q1].contiguous(),qs,ks,vs,o[q0:q1].contiguous(),lse[:,q0:q1].contiguous(),
                    local_dq,local_dk,local_dv,local_q,local_k,hi-lo,k1-ka,0.,scale,True,-1,-1,0.,None,True,rng,zero)
                dq[q0:q1].copy_(local_dq)
                dk[ka:k1].add_(local_dk.float()); dv[ka:k1].add_(local_dv.float())

    # Only opt-in construction pays for adapting the native autograd function.
    source = adapt_backward_source(inspect.getsource(original))
    namespace = dict(packed.__dict__); namespace['_adaptive_canonical'] = canonical
    exec(compile(source,'<adaptive-canonical-backward>','exec'),namespace)
    fused = namespace['backward']
    def backward(ctx, grad):
        return original(ctx,grad) if ctx.split_attention_backward else fused(ctx,grad)
    cls.backward = staticmethod(backward)
    cls._adaptive_chunk_installed = True

    for name in ('bias_dropout_add_fused_train','bias_dropout_add_fused_inference'):
        native = getattr(transformer,name)
        def dropout(x,bias,residual,prob,_native=native):
            return residual+(x if bias is None else x+bias) if prob==0. else _native(x,bias,residual,prob)
        setattr(transformer,name,dropout)
