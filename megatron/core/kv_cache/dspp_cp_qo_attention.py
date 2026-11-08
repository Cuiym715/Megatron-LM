"""DSPP CP-QO: owner-local KV history, current-query/output communication.

Uses FA2 with global O/LSE in backward. Disjoint Q intervals attend to
owner-local KV prefixes. Global positions, not local KV lengths, define
the causal boundaries. No dense attention mask or global KV is built.
"""
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist

from . import dspp_packed_attention as packed

IMPLEMENTATION = 'owner-kv-query-out-fa-dh-v2'



@dataclass
class _QOHistoryJob(packed.HistoryGradientJob):
    state: object
    layer: int
    layouts: tuple
    current_shape: tuple
    parts: tuple
    prototype: torch.Tensor

    def run(self):
        from .fa_split import backward_dkv_range
        offsets, total = [], 0
        for layout in self.layouts:
            offsets.append(total)
            total += sum(n for _,n in layout.history)
        dk = self.prototype.new_zeros((total,*self.prototype.shape[1:]),dtype=torch.float32)
        dv = torch.zeros_like(dk)
        for i,prepared,history in self.parts:
            ki,vi = backward_dkv_range(prepared,0,history)
            start = offsets[i]
            dk[start:start+ki.size(0)].add_(ki)
            dv[start:start+vi.size(0)].add_(vi)
        self.state._route_key_value_grads(self.layer,self.layouts,dk.to(self.prototype.dtype),dv.to(self.prototype.dtype),self.current_shape,
            route_history=True,return_current=False,compact_region='history')

def prepare_qo_metadata(state, metas, size, rank, device):
    from .dspp_cp_attention import _prepared
    shard_lengths = {}
    for meta in metas:
        runs, _, _, _ = _prepared(state, meta, size, rank, device)
        for i,item in enumerate(meta.items):
            shard_lengths[item.sequence_id,item.segment_id] = sum(
                r.local_end-r.local_begin for r in runs if r.item_index == i)
    for meta in metas:
        runs, offsets, _, transport = _prepared(state, meta, size, rank, device)
        items, parts, indices, global_indices = [], [], [], []
        for i, item in enumerate(meta.items):
            own = tuple(sorted((r for r in runs if r.item_index == i),key=lambda r:r.position_begin))
            length = sum(r.local_end-r.local_begin for r in own)
            history = sum(shard_lengths[item.sequence_id,j] for j in range(item.segment_id))
            items.append((own, length, history))
            indices.extend(j for r in own for j in range(r.local_begin, r.local_end))
            global_indices.extend(offsets[i]+r.position_begin-item.token_offset+j
                                  for r in own for j in range(r.local_end-r.local_begin))
            # Partition Q into disjoint intervals. Each interval attends to
            # one local-KV prefix, eliminating repeated local softmax merges.
            cursor, qcursor = history, offsets[i]
            for r in own:
                n = r.local_end-r.local_begin
                begin = offsets[i]+r.position_begin-item.token_offset
                if begin > qcursor and cursor:
                    parts.append((i,qcursor,begin,0,cursor,False))
                if parts and parts[-1][0] == i and parts[-1][-1] and parts[-1][2] == begin:
                    parts[-1] = (i,parts[-1][1],begin+n,0,cursor+n,True)
                else:
                    parts.append((i,begin,begin+n,0,cursor+n,True))
                cursor += n
                qcursor = begin+n
            if qcursor < offsets[i]+item.token_length and cursor:
                parts.append((i,qcursor,offsets[i]+item.token_length,0,cursor,False))
        # One setup copy; never allocate CUDA metadata while a PP send is pending.
        values = [[0, qb-qa, 0, kb-ka] for _,qa,qb,ka,kb,_ in parts]
        cu = torch.tensor(values, device=device, dtype=torch.int32).reshape(-1, 4)
        state._cp_prepared['qo', meta.microbatch_id, rank, device] = (
            tuple(items), tuple(parts), cu,
            torch.tensor(indices, device=device, dtype=torch.long),
            torch.tensor(global_indices, device=device, dtype=torch.long), transport)


def _lse_global(local, meta, transport, group, size):
    if transport is None:
        selected = local
    elif not transport['width']:
        return local
    else:
        selected = local.index_select(0, transport['shared_global'])
    selected = selected.contiguous()
    gathered = selected.new_empty((size*selected.size(0), *selected.shape[1:]))
    dist.all_gather_into_tensor(gathered, selected, group=group)
    merged = torch.logsumexp(gathered.view(size, *selected.shape), dim=0)
    if transport is None:
        return merged
    result = local.clone()
    result.index_copy_(0, transport['shared_global'], merged)
    return result


class _QOAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, state, meta, layer, group, scale):
        from .dspp_cp_attention import _gather, _scatter
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        cache_key = ('qo', meta.microbatch_id, rank, query.device)
        item_info, parts, cu, indices, global_indices, transport = state._cp_prepared[cache_key]
        q = _gather(query[:, 0], group, size, transport)
        kv, layouts = [], []
        for item, (runs, length, history_length) in zip(meta.items, item_info):
            entry_key = state._entry_key(layer, item.sequence_id)
            entry = state._entries.get(entry_key)
            if (entry is None and item.segment_id != 0) or (
                    entry is not None and entry.next_segment_id != item.segment_id):
                raise RuntimeError('CP-QO sequence forward is out of order')
            history = tuple(entry.segments) if entry else ()
            current = []
            for tensor in (key, value):
                slices = [tensor[r.local_begin:r.local_end, 0] for r in runs]
                contiguous = runs and all(a.local_end == b.local_begin for a,b in zip(runs,runs[1:]))
                current.append(tensor[runs[0].local_begin:runs[-1].local_end,0] if contiguous else
                               torch.cat(slices) if slices else tensor[:0,0])
            for j in range(2):
                tensors = [getattr(s, ('key','value')[j]) for s in history]+[current[j]]
                kv.append(torch.cat(tensors) if len(tensors)>1 else tensors[0])
            layouts.append(packed._ItemLayout(replace(item, token_length=length),
                tuple((s.segment_id, s.key.size(0)) for s in history)))
            if not item.is_last_segment:
                if entry is None:
                    entry = packed._SequenceLayerEntry()
                    state._entries[entry_key] = entry
                entry.segments.append(packed._CachedSegment(item.segment_id, *current))
                entry.next_segment_id = item.segment_id+1
        out = q.new_zeros(q.shape, dtype=torch.float32)
        lse = q.new_full(q.shape[:2], -torch.inf, dtype=torch.float32)
        rngs = []
        for n, (i, qa, qb, ka, kb, causal) in enumerate(parts):
            cq, ck = cu[n, :2], cu[n, 2:]
            oi, li, _, rng = packed._wrapped_flash_attn_varlen_forward(
                q[qa:qb].contiguous(), kv[2*i][ka:kb].contiguous(), kv[2*i+1][ka:kb].contiguous(),
                cq, ck, qb-qa, kb-ka, 0., scale, causal=causal,
                window_size_left=-1, window_size_right=-1, softcap=0.,
                alibi_slopes=None, return_softmax=False, block_table=None,
                leftpad_k=None, seqused_k=None, zero_tensors=False)
            out[qa:qb] = oi
            lse[qa:qb] = li.T
            rngs.append(rng)
        global_lse = _lse_global(lse, meta, transport, group, size)
        weight = torch.where(torch.isfinite(lse), torch.exp(lse-global_lse), 0.)
        local_out = _scatter((out*weight[...,None]).to(query.dtype),
                             meta.chunk_size, group, size, transport).unsqueeze(1)
        # Save sharded Q/O/LSE, not a full-current-Q copy at every layer.
        local_lse = global_lse.new_full((query.size(0), query.size(2)), -torch.inf)
        local_lse.index_copy_(0, indices, global_lse.index_select(0, global_indices))
        ctx.save_for_backward(query, local_out, local_lse, *kv)
        ctx.state, ctx.meta, ctx.layer = state, meta, layer
        ctx.group, ctx.size, ctx.scale, ctx.transport = group, size, scale, transport
        ctx.parts, ctx.cu, ctx.layouts, ctx.indices, ctx.rngs = parts, cu, tuple(layouts), indices, rngs
        ctx.key_shape = key.shape
        ctx.after_dx = state.task_after_dx(meta.microbatch_id)
        return local_out

    @staticmethod
    def backward(ctx, grad_output):
        from .dspp_cp_attention import _gather, _scatter
        query, output, local_lse, *kv = ctx.saved_tensors
        q, out, do = _gather(torch.stack((query[:,0], output[:,0], grad_output[:,0]),dim=1),
                            ctx.group, ctx.size, ctx.transport).unbind(1)
        lse = _gather(local_lse, ctx.group, ctx.size, ctx.transport)
        dq = torch.zeros_like(q, dtype=torch.float32)
        split = ctx.after_dx and any(i.has_history for i in ctx.meta.items)
        if split and ctx.state._history_task_id != ctx.meta.microbatch_id:
            raise RuntimeError('CP-QO D requires its active physical task')
        history_sizes = tuple(sum(n for _,n in layout.history) for layout in ctx.layouts)
        grads = [torch.zeros_like(t[h:] if split else t,dtype=torch.float32)
                 for t,h in zip(kv,(n for h in history_sizes for n in (h,h)))]
        history_parts = []
        for n,(i,qa,qb,ka,kb,causal) in enumerate(ctx.parts):
            qi,ki,vi = q[qa:qb].contiguous(),kv[2*i][ka:kb].contiguous(),kv[2*i+1][ka:kb].contiguous()
            if split:
                from .fa_split import prepare_partial_backward, backward_dq, backward_dq_current
                prepared = prepare_partial_backward(qi,ki,vi,out[qa:qb].contiguous(),
                    do[qa:qb].contiguous(),lse[qa:qb].T.contiguous(),
                    softmax_scale=ctx.scale,causal=causal)
                if kb <= history_sizes[i]:
                    dqi = backward_dq(prepared)
                else:
                    dqi,dki,dvi = backward_dq_current(prepared,current_begin=history_sizes[i])
                    end = kb-history_sizes[i]
                    grads[2*i][:end].add_(dki)
                    grads[2*i+1][:end].add_(dvi)
                if history_sizes[i]:
                    history_parts.append((i,prepared,history_sizes[i]))
                dq[qa:qb].add_(dqi)
                continue
            dqi,dki,dvi = torch.empty_like(qi),torch.empty_like(ki),torch.empty_like(vi)
            packed._wrapped_flash_attn_varlen_backward(
                do[qa:qb].contiguous(), qi, ki, vi, out[qa:qb].contiguous(),
                lse[qa:qb].T.contiguous(), dqi,dki,dvi,ctx.cu[n,:2],ctx.cu[n,2:],
                qb-qa,kb-ka,0.,ctx.scale,causal,-1,-1,0.,None,False,ctx.rngs[n],False)
            dq[qa:qb].add_(dqi)
            grads[2*i][ka:kb].add_(dki)
            grads[2*i+1][ka:kb].add_(dvi)
        # Local shard lengths replace global lengths only in the gradient router.
        dk = torch.cat(grads[::2]).to(query.dtype)
        dv = torch.cat(grads[1::2]).to(query.dtype)
        current_shape = (ctx.indices.numel(), *ctx.key_shape[2:])
        if split:
            ctx.state._history_jobs.append(_QOHistoryJob(ctx.state,ctx.layer,ctx.layouts,
                current_shape,tuple(history_parts),query.new_empty((0,*ctx.key_shape[2:]))))
            ck,cv = ctx.state._route_key_value_grads(ctx.layer,ctx.layouts,dk,dv,current_shape,
                route_history=False,return_current=True,compact_region='current')
        else:
            ck,cv = ctx.state.distribute_key_value_grads(ctx.layer,ctx.layouts,dk,dv,current_shape)
        local_k = query.new_zeros(ctx.key_shape)
        local_v = torch.zeros_like(local_k)
        local_k[:,0].index_copy_(0,ctx.indices,ck)
        local_v[:,0].index_copy_(0,ctx.indices,cv)
        local_q = _scatter(dq.to(query.dtype),ctx.meta.chunk_size,ctx.group,ctx.size,ctx.transport)
        return local_q.unsqueeze(1),local_k,local_v,None,None,None,None,None


def dspp_cp_qo_attention(query, key, value, *, state, meta, layer_idx, cp_group, softmax_scale):
    return _QOAttention.apply(query,key,value,state,meta,layer_idx,cp_group,softmax_scale)
