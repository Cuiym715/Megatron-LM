"""Fused and D/H DSPP CP attention for the CP2 training path.

Each rank caches global K/V and its own query-rank partial history gradients.
At a segment's backward, pending partials are added before SUM reduce-scatter.
No historical gradient is reduced twice. KV and dKV each use one packed
collective, including empty-query ranks. Callers use identical task orders.
"""
import torch
import torch.distributed as dist
from dataclasses import dataclass

from megatron.core.datasets.dspp_cp_layout import cp_query_runs, cp_token_ranges, owned_query_runs
from . import dspp_packed_attention as packed


@dataclass
class _CpHistoryJob(packed.HistoryGradientJob):
    state: object
    layer: int
    layouts: tuple
    current_shape: tuple
    sequences: tuple
    prototype: torch.Tensor

    def run(self):
        from .fa_split import backward_dkv_range
        offsets = []
        total = 0
        for layout in self.layouts:
            offsets.append(total)
            total += sum(length for _, length in layout.history)
        dk = self.prototype.new_zeros((total, *self.prototype.shape[1:]))
        dv = torch.zeros_like(dk)
        for item_index, split, history in self.sequences:
            ki, vi = backward_dkv_range(split, 0, history)
            begin = offsets[item_index]
            dk[begin:begin + history].add_(ki)
            dv[begin:begin + history].add_(vi)
        self.state._route_key_value_grads(
            self.layer, self.layouts, dk, dv, self.current_shape,
            route_history=True, return_current=False, compact_region='history')


def _prepared(state, meta, size, rank, device):
    key = (meta.microbatch_id, size, rank, device)
    result = state._cp_prepared.get(key)
    if result is None:
        transport = None
        if state.cp_layout == 'hybrid':
            ownership = state.cp_ownership(meta, size)
            runs = owned_query_runs(meta, ownership.rank_indices[rank])
            ranges = tuple(zip(meta.cu_seqlens_q, meta.cu_seqlens_q[1:]))
            shared_items = set(ownership.shared_items)
            if state.cp_impl == 'query-out':
                shared_items.update(i for i,item in enumerate(meta.items) if item.segment_count > 1)
            shared_tokens = {i for item in shared_items for i in range(*ranges[item])}
            shared = tuple(tuple((local, token) for local, token in enumerate(row)
                                  if token in shared_tokens) for row in ownership.rank_indices)
            width = max(map(len, shared))
            tensor = lambda values: torch.tensor(values, dtype=torch.long, device=device)
            transport = dict(capacity=meta.chunk_size, width=width,
                global_indices=tensor(ownership.rank_indices[rank]),
                shared_local=tensor([local for local, _ in shared[rank]]),
                shared_global=tensor([token for row in shared for _, token in row]),
                packed_positions=tensor([r*width+i for r, row in enumerate(shared)
                                         for i in range(len(row))]))
        else:
            runs = cp_query_runs(meta, size, rank)
        offsets = meta.cu_seqlens_q[:-1]
        cu = tuple((torch.tensor([0, r.local_end-r.local_begin], dtype=torch.int32, device=device),
                    torch.tensor([0, r.key_end-r.key_begin], dtype=torch.int32, device=device))
                   for r in runs) if state.cp_impl != 'query-out' else ()
        result = runs, offsets, cu, transport
        state._cp_prepared[key] = result
    return result


def _rank_order(global_tensor, size):
    chunks = global_tensor.chunk(2 * size, dim=0)
    return torch.cat([chunks[i] for r in range(size)
                      for i in (2 * r, 2 * size - 1 - 2 * r)], dim=0)


def prepare_cp_metadata(state, metas, group):
    """Prepare H2D metadata before PP work, once per iteration/model chunk.

    torch.tensor(..., device='cuda') can synchronize its calling stream. A
    lazy allocation inside F could block the host while queued PP sends still
    need cooperative progress to release that stream's CP dependencies.
    """
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    device = torch.device('cuda', torch.cuda.current_device())
    for meta in metas:
        _prepared(state, meta, size, rank, device)
        if state.cp_layout == 'hybrid':
            rows = state.cp_ownership(meta, size).rank_indices
        else:
            rows = tuple(tuple(i for a,b in cp_token_ranges(meta.chunk_size,size,r)
                               for i in range(a,b)) for r in range(size))
        indices = tuple(torch.tensor(row,dtype=torch.long,device=device) for row in rows)
        state._cp_prepared['input-index',meta.microbatch_id,rank,device] = indices[rank]
        if state.cp_layout == 'hybrid':
            state._cp_prepared['output-index',meta.microbatch_id,device] = indices

    if state.cp_impl == 'query-out':
        from .dspp_cp_qo_attention import prepare_qo_metadata
        prepare_qo_metadata(state, metas, size, rank, device)


def _global_order(rank_tensor, size):
    chunks = rank_tensor.chunk(2 * size, dim=0)
    ordered = [None] * (2 * size)
    for r in range(size):
        ordered[2 * r] = chunks[2 * r]
        ordered[2 * size - 1 - 2 * r] = chunks[2 * r + 1]
    return torch.cat(ordered, dim=0)


def _gather(local, group, size, transport=None):
    if transport is not None:
        result = local.new_zeros((transport['capacity'], *local.shape[1:]))
        indices = transport['global_indices']
        result.index_copy_(0, indices, local[:indices.numel()])
        width = transport['width']
        if width:
            selected = transport['shared_local']
            send = local.new_zeros((width, *local.shape[1:]))
            send[:selected.numel()].copy_(local.index_select(0, selected))
            gathered = local.new_empty((size*width, *local.shape[1:]))
            dist.all_gather_into_tensor(gathered, send, group=group)
            result.index_copy_(0, transport['shared_global'],
                               gathered.index_select(0, transport['packed_positions']))
        return result
    rank_order = local.new_empty((local.size(0) * size, *local.shape[1:]))
    dist.all_gather_into_tensor(rank_order, local.contiguous(), group=group)
    return _global_order(rank_order, size)


def _scatter(global_grad, capacity, group, size, transport=None):
    padded = global_grad.new_zeros((capacity, *global_grad.shape[1:]))
    padded[:global_grad.size(0)].copy_(global_grad)
    if transport is not None:
        local = padded.new_zeros((capacity//size, *padded.shape[1:]))
        indices = transport['global_indices']
        local[:indices.numel()].copy_(padded.index_select(0, indices))
        width = transport['width']
        if width:
            packed = padded.new_zeros((size*width, *padded.shape[1:]))
            packed.index_copy_(0, transport['packed_positions'],
                               padded.index_select(0, transport['shared_global']))
            reduced = padded.new_empty((width, *padded.shape[1:]))
            dist.reduce_scatter_tensor(reduced, packed, op=dist.ReduceOp.SUM, group=group)
            selected = transport['shared_local']
            local.index_copy_(0, selected, reduced[:selected.numel()])
        return local
    local = padded.new_empty((capacity // size, *padded.shape[1:]))
    dist.reduce_scatter_tensor(local, _rank_order(padded, size),
                              op=dist.ReduceOp.SUM, group=group)
    return local


class _CpGatherOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, state, group):
        meta = state.current_meta
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        key = ('output-index', meta.microbatch_id, output.device)
        indices = state._cp_prepared.get(key)
        if indices is None:
            ownership = state.cp_ownership(meta, size)
            indices = tuple(torch.tensor(row, dtype=torch.long, device=output.device)
                            for row in ownership.rank_indices)
            state._cp_prepared[key] = indices
        local = output.transpose(0, 1).contiguous()
        gathered = local.new_empty((local.size(0)*size, *local.shape[1:]))
        dist.all_gather_into_tensor(gathered, local, group=group)
        result = local.new_zeros((meta.chunk_size, *local.shape[1:]))
        for r, index in enumerate(indices):
            result.index_copy_(0, index, gathered[r*local.size(0):r*local.size(0)+index.numel()])
        ctx.save_for_backward(indices[rank])
        ctx.capacity = local.size(0)
        return result.transpose(0, 1)

    @staticmethod
    def backward(ctx, grad):
        index, = ctx.saved_tensors
        valid = grad.index_select(1, index)
        return torch.nn.functional.pad(valid, (0, ctx.capacity-index.numel())), None, None


def dspp_cp_gather_output(output, state, group):
    return _CpGatherOutput.apply(output, state, group)


class _DsppCpAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, state, meta, layer_idx, group, scale):
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        runs, current_offsets, cu, transport = _prepared(state, meta, size, rank, query.device)
        # Gather fixed capacities, including padding: identical collective
        # shapes/order on all ranks, independent of local packed intersections.
        global_k, global_v = _gather(torch.stack((key[:,0],value[:,0]),dim=1),
                                    group,size,transport).unbind(1)
        assembled_k, assembled_v, layouts = state.build_key_value(
            layer_idx, meta, global_k[:meta.valid_token_count],
            global_v[:meta.valid_token_count])
        output = torch.zeros_like(query)
        saved = []
        for run, (cq, ck) in zip(runs, cu):
            q = query[run.local_begin:run.local_end, 0].contiguous()
            k = assembled_k[run.key_begin:run.key_end].contiguous()
            v = assembled_v[run.key_begin:run.key_end].contiguous()
            out, lse, _, rng = packed._wrapped_flash_attn_varlen_forward(
                q, k, v, cq, ck, q.size(0), k.size(0), 0.0, scale,
                causal=True, window_size_left=-1, window_size_right=-1,
                softcap=0.0, alibi_slopes=None, return_softmax=False,
                block_table=None, leftpad_k=None, seqused_k=None, zero_tensors=False)
            output[run.local_begin:run.local_end, 0] = out
            saved.extend((q, k, v, out, lse, cq, ck, rng))
        ctx.save_for_backward(assembled_k, *saved)
        ctx.runs, ctx.layouts = runs, layouts
        ctx.state, ctx.meta, ctx.layer_idx = state, meta, layer_idx
        ctx.group, ctx.size, ctx.scale = group, size, scale
        ctx.query_shape = query.shape
        ctx.key_shape = key.shape
        ctx.current_offsets = current_offsets
        ctx.after_dx = state.task_after_dx(meta.microbatch_id)
        ctx.transport = transport
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assembled_k, *saved = ctx.saved_tensors
        dq = grad_output.new_zeros(ctx.query_shape)
        current_shape = (ctx.meta.valid_token_count, *ctx.key_shape[2:])
        has_history = any(layout.history for layout in ctx.layouts)
        split = ctx.after_dx and has_history
        if split and ctx.state._history_task_id != ctx.meta.microbatch_id:
            raise RuntimeError('CP D requires its active physical task')
        dk = assembled_k.new_zeros(current_shape) if split else torch.zeros_like(assembled_k)
        dv = torch.zeros_like(dk)
        history_states = []
        for index, run in enumerate(ctx.runs):
            q, k, v, out, lse, cq, ck, rng = saved[index * 8:(index + 1) * 8]
            do = grad_output[run.local_begin:run.local_end, 0].contiguous()
            if split:
                from .fa_split import prepare_backward, backward_dq_current
                prepared = prepare_backward(
                    *(t.detach() for t in (q, k, v, out, do, lse.contiguous())),
                    softmax_scale=ctx.scale)
                dqi, dki, dvi = backward_dq_current(prepared, current_begin=run.history_length)
                dq[run.local_begin:run.local_end, 0] = dqi
                begin = ctx.current_offsets[run.item_index]
                dk[begin:begin + dki.size(0)].add_(dki)
                dv[begin:begin + dvi.size(0)].add_(dvi)
                if run.history_length:
                    history_states.append((run.item_index, prepared, run.history_length))
                continue
            dqi, dki, dvi = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            packed._wrapped_flash_attn_varlen_backward(
                do,
                q, k, v, out, lse, dqi, dki, dvi, cq, ck,
                q.size(0), k.size(0), 0.0, ctx.scale, True, -1, -1,
                0.0, None, False, rng, False)
            dq[run.local_begin:run.local_end, 0] = dqi
            dk[run.key_begin:run.key_end].add_(dki)
            dv[run.key_begin:run.key_end].add_(dvi)
        if split:
            ctx.state._history_jobs.append(_CpHistoryJob(
                ctx.state, ctx.layer_idx, ctx.layouts, current_shape,
                tuple(history_states), assembled_k[:0]))
            current_k, current_v = ctx.state._route_key_value_grads(
                ctx.layer_idx, ctx.layouts, dk, dv, current_shape,
                route_history=False, return_current=True, compact_region='current')
        else:
            current_k, current_v = ctx.state.distribute_key_value_grads(
                ctx.layer_idx, ctx.layouts, dk, dv, current_shape)
        local_k, local_v = _scatter(torch.stack((current_k,current_v),dim=1),
            ctx.meta.chunk_size,ctx.group,ctx.size,ctx.transport).unbind(1)
        return dq, local_k.unsqueeze(1), local_v.unsqueeze(1), None, None, None, None, None


def dspp_cp_flash_attention(query, key, value, *, state, meta, layer_idx,
                            cp_group, softmax_scale=None):
    """Local physical Q/K/V -> local output, preserving global packed metadata.

    D retains local history jobs. H has no collectives; the consumer reduces
    pending history partials together with its own current gradient.
    """
    size, rank = dist.get_world_size(cp_group), dist.get_rank(cp_group)
    cp_token_ranges(meta.chunk_size, size, rank)
    if size < 2:
        raise ValueError('DSPP CP operator requires CP>=2')
    if state.split_attention_backward and (not state.history_dkv_after_dx
                                           or query.dtype != torch.bfloat16 or query.size(-1) != 128):
        raise ValueError('DSPP CP split requires after-dx, BF16 and head dimension 128')
    if packed._wrapped_flash_attn_varlen_forward is None:
        raise RuntimeError('FlashAttention is required for DSPP CP')
    if (query.ndim != 4 or key.ndim != 4 or value.shape != key.shape
            or query.size(1) != 1 or key.size(1) != 1
            or query.size(0) != meta.chunk_size // size
            or key.size(0) != query.size(0) or key.size(-1) != query.size(-1)
            or key.size(2) <= 0 or query.size(2) % key.size(2)
            or query.size(-1) % 8 or query.size(-1) > 256
            or query.device != key.device or key.device != value.device
            or query.dtype != key.dtype or key.dtype != value.dtype
            or not query.is_cuda or query.dtype not in (torch.float16, torch.bfloat16)):
        raise ValueError('invalid DSPP CP Q/K/V shape, dtype or device')
    if state.cp_impl == 'query-out':
        from .dspp_cp_qo_attention import dspp_cp_qo_attention
        return dspp_cp_qo_attention(query,key,value,state=state,meta=meta,
            layer_idx=layer_idx,cp_group=cp_group,
            softmax_scale=float(softmax_scale or query.size(-1)**-0.5))
    return _DsppCpAttention.apply(query, key, value, state, meta, int(layer_idx),
                                  cp_group, float(softmax_scale or query.size(-1)**-0.5))
