from .dispatch_flash_attn import flash_attn_func, _flash_attn_forward, _flash_attn_backward
from megatron.core.cudnn_attn import cudnn_attn_check_capability, CudnnAttnFunc
from megatron.core.parallel_state import get_context_parallel_group_slow

import nvtx
import torch
import torch.distributed as dist


# example before flip, rank 0: [0 1 2 3],   rank 1: [4 5 6 7],   rank 2: [8 9 10 11], rank 3: [12 13 14 15]
#          after flip, rank 0: [0 1 14 15], rank 1: [4 5 10 11], rank 2: [8 9 6 7],   rank 3: [12 13 2 3]

def flip_cp_(x, dim, world_size):
    if world_size == 1:
        return x
    batch_size = x.shape[:dim].numel()
    v = x.view(batch_size, world_size, 2, -1)[:, :, 1]

    # Fast v.copy_(v.flip(1))
    import fast_flip_cuda
    assert v.device.type == "cuda", "the fused op only supports CUDA"
    assert v.stride(2) == 1, "the fused op requires the last dim to be contiguous"
    fast_flip_cuda.flip(v.data_ptr(), v.data_ptr(), v.shape[0], v.stride(0), v.shape[1], v.stride(1), v.shape[2], v.element_size(), torch.cuda.current_stream().cuda_stream)

    return x


def flip_cp(x, dim, world_size):
    if world_size == 1:
        return x
    vx = x.view(*x.shape[:dim], world_size, 2, x.shape[dim] // world_size // 2, *x.shape[dim + 1:])
    vo = torch.cat([vx.select(dim + 1, 0), vx.select(dim + 1, 1).flip(dim)], dim=dim + 1)
    o = vo.view_as(x)
    return o


def slice_cp(x, dim, world_size, rank):
    if world_size == 1:
        assert rank == 0
        return x
    vs = x.chunk(world_size * 2, dim=dim)
    return torch.cat([vs[rank * 2], vs[2 * world_size - 1 - 2 * rank]], dim=dim)


def all_gather_along_dim(input, dim, group):
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input
    output = torch.empty(world_size, *input.shape, dtype=input.dtype, device=input.device)
    dist.all_gather_into_tensor(output, input.contiguous(), group=group)
    output = output.permute(*range(1, dim + 1), 0, *range(dim + 1, input.dim() + 1))
    output = output.reshape(*input.shape[:dim], world_size * input.shape[dim], *input.shape[dim + 1:])
    return output


def reduce_scatter_along_dim(input, dim, group):
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input
    output = torch.empty(input.shape[dim] // world_size, *input.shape[:dim], *input.shape[dim + 1:], dtype=input.dtype, device=input.device)
    dist.reduce_scatter_tensor(output, input.permute(dim, *range(dim), *range(dim + 1, input.dim())).contiguous(), group=group)
    output = output.permute(*range(1, dim + 1), 0, *range(dim + 1, input.dim()))
    return output


_CP_STREAM = None


def get_cp_stream():
    global _CP_STREAM
    if _CP_STREAM is None:
        _CP_STREAM = torch.cuda.Stream()
    return _CP_STREAM


# The qi refers to the i-th shard of q.
# The qi is sharded along the second axis (the seqlen axis).
# The kvT refers to kv.transpose(0, 1) whose shape is (s, b, 2, num_heads, head_dim).
# The kvTi is sharded along the first axis (the seqlen axis).
# Sharding on kvT (instead of kv) helps to avoid transpose before NCCL communication.
# Flip seqlen (call flip_cp_) before shard tensors.


class DAttentionPreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kTi, vTi, cp_group):
        nvtx.push_range("dattention forward")
        ctx.cp_group = cp_group
        CP = dist.get_world_size(ctx.cp_group)
        ctx.n = len(kTi._tensors) if hasattr(kTi, "_tensors") else 1
        kTi = kTi.concat() if hasattr(kTi, "concat") else kTi
        vTi = vTi.concat() if hasattr(vTi, "concat") else vTi
        kTi = kTi.unflatten(0, (ctx.n, -1)).transpose(0, 1).contiguous()
        kT = kTi.new_empty(CP * kTi.shape[0], *kTi.shape[1:])
        req_k = dist.all_gather_into_tensor(kT, kTi, group=ctx.cp_group, async_op=True)
        vTi = vTi.unflatten(0, (ctx.n, -1)).transpose(0, 1).contiguous()
        vT = kTi.new_empty(CP * vTi.shape[0], *vTi.shape[1:])
        req_v = dist.all_gather_into_tensor(vT, vTi, group=ctx.cp_group, async_op=True)
        req_k.wait()
        flip_cp_(kT, 0, CP)
        kT = kT.transpose(0, 1).flatten(0, 1)
        req_v.wait()
        flip_cp_(vT, 0, CP)
        vT = vT.transpose(0, 1).flatten(0, 1)
        return kT, vT

    @staticmethod
    def backward(ctx, grad_kT, grad_vT):
        CP = dist.get_world_size(ctx.cp_group)
        grad_kT = grad_kT.unflatten(0, (ctx.n, -1)).transpose(0, 1).contiguous()
        flip_cp_(grad_kT, 0, CP)
        grad_kTi = grad_kT.new_empty(grad_kT.shape[0] // CP, *grad_kT.shape[1:])
        req_k = dist.reduce_scatter_tensor(grad_kTi, grad_kT, group=ctx.cp_group, async_op=True)
        grad_vT = grad_vT.unflatten(0, (ctx.n, -1)).transpose(0, 1).contiguous()
        flip_cp_(grad_vT, 0, CP)
        grad_vTi = grad_vT.new_empty(grad_vT.shape[0] // CP, *grad_vT.shape[1:])
        req_v = dist.reduce_scatter_tensor(grad_vTi, grad_vT, group=ctx.cp_group, async_op=True)
        req_k.wait()
        grad_kTi = grad_kTi.transpose(0, 1).flatten(0, 1)
        req_v.wait()
        grad_vTi = grad_vTi.transpose(0, 1).flatten(0, 1)
        nvtx.pop_range()
        return grad_kTi, grad_vTi, None


class DAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qi, k, v, save_kv, cp_group, alibi_bias_max, tp_world_size, tp_rank):
        ctx.cp_group = cp_group
        ctx.alibi_bias_max = alibi_bias_max
        ctx.tp_world_size = tp_world_size
        ctx.tp_rank = tp_rank
        CP = dist.get_world_size(ctx.cp_group)
        cp_rank = dist.get_rank(ctx.cp_group)
        b, seqlen_qi, a, d = qi.shape
        seqlen_q = seqlen_qi * CP
        prelen = k.shape[1] - seqlen_q

        ctx.use_torch_sdpa = False
        ctx.use_cudnn_sdpa = not alibi_bias_max and cudnn_attn_check_capability(use_causal_mask_bottom_right=True)

        if ctx.use_torch_sdpa:
            if alibi_bias_max:
                raise NotImplementedError("not implemented alibi for torch SDPA")
            attn_bias = torch.full((seqlen_qi, seqlen_kv), float("-inf"), dtype=qi.dtype, device=qi.device)
            attn_bias[:seqlen_qi // 2].triu_(1 + cp_rank * seqlen_kv // CP)
            attn_bias[seqlen_qi // 2:].triu_(1 + (2 * CP - 1 - 2 * cp_rank) * seqlen_kv // CP // 2)
            attn_bias = attn_bias.expand(b, a, *attn_bias.shape)
            compute_log_sumexp = True
            dropout_p = 0.
            is_causal = False
            output, log_sumexp, philox_seed, philox_offset = \
                torch.ops.aten._scaled_dot_product_efficient_attention(
                    qi.transpose(1, 2), kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2),
                    attn_bias, compute_log_sumexp, dropout_p, is_causal)
            oi = output.transpose(1, 2)

            ctx.save_for_backward(qi, log_sumexp, attn_bias, oi, log_sumexp, philox_seed, philox_offset)
            ctx.dropout_p = dropout_p
            ctx.is_causal = is_causal
            nvtx.pop_range()
            return oi, data_to_save

        # kv_ = kv.concat() if hasattr(kv, "concat") else kv
        dropout_p = 0.
        softmax_scale = d ** -.5
        causal = True
        window_size = (-1, -1)
        return_softmax = False

        qi0 = qi[:, :seqlen_qi // 2]
        k0 = k[:, :prelen + (2 * cp_rank + 1) * seqlen_q // (2 * CP)]
        v0 = v[:, :prelen + (2 * cp_rank + 1) * seqlen_q // (2 * CP)]

        qi1 = qi[:, seqlen_qi // 2:]
        k1 = k[:, :prelen + (CP - cp_rank) * seqlen_q // CP]
        v1 = v[:, :prelen + (CP - cp_rank) * seqlen_q // CP]

        oi0, softmax_lse0, S_dmask0, rng_state0 = (None,) * 4
        oi1, softmax_lse1, S_dmask1, rng_state1 = (None,) * 4

        if ctx.use_cudnn_sdpa:
            oi = torch.empty_like(qi)

        def attn_func0():
            nonlocal oi0, softmax_lse0, S_dmask0, rng_state0
            if ctx.use_cudnn_sdpa:
                oi0 = oi[:, :seqlen_qi // 2]
                softmax_lse0 = CudnnAttnFunc.forward_no_ctx(qi0, k0, v0, causal, False, False, None, None, None, None, oi0)
                S_dmask0, rng_state0 = None, None
            else:
                oi0, qi_padded0, k_padded0, v_padded0, out_padded0, softmax_lse0, S_dmask0, rng_state0 = _flash_attn_forward(
                    qi0,
                    k0,
                    v0,
                    dropout_p,
                    softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    return_softmax=return_softmax and dropout_p > 0,
                    alibi_bias_max=alibi_bias_max,
                    tp_world_size=tp_world_size,
                    tp_rank=tp_rank,
                )
                assert (qi0.shape, k0.shape, v0.shape) == (qi_padded0.shape, k_padded0.shape, v_padded0.shape), "no support padding"
                assert (oi0.data_ptr(), oi0.shape, oi0.stride()) == (out_padded0.data_ptr(), out_padded0.shape, out_padded0.stride()), "no support padding"

        def attn_func1():
            nonlocal oi1, softmax_lse1, S_dmask1, rng_state1
            if ctx.use_cudnn_sdpa:
                oi1 = oi[:, seqlen_qi // 2:]
                softmax_lse1 = CudnnAttnFunc.forward_no_ctx(qi1, k1, v1, causal, False, False, None, None, None, None, oi1)
                S_dmask1, rng_state1 = None, None
            else:
                oi1, qi_padded1, k_padded1, v_padded1, out_padded1, softmax_lse1, S_dmask1, rng_state1 = _flash_attn_forward(
                    qi1,
                    k1,
                    v1,
                    dropout_p,
                    softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    return_softmax=return_softmax and dropout_p > 0,
                    alibi_bias_max=alibi_bias_max,
                    tp_world_size=tp_world_size,
                    tp_rank=tp_rank,
                )
                assert (qi1.shape, k1.shape, v1.shape) == (qi_padded1.shape, k_padded1.shape, v_padded1.shape), "no support padding"
                assert (oi1.data_ptr(), oi1.shape, oi1.stride()) == (out_padded1.data_ptr(), out_padded1.shape, out_padded1.stride()), "no support padding"

        get_cp_stream().wait_stream(torch.cuda.current_stream())
        if k0.shape[1] >= k1.shape[1]:  # call the longer kernel first
            attn_func0()
            with torch.cuda.stream(get_cp_stream()):
                attn_func1()
        else:
            with torch.cuda.stream(get_cp_stream()):
                attn_func1()
            attn_func0()
        torch.cuda.current_stream().wait_stream(get_cp_stream())

        if not ctx.use_cudnn_sdpa:
            # Fused version of oi = torch.concat([oi0, oi1], dim=1)
            import fast_cat_cuda
            oi = torch.empty(oi0.shape[0], oi0.shape[1] * 2, *oi0.shape[2:], dtype=oi0.dtype, device=oi0.device)
            fast_cat_cuda.cat([oi0.data_ptr(), oi1.data_ptr()], oi.data_ptr(),
                               oi0.shape[0], oi0.shape[1:].numel() * oi0.element_size(), torch.cuda.current_stream().cuda_stream)
        if save_kv:
            ctx.kv = (k, v)
            k = v = None
        else:
            k = k.new_empty(()).expand_as(k)
            v = v.new_empty(()).expand_as(v)
        ctx.save_for_backward(qi, oi, softmax_lse0, rng_state0, softmax_lse1, rng_state1)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        if save_kv:
            k = v = None
        else:
            k = k.new_empty(()).expand_as(k)
            v = v.new_empty(()).expand_as(v)
        nvtx.pop_range()
        return oi, k, v

    @staticmethod
    def backward(ctx, grad_oi, k, v):
        nvtx.push_range("dattention backward")
        CP = dist.get_world_size(ctx.cp_group)
        cp_rank = dist.get_rank(ctx.cp_group)
        qi, oi, softmax_lse0, rng_state0, softmax_lse1, rng_state1 = ctx.saved_tensors

        k, v = getattr(ctx, "kv", (k, v))

        if ctx.use_torch_sdpa:
            qi, log_sumexp, attn_bias, oi, log_sumexp, philox_seed, philox_offset = ctx.saved_tensors
            b, seqlen_qi, a, d = qi.shape
            seqlen_kv = seqlen_qi * CP

            grad_input_mask = ctx.needs_input_grad[:3] + (False,)
            grad_qi, grad_k, grad_v, grad_bias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
                grad_oi.transpose(1, 2), qi.transpose(1, 2), kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2), attn_bias, oi.transpose(1, 2),
                log_sumexp, philox_seed, philox_offset, ctx.dropout_p, grad_input_mask, ctx.is_causal)
            grad_qi, grad_k, grad_v = grad_qi.transpose(1, 2), grad_k.transpose(1, 2), grad_v.transpose(1, 2)
            grad_kv = torch.empty_strided(kv.shape, kv.stride(), dtype=kv.dtype, device=kv.device)
            grad_kv[:, :, 0] = grad_k
            grad_kv[:, :, 1] = grad_v
            return grad_qi, grad_kv, None, None, None, None

        out_padded0, out_padded1 = oi.chunk(2, dim=1)
        b, seqlen_qi, a, d = qi.shape
        seqlen_q = seqlen_qi * CP
        prelen = k.shape[1] - seqlen_q

        def wait_flip(x):
            if not hasattr(x, "_handle"):
                return x
            x._handle.wait()
            del x._handle  # break circular reference
            x = x.unflatten(1, (seqlen_q, -1))
            flip_cp_(x, 1, CP)
            return x.transpose(1, 2).flatten(1, 2)
        k, v = [wait_flip(x) for x in (k, v)]

        dqi = torch.empty_like(qi)
        dk = torch.empty_strided(k.shape, k.stride(), dtype=k.dtype, device=k.device)
        dv = torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device=v.device)

        qi0 = qi[:, :seqlen_qi // 2]
        k0 = k[:, :prelen + (2 * cp_rank + 1) * seqlen_q // (2 * CP)]
        v0 = v[:, :prelen + (2 * cp_rank + 1) * seqlen_q // (2 * CP)]
        doi0 = grad_oi[:, :seqlen_qi // 2]
        dqi0 = dqi[:, :seqlen_qi // 2]

        qi1 = qi[:, seqlen_qi // 2:]
        k1 = k[:, :prelen + (CP - cp_rank) * seqlen_q // CP]
        v1 = v[:, :prelen + (CP - cp_rank) * seqlen_q // CP]
        doi1 = grad_oi[:, seqlen_qi // 2:]
        dqi1 = dqi[:, seqlen_qi // 2:]

        kv0_is_longer = k0.shape[1] >= k1.shape[1]
        if kv0_is_longer:
            dk0 = dk[:, :k0.shape[1]]
            dv0 = dv[:, :k0.shape[1]]
            dk1 = torch.empty_like(k1)
            dv1 = torch.empty_like(v1)
        else:
            dk0 = torch.empty_like(k0)
            dv0 = torch.empty_like(v0)
            dk1 = dk[:, :k1.shape[1]]
            dv1 = dv[:, :k1.shape[1]]

        get_cp_stream().wait_stream(torch.cuda.current_stream())
        if ctx.use_cudnn_sdpa:
            CudnnAttnFunc.backward_no_ctx(
                doi0,
                qi0,
                k0,
                v0,
                out_padded0,
                softmax_lse0,
                ctx.causal,
                False, None, None, None, None,
                dqi0,
                dk0,
                dv0,
            )
        else:
            _flash_attn_backward(
                doi0,
                qi0,
                k0,
                v0,
                out_padded0,
                softmax_lse0,
                dqi0,
                dk0,
                dv0,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                ctx.window_size,
                rng_state=rng_state0,
                alibi_bias_max=ctx.alibi_bias_max,
                tp_world_size=ctx.tp_world_size,
                tp_rank=ctx.tp_rank,
            )
        with torch.cuda.stream(get_cp_stream()):
            if ctx.use_cudnn_sdpa:
                CudnnAttnFunc.backward_no_ctx(
                    doi1,
                    qi1,
                    k1,
                    v1,
                    out_padded1,
                    softmax_lse1,
                    ctx.causal,
                    False, None, None, None, None,
                    dqi1,
                    dk1,
                    dv1,
                )
            else:
                _flash_attn_backward(
                    doi1,
                    qi1,
                    k1,
                    v1,
                    out_padded1,
                    softmax_lse1,
                    dqi1,
                    dk1,
                    dv1,
                    ctx.dropout_p,
                    ctx.softmax_scale,
                    ctx.causal,
                    ctx.window_size,
                    rng_state=rng_state1,
                    alibi_bias_max=ctx.alibi_bias_max,
                    tp_world_size=ctx.tp_world_size,
                    tp_rank=ctx.tp_rank,
                )
        torch.cuda.current_stream().wait_stream(get_cp_stream())

        if kv0_is_longer:
            dk[:, :dk1.shape[1]] += dk1
            dv[:, :dv1.shape[1]] += dv1
        else:
            dk[:, :dk0.shape[1]] += dk0
            dv[:, :dv0.shape[1]] += dv0
        dk[:, max(dk0.shape[1], dk1.shape[1]):] = 0
        dv[:, max(dv0.shape[1], dv1.shape[1]):] = 0
        return dqi, dk, dv, None, None, None, None, None


class ShardSaveForBackwardFunction(torch.autograd.Function):
    @staticmethod
    def _invert(perm):
        inv = [0] * len(perm)
        for i, p in enumerate(perm):
            inv[p] = i
        return inv

    @staticmethod
    def forward(ctx, x, data_to_save, group):
        ctx.group = group
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        # convert data_to_save into contiguous.
        ctx.dim_order = data_to_save.dim_order()
        data_to_save = data_to_save.permute(ctx.dim_order)
        data_shard = data_to_save.view(world_size, -1)[rank].clone()
        ctx.shape = data_to_save.shape
        ctx.save_for_backward(data_shard)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        data_shard, = ctx.saved_tensors
        saved_data = data_shard.new_empty(ctx.shape)
        handle = dist.all_gather_into_tensor(saved_data, data_shard, group=ctx.group, async_op=True)
        inv_order = __class__._invert(ctx.dim_order)
        saved_data = saved_data.permute(inv_order)
        # NOTE(lizhouyang): must delete the `_handle` manually to break circular reference.
        saved_data._handle = handle
        return grad_output, saved_data, None


class FlipInplaceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, CP):
        nvtx.push_range("dattention forward")
        ctx.CP = CP
        flip_cp_(k, 0, CP)
        flip_cp_(v, 0, CP)
        return k, v

    @staticmethod
    def backward(ctx, dk, dv):
        flip_cp_(dk, 0, ctx.CP)
        flip_cp_(dv, 0, ctx.CP)
        nvtx.pop_range()
        return dk, dv, None


class ForwardGatherBackwardSliceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, cp_group):
        CP = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)
        ctx.dim = dim
        ctx.CP = CP
        ctx.cp_rank = cp_rank
        x = all_gather_along_dim(x, dim, cp_group)
        flip_cp_(x, dim, CP)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return slice_cp(grad_output, ctx.dim, ctx.CP, ctx.cp_rank), None, None


class SaveShardedForBackwardFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, v, kTi, vTi, group):
        ctx.cp_group = group
        ctx.shape = k.shape
        ctx.save_for_backward(kTi, vTi)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        kTi, vTi = ctx.saved_tensors
        CP = dist.get_world_size(ctx.cp_group)
        n = len(kTi._tensors) if hasattr(kTi, "_tensors") else 1
        kTi = kTi.concat() if hasattr(kTi, "concat") else kTi
        vTi = vTi.concat() if hasattr(vTi, "concat") else vTi
        kTi = kTi.unflatten(0, (n, -1)).transpose(0, 1).contiguous()
        vTi = vTi.unflatten(0, (n, -1)).transpose(0, 1).contiguous()
        kT = kTi.new_empty(CP * kTi.shape[0], *kTi.shape[1:])
        vT = kTi.new_empty(CP * vTi.shape[0], *vTi.shape[1:])
        with dist._coalescing_manager(ctx.cp_group, async_ops=True) as cm:
            dist.all_gather_into_tensor(kT, kTi, group=ctx.cp_group, async_op=True)
            dist.all_gather_into_tensor(vT, vTi, group=ctx.cp_group, async_op=True)
        # delay to DAttentionBackward
        # cm.wait()
        # flip_cp_(kT, 0, CP)
        # flip_cp_(vT, 0, CP)
        # kT = kT.transpose(0, 1).flatten(0, 1)
        # vT = vT.transpose(0, 1).flatten(0, 1)
        kT = kT.flatten(0, 1)
        vT = vT.flatten(0, 1)
        k = kT.transpose(0, 1)
        v = vT.transpose(0, 1)
        k._handle = cm
        v._handle = cm

        return grad_x, k, v, None, None, None


def dattention(qTi, kTi, vTi, cp_group, *, kv_cache, layer_idx, alibi_bias_max, tp_world_size, tp_rank):
    if dist.get_world_size(cp_group) == 1:
        return flash_attn_func(qTi.transpose(0, 1), kTi.transpose(0, 1), vTi.transpose(0, 1), causal=True,
                               alibi_bias_max=alibi_bias_max, tp_world_size=tp_world_size, tp_rank=tp_rank)
    CudnnAttnFunc._ALLOW_GRAPH_CREATION = True
    qi = qTi.transpose(0, 1)
    kTi, vTi = kv_cache.update(layer_idx, [kTi, vTi])
    kT, vT = DAttentionPreFunction.apply(kTi, vTi, cp_group)
    k = kT.transpose(0, 1)
    v = vT.transpose(0, 1)
    oi, k, v = DAttentionFunction.apply(qi, k, v, False, cp_group, alibi_bias_max, tp_world_size, tp_rank)
    oTi = oi.transpose(0, 1)
    def save(x):
        return SaveShardedForBackwardFunc.apply(x, k, v, kTi, vTi, get_context_parallel_group_slow())
    return oTi, save


def dattention_overlap(qTi, kv_2sbad, cp_group, *, kv_cache, layer_idx, alibi_bias_max, tp_world_size, tp_rank):
    """The layout of kv_2sbad is (2, s, b, num_heads, head_dim).
    This is the native layout after gathering V and K respectively.
    """
    # TODO(lizhouyang):
    assert False, "not impl"
    qi = qTi.transpose(0, 1)
    kv = kv_2sbad.permute(2, 1, 0, 3, 4)
    CP = dist.get_world_size(cp_group)
    assert CP >= 2, "dattention overlap is not optimized for CP=1"
    kv = FlipInplaceFunction.apply(kv, CP)
    kv = kv_cache.update(layer_idx, [kv.transpose(0, 1)])[0].transpose(0, 1)
    oi, kv = DAttentionFunction.apply(qi, kv, kv_cache, cp_group, alibi_bias_max, tp_world_size, tp_rank)
    oTi = oi.transpose(0, 1)
    return oTi, kv


    @staticmethod
    def backward(ctx, grad_x):
        kTi, vTi = ctx.saved_tensors
        kT = kTi.new_empty(ctx.shape)
        vT = vTi.new_empty(ctx.shape)
        with dist._coalescing_manager(ctx.group, async_ops=True) as cm:
            dist.all_gather_into_tensor(kT, kTi, group=ctx.group, async_op=True)
            dist.all_gather_into_tensor(vT, vTi, group=ctx.group, async_op=True)
        k = kT.transpose(0, 1)
        v = vT.transpose(0, 1)
        k._handle = cm
        v._handle = cm
        return grad_x, k, v, None, None, None


def dattention(qTi, kTi, vTi, cp_group, *, overlap, kv_cache, layer_idx, alibi_bias_max, tp_world_size, tp_rank):
    if dist.get_world_size(cp_group) == 1 and not kv_cache:
        return flash_attn_func(qTi.transpose(0, 1), kTi.transpose(0, 1), vTi.transpose(0, 1), causal=True,
                               alibi_bias_max=alibi_bias_max, tp_world_size=tp_world_size, tp_rank=tp_rank)
    if kv_cache:
        assert not kv_cache.ctx_pair, 'context exchange is not implemented for CP-KV.'
    if overlap: # k and v are already all-gathered in to_qkv
        CP = dist.get_world_size(cp_group)
        assert CP >= 2, "dattention overlap is not optimized for CP=1"
        kT, vT = FlipInplaceFunction.apply(kTi, vTi, CP)
    else:
        kT, vT = DAttentionPreFunction.apply(kTi, vTi, cp_group)
    kT, vT = kv_cache.update(layer_idx, [kT, vT])
    qi = qTi.transpose(0, 1)
    k = kT.transpose(0, 1)
    v = vT.transpose(0, 1)
    oi, k, v = DAttentionFunction.apply(qi, k, v, kv_cache, cp_group, alibi_bias_max, tp_world_size, tp_rank)
    oTi = oi.transpose(0, 1)
    if kv_cache:
        save = None
    else:
        if overlap:
            cp_rank = dist.get_rank(cp_group)
            kTi = slice_cp(kT, 0, CP, cp_rank)
            vTi = slice_cp(vT, 0, CP, cp_rank)
        def save(x):
            return SaveShardedForBackwardFunc.apply(x, k, v, kTi, vTi, get_context_parallel_group_slow())
    return oTi, save


def forward_gather_backward_slice(x, dim, cp_group):
    return ForwardGatherBackwardSliceFunction.apply(x, dim, cp_group)
