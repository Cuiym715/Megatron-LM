import contextlib
import functools
import math
import os
import threading
import torch
import torch.utils.checkpoint
import traceback
import warnings

from collections import defaultdict

try:
    from packaging.version import Version
except ModuleNotFoundError:
    pass

try:
    import cudnn
except ModuleNotFoundError:
    cudnn = None


@contextlib.contextmanager
def rewrite_env_if_not_set(name, value):
    env_exists = name in os.environ
    if env_exists:
        value_orig = os.environ[name]
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if env_exists:
            os.environ[name] = value_orig
        else:
            os.environ.pop(name)


class CudnnAttnFunc(torch.autograd.Function):
    _GRAPHS = dict()
    _ALLOW_GRAPH_CREATION = False
    _ONLY_GRAPH_CREATION = False
    _WARNED = False
    _CUDNN_HANDLES = defaultdict(lambda: cudnn.create_handle())

    @classmethod
    def forward(cls, ctx, q_gpu, k_gpu, v_gpu, causal, varlen, max_seq_len_q, max_seq_len_kv, cu_seq_len_q, cu_seq_len_kv):
        o_gpu = torch.empty_like(q_gpu)
        is_inference = not q_gpu.requires_grad and not k_gpu.requires_grad and not v_gpu.requires_grad
        stats_gpu = cls.forward_no_ctx(q_gpu, k_gpu, v_gpu, causal, varlen, is_inference, max_seq_len_q, max_seq_len_kv, cu_seq_len_q, cu_seq_len_kv, o_gpu)
        ctx.causal = causal
        ctx.varlen = varlen
        ctx.max_seq_len_q = max_seq_len_q
        ctx.max_seq_len_kv = max_seq_len_kv
        ctx.save_for_backward(q_gpu, k_gpu, v_gpu, o_gpu, stats_gpu, cu_seq_len_q, cu_seq_len_kv)
        return o_gpu

    @classmethod
    def backward(cls, ctx, dO_gpu):
        dO_gpu = dO_gpu.contiguous()  # Workaround for cuDNN bug: produces wrong result when dO is not contiguous
        q_gpu, k_gpu, v_gpu, o_gpu, stats_gpu, cu_seq_len_q, cu_seq_len_kv = ctx.saved_tensors
        dQ_gpu = torch.empty_like(q_gpu)
        dK_gpu = torch.empty_like(k_gpu)
        dV_gpu = torch.empty_like(v_gpu)
        cls.backward_no_ctx(dO_gpu, q_gpu, k_gpu, v_gpu, o_gpu, stats_gpu, ctx.causal, ctx.varlen, ctx.max_seq_len_q, ctx.max_seq_len_kv, cu_seq_len_q, cu_seq_len_kv, dQ_gpu, dK_gpu, dV_gpu)
        return dQ_gpu, dK_gpu, dV_gpu, None, None, None, None, None, None

    @classmethod
    def forward_no_ctx(cls, q_gpu, k_gpu, v_gpu, causal, varlen, is_inference, max_seq_len_q, max_seq_len_kv, cu_seq_len_q, cu_seq_len_kv, o_gpu):
        if not cls._WARNED and not cls._ALLOW_GRAPH_CREATION:
            msg = "Use CudnnAttnFunc."
            if torch.distributed.is_initialized():
                msg = f"[Rank {torch.distributed.get_rank()}] " + msg
            print(msg, flush=True)
            cls._WARNED = True

        b, s_q, a, d = q_gpu.shape
        _, s_kv, g, _ = k_gpu.shape
        dtype = q_gpu.dtype
        if varlen:
            s_q = max_seq_len_q
            s_kv = max_seq_len_kv
            cu_seq_len_q = cu_seq_len_q.view(b, 1, 1, 1).contiguous()
            cu_seq_len_kv = cu_seq_len_kv.view(b, 1, 1, 1).contiguous()

        q_gpu_transposed = q_gpu.transpose(1, 2)
        k_gpu_transposed = k_gpu.transpose(1, 2)
        v_gpu_transposed = v_gpu.transpose(1, 2)
        o_gpu_transposed = o_gpu.transpose(1, 2)
        # XXX(lizhouyang) transposed `stats_gpu` is not supported.
        stats_gpu = None if is_inference else torch.empty(b, a, s_q, 1, dtype=torch.float32, device="cuda")

        graph_args = (b, s_q, s_kv, a, g, d, causal, varlen, dtype, is_inference,
                      q_gpu_transposed.stride(), k_gpu_transposed.stride(), v_gpu_transposed.stride(), o_gpu_transposed.stride())
        graph, q, k, v, seq_len_q, seq_len_kv, o, stats = cls.get_graph_forward(*graph_args)
        if cls._ONLY_GRAPH_CREATION:
            return stats_gpu

        variant_pack = {
            q: q_gpu_transposed,
            k: k_gpu_transposed,
            v: v_gpu_transposed,
            o: o_gpu_transposed,
        }
        if varlen:
            variant_pack.update({
                seq_len_q: cu_seq_len_q,
                seq_len_kv: cu_seq_len_kv,
            })
        if not is_inference:
            assert stats.get_data_type() == cudnn.data_type.FLOAT, str(stats.get_data_type())
            variant_pack[stats] = stats_gpu
        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        cls.execute_graph_on_current_stream(graph, variant_pack, workspace)
        return stats_gpu

    @classmethod
    def backward_no_ctx(cls, dO_gpu, q_gpu, k_gpu, v_gpu, o_gpu, stats_gpu, causal, varlen, max_seq_len_q, max_seq_len_kv, cu_seq_len_q, cu_seq_len_kv, dQ_gpu, dK_gpu, dV_gpu):
        b, s_q, a, d = q_gpu.shape
        _, s_kv, g, _ = k_gpu.shape
        dtype = q_gpu.dtype
        if varlen:
            s_q = max_seq_len_q
            s_kv = max_seq_len_kv
            cu_seq_len_q = cu_seq_len_q.view(b, 1, 1, 1).contiguous()
            cu_seq_len_kv = cu_seq_len_kv.view(b, 1, 1, 1).contiguous()

        q_gpu_transposed = q_gpu.transpose(1, 2)
        k_gpu_transposed = k_gpu.transpose(1, 2)
        v_gpu_transposed = v_gpu.transpose(1, 2)
        o_gpu_transposed = o_gpu.transpose(1, 2)
        dO_gpu_transposed = dO_gpu.transpose(1, 2)
        dQ_gpu_transposed = dQ_gpu.transpose(1, 2)
        dK_gpu_transposed = dK_gpu.transpose(1, 2)
        dV_gpu_transposed = dV_gpu.transpose(1, 2)
        # XXX(lizhouyang) transposed `stats_gpu` is not supported.
        stats_gpu = stats_gpu.contiguous()

        graph_args = (b, s_q, s_kv, a, g, d, causal, varlen, dtype,
                      q_gpu_transposed.stride(), k_gpu_transposed.stride(), v_gpu_transposed.stride(), o_gpu_transposed.stride(),
                      dQ_gpu_transposed.stride(), dK_gpu_transposed.stride(), dV_gpu_transposed.stride(), dO_gpu_transposed.stride(),)
        graph_backward, q_backward, k_backward, v_backward, o_backward, dO_backward, stats_backward, seq_len_q_backward, seq_len_kv_backward, dQ_backward, dK_backward, dV_backward = cls.get_graph_backward(*graph_args)
        if cls._ONLY_GRAPH_CREATION:
            return

        variant_pack_backward = {
            q_backward: q_gpu_transposed,
            k_backward: k_gpu_transposed,
            v_backward: v_gpu_transposed,
            o_backward: o_gpu_transposed,
            dO_backward: dO_gpu_transposed,
            stats_backward: stats_gpu,
            dQ_backward: dQ_gpu_transposed,
            dK_backward: dK_gpu_transposed,
            dV_backward: dV_gpu_transposed,
        }
        if varlen:
            variant_pack_backward.update({
                seq_len_q_backward: cu_seq_len_q,
                seq_len_kv_backward: cu_seq_len_kv,
            })
        workspace = torch.empty(graph_backward.get_workspace_size(), device="cuda", dtype=torch.uint8)
        cls.execute_graph_on_current_stream(graph_backward, variant_pack_backward, workspace)

    @classmethod
    def get_graph_forward(cls, b, s_q, s_kv, a, g, d, causal, varlen, dtype, is_inference, q_stride, k_stride, v_stride, o_stride):
        graph_key = "forward", b, s_q, s_kv, a, g, d, causal, varlen, dtype, is_inference, q_stride, k_stride, v_stride, o_stride
        if graph_key in cls._GRAPHS:
            return cls._GRAPHS[graph_key]
        if not cls._ALLOW_GRAPH_CREATION:
            print('forwrad graph_key', graph_key)
            raise ValueError("Unexpected graph creation. Please create graphs in the cudnn_attn_allow_graph_creation context.")

        attn_scale = 1 / math.sqrt(d)

        graph_forward = cudnn.pygraph(
            io_data_type=cudnn.datatypes._torch_to_cudnn_data_type(dtype),
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )

        q_forward = graph_forward.tensor(dim=(b, a, s_q, d), stride=q_stride)
        k_forward = graph_forward.tensor(dim=(b, g, s_kv, d), stride=k_stride)
        v_forward = graph_forward.tensor(dim=(b, g, s_kv, d), stride=v_stride)
        if varlen:
            seq_len_q_forward = graph_forward.tensor(dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
            seq_len_kv_forward = graph_forward.tensor(dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
        else:
            seq_len_q_forward = seq_len_kv_forward = None
        kwargs = dict()
        if causal == 1:         # bottom right.
            if s_q == s_kv:     # optimize for speed.
                kwargs["use_causal_mask"] = True
            else:
                assert not varlen, "cannot use bottom right mask for varlen"
                kwargs["use_causal_mask_bottom_right"] = True  # nvidia-cudnn-frontend>=1.5.0
        elif causal == -1:      # top left.
            kwargs["use_causal_mask"] = True

        o_forward, stats_forward = graph_forward.sdpa(
            name="sdpa",
            q=q_forward, k=k_forward, v=v_forward,
            is_inference=is_inference,
            attn_scale=attn_scale,
            use_padding_mask=varlen,
            seq_len_q=seq_len_q_forward, seq_len_kv=seq_len_kv_forward,
            **kwargs,
        )

        o_forward.set_output(True).set_dim(q_forward.get_dim()).set_stride(o_stride)
        if not is_inference:
            stats_forward.set_output(True).set_dim((b, a, s_q, 1)).set_stride((a*s_q, s_q, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

        graph_forward.validate()
        graph_forward.build_operation_graph()
        graph_forward.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph_forward.check_support()
        graph_forward.build_plans()

        res = graph_forward, q_forward, k_forward, v_forward, seq_len_q_forward, seq_len_kv_forward, o_forward, stats_forward
        cls._GRAPHS[graph_key] = res
        return res

    @classmethod
    def get_graph_backward(cls, b, s_q, s_kv, a, g, d, causal, varlen, dtype, q_stride, k_stride, v_stride, o_stride, dQ_stride, dK_stride, dV_stride, dO_stride):
        graph_key = "backward", b, s_q, s_kv, a, g, d, causal, varlen, dtype, q_stride, k_stride, v_stride, o_stride, dQ_stride, dK_stride, dV_stride, dO_stride
        if graph_key in cls._GRAPHS:
            return cls._GRAPHS[graph_key]
        if not cls._ALLOW_GRAPH_CREATION:
            print('backward graph_key', graph_key)
            raise ValueError("Unexpected graph creation. Please create graphs in the cudnn_attn_allow_graph_creation context.")

        attn_scale = 1 / math.sqrt(d)

        with rewrite_env_if_not_set("CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT", "0"):
            graph_backward = cudnn.pygraph(
                io_data_type=cudnn.datatypes._torch_to_cudnn_data_type(dtype),
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
            )

            q_backward = graph_backward.tensor(dim=(b, a, s_q, d), stride=q_stride)
            k_backward = graph_backward.tensor(dim=(b, g, s_kv, d), stride=k_stride)
            v_backward = graph_backward.tensor(dim=(b, g, s_kv, d), stride=v_stride)
            o_backward = graph_backward.tensor(dim=(b, a, s_q, d), stride=o_stride)
            dO_backward = graph_backward.tensor(dim=(b, a, s_q, d), stride=dO_stride)
            stats_backward = graph_backward.tensor(dim=(b, a, s_q, 1), stride=(a*s_q, s_q, 1, 1), data_type=cudnn.data_type.FLOAT)
            if varlen:
                seq_len_q_backward = graph_backward.tensor(dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
                seq_len_kv_backward = graph_backward.tensor(dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
            else:
                seq_len_q_backward = seq_len_kv_backward = None
            kwargs = dict()
            if causal == 1:         # bottom right.
                if s_q == s_kv:     # optimize for speed.
                    kwargs["use_causal_mask"] = True
                else:
                    assert not varlen, "cannot use bottom right mask for varlen"
                    kwargs["use_causal_mask_bottom_right"] = True  # nvidia-cudnn-frontend>=1.5.0
            elif causal == -1:      # top left.
                kwargs["use_causal_mask"] = True

            dQ_backward, dK_backward, dV_backward = graph_backward.sdpa_backward(
                name="sdpa_backward",
                q=q_backward, k=k_backward, v=v_backward,
                o=o_backward, dO=dO_backward, stats=stats_backward,
                attn_scale=attn_scale,
                use_padding_mask=varlen,
                seq_len_q=seq_len_q_backward, seq_len_kv=seq_len_kv_backward,
                **kwargs,
            )

            dQ_backward.set_output(True).set_dim((b, a, s_q, d)).set_stride(dQ_stride)
            dK_backward.set_output(True).set_dim((b, g, s_kv, d)).set_stride(dK_stride)
            dV_backward.set_output(True).set_dim((b, g, s_kv, d)).set_stride(dV_stride)

            graph_backward.validate()
            graph_backward.build_operation_graph()
            graph_backward.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            graph_backward.check_support()
            graph_backward.build_plans()

        res = graph_backward, q_backward, k_backward, v_backward, o_backward, dO_backward, stats_backward, seq_len_q_backward, seq_len_kv_backward, dQ_backward, dK_backward, dV_backward
        cls._GRAPHS[graph_key] = res
        return res

    @classmethod
    def execute_graph_on_current_stream(cls, graph, variant_pack, workspace):
        handle = cls._CUDNN_HANDLES[threading.current_thread()]
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        graph.execute(variant_pack, workspace, handle=handle)


def _cudnn_attn_backward(dO_gpu, q_gpu, k_gpu, v_gpu, o_gpu, stats_gpu, dQ_gpu, dK_gpu, dV_gpu, causal):
    with rewrite_env_if_not_set("CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT", "0"):
        graph_backward, q_backward, k_backward, v_backward, o_backward, dO_backward, stats_backward, seq_len_q, seq_len_kv, dQ_backward, dK_backward, dV_backward = CudnnAttnFunc.get_graph_backward(
            q_gpu.shape[0], q_gpu.shape[1], k_gpu.shape[1], q_gpu.shape[2], k_gpu.shape[2], q_gpu.shape[3], causal, False, q_gpu.dtype,
            q_gpu.transpose(1, 2).stride(), k_gpu.transpose(1, 2).stride(), v_gpu.transpose(1, 2).stride(), o_gpu.transpose(1, 2).stride(), dQ_gpu.transpose(1, 2).stride(), dK_gpu.transpose(1, 2).stride(), dV_gpu.transpose(1, 2).stride(), dO_gpu.transpose(1, 2).stride(),
        )
        assert q_gpu.stride() == o_gpu.stride()
        variant_pack_backward = {
            q_backward: q_gpu.transpose(1, 2),
            k_backward: k_gpu.transpose(1, 2),
            v_backward: v_gpu.transpose(1, 2),
            o_backward: o_gpu.transpose(1, 2),
            dO_backward: dO_gpu.contiguous().transpose(1, 2),
            stats_backward: stats_gpu,
            dQ_backward: dQ_gpu.transpose(1, 2),
            dK_backward: dK_gpu.transpose(1, 2),
            dV_backward: dV_gpu.transpose(1, 2),
        }
        workspace = torch.empty(graph_backward.get_workspace_size(), device="cuda", dtype=torch.uint8)
        CudnnAttnFunc.execute_graph_on_current_stream(graph_backward, variant_pack_backward, workspace)


@contextlib.contextmanager
def cudnn_attn_allow_graph_creation(only_graph_creation=False):
    allow = CudnnAttnFunc._ALLOW_GRAPH_CREATION
    CudnnAttnFunc._ALLOW_GRAPH_CREATION = True
    only = CudnnAttnFunc._ONLY_GRAPH_CREATION
    CudnnAttnFunc._ONLY_GRAPH_CREATION = only_graph_creation
    try:
        yield
    finally:
        CudnnAttnFunc._ALLOW_GRAPH_CREATION = allow
        CudnnAttnFunc._ONLY_GRAPH_CREATION = only


@functools.lru_cache
def cudnn_attn_check_capability(use_causal_mask_bottom_right):
    if cudnn is None:
        return False
    if cudnn.backend_version() < 90101:
        return False
    if use_causal_mask_bottom_right and Version(cudnn.__version__) < Version("1.5.0"):
        return False
    if torch.cuda.get_device_capability() < (9, 0):
        return False
    return True


def cudnn_attn_func(q, k, v, causal):
    return CudnnAttnFunc.apply(q, k, v, causal, False, None, None, None, None)


def cudnn_attn_varlen_func(q, k, v, causal, max_seq_len_q, max_seq_len_kv, cu_seq_len_q, cu_seq_len_kv):
    return CudnnAttnFunc.apply(q, k, v, causal, True, max_seq_len_q, max_seq_len_kv, cu_seq_len_q, cu_seq_len_kv)


def torch_attn_func(q, k, v, causal):
    if causal == 1:
        s_q, s_kv = q.shape[1], k.shape[1]
        attn_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=q.device).tril_(s_kv - s_q)
    else:
        attn_mask = None
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask, is_causal=causal==-1).transpose(1, 2)


def dry_run():
    from flash_attn.flash_attn_interface import flash_attn_func
    dtype = torch.bfloat16
    do_backward = True
    b, s_q, s_kv, a, g, d, causal = -1, 4096, 4096, 64, 8, 128, True
    max_seq_len_q = 8192
    max_seq_len_kv = 8192

    q = torch.randn(s_q, b, a, d, dtype=dtype, device="cuda").transpose(0, 1).requires_grad_()
    k = torch.randn(s_kv, b, g, d, dtype=dtype, device="cuda").transpose(0, 1).requires_grad_()
    v = torch.randn(s_kv, b, g, d, dtype=dtype, device="cuda").transpose(0, 1).requires_grad_()
    grad_o = torch.randn(s_q, b, a, d, dtype=dtype, device="cuda").transpose(0, 1)

    q_float = q.detach().float().requires_grad_()
    k_float = k.detach().float().requires_grad_()
    v_float = v.detach().float().requires_grad_()
    k_float_a = k_float.repeat_interleave(a // g, dim=2)
    v_float_a = v_float.repeat_interleave(a // g, dim=2)
    o_ref = torch_attn_func(q_float, k_float_a, v_float_a, causal=causal).to(q.dtype)
    o_ref.backward(grad_o)
    grad_q_ref = q_float.grad.to(q.dtype)
    grad_k_ref = k_float.grad.to(k.dtype)
    grad_v_ref = v_float.grad.to(v.dtype)

    for attn_func_lib_name, varlen, include_pad_time in [
        ("cudnn", False, None),
        ("cudnn", True, False),
        ("cudnn", True, True),
        ("flash", False, None),
        ("flash", True, None),
    ]:
        if attn_func_lib_name == "cudnn" and varlen and causal and s_q != s_kv:
            print("cuDNN SDPA does not support varlen + causal_mask_bottom_right, skipping")
            print()
            continue
        if attn_func_lib_name == "cudnn":
            if varlen:
                cu_seq_len_q = torch.full((b,), s_q, dtype=torch.int32, device="cuda")
                cu_seq_len_kv = cu_seq_len_q if s_q == s_kv else torch.full((b,), s_kv, dtype=torch.int32, device="cuda")

            # Compile cuDNN graphs
            with cudnn_attn_allow_graph_creation():
                if varlen:
                    q_padded = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, max_seq_len_q - s_q))
                    k_padded = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, max_seq_len_kv - s_kv))
                    v_padded = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, max_seq_len_kv - s_kv))
                    o_padded = cudnn_attn_varlen_func(q_padded, k_padded, v_padded, causal, max_seq_len_q, max_seq_len_kv, cu_seq_len_q, cu_seq_len_kv)
                    o = o_padded[:, :s_q]
                else:
                    o = cudnn_attn_func(q, k, v, causal)
                if do_backward:
                    o.backward(grad_o)

        warmup_times = 5
        run_times = 10
        ev1 = torch.cuda.Event(enable_timing=True)
        ev2 = torch.cuda.Event(enable_timing=True)
        for i in range(warmup_times + run_times):
            if i == warmup_times:
                ev1.record()
            q.grad = None
            k.grad = None
            v.grad = None
            if varlen:
                if attn_func_lib_name == "cudnn":
                    if include_pad_time:
                        # TODO: ragged
                        q_padded = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, max_seq_len_q - s_q))
                        k_padded = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, max_seq_len_kv - s_kv))
                        v_padded = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, max_seq_len_kv - s_kv))
                    o_padded = cudnn_attn_varlen_func(q_padded, k_padded, v_padded, causal, max_seq_len_q, max_seq_len_kv, cu_seq_len_q, cu_seq_len_kv)
                    o = o_padded[:, :s_q]
                elif attn_func_lib_name == "flash":
                    o = flash_attn_func(q, k, v, causal=causal)
                else:
                    raise ValueError(f"unknown {attn_func_lib_name=}")
            else:
                if attn_func_lib_name == "cudnn":
                    o = cudnn_attn_func(q, k, v, causal=causal)
                elif attn_func_lib_name == "flash":
                    o = flash_attn_func(q, k, v, causal=causal)
                else:
                    raise ValueError(f"unknown {attn_func_lib_name=}")
            if do_backward:
                o.backward(grad_o)
        ev2.record()
        ev2.synchronize()
        dt = ev1.elapsed_time(ev2) / 1000. / run_times
        flops = (1 + 2 * do_backward) * 2 * b * s_q * (s_kv + s_kv - causal * s_q) * (a * d)
        print(f"{attn_func_lib_name} {varlen=} {b=} {s_q=} {s_kv=} {a=} {g=} {d=}")
        print(f"{dt:.6f} sec {flops / dt / 1e12:.3f} TFLOPS")

        print(f"output diff max {(o - o_ref).abs().max()} mean {(o - o_ref).abs().double().mean()}")
        if do_backward:
            print(f"grad_q diff max {(q.grad - grad_q_ref).abs().max()} mean {(q.grad - grad_q_ref).abs().double().mean()}")
            print(f"grad_k diff max {(k.grad - grad_k_ref).abs().max()} mean {(k.grad - grad_k_ref).abs().double().mean()}")
            print(f"grad_v diff max {(v.grad - grad_v_ref).abs().max()} mean {(v.grad - grad_v_ref).abs().double().mean()}")
        print()


if __name__ == "__main__":
    dry_run()
