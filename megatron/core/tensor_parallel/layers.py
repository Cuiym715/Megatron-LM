# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

from __future__ import annotations

import math
import os
from pkg_resources import packaging
import sys
from typing import Callable, Optional, Union
import warnings

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import transformer_engine
if "transformer_engine_torch" in sys.modules:
    import transformer_engine_torch as tex
else:
    import transformer_engine_extensions as tex
from transformer_engine.pytorch.cpp_extensions import gemm as te_gemm

from torch.amp import custom_fwd, custom_bwd

from megatron.core.parallel_state import (
    get_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_global_memory_buffer,
    get_context_parallel_world_size,
    get_context_parallel_group,
    get_global_te_user_buffer,
)
from megatron.core.utils import _te_version
from .mappings import (
    reduce_from_model_parallel_region,
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    _reduce_scatter_along_first_dim,
)

from .random import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from .utils import (
    divide,
    split_tensor_along_last_dim,
    VocabUtility,
    get_workspace,
    get_ub_copy_stream
)

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}


_TE_OVERLAP_ALGO = getattr(tex, "UbufOverlapAlgo", getattr(tex, "CommOverlapAlgo", None))
if _TE_OVERLAP_ALGO is None:
    SPLIT_PIPELINED_AG_P2P = None
elif _te_version >= packaging.version.Version("1.6.0"):
    SPLIT_PIPELINED_AG_P2P = getattr(
        _TE_OVERLAP_ALGO,
        "SPLIT_PIPELINED_AG_P2P",
        getattr(_TE_OVERLAP_ALGO, "SPLIT_PIPELINED_AG", None),
    )
else:
    SPLIT_PIPELINED_AG_P2P = getattr(_TE_OVERLAP_ALGO, "SPLIT_PIPELINED_AG", None)


def param_is_not_tensor_parallel_duplicate(param):
    return (hasattr(param, 'tensor_model_parallel') and
            param.tensor_model_parallel) or (
                get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute,
                    getattr(source_tensor, attribute))
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(
    weight, init_method, partition_dim, stride=1, expert_parallel=False
):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    if not expert_parallel:
        with get_cuda_rng_tracker().fork():
            init_method(weight)
    else:
        with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
            init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False,
                                  *, params_dtype=torch.float32):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    master_weight = master_weight.to(dtype=params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.

    Keyword Arguments:
        init_method: method to initialize weights.
        params_dtype
        use_cpu_initialization
        perform_initialization
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, *,
                 init_method=init.xavier_normal_,
                 params_dtype: torch.dtype=torch.float32,
                 use_cpu_initialization: bool=False,
                 perform_initialization: bool=True,
                 padding_idx: int=None,
                 vocab_in_pp: bool=False):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        # Set the detauls for compatibility.
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.vocab_in_pp = vocab_in_pp
        if self.vocab_in_pp:
            self.model_parallel_size = torch.distributed.get_world_size(get_model_parallel_group())
            self.model_parallel_rank = torch.distributed.get_rank(get_model_parallel_group())
        else:
            self.model_parallel_size = get_tensor_model_parallel_world_size()
            self.model_parallel_rank = get_tensor_model_parallel_rank()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, self.model_parallel_rank,
                self.model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight, self.num_embeddings, self.embedding_dim,
                    self.num_embeddings_per_partition, 0, init_method,
                    params_dtype=params_dtype)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method,
                                              partition_dim=0, stride=1)

    def forward(self, input_):
        if self.model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        if self.vocab_in_pp:
            output = reduce_from_model_parallel_region(output_parallel)
        else:
            output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, input, weight, bias, gradient_accumulation_fusion,
                async_grad_allreduce, sequence_parallel, ub_fw_obj, ub_bw_obj,
                recompute_func, pipeline_parallel):
        ctx.save_for_backward(input, weight)
        ctx.recompute_func = recompute_func
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel
        ctx.group = get_model_parallel_group() if pipeline_parallel \
            else get_tensor_model_parallel_group()
        if weight.requires_grad:
            ctx.main_grad = weight.main_grad
        ctx.ub_bw_obj = ub_bw_obj
        if recompute_func is not None:
            with torch.no_grad():
                input = recompute_func(input)
        world_size = torch.distributed.get_world_size(ctx.group)
        if sequence_parallel and ub_fw_obj is None:

            dim_sizes = list(input.size())
            dim_sizes[0] = dim_sizes[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_sizes, input.dtype, "mpu")
            torch.distributed.all_gather_into_tensor(
                all_gather_buffer,
                input,
                group=ctx.group)
            total_input = all_gather_buffer
        else:
            total_input = input

        if ub_fw_obj is None:
            output = torch.matmul(total_input, weight.t())
        else:
            partitial_input = None
            out_dim_sizes = list(total_input.shape)
            if(not sequence_parallel): # rs
                output = torch.matmul(total_input, weight.t())
                output = _reduce_scatter_along_first_dim(output)
            else: # ag
                out_dim_sizes[0]  = out_dim_sizes[0] * world_size
                out_dim_sizes[-1] = weight.shape[0]
                out = None
                rs_out = None
                ub_algo = SPLIT_PIPELINED_AG_P2P
                ub_fw_obj.copy_input_to_ubuf(input, True)
                total_input = ub_fw_obj.get_ubuf_output(1)

                output, _, _ = te_gemm(
                    weight,
                    total_input,
                    total_input.dtype,
                    get_workspace(),
                    bias=None,
                    use_bias=False,
                    layout="TN",
                    out=out,
                    grad=False,
                    ub_algo=ub_algo,
                    ub=ub_fw_obj,
                    extra_output_tensor=rs_out
                )
                output = output.view(out_dim_sizes)

        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        ub_bw_obj = ctx.ub_bw_obj
        use_bias = ctx.use_bias
        recompute_func = ctx.recompute_func

        if recompute_func is not None:
            recompute_input, weight = ctx.saved_tensors
            detach_inputs = recompute_input.detach()
            detach_inputs.requires_grad = recompute_input.requires_grad
            with torch.enable_grad():
                recompute_out = recompute_func(detach_inputs)
            input = recompute_out.detach()
        else:
            input, weight = ctx.saved_tensors

        if not weight.requires_grad:
            pass
        elif ctx.sequence_parallel:
            world_size = torch.distributed.get_world_size(ctx.group)
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            handle = torch.distributed.all_gather_into_tensor(
                all_gather_buffer,
                input,
                group=ctx.group, async_op=True)

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer
        else:
            total_input = input

        grad_input = None
        if ub_bw_obj is None:
            grad_input = grad_output.matmul(weight)
        else:
            ub_algo = SPLIT_PIPELINED_AG_P2P
            dim_sizes = list(grad_output.shape)
            # grad_output_all = torch.empty_like(ub_bw_obj.get_ubuf_output(1))
            # torch.distributed.all_gather_into_tensor(grad_output_all, grad_output, group=get_tensor_model_parallel_group())

            ub_bw_obj.copy_input_to_ubuf(grad_output, True)
            grad_output_all = ub_bw_obj.get_ubuf_output(1)
            grad_input, _, _ = te_gemm(
                weight,
                grad_output_all,
                grad_output_all.dtype,
                get_workspace(),
                bias=None,
                use_bias=False,
                layout="NN",
                out=None,
                grad=False,
                ub_algo=ub_algo,
                ub=ub_bw_obj,
                extra_output_tensor=None
            )
            grad_output = grad_output_all.view([-1] + dim_sizes[1:])
            # grad_input = grad_output.matmul(weight)
            grad_input = grad_input.view(list(grad_output.shape)[0:-1] + [-1])

        if weight.requires_grad and ctx.sequence_parallel:
            handle.wait()

        # Doing gather + slicing during the NeMo forward pass can make this tensor
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
        # clones it if it's not contiguous:
        # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1],
                                       grad_output.shape[2])
        if weight.requires_grad:
            # XXX(lizhouyang): there may be a layout transform.
            total_input = total_input.reshape(total_input.shape[0] * total_input.shape[1],
                                              total_input.shape[2])

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                    grad_input, group=ctx.group, async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(dim_size, dtype=input.dtype,
                                         device=torch.cuda.current_device(),
                                         requires_grad=False)
            # reduce_scatter
            handle = torch.distributed.reduce_scatter_tensor(sub_grad_input, grad_input,
                                                             group=ctx.group,
                                                             async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation


        if not weight.requires_grad:
            grad_weight = None
        elif ctx.gradient_accumulation_fusion:
            if ctx.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, ctx.main_grad)
            elif ctx.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output, ctx.main_grad)
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
            grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            if recompute_func is not None:
                torch.autograd.backward(recompute_out, sub_grad_input)
                sub_grad_input = detach_inputs.grad
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()

        if recompute_func is not None:
            torch.autograd.backward(recompute_out, grad_input)
            grad_input = detach_inputs.grad

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel_enabled: bool,
    ub_fw_obj: Union[tex.UbufCommOverlap, tex.UbufP2PCommOverlap] = None,
    ub_bw_obj: Optional[tex.UbufP2PCommOverlap] = None,
    recompute_func: Optional[Callable] = None,
    pipeline_parallel: bool = False,
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calcluation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Arguments:

    input (torch.Tensor required): input like torch.nn.functional.linear

    weight (torch.Tensor required): weight like torch.nn.functional.linear

    bias (torch.Tensor optional): bias like torch.nn.functional.linear

    gradient_accumulation_fusion (bool required): Perform the gradient
        accumulation fusion, requires the custom CUDA extension
        fused_weight_gradient_mlp_cuda module. To use
        gradient_accumulation_fusion you must install APEX with
        --cpp_ext and --cuda_ext. For example: "pip install
        --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
        " Note that the extension requires CUDA>=11. Otherwise, you
        must turn off gradient accumulation fusion."

    async_grad_allreduce (bool required): Do the allreduce of input
        gradients asyncronously with the computation of weight
        gradients. If sequence_parallel_enabled is True, this must be
        False, as no all reduce is performed.

    sequence_parallel_enabled (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.
    """
    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel_enabled,
        ub_fw_obj,
        ub_bw_obj,
        recompute_func,
        pipeline_parallel,
    ]

    if not linear_with_grad_accumulation_and_async_allreduce.warned:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            if sequence_parallel_enabled:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup")
                linear_with_grad_accumulation_and_async_allreduce.warned = True

            if async_grad_allreduce:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup")
                linear_with_grad_accumulation_and_async_allreduce.warned = True

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)

linear_with_grad_accumulation_and_async_allreduce.warned = False


class LinearQKVWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, input, weight, bias, gradient_accumulation_fusion,
                async_grad_allreduce, sequence_parallel, hidden_size_per_attention_head, k_pos_emb, ub_fw_obj, recompute_func):
        ctx.save_for_backward(input, weight, k_pos_emb)
        ctx.recompute_func = recompute_func
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel
        if weight.requires_grad:
            ctx.main_grad = weight.main_grad
        ctx.ub_fw_obj = ub_fw_obj
        if recompute_func is not None:
            with torch.no_grad():
                input = recompute_func(input)

        if sequence_parallel and ub_fw_obj is None:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed.all_gather_into_tensor(
                all_gather_buffer,
                input,
                group=get_tensor_model_parallel_group())
            total_input = all_gather_buffer
        else:
            if ub_fw_obj is not None:
                ub_fw_obj.copy_input_to_ubuf(input, True)
                _, b, h = input.shape
                total_input = ub_fw_obj.get_ubuf_output(1)
                total_input = total_input.view(-1, b, h)
            else:
                total_input = input

        s, b, h = total_input.shape
        a = weight.shape[0] // 3 // hidden_size_per_attention_head
        d = hidden_size_per_attention_head
        k = torch.empty(s * get_context_parallel_world_size(), b, a, d, dtype=total_input.dtype, device=total_input.device)
        v = torch.empty(s * get_context_parallel_world_size(), b, a, d, dtype=total_input.dtype, device=total_input.device)
        weight_q, weight_k, weight_v = weight.view(3, a * d, h)
        if ctx.ub_fw_obj is None:
            vi = torch.matmul(total_input, weight_v.t()).view(s, b, a, d)
        else:
            ub_algo = SPLIT_PIPELINED_AG_P2P
            vi, _, _ = te_gemm(
                weight_v,
                total_input,
                total_input.dtype,
                get_workspace(),
                bias=None,
                use_bias=False,
                layout="TN",
                out=None,
                grad=False,
                ub_algo=ub_algo,
                ub=ctx.ub_fw_obj,
                extra_output_tensor=None
            )
            vi = vi.view(s, b, a, d)
        assert vi is not None and vi.is_contiguous()
        handle = torch.distributed.all_gather_into_tensor(v, vi, group=get_context_parallel_group(), async_op=True)
        ki = torch.matmul(total_input, weight_k.t()).view(s, b, a, d)
        if k_pos_emb is not None:
            import fast_rotary_pos_emb
            ki = fast_rotary_pos_emb.forward(ki, k_pos_emb, True)
        if b >= 2:
            warnings.warn("There is a performance regression issue that can be fixed by customizing the fast_roraty_pos_emb output layout")
            assert not ki.is_contiguous(), "Remove the branch if the performance regression issue is fixed"
            ki = ki.contiguous()
        handle.wait()
        assert ki.is_contiguous()
        handle = torch.distributed.all_gather_into_tensor(k, ki, group=get_context_parallel_group(), async_op=True)
        qi = torch.matmul(total_input, weight_q.t()).view(s, b, a, d)
        handle.wait()
        return qi, k, v

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, *args):
        use_fast_cat = True
        use_addmm_inplace = True

        if use_addmm_inplace:
            from megatron.fused_kernels.wrap_gemm_api import addmm_inplace as addmm
        else:
            from torch import addmm

        grad_qi, grad_k, grad_v = args
        use_bias = ctx.use_bias
        recompute_func = ctx.recompute_func

        if recompute_func is not None:
            recompute_input, weight, k_pos_emb = ctx.saved_tensors
            detach_inputs = recompute_input.detach()
            detach_inputs.requires_grad = recompute_input.requires_grad
            with torch.enable_grad():
                recompute_out = recompute_func(detach_inputs)
            input = recompute_out.detach()
        else:
            input, weight, k_pos_emb = ctx.saved_tensors

        s, b, a, d = grad_qi.shape
        h = weight.shape[1]
        weight_q, weight_k, weight_v = weight.view(3, a * d, h)
        assert grad_qi.is_contiguous()
        grad_ki = torch.empty_like(grad_qi)
        assert grad_k.is_contiguous()
        handle = torch.distributed.reduce_scatter_tensor(grad_ki, grad_k, group=get_context_parallel_group(), async_op=True)
        grad_input = grad_qi.view(s, b, a * d).matmul(weight_q)
        handle.wait()
        grad_vi = torch.empty_like(grad_qi)
        assert grad_v.is_contiguous()
        handle = torch.distributed.reduce_scatter_tensor(grad_vi, grad_v, group=get_context_parallel_group(), async_op=True)
        if k_pos_emb is not None:
            import fast_rotary_pos_emb
            grad_ki = fast_rotary_pos_emb.backward(grad_ki, k_pos_emb, True)
        grad_input = addmm(grad_input.view(s * b, h), grad_ki.view(s * b, a * d), weight_k).view(s, b, h)
        handle.wait()

        if not weight.requires_grad:
            pass
        elif ctx.sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            handle = torch.distributed.all_gather_into_tensor(
                all_gather_buffer,
                input,
                group=get_tensor_model_parallel_group(), async_op=True)

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer
        else:
            total_input = input

        grad_input = addmm(grad_input.view(s * b, h), grad_vi.view(s * b, a * d), weight_v).view(s, b, h)
        if use_fast_cat:
            import fast_cat_cuda
            grad_output = torch.empty(s, b, 3 * a * d, dtype=grad_qi.dtype, device=grad_qi.device)
            fast_cat_cuda.cat([grad_qi.data_ptr(), grad_ki.data_ptr(), grad_vi.data_ptr()], grad_output.data_ptr(),
                              s * b, a * d * grad_output.element_size(), torch.cuda.current_stream().cuda_stream)
        else:
            grad_output = torch.concat([grad_qi, grad_ki, grad_vi], dim=2)
            grad_output = grad_output.view(s, b, 3 * a * d)

        if weight.requires_grad and ctx.sequence_parallel:
            handle.wait()

        # Doing gather + slicing during the NeMo forward pass can make this tensor
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
        # clones it if it's not contiguous:
        # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1],
                                       grad_output.shape[2])
        if weight.requires_grad:
            total_input = total_input.view(total_input.shape[0] * total_input.shape[1],
                                           total_input.shape[2])

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                    grad_input, group=get_tensor_model_parallel_group(), async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(dim_size, dtype=input.dtype,
                                         device=torch.cuda.current_device(),
                                         requires_grad=False)
            # reduce_scatter
            handle = torch.distributed.reduce_scatter_tensor(sub_grad_input, grad_input,
                                                             group=get_tensor_model_parallel_group(),
                                                             async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation


        if not weight.requires_grad:
            grad_weight = None
        elif ctx.gradient_accumulation_fusion:
            if ctx.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, ctx.main_grad)
            elif ctx.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output, ctx.main_grad)
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
            grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            if recompute_func is not None:
                torch.autograd.backward(recompute_out, sub_grad_input)
                sub_grad_input = detach_inputs.grad
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()

        if recompute_func is not None:
            torch.autograd.backward(recompute_out, grad_input)
            grad_input = detach_inputs.grad

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


def linear_qkv_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel_enabled: bool,
    hidden_size_per_attention_head: int,
    k_pos_emb: torch.Tensor,
    ub_fw_obj: Optional[tex.UbufP2PCommOverlap] = None,
    recompute_func: Optional[Callable] = None,
) -> torch.Tensor:
    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel_enabled,
        hidden_size_per_attention_head,
        k_pos_emb,
        ub_fw_obj,
        recompute_func,
    ]

    if not linear_qkv_with_grad_accumulation_and_async_allreduce.warned:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            if sequence_parallel_enabled:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup")
                linear_qkv_with_grad_accumulation_and_async_allreduce.warned = True

            if async_grad_allreduce:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup")
                linear_qkv_with_grad_accumulation_and_async_allreduce.warned = True

    return LinearQKVWithGradAccumulationAndAsyncCommunication.apply(*args)

linear_qkv_with_grad_accumulation_and_async_allreduce.warned = False


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
        is_expert: If True, the layer is treated as an MoE expert layer.
        async_tensor_model_parallel_allreduce:
        params_dtype:
        use_cpu_initialization:
        gradient_accumulation_fusion:
        sequence_parallel_enabled:
    """

    def __init__(self, input_size, output_size, *,
                 bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 is_expert: bool = False,
                 async_tensor_model_parallel_allreduce=True,
                 params_dtype=torch.float32,
                 use_cpu_initialization=False,
                 perform_initialization=True,
                 gradient_accumulation_fusion=False,
                 sequence_parallel_enabled: bool = False,
                 cp_overlap: bool = False,
                 hidden_size_per_attention_head: int = -1,
                 ub_obj: Optional[tex.UbufP2PCommOverlap] = None
                 ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        from megatron import get_args; self.expert_parallel = get_args().expert_model_parallel_size > 1
        self.ub_obj = ub_obj

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=params_dtype))
            if perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight, self.output_size, self.input_size,
                    self.output_size_per_partition, 0, init_method,
                    stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=0,
                    stride=stride,
                    expert_parallel=(self.is_expert and self.expert_parallel),
                )

        setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))

        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
        else:
            self.register_parameter('bias', None)

        self.cp_overlap = cp_overlap
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        if cp_overlap and perform_initialization:
            self.convert_to_cp_overlap_weights()

        self.async_tensor_model_parallel_allreduce = (
                async_tensor_model_parallel_allreduce and
                world_size > 1)
        if sequence_parallel_enabled:
            if world_size <= 1:
                warnings.warn(
                    f"`sequence_parallel_enabled` is set to `True`, but tensor model parallel size is {world_size}. "
                    f"Disabling sequence parallel."
                )
                sequence_parallel_enabled = False
        self.sequence_parallel_enabled = sequence_parallel_enabled

        if gradient_accumulation_fusion:
            if not _grad_accum_fusion_available:
                raise RuntimeError(
                    "ColumnParallelLinear was called with gradient_accumulation_fusion set "
                    "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
                    "module is not found. To use gradient_accumulation_fusion you must "
                    "install APEX with --cpp_ext and --cuda_ext. For example: "
                    "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\" "
                    "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
                    "gradient accumulation fusion."
                )
        self.gradient_accumulation_fusion = gradient_accumulation_fusion

        self.explicit_expert_comm = self.is_expert and (
            self.sequence_parallel_enabled or self.expert_parallel
        )

        if self.async_tensor_model_parallel_allreduce and self.sequence_parallel_enabled:
            raise RuntimeError(
                "`async_tensor_model_parallel_allreduce` and `sequence_parallel_enabled` "
                "cannot be enabled at the same time."
            )


    def forward(self, input_, k_pos_emb=None, recompute_func=None):
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        bias = self.bias if not self.skip_bias_add else None

        if self.async_tensor_model_parallel_allreduce or \
                self.sequence_parallel_enabled or \
                self.explicit_expert_comm:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        if self.cp_overlap:
            output_parallel = linear_qkv_with_grad_accumulation_and_async_allreduce(
                input=input_parallel,
                weight=self.weight,
                bias=bias,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                async_grad_allreduce=False
                if self.explicit_expert_comm
                else self.async_tensor_model_parallel_allreduce,
                sequence_parallel_enabled=False if self.explicit_expert_comm else self.sequence_parallel_enabled,
                hidden_size_per_attention_head=self.hidden_size_per_attention_head,
                k_pos_emb=k_pos_emb,
                ub_fw_obj=self.ub_obj,
                recompute_func=recompute_func,
            )
        else:
            output_parallel = linear_with_grad_accumulation_and_async_allreduce(
                input=input_parallel,
                weight=self.weight,
                bias=bias,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                async_grad_allreduce=False
                if self.explicit_expert_comm
                else self.async_tensor_model_parallel_allreduce,
                sequence_parallel_enabled=False if self.explicit_expert_comm else self.sequence_parallel_enabled,
                ub_fw_obj=self.ub_obj,
                recompute_func=recompute_func,
            )
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel_enabled
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def convert_to_cp_overlap_weights(self):
        assert self.bias is None, "bias is not implemented when cp_overlap is enabled"
        weight_qkv_3adh = self.weight.data.view(-1, 3, self.hidden_size_per_attention_head, self.weight.shape[1]).transpose(0, 1)
        self.weight.data.copy_(weight_qkv_3adh.reshape(self.weight.shape).clone())

    def convert_from_cp_overlap_weights(self):
        assert self.bias is None, "bias is not implemented when cp_overlap is enabled"
        weight_qkv_a3dh = self.weight.data.view(3, -1, self.hidden_size_per_attention_head, self.weight.shape[1]).transpose(0, 1)
        self.weight.data.copy_(weight_qkv_a3dh.reshape(self.weight.shape).clone())


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments:
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
        params_dtype:
        use_cpu_initialization:
        perform_initialization:
        gradient_accumulation_fusion:
        sequence_parallel_enabled:
    """

    def __init__(self, input_size, output_size, *,
                 bias=True, input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 params_dtype=torch.float32,
                 use_cpu_initialization=False,
                 perform_initialization=True,
                 gradient_accumulation_fusion=False,
                 sequence_parallel_enabled: bool = False,
                 is_expert: bool = False,
                 ub_fw_obj=None,
                 ub_bw_obj=None
                 ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        from megatron import get_args; self.expert_parallel = get_args().expert_model_parallel_size > 1
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.ub_fw_obj = ub_fw_obj
        self.ub_bw_obj = ub_bw_obj
        if self.sequence_parallel_enabled and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel_enabled`, `input_is_parallel` must be `True`")

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=params_dtype))
            if perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight, self.output_size, self.input_size,
                    self.input_size_per_partition, 1, init_method,
                    stride=stride, return_master_weight=keep_master_weight_for_test,
                    params_dtype=params_dtype)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=1,
                    stride=stride,
                    expert_parallel=(self.is_expert and self.expert_parallel),
                )
        setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))

        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=params_dtype))
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
            setattr(self.bias, 'sequence_parallel', sequence_parallel_enabled)

            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        self.explicit_expert_comm = self.is_expert and (
            self.sequence_parallel_enabled or self.expert_parallel
        )


    def forward(self, input_, cp_data_to_save=None, recompute_func=None):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel_enabled
            input_parallel = scatter_to_tensor_model_parallel_region(input_)

        # Matrix multiply.
        output_parallel = linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
            ub_fw_obj=self.ub_fw_obj,
            ub_bw_obj=self.ub_bw_obj,
            recompute_func=recompute_func,
        )

        if isinstance(cp_data_to_save, Callable):
            output_parallel = cp_data_to_save(output_parallel)

        # All-reduce across all the partitions.
        if self.explicit_expert_comm:
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel_enabled:
            if self.ub_fw_obj is None:
                output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
            else:
                output_ = output_parallel
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias
