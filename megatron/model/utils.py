# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Utilities for models."""

import math

import torch

from megatron import get_args
from megatron.core import mpu
from megatron.core.context_parallel import dattention

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_

def init_method_xavier_normal(beta=1.0):
    """Init method based on N(0, beta*sqrt(2/(fan_in+fan_out)))."""
    def init_(tensor):
        return torch.nn.init.xavier_normal_(tensor, gain=beta)

    return init_

def scaled_init_method_xavier_normal(beta, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)）. sigma=beta*sqrt(2/(fan_in+fan_out))"""
    scaled_beta = beta / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.xavier_normal_(tensor, gain=scaled_beta)

    return init_

def attention_mask_func(attention_scores, attention_mask, alibi_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)

    # if torch.distributed.get_rank() == 0:
    #     import ipdb
    #     ipdb.set_trace()

    if alibi_mask is not None:
        attention_scores = attention_scores + alibi_mask

    return attention_scores

def get_linear_layer(rows, columns, init_method):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if get_args().perform_initialization:
        init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer

@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))
def openai_gelu(x):
    return gelu_impl(x)

#This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype)+torch.ones_like(x).to(dtype=x.dtype))


def swiglu(x):
    x = torch.chunk(x, 2, dim=-1)
    return torch.nn.functional.silu(x[0]) * x[1]


TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])
try:
    import triton
except ModuleNotFoundError:
    triton = None
_compile_swiglu = (TORCH_MAJOR, TORCH_MINOR) >= (2, 0) and triton is not None
if _compile_swiglu:
    swiglu = torch.compile(swiglu, dynamic=True)


def slice_lm_inputs_along_cp(input_ids, position_ids, attention_mask, labels):
    CP = mpu.get_context_parallel_world_size()
    if CP >= 2:
        # Check inputs with the same context parallel rank are equal
        args = get_args()
        if args.curr_iteration < args.iteration + args.kaimm_warmup_iters:
            max_input_ids = input_ids.clone()
            torch.distributed.all_reduce(max_input_ids, op=torch.distributed.ReduceOp.MAX,
                                        group=mpu.get_context_parallel_group())
            if (max_input_ids != input_ids).any():
                raise ValueError("Inputs with the same get_data_parallel_for_sample_rank() should be equal. "
                                    "Please check the dataloader.")
        cp_rank = mpu.get_context_parallel_rank()
        input_ids = dattention.slice_cp(input_ids, 1, CP, cp_rank)
        if isinstance(position_ids, torch.Tensor):
            position_ids = dattention.slice_cp(position_ids, -1, CP, cp_rank)
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = dattention.slice_cp(attention_mask, -1, CP, cp_rank)
        labels = dattention.slice_cp(labels, 1, CP, cp_rank)
    return input_ids, position_ids, attention_mask, labels


def gather_post_lm_output_along_cp(output):
    return dattention.forward_gather_backward_slice(output, 1, mpu.get_context_parallel_group())


def pad_to_be_divisible(input: torch.Tensor, divisor, dim):
    shape = list(input.shape)
    rem = shape[dim] % divisor
    if rem == 0:
        return input
    shape[dim] += divisor - rem
    output = input.new_zeros(shape)
    output.narrow(dim, 0, input.shape[dim]).copy_(input)
    return output


def pad_and_permute(input: torch.Tensor, dim):
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    input = pad_to_be_divisible(input, pp_size, dim)
    input = input.unflatten(dim, (tp_size, pp_size, -1)).transpose(dim, dim + 1).flatten(dim, dim + 2)
    return input


def unpermute_and_unpad(input: torch.Tensor, dim, shape):
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    input = input.unflatten(dim, (pp_size, tp_size, -1)).transpose(dim, dim + 1).flatten(dim, dim + 2)
    input = input.narrow(dim, 0, shape[dim])
    return input
