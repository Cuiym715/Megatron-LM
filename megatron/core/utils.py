# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Utility functions used throughout Megatron core"""
from functools import reduce
from pkg_resources import packaging
import importlib.metadata
import math
import operator
import os
import sys

import torch
try:
    import transformer_engine
    if "transformer_engine_torch" in sys.modules:
        import transformer_engine_torch as tex
    else:
        import transformer_engine_extensions as tex
except:
    tex = None

from megatron import get_args
from megatron.core import parallel_state


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def get_attr_wrapped_model(model, attr):
    """Get an attribute from a wrapped model"""
    if isinstance(model, list):
        raise RuntimeError("_get_attr_wrapped_model given a list of models")

    while not hasattr(model, attr):
        if not hasattr(model, "module"):
            raise RuntimeError(f"_get_attr_wrapped_model couldn't find attribute {attr}")

        model = model.module
    return getattr(model, attr)

def get_model_type(model):
    return get_attr_wrapped_model(model, 'model_type')


def get_te_version():
    def get_te_version_str():
        if hasattr(transformer_engine, '__version__'):
            return str(transformer_engine.__version__)
        else:
            return importlib.metadata.version("transformer-engine")

    return packaging.version.Version(get_te_version_str())


_te_version = get_te_version()


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if self.buffer.get((name, dtype), None) is None or \
                self.buffer[(name, dtype)].numel() < required_len:
            self.buffer[(name, dtype)] = \
                torch.empty(required_len,
                            dtype=dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False)

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)

class GlobalTEUserBuffer:
    """Global Transformer Engine UserBuffer """

    def __init__(self):
        self.buffer_ag = {}
        self.buffer_rs = {}
        assert tex is not None, "Using Transformer Engine userbuffer, please install transformer engine first."
        self.ag_sm_margin = int(os.getenv('UB_AG_SM_MARGIN', '0'))
        self.rs_sm_margin = int(os.getenv('UB_RS_SM_MARGIN', '0'))
        self.cga_size = 2
        self._NUM_MAX_UB_STREAMS = int(os.getenv('UB_MAX_STREAMS', '1'))
        self.aggregate = 0

    def get_ub(self, name, shape, dtype, tp_world_size, tp_rank_id, ag):
        if _te_version >= packaging.version.Version("1.9.0.dev0"):
            assert tex.ubuf_built_with_mpi()
            assert torch.distributed.is_mpi_available()
            mpi_group = torch.distributed.new_group(backend="mpi")
            world_rank = torch.distributed.get_rank(mpi_group)
            world_size = torch.distributed.get_world_size(mpi_group)
            local_rank = world_rank % tp_world_size
            local_size = tp_world_size
            node_id = world_rank // tp_world_size
            num_nodes = world_size // tp_world_size
            atomic_gemm = False
            ub_callbacks = tex.UbufBootstrapCallbacks()
        if(ag):
            if(name not in self.buffer_ag):
                sample_buffer = torch.empty(shape, dtype = dtype, device="cuda")
                if _te_version >= packaging.version.Version("1.9.0.dev0"):
                    is_reduce_scatter = False
                    use_ce = True
                    self.buffer_ag[name] = tex.UbufP2PCommOverlap(
                        sample_buffer,                          # Sample userbuffer
                        world_rank,                             # World rank
                        world_size,                             # World size
                        local_rank,                             # Rank within the node
                        local_size,                             # Number of ranks/GPUs per node
                        node_id,                                # Node ID
                        num_nodes,                              # Number of nodes
                        tp_world_size,                          # Tensor-parallel group size (may be different than local_size)
                        tp_world_size,                          # Number of communication SMs
                        self.cga_size,                          # CGA cluster size
                        self.ag_sm_margin,                      # Set SM margin
                        self.aggregate,                         # Aggregate 2X GEMM chunks
                        self._NUM_MAX_UB_STREAMS,               # Max concurrent GEMM streams
                        is_reduce_scatter,                      # Overlap with reduce scatter
                        atomic_gemm,                            # Use a single GEMM with atomic-counters
                        use_ce,                                 # Use copy engine for P2P communications
                        ub_callbacks,
                    )
                else:
                    self.buffer_ag[name] = tex.UbufP2PCommOverlap(
                        sample_buffer,                          # Sample userbuffer
                        tp_rank_id,                             # Rank id
                        tp_world_size,                          # TP size
                        tp_world_size,                          # Number of communication SMs
                        self.cga_size,                          # CGA cluster size
                        self.ag_sm_margin,                      # Set SM margin
                        self.aggregate,                         # Aggregate 2X GEMM chunks
                        self._NUM_MAX_UB_STREAMS,               # Max concurrent GEMM streams
                        torch.Tensor(),                         # empty tensor to pass to counters
                    )
            return self.buffer_ag[name]
        else:
            if(name not in self.buffer_rs):
                sample_buffer = torch.empty(shape, dtype = dtype, device="cuda")
                if _te_version >= packaging.version.Version("1.9.0.dev0"):
                    self.buffer_rs[name] = tex.UbufCommOverlap(
                        sample_buffer,                          # Sample userbuffer
                        world_rank,                             # World rank
                        world_size,                             # World size
                        local_rank,                             # Rank within the node
                        local_size,                             # Number of ranks/GPUs per node
                        node_id,                                # Node ID
                        num_nodes,                              # Number of nodes
                        tp_world_size,                          # Tensor-parallel group size (may be different than local_size)
                        tp_world_size * 2,                      # Number of communication SMs
                        self.cga_size,                          # CGA cluster size
                        tp_world_size,                          # Number of communication splits
                        self.rs_sm_margin,                      # Set SM margin
                        self._NUM_MAX_UB_STREAMS,               # Max concurrent GEMM streams
                        atomic_gemm,                            # Use a single GEMM with atomic-counters
                        ub_callbacks,
                    )
                else:
                    self.buffer_rs[name] = tex.UbufCommOverlap(
                        sample_buffer,                          # Sample userbuffer
                        tp_rank_id,                             # Rank id
                        tp_world_size,                          # TP size
                        tp_world_size * 2,                      # Number of communication SMs
                        self.cga_size,                          # CGA cluster size
                        tp_world_size,                          # Number of communication splits
                        self.rs_sm_margin,                      # Set SM margin
                        self._NUM_MAX_UB_STREAMS,               # Max concurrent GEMM streams
                        torch.Tensor(),                         # empty tensor to pass to counters
                    )
            return self.buffer_rs[name]

def _kernel_make_viewless_tensor(inp, requires_grad):
    '''Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    '''
    out = torch.empty(
        (1,),
        dtype = inp.dtype,
        device = inp.device,
        requires_grad = requires_grad,
    )
    out.data = inp.data
    return out

class MakeViewlessTensor(torch.autograd.Function):
    '''
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    '''
    @staticmethod
    def forward(ctx, inp, requires_grad):
        return _kernel_make_viewless_tensor(inp, requires_grad)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def make_viewless_tensor(inp, requires_grad, keep_graph):
    '''
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    '''

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)

def assert_viewless_tensor(tensor, extra_msg = None):
    '''Assert that a tensor is not a view (i.e., its '._base' field is
    not set).'''
    if isinstance(tensor, list):
        [ assert_viewless_tensor(t) for t in tensor ]
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, (
        "Ensure tensor._base is None before setting tensor.data or storing "
        "tensor to memory buffer. Otherwise, a memory leak will occur (and "
        "likely accumulate over iterations). %s"
    ) % extra_msg
    return tensor

def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    '''Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    '''
    assert_viewless_tensor(tensor, extra_msg = "FYI, tensor._base has shape %s, and new_data_tensor has shape %s." % ("--" if tensor._base is None else tensor._base.shape, new_data_tensor.shape))
    tensor.data = new_data_tensor


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


_SYNC_EVENT = None


class SyncAtBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sync_level):
        ctx.sync_level = sync_level
        return x

    def backward(ctx, grad_output):
        cuda_sync_and_record(sync_level=ctx.sync_level)
        return grad_output, None


def cuda_sync_and_record(*, sync_level):
    if sync_level <= get_args().kaimm_cuda_synchronize_level:
        global _SYNC_EVENT 
        if _SYNC_EVENT is None:
            _SYNC_EVENT = torch.cuda.Event()
        _SYNC_EVENT.synchronize()
        _SYNC_EVENT.record()


def cuda_sync_and_record_at_backward(x, *, sync_level):
    return SyncAtBackwardFunction.apply(x, sync_level)
