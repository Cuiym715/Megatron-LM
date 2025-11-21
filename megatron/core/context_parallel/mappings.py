import torch

from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_world_size,
)


def _reduce(input_):
    """All-reduce the input tensor across context parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_context_parallel_world_size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_context_parallel_group())

    return input_


class _ReduceFromContextParallelRegion(torch.autograd.Function):
    """All-reduce the input from the context parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def reduce_from_context_parallel_region(input_):
    return _ReduceFromContextParallelRegion.apply(input_)
