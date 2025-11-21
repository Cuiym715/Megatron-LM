# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC
from abc import abstractmethod
import math

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron import get_args
from megatron.core import mpu
from .module import MegatronModule


class MemoryBuffer:

    def __init__(self, numel, numel_padded, dtype):
        self.numel = numel
        self.numel_padded = numel_padded
        self.dtype = dtype
        self.data = torch.zeros(self.numel_padded,
                                dtype=self.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False)

    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()


    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor


def aligned_numel(numel, dtype, num_bytes_divisible_by):
    element_size = torch.empty(0, dtype=dtype).element_size()
    if element_size % num_bytes_divisible_by == 0:
        return numel
    if num_bytes_divisible_by % element_size != 0:
        raise ValueError(f"num_bytes_divisible_by ({num_bytes_divisible_by}) should be multiple of element_size ({element_size})")
    numel_divisible_by = num_bytes_divisible_by // element_size
    return (numel + numel_divisible_by - 1) // numel_divisible_by * numel_divisible_by


class DistributedDataParallelBase(MegatronModule, ABC):
    """Abstract class for DDP."""

    def __init__(self, module):
        super(DistributedDataParallelBase, self).__init__()
        # Keep a pointer to the model.
        self.module = module


    @abstractmethod
    def allreduce_gradients(self):
        pass


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


    def state_dict(self, prefix='', keep_vars=False):
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(prefix=prefix,
                                                          keep_vars=keep_vars)


    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)



class DistributedDataParallel(DistributedDataParallelBase):
    """DDP with contiguous buffers options to storre and accumulate gradients.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (for example fp32)

    Arguments:
        module: input model.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and the gradient all-reduce all in in float32. If this option is
            true, we require `use_contiguous_buffers` to be true too.
        use_contiguous_buffers: if true, use a contiguous buffer to store the
            gradients.
        track_expert_parallel_params: if true, create grad buffers based on
            expert parallel parameters; if false, create grad buffers based on
            dense parameters.
    """

    def __init__(self, module,
                 accumulate_allreduce_grads_in_fp32,
                 use_contiguous_buffers,
                 make_main_grad_addresss_divisible_by,
                 track_expert_parallel_params=False):

        super(DistributedDataParallel, self).__init__(module)

        self.accumulate_allreduce_grads_in_fp32 \
            = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers
        self.track_expert_parallel_params = track_expert_parallel_params
        # If we are using fp32-accumulate-allreduce explicitly
        # this means we need main grads in a continous buffer.
        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers

        # ===================================
        # Rest of this part applies only to
        # the case we use continuous buffers.
        # ===================================
        self._grad_buffers = None
        self._grad_buffer_param_index_map = None
        self._grad_buffer_param_index_map_tight = None
        if self.use_contiguous_buffers:
            self._grad_buffers = {}
            self._grad_buffer_param_index_map = {}
            self._grad_buffer_param_index_map_tight = {}
            if track_expert_parallel_params:
                data_parallel_world_size = mpu.get_data_modulo_expert_parallel_world_size()
            else:
                data_parallel_world_size = mpu.get_data_parallel_world_size()

            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return torch.float if \
                    self.accumulate_allreduce_grads_in_fp32 else param.dtype

            # First calculate total number of elements per type.
            type_num_elements = {}
            type_num_elements_tight = {}
            for param in self.module.parameters():
                is_dense_param = getattr(param, 'allreduce', True)
                if param.requires_grad and track_expert_parallel_params == (not is_dense_param):
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                               + aligned_numel(param.data.nelement(), dtype, make_main_grad_addresss_divisible_by)
                    type_num_elements_tight[dtype] = type_num_elements_tight.get(dtype, 0) \
                                                     + param.data.nelement()

            # Allocate the buffer.
            for dtype, num_elements in type_num_elements.items():

                # If using distributed optimizer, pad memory buffer to be
                # multiple of data_parallel_world_size. (This padding is done
                # due to a constraint with the reduce_scatter op, which requires
                # all tensors have equal size. See: optimizer.py.)
                num_elements_padded = data_parallel_world_size * \
                    int(math.ceil(num_elements / data_parallel_world_size))

                # Allocate grad buffer.
                self._grad_buffers[dtype] = MemoryBuffer(num_elements,
                                                         num_elements_padded,
                                                         dtype)

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            for param in self.module.parameters():
                is_dense_param = getattr(param, 'allreduce', True)
                if param.requires_grad and track_expert_parallel_params == (not is_dense_param):
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] -= aligned_numel(param.data.nelement(), dtype, make_main_grad_addresss_divisible_by)
                    type_num_elements_tight[dtype] -= param.data.nelement()
                    param.main_grad = self._grad_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype])
                    if dtype not in self._grad_buffer_param_index_map:
                        self._grad_buffer_param_index_map[dtype] = {}
                    self._grad_buffer_param_index_map[dtype][param] = (
                        type_num_elements[dtype],
                        type_num_elements[dtype] + param.data.nelement(),
                    )
                    if dtype not in self._grad_buffer_param_index_map_tight:
                        self._grad_buffer_param_index_map_tight[dtype] = {}
                    self._grad_buffer_param_index_map_tight[dtype][param] = (
                        type_num_elements_tight[dtype],
                        type_num_elements_tight[dtype] + param.data.nelement(),
                    )

            # Backward hook.
            # Accumalation function for the gradients. We need
            # to store them so they don't go out of scope.
            self.grad_accs = []
            # Loop over all the parameters in the model.
            for param in self.module.parameters():
                is_dense_param = getattr(param, 'allreduce', True)
                if param.requires_grad and track_expert_parallel_params == (not is_dense_param):
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator functtion.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(param))
                    self.grad_accs.append(grad_acc)

        if not track_expert_parallel_params:
            # Use `model_ep' to build _grad_buffers for parallel MoE params.
            self.model_ep = DistributedDataParallel(module,
                                                    accumulate_allreduce_grads_in_fp32,
                                                    use_contiguous_buffers,
                                                    make_main_grad_addresss_divisible_by,
                                                    track_expert_parallel_params=True)


    def _make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""
        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad is not None:
                # The gradient function of linear layers is fused with GEMMs
                param.main_grad.add_(param.grad.data)
                # Now we can deallocate grad memory.
                param.grad = None
        return param_hook


    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        begining of each iteration."""
        assert self._grad_buffers is not None, 'buffers are not initialized.'
        for _, buffer_ in self._grad_buffers.items():
            buffer_.zero()
        assert self.model_ep._grad_buffers is not None, 'buffers are not initialized.'
        for _, buffer_ in self.model_ep._grad_buffers.items():
            buffer_.zero()


    def broadcast_params(self):
        for param in self.module.parameters():
            if getattr(param, 'allreduce', True):
                torch.distributed.broadcast(param.data,
                                            src=mpu.get_data_parallel_src_rank(),
                                            group=mpu.get_data_parallel_group())
            else:
                raise NotImplementedError("EP + --data-parallel-random-init is not implemented")


    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks."""
        # If we have buffers, simply reduce the data in the buffer.
        if self._grad_buffers is not None:
            if self.track_expert_parallel_params:
                data_parallel_group = mpu.get_data_modulo_expert_parallel_group()
            else:
                data_parallel_group = mpu.get_data_parallel_group()
            for _, buffer_ in self._grad_buffers.items():
                buffer_.data /= mpu.get_data_parallel_world_size() // mpu.get_context_parallel_world_size()
                torch.distributed.all_reduce(
                    buffer_.data, group=data_parallel_group)
        else:
            # Otherwise, bucketize and all-reduce
            if self.track_expert_parallel_params:
                raise NotImplementedError("EP w/o use_contiguous_buffers is not implemented")
            buckets = {}
            # Pack the buckets.
            for param in self.module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.data.type()
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
                    param.main_grad = param.grad

            # For each bucket, all-reduce and copy all-reduced grads.
            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                coalesced /= mpu.get_data_parallel_world_size() // mpu.get_context_parallel_world_size()
                torch.distributed.all_reduce(
                    coalesced, group=mpu.get_data_parallel_group())
                for buf, synced in zip(grads, _unflatten_dense_tensors(
                        coalesced, grads)):
                    buf.copy_(synced)
