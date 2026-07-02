# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
import math
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Deque, Dict, Iterator, List, Optional, Tuple, Union, Type

import torch
import torch.distributed as dist
from torch.autograd.variable import Variable
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import offload, p2p_communication
from megatron.core.kv_cache import Cache
from megatron.core.pipeline_parallel.slice_v import build_slice_v_schedule
from megatron.core.zbpp_utils import WeightGradStore
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import cuda_sync_and_record, get_attr_wrapped_model, get_model_type
from megatron.model.utils import slice_lm_inputs_along_cp, pad_to_be_divisible
from megatron.profile_utils import annotate_forward_range, annotate_backward_range

# Types
Shape = Union[List[int], torch.Size]

def get_forward_backward_func(slicing=False, variable_slicing=False):
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of
        modules in the case of interleaved pipeline parallelism.

    num_microbatches (int, required):
        The number of microbatches to go through

    dtype (required when using pipeline parallelism): dtype used in
        p2p communication, usually params_dtype

    tensor_shape (required when using pipeline parallelism): Shape of
        tensor. The tensor is expected to be 3D and its order of
        dimension is supposed to be ``(sequence, batch, hidden)``.

    decoder_seq_length (int, required for ModelType.encoder_and_decoder models):
        Sequence length of the decoder portion, used to determine tensor shapes.

    grad_scaler (optional, default=None): If using loss scaling,
        this function should take the loss and return the scaled
        loss. If None, no function is called on the loss.

    sequence_parallel (optional, default=False):
        Set to :obj:`True` for this function to handle sequence
        length.  When :obj:`True`, the sequence length on each tensor
        model parallel rank is updated to
        :math:`original\_sequence\_length /
        tensor\_model\_parallel\_world\_size`.
        TODO: Do we need this? Just roll into tensor_shape arg?

    overlap_p2p_comm (optional, default=False): When True
        some of the peer to peer communication for pipeline
        parallelism will overlap with computation. Must be False if
        batch_p2p_comm is true.

    batch_p2p_comm (optional, default=True): When true use
        batch_isend_irecv, otherwise use individual isend and irecv
        calls. Must be false if overlap_p2p_comm is True.

    forward_only (optional, default=False): Perform only the forward step

    timers (optional, default=None): TODO

    collect_non_loss_data: TODO

    enable_autocast (optional, default=False): If True, runs the
        forward_step_func call inside torch.autocast context

    deallocate_pipeline_outputs (optional, default=False): If True, output data
        is deallocated after the tensor is sent to the next pipeline stage.
        Helps with saving memory, does nothing when pipeline parallel is
        not used.

    no_sync_func (optional): Function that creates a context that
        suppresses asynchronous data-parallel communication. If the
        model is an instance of torch.nn.DistributedDataParallel, the
        default is to use torch.nn.DistributedDataParallel.no_sync.

    grad_sync_func (optional): Function that launches asynchronous
        gradient reductions (e.g. distributed optimizer gradient
        reduce-scatters). The function should take one argument: an
        iterable of parameters whose gradients are to be synchronized.

    param_sync_func (optional): Function that launches asynchronous
        parameter synchronizations (e.g. distributed optimizer
        parameter all-gathers). The function should take one argument:
        an iterable of parameters to be synchronized.

    pre_p2p_func (optional): Function that is to be called before p2p
        communication in warmup passes and cooldown passes.

    post_p2p_async_func (optional): Function that is to be called after
        p2p communication in warmup passes and cooldown passes. The
        function should take one argument: the list of p2p communication
        handles.

    offload_delay_to_next_stage (optional, default=False): If True, delay
        the timing of offload to the next stage of the same microbatch.

    """
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            if variable_slicing:
                args = get_args()
                if getattr(args, 'variable_seq_schedule', '1f1b') == 'slice-v':
                    forward_backward_func = pipelining_with_variable_slicing_slice_v
                else:
                    raise NotImplementedError(
                        "variable sequence slicing 1f1b is implemented for "
                        "non-interleaved pipeline parallelism; unset "
                        "--num-layers-per-virtual-pipeline-stage or set "
                        "--variable-seq-schedule slice-v."
                    )
            elif slicing:
                forward_backward_func = pipelining_with_interleaved_slicing
            else:
                forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            if variable_slicing:
                forward_backward_func = pipelining_with_variable_slicing
            elif slicing:
                forward_backward_func = pipelining_with_slicing
            else:
                forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func

def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), \
        "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, \
        "counter-productive to free a view of another tensor."
    out.data = torch.empty(
        (1,),
        device = out.device,
        dtype = out.dtype,
    )

def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, \
        "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), \
        "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), \
        "grad_output == '%s'." % type(grad_output).__name__

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(
            output,
            memory_format = torch.preserve_format,
        )

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors = (output,),
        grad_tensors = (grad_output,),
        keep_graph = False,
        create_graph = False,
        inputs = tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )





def forward_step(data,
                 input_tensor,
                 kv_cache,
                 forward_step_func,
                 model,
                 timers,
                 autocast_dtype=torch.float,
                 enable_autocast=False):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""

    cuda_sync_and_record(sync_level=2)

    if timers is not None:
        timers('forward-compute', log_level=2).start()

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if enable_autocast:
        context_manager = torch.autocast("cuda", dtype=autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        with annotate_forward_range("forward"):
            output_tensor = forward_step_func(data, kv_cache, model)

    # Unset the input tensor to release memory.
    set_input_tensor(None)

    if timers is not None:
        timers('forward-compute').stop()

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)

    if parallel_state.is_pipeline_stage_after_split() and \
            model_type == ModelType.encoder_and_decoder:
        return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def backward_step(input_tensor, output_tensor, output_tensor_grad,
                  model_type, timers, deallocate_pipeline_outputs=False):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    cuda_sync_and_record(sync_level=2)

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if timers is not None:
        timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    with annotate_backward_range("backward"):
        # No backward pass is needed on the first pipeline stage if all parameters do not require gradients.
        if output_tensor[0].requires_grad:
            if deallocate_pipeline_outputs:
                custom_backward(output_tensor[0], output_tensor_grad[0])
            else:
                args = get_args()
                retain_graph = args.variable_seq_slicing and not args.use_flash_attn
                torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad,
                                        retain_graph=retain_graph)

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if parallel_state.get_pipeline_model_parallel_world_size() > 1 and \
            parallel_state.is_pipeline_stage_after_split() and \
            model_type == ModelType.encoder_and_decoder:
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if timers is not None:
        timers('backward-compute').stop()

    return input_tensor_grad


def pre_process_forward(model, data, input_shape):
    group = parallel_state.get_pipeline_model_parallel_group()
    first = parallel_state.get_pipeline_model_parallel_first_rank()
    with annotate_forward_range("fwd pre"):
        if data is None:
            input_ids = torch.empty(input_shape, dtype=torch.int64, device="cuda")
        else:
            data = slice_lm_inputs_along_cp(*data)
            input_ids = data[0]
            assert input_ids.shape == input_shape
        torch.distributed.broadcast(input_ids, first, group)
        pre_process = get_attr_wrapped_model(model, "pre_process_forward")
        encoder_input = pre_process(input_ids)
    dummy_input = offload.forward_empty_backward_identity(encoder_input)
    return encoder_input, dummy_input, data


def pre_process_backward(encoder_input, encoder_input_grad):
    group = parallel_state.get_pipeline_model_parallel_group()
    first = parallel_state.get_pipeline_model_parallel_first_rank()
    with annotate_backward_range("bwd pre"):
        if encoder_input_grad is None:
            encoder_input_grad = torch.empty_like(encoder_input)
        torch.distributed.broadcast(encoder_input_grad, first, group)
        torch.autograd.backward(encoder_input, encoder_input_grad)


def post_process_forward(model, output, tensor_shape, labels_shape, dtype, offload_ratio=0):
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    group = parallel_state.get_pipeline_model_parallel_group()
    last = parallel_state.get_pipeline_model_parallel_last_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    output_shape = ((tensor_shape[0] + pp_size - 1) // pp_size,) + tensor_shape[1:]
    with annotate_forward_range("fwd post"):
        if output is None:
            output = torch.empty(output_shape, dtype=dtype, device="cuda")
            labels = torch.empty(labels_shape, dtype=torch.int64, device="cuda")
            scatter_list = None
        else:
            labels = output._labels.contiguous(); del output._labels
            assert output.shape == tensor_shape
            assert labels.shape == labels_shape
            output = output.detach()
            output = pad_to_be_divisible(output, pp_size, dim=0)
            scatter_list = list(output.chunk(pp_size))
            output = torch.empty(output_shape, dtype=dtype, device="cuda")
        torch.distributed.scatter(output, scatter_list, last, group)
        torch.distributed.broadcast(labels, last, group)
        output.requires_grad_()
        output, dummy_output = offload.get_forward_tensor_and_backward_handle(output)
        post_process = get_attr_wrapped_model(model, "post_process_forward")
        om = offload.OffloadManager(offload_ratio)
        with offload.offload_manager(om):
            loss = post_process(output, labels, pipeline_parallel=True)
    return dummy_output, loss, om


def post_process_backward(loss, grad_loss, output, output_shape):
    world_size = parallel_state.get_pipeline_model_parallel_world_size()
    group = parallel_state.get_pipeline_model_parallel_group()
    last = parallel_state.get_pipeline_model_parallel_last_rank()
    with annotate_backward_range("bwd post"):
        if grad_loss is None:
            grad_loss = torch.empty_like(loss)
            grad = None
            gather_list = None
        else:
            grad_loss = grad_loss.contiguous()
            grad_shape = list(output.shape)
            grad_shape[0] *= world_size
            grad = output.new_empty(grad_shape)
            gather_list = list(grad.chunk(world_size))
        torch.distributed.broadcast(grad_loss, last, group)
        output.retain_grad()
        torch.autograd.backward(loss, grad_loss)
        torch.distributed.gather(output.grad, gather_list, last, group=group)
        if grad is not None:
            grad = grad.narrow(0, 0, output_shape[0])   # unpad
    return grad


class MicroBatch:
    def __init__(self,
                 slices: List,
                 kv_cache: Cache,
                 offload_ratio: float,
                 forward_func: Callable,
                 backward_func: Optional[Callable],
                 loss_func: Optional[Callable],
                 grad_scaler: Optional[Callable],
                 batch_idx: Optional[int] = None,
                 timers: Optional[Callable] = None):
        self.slices = deque(slices)
        self.num_slices = len(slices)
        self.batch_idx = batch_idx
        self.timers = timers
        self.kv_cache = kv_cache
        self.offload_ratio = offload_ratio
        self.forward_func = forward_func
        self.backward_func = backward_func
        self.loss_func = loss_func
        self.grad_scaler = grad_scaler
        assert self.forward_func is not None
        if self.backward_func is not None:
            self.om_stack = []
            self.input_stack = []
            self.output_stack = []
            self.kv_grad = []
        if self.loss_func is not None:
            self.output_tensors = []

    @property
    def num_slices_to_forward(self):
        return len(self.slices)

    @property
    def num_slices_to_backward(self):
        assert self.backward_func is not None
        return len(self.om_stack)

    @property
    def slice_idx(self):
        return self.cache_len()

    def cache_len(self, offset=0):
        return (len(self.om_stack) + offset) % self.num_slices

    def curr_om(self):
        assert self.backward_func is not None
        return self.om_stack[-1]

    def update_kv_cache(self, ctx_pair):
        if self.kv_cache:
            self.kv_cache = self.kv_cache.copy(ctx_pair=ctx_pair)

    def forward(self, input_tensor):
        """Run forward on one slice of micro-batch."""
        slice_idx = self.num_slices - len(self.slices)
        slice = self.slices.popleft()
        if self.backward_func is not None:
            dummy_input = input_tensor
            if input_tensor is not None:
                input_tensor, dummy_input = offload.get_forward_tensor_and_backward_handle(input_tensor)
            input_kv = self.kv_cache.detach().dump()
            self.input_stack.append([dummy_input] + input_kv)
        om = offload.OffloadManager(self.offload_ratio)
        timers = self.timers
        record_context = timers is not None and timers.is_recording_active()
        if record_context:
            timers.set_record_context(mb=self.batch_idx,
                                      chunk=slice_idx,
                                      num_chunks=self.num_slices)
        with offload.offload_manager(om):
            output_tensor = self.forward_func(slice, input_tensor, self.kv_cache)
        if record_context:
            timers.clear_record_context()
        self.om_stack.append(om)
        if self.backward_func is not None:
            dummy_output = offload.forward_empty_backward_identity(output_tensor)
            output_kv = self.kv_cache.dump() # if self.num_slices_to_forward else []
            self.output_stack.append([dummy_output] + output_kv)
        if self.loss_func is not None:
            self.output_tensors.append(output_tensor)
            output_tensor = None
        if self.num_slices_to_forward == 0 and self.kv_cache:
            self.kv_cache = self.kv_cache.copy(retain_kv=False)
        return output_tensor

    def pop_data(self):
        return self.slices.popleft()

    def append_data(self, data):
        self.slices.appendleft(data)

    def pop_output(self):
        assert self.loss_func is not None
        return self.output_tensors.pop()

    def append_output(self, output):
        assert self.loss_func is not None
        self.output_tensors.append(output)

    def pop_output_grad(self):
        assert self.loss_func is not None
        return self.output_tensor_grads.pop()

    def compute_loss(self, loss_div):
        """Compute the loss if in the last stage, otherwise do nothing."""
        # Set the loss scale for the auxiliary loss of the MoE layer.
        # Since we use a trick to do backward on the auxiliary loss, we need to set the scale explicitly.
        if get_args().num_experts is not None:
            # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
            loss_scale = (
                self.grad_scaler(torch.ones(())) if self.grad_scaler is not None else torch.ones(())
            )
            # Set the loss scale
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale / loss_div)
        if self.loss_func is None:
            return None
        output_tensor = torch.cat(self.output_tensors, dim=-1)
        # TODO(lizhiouyang): support the collect_non_loss_data logic.
        loss, loss_reduced = self.loss_func(output_tensor)
        if self.backward_func is not None:
            loss /= loss_div
            if self.grad_scaler is not None:
                loss = self.grad_scaler(loss)
            self.output_tensor_grads = list(torch.autograd.grad(loss, self.output_tensors))
        del self.output_tensors
        return loss_reduced

    def backward(self, output_tensor_grad):
        """Run backward on one slice of micro-batch (in reversed order of forward)."""
        assert self.backward_func is not None
        slice_idx = len(self.om_stack) - 1
        om = self.om_stack.pop()
        assert om.is_complete()
        def _missing_grad(grad):
            if grad is None:
                return True
            if isinstance(grad, (list, tuple)):
                return all(item is None for item in grad)
            return False
        if self.loss_func is not None or _missing_grad(output_tensor_grad):
            if hasattr(self, 'output_tensor_grads') and self.output_tensor_grads:
                output_tensor_grad = self.output_tensor_grads.pop()
            elif _missing_grad(output_tensor_grad):
                raise RuntimeError("Missing output_tensor_grad for a non-last pipeline stage microbatch.")
        if isinstance(output_tensor_grad, (list, tuple)):
            assert len(output_tensor_grad) == 1, \
                f"Expected one output tensor grad, got {len(output_tensor_grad)}."
            output_tensor_grad = output_tensor_grad[0]
        inputs = self.input_stack.pop()
        outputs = self.output_stack.pop()
        output_tensor_grads = [output_tensor_grad] + self.kv_grad
        outputs_and_grads = [
            (output, grad)
            for output, grad in zip(outputs, output_tensor_grads, strict=False)
            if grad is not None
        ]
        assert outputs_and_grads, "No tensor gradients available for this backward slice."
        outputs, output_tensor_grad = map(list, zip(*outputs_and_grads))
        timers = self.timers
        record_context = timers is not None and timers.is_recording_active()
        if record_context:
            timers.set_record_context(mb=self.batch_idx,
                                      chunk=slice_idx,
                                      num_chunks=self.num_slices)
        input_tensor_grad, *self.kv_grad = self.backward_func(inputs, outputs, output_tensor_grad)
        if record_context:
            timers.clear_record_context()
        om.check_ref()
        return input_tensor_grad

    def backward_b(self, output_tensor_grad, chunk=0):
        """Run only activation-gradient work and defer weight gradients."""
        WeightGradStore.assert_supported()
        with WeightGradStore.set_split_bw(True):
            input_tensor_grad = self.backward(output_tensor_grad)
            WeightGradStore.flush(chunk=chunk)
        return input_tensor_grad

    def weight_grad(self, pop_num=1, chunk=0):
        """Run deferred weight-gradient work for this microbatch."""
        WeightGradStore.assert_supported()
        return WeightGradStore.pop(chunk=chunk, pop_num=pop_num, timers=self.timers)


class GroupedBatch:
    def __init__(self,
                 mbatches: List[MicroBatch],
                 num_stages: int,
                 group_size: int):
        self.mbatches = mbatches
        self.num_stages = num_stages
        self.group_size = group_size
        self.fwd_batch_idx = 0
        self.bwd_batch_idx = 0

    def forward_stage_idx(self, offset=0):
        return (self.fwd_batch_idx + offset) // self.group_size % self.num_stages

    def backward_stage_idx(self, offset=0):
        return self.num_stages - 1 - (self.bwd_batch_idx + offset) // self.group_size % self.num_stages

    def prev_mbatch(self):
        return self.mbatches[self.fwd_batch_idx - 1]

    def curr_fwd_mbatch(self):
        return self.mbatches[self.fwd_batch_idx]

    def curr_bwd_mbatch(self):
        stage_idx = (self.num_stages - 1 - self.bwd_batch_idx // self.group_size)
        batch_idx = self.bwd_batch_idx % self.group_size
        return self.mbatches[stage_idx * self.group_size + batch_idx]

    @property
    def num_batches_to_forward(self):
        return sum(b.num_slices_to_forward for b in self.mbatches)

    @property
    def num_batches_to_backward(self):
        return sum(b.num_slices_to_backward for b in self.mbatches)

    def forward(self, input_tensor):
        parallel_state.set_virtual_pipeline_model_parallel_rank(self.forward_stage_idx())
        output_tensor = self.curr_fwd_mbatch().forward(input_tensor)
        self.fwd_batch_idx += 1
        return output_tensor

    def compute_loss(self, loss_div):
        loss = self.prev_mbatch().compute_loss(loss_div)
        return loss

    def backward(self, output_tensor_grad):
        parallel_state.set_virtual_pipeline_model_parallel_rank(self.backward_stage_idx())
        input_tensor_grad = self.curr_bwd_mbatch().backward(output_tensor_grad)
        self.bwd_batch_idx += 1
        return input_tensor_grad


class CycledBatch:
    def __init__(self,
                 batch_idx: int,
                 mbatches: List[MicroBatch],
                 group_size: int):
        self.batch_idx = batch_idx
        self.mbatches = mbatches
        self.group_size = group_size
        self.num_slices_per_cycle = len(mbatches) * self.group_size
        self.num_cycles = mbatches[0].num_slices // group_size
        self.num_slices = len(mbatches) * mbatches[0].num_slices
        self.slice_idx = 0

    def stage_idx(self, offset=0):
        return (self.slice_idx + offset) % self.num_slices_per_cycle // self.group_size

    def cache_len(self, offset=0):
        idx = self.slice_idx + offset
        return (idx // self.num_slices_per_cycle) % self.num_cycles * self.group_size + idx % self.group_size

    def prev_mbatch(self):
        return self.mbatches[self.stage_idx(offset=-1)]

    def curr_mbatch(self):
        return self.mbatches[self.stage_idx()]

    @property
    def num_slices_to_forward(self):
        return sum(b.num_slices_to_forward for b in self.mbatches)

    @property
    def num_slices_to_backward(self):
        return sum(b.num_slices_to_backward for b in self.mbatches)

    def forward(self, input_tensor):
        parallel_state.set_virtual_pipeline_model_parallel_rank(self.stage_idx())
        output_tensor = self.curr_mbatch().forward(input_tensor)
        self.slice_idx += 1
        return output_tensor

    def compute_loss(self, loss_div):
        assert self.slice_idx == self.num_slices
        self.slice_idx -= 1
        return self.prev_mbatch().compute_loss(loss_div)

    def backward(self, output_tensor_grad):
        parallel_state.set_virtual_pipeline_model_parallel_rank(self.stage_idx())
        input_tensor_grad = self.curr_mbatch().backward(output_tensor_grad)
        self.slice_idx -= 1
        return input_tensor_grad


@dataclass(frozen=True)
class CtxPair:
    """Informations for attention workload balance."""
    peer: int
    # mbid: int
    stage_idx: int
    cache_len: int
    nfwd: int
    nbwd: int
    cache: bool
    local_qo: List = field(default_factory=list, repr=False, hash=False, compare=False)
    other_kv: List = field(default_factory=list, repr=False, hash=False, compare=False)
    other_qo: List = field(default_factory=list, repr=False, hash=False, compare=False)
    reqs: List = field(default_factory=list, repr=False, hash=False, compare=False)


class AttnBalancer:
    """Redistribute the attention workloads by exchange key-value with a peer."""
    def __init__(self, num_mb, rank, size, threshold):
        self.num_mb = num_mb
        self.pp_rank = rank
        self.pp_size = size
        self.threshold = threshold

    def calc_ctx_pair(self, batch) -> CtxPair:
        """Calculate the peer to exchange key-value with."""
        if not self.threshold:
            return None

        # batch_idx = [batch.batch_idx(offset=self.pp_rank - i + prior) for i in range(self.pp_size)]
        stage_idx = [batch.stage_idx(offset=self.pp_rank - i) for i in range(self.pp_size)]
        cache_len = [batch.cache_len(offset=self.pp_rank - i) for i in range(self.pp_size)]
        # print(f'{self.pp_rank}: {batch_idx=}')
        # print(f'{self.pp_rank}: {cache_len=}')
        # avg = sum(cache_len) / self.pp_size
        # len = cache_len[self.pp_rank]
        # com = avg * 2 - len
        # assert com in cache_len, (self.pp_rank, cache_len)
        # idx = cache_len.index(com)
        # vol = int(avg - len)

        ord = sorted(cache_len)
        len = cache_len[self.pp_rank]
        com = ord[self.pp_size - 1 - ord.index(len)]
        peer = cache_len.index(com)
        avg = (com + len) / 2
        vol = int(avg - len)
        if abs(vol) > abs(self.threshold):    # clip to vaule of threshold
            vol = int(math.copysign(self.threshold, vol))
        # mbid = batch_idx[self.pp_rank] if vol < 0 else batch_idx[peer]
        ckid = stage_idx[self.pp_rank] # if vol < 0 else stage_idx[peer]
        clen = cache_len[self.pp_rank]

        # cnt_mb = batch_idx[self.pp_rank]
        slice_idx = batch.slice_idx
        cnt_mb = batch.batch_idx
        if slice_idx >= batch.num_slices:   # forward to next batch
            slice_idx -= batch.num_slices
            cnt_mb += 1
        if slice_idx < 0:                   # backward to next batch
            slice_idx += batch.num_slices
            cnt_mb += 1
        if cnt_mb >= self.num_mb:
            return CtxPair(peer, ckid, clen, False, False, False)

        fwd = not (slice_idx + 1 < self.pp_size - self.pp_rank and cnt_mb == 0 or
                    batch.num_slices - slice_idx <= self.pp_rank and cnt_mb + 1 == self.num_mb)
        bwd = not (slice_idx + 1 < self.pp_size - self.pp_rank and cnt_mb + 1 == self.num_mb or
                    batch.num_slices - slice_idx <= self.pp_rank and cnt_mb == 0)
        nfwd = vol if fwd else 0
        nbwd = vol if bwd else 0
        junc = slice_idx + 1 < self.pp_size - self.pp_rank or batch.num_slices - slice_idx <= self.pp_rank

        cache = self.threshold < 0 and not junc
        # if fwd or bwd:
        #     print(f"rank{self.pp_rank}: {msg}: {idx, vol, fwd, bwd} {cache_len}", flush=True)
        return CtxPair(peer, ckid, clen, nfwd, nbwd, cache)


def forward_backward_no_pipelining(*,
                                   forward_step_func,
                                   get_batch_func,
                                   data_iterator: Union[Iterator, List[Iterator]],
                                   model: Union[torch.nn.Module, List[torch.nn.Module]],
                                   num_microbatches: int,
                                   micro_seq_length: int,
                                   kv_cache_class: Type[Cache],
                                   dtype: Optional[torch.dtype] = None,
                                   tensor_shape: Optional[Shape] = None, # unused
                                   decoder_seq_length: Optional[int] = None, # unused
                                   grad_scaler: Callable = None,
                                   sequence_parallel: bool = False, # unused
                                   overlap_p2p_comm: bool = False, # unused
                                   batch_p2p_comm: bool = True, # unused
                                   attn_balance: int = 0, # unused
                                   vocab_in_pp: bool = False, # unused
                                   forward_only: bool = False,
                                   timers: Callable = None,
                                   collect_non_loss_data: bool = False,
                                   enable_autocast: bool = False,
                                   deallocate_pipeline_outputs: bool = False,
                                   no_sync_func: Optional[Callable] = None,
                                   grad_sync_func: Optional[Callable] = None, # unused
                                   param_sync_func: Optional[Callable] = None, # unused
                                   pre_p2p_func: Optional[Callable] = None, # unused
                                   post_p2p_async_func: Optional[Callable] = None, # unused
                                   offload_ratio: float = 0,
                                   offload_delay_to_next_stage: bool = False, # unused
                                   ):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):
        assert len(model) == 1, \
            "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, \
            "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]
    assert not isinstance(model, torchDDP), "torchDDP is no longer supported."
    model_type = get_model_type(model)

    forward_func = partial(forward_step, forward_step_func=forward_step_func, model=model,
                           timers=timers, enable_autocast=enable_autocast)
    backward_func = None if forward_only else partial(backward_step, model_type=model_type,
                            timers=timers, deallocate_pipeline_outputs=deallocate_pipeline_outputs)

    cnt_onload = 0
    make_microbatch_idx = 0
    def make_microbatch():
        nonlocal make_microbatch_idx
        if timers is not None:
            record_context = timers.is_recording_active()
            if record_context:
                timers.set_record_context(mb=make_microbatch_idx)
            timers('batch-generator', log_level=2).start()
        sliced_batch, loss_func = get_batch_func(data_iterator)
        if timers is not None:
            timers('batch-generator').stop()
            if record_context:
                timers.clear_record_context()
        slices = sliced_batch(micro_seq_length)
        kv_cache = kv_cache_class()
        mb = MicroBatch(slices, kv_cache, offload_ratio, forward_func,
                        backward_func, loss_func, grad_scaler,
                        batch_idx=make_microbatch_idx,
                        timers=timers)
        make_microbatch_idx += 1
        return mb

    forward_data_store = []
    for _ in range(num_microbatches):
        mb = make_microbatch()
        # forward
        offload_req = None
        while mb.num_slices_to_forward:
            mb.forward(None)
            if offload_req:
                offload_req.wait(); offload_req = None
            curr_om = mb.curr_om()
            if mb.num_slices_to_forward > 1:
                offload_req = curr_om.offload(prior_works=[torch.cuda.current_stream().record_event()])
            else:
                curr_om.reset()
        # compute loss
        forward_data_store.append(mb.compute_loss(num_microbatches))
        if forward_only:
            continue
        # backward
        prior_works = [torch.cuda.current_stream().record_event()]
        while mb.num_slices_to_backward:
            next_om = mb.curr_om()
            onload_req = next_om.onload(prior_works=prior_works, buffer_name="onload", buffer_idx=cnt_onload % 2)
            cnt_onload += 1
            if onload_req:
                onload_req.wait(); onload_req = None
            prior_works = [torch.cuda.current_stream().record_event()]
            mb.backward(None)

    return forward_data_store


def get_actual_tensor_shape(tensor_shape, sequence_parallel, micro_seq_length=0):
    seq_length, batch_size, hidden_size = tensor_shape
    if micro_seq_length:
        seq_length = micro_seq_length
    seq_length //= parallel_state.get_context_parallel_world_size()
    if sequence_parallel:
        seq_length //= parallel_state.get_tensor_model_parallel_world_size()
    return (seq_length, batch_size, hidden_size)


def pipelining_with_slicing(*,
                            forward_step_func,
                            get_batch_func,
                            data_iterator: Union[Iterator, List[Iterator]],
                            model: Union[torch.nn.Module, List[torch.nn.Module]],
                            num_microbatches: int,
                            micro_seq_length: int,
                            kv_cache_class: Type[Cache],
                            dtype: Optional[torch.dtype] = None,
                            tensor_shape: Optional[Shape] = None,
                            decoder_seq_length: Optional[int] = None, # unused
                            grad_scaler: Callable = None,
                            sequence_parallel: bool = False,
                            overlap_p2p_comm: bool = False,
                            batch_p2p_comm: bool = True,
                            attn_balance: int = 0,
                            vocab_in_pp: bool = False, # unused
                            forward_only: bool = False,
                            timers: Callable = None,
                            collect_non_loss_data: bool = False, # unused
                            enable_autocast: bool = False,
                            deallocate_pipeline_outputs: bool = False,
                            no_sync_func: Optional[Callable] = None, # unused
                            grad_sync_func: Optional[Callable] = None, # unused
                            param_sync_func: Optional[Callable] = None, # unused
                            pre_p2p_func: Optional[Callable] = None, # unused
                            post_p2p_async_func: Optional[Callable] = None, # unused
                            offload_ratio: float = 0,
                            offload_delay_to_next_stage: bool = False, # unused
                            _variable_slicing: bool = False,
                            ):
    """Run 1F1B schedule, with batch and token level pipeline parallelism.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert len(model) == 1, \
            "Non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, \
            "Non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]
    assert overlap_p2p_comm, \
        "Slicing pipeline parallelism only supports overlapping p2p communication"
    assert not batch_p2p_comm, \
        "Slicing pipeline parallelism does not support using batched p2p communication"

    assert not isinstance(model, torchDDP), "torchDDP is no longer supported."
    model_type = get_model_type(model)
    assert model_type != ModelType.encoder_and_decoder, "encoder_and_decoder model is not supported yet."
    assert not attn_balance, "not implemented"

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    first_stage = parallel_state.is_pipeline_first_stage()
    last_stage = parallel_state.is_pipeline_last_stage()
    args = get_args()
    # DEBUG for variable length training.
    variable_seq_debug_limit = getattr(args, 'variable_seq_debug_num_batches', 0)

    num_slices = tensor_shape[0] // micro_seq_length
    # shape of input_ids.
    input_shape = (tensor_shape[1], micro_seq_length // parallel_state.get_context_parallel_world_size())
    tensor_shape = get_actual_tensor_shape(tensor_shape, sequence_parallel, micro_seq_length)

    forward_func = partial(forward_step, forward_step_func=forward_step_func, model=model,
                           timers=timers, enable_autocast=enable_autocast)
    backward_func = None if forward_only else partial(backward_step, model_type=model_type,
                            timers=timers, deallocate_pipeline_outputs=deallocate_pipeline_outputs)

    cnt_microbatches = 0
    loss_div = num_microbatches
    forward_data_store = []
    # minimun number of slices to fullfill the pipeline.
    num_slices_preset = 1 if _variable_slicing else num_slices
    # warm up to prefill the pipeline.
    num_slices_warmup = 2 * (pipeline_parallel_size - pipeline_parallel_rank - 1)
    # in-flight slices need to gain.
    num_slices_target = 4096 if forward_only else num_slices_preset + num_slices_warmup
    # current in-flight slices.
    num_slices_flight = 0
    mb_queue = deque()
    batch_fwd: Optional[MicroBatch] = None
    batch_bwd: Optional[MicroBatch] = None
    input_tensor_grad = None
    output_tensor_grad = None
    offload_req = None
    onload_req = None
    cnt_onload = 0
    variable_microbatch_specs = deque()

    def read_microbatch_slices(cnt_microbatches):
        if timers is not None:
            record_context = timers.is_recording_active()
            if record_context:
                timers.set_record_context(mb=cnt_microbatches)
            timers('batch-generator', log_level=2).start()
        sliced_batch, loss_func = get_batch_func(data_iterator)
        if timers is not None:
            timers('batch-generator').stop()
            if record_context:
                timers.clear_record_context()
        if not last_stage:
            loss_func = None
        slices = sliced_batch(micro_seq_length)
        return slices, loss_func

    def make_microbatch(cnt_microbatches):
        nonlocal num_slices_target
        assert cnt_microbatches < num_microbatches, "No more microbatches."
        if variable_microbatch_specs:
            slices, loss_func = variable_microbatch_specs.popleft()
        else:
            slices, loss_func = read_microbatch_slices(cnt_microbatches)
        if _variable_slicing:
            assert len(slices) >= 1, "variable sequence slicing produced no slices."
            assert len(slices) % pipeline_parallel_size == 0, \
                "variable sequence slicing currently requires chunk count to be divisible by pipeline size."
            # DEBUG for variable length training.
            if cnt_microbatches < variable_seq_debug_limit:
                print(
                    "[variable-seq][schedule] "
                    f"rank={pipeline_parallel_rank}/{pipeline_parallel_size}, "
                    f"microbatch={cnt_microbatches}, slices={len(slices)}, "
                    f"warmup_slices={num_slices_warmup}, "
                    f"target_inflight_slices={num_slices_target}, "
                    f"current_inflight_slices={num_slices_flight}, "
                    f"forward_only={forward_only}",
                    flush=True,
                )
        num_slices_total = num_slices_flight + (num_microbatches - cnt_microbatches) * len(slices)
        if not _variable_slicing:
            assert num_slices_total >= num_slices_target, "number of total slices is not enough."
        kv_cache = kv_cache_class()
        mb = MicroBatch(slices, kv_cache, offload_ratio, forward_func,
                        backward_func, loss_func, grad_scaler,
                        batch_idx=cnt_microbatches,
                        timers=timers)
        # mb._bwd = []
        return cnt_microbatches + 1, mb

    if _variable_slicing:
        variable_microbatch_specs_list = []
        for microbatch_idx in range(num_microbatches):
            slices, loss_func = read_microbatch_slices(microbatch_idx)
            assert len(slices) >= 1, "variable sequence slicing produced no slices."
            variable_microbatch_specs_list.append([slices, loss_func])
        max_num_slices = max(len(slices) for slices, _ in variable_microbatch_specs_list)
        assert max_num_slices >= pipeline_parallel_size, \
            "variable sequence slicing currently requires at least one chunk per pipeline stage."
        assert max_num_slices % pipeline_parallel_size == 0, \
            "variable sequence slicing currently requires chunk count to be divisible by pipeline size."
        for idx, (slices, _) in enumerate(variable_microbatch_specs_list):
            assert len(slices) % pipeline_parallel_size == 0, \
                "variable sequence slicing currently requires chunk count to be divisible by pipeline size."
            # DEBUG for variable length training.
            if idx < variable_seq_debug_limit:
                print(
                    "[variable-seq][schedule-fixed] "
                    f"rank={pipeline_parallel_rank}/{pipeline_parallel_size}, "
                    f"microbatch={idx}, slices={len(slices)}, max_slices={max_num_slices}, "
                    f"warmup_slices={num_slices_warmup}, "
                    f"target_inflight_slices={max_num_slices + num_slices_warmup}, "
                    f"forward_only={forward_only}",
                    flush=True,
                )
        variable_microbatch_specs.extend(variable_microbatch_specs_list)
        num_slices_preset = max_num_slices
        num_slices_target = 4096 if forward_only else max_num_slices + num_slices_warmup

    def calc_ctx_pair(batch: MicroBatch, attn_balance):
        assert attn_balance >= 0, 'attn_balance with cache is not implemented.'
        if not attn_balance:
            return None
        cache_len = [batch.cache_len(offset=pipeline_parallel_rank - i) for i in range(pipeline_parallel_size)]
        # avg = sum(cache_len) / pipeline_parallel_size
        # len = cache_len[pipeline_parallel_rank]
        # com = avg * 2 - len
        # assert com in cache_len, (pipeline_parallel_rank, cache_len)
        # idx = cache_len.index(com)
        # vol = int(avg - len)

        ord = sorted(cache_len)
        len = cache_len[pipeline_parallel_rank]
        com = ord[pipeline_parallel_size - 1 - ord.index(len)]
        idx = cache_len.index(com)
        avg = (com + len) / 2
        vol = int(avg - len)
        fwd = not (batch.slice_idx + 1 < pipeline_parallel_size - pipeline_parallel_rank and cnt_microbatches == 1 or
                    batch.num_slices - batch.slice_idx <= pipeline_parallel_rank and cnt_microbatches == num_microbatches)
        bwd = not (batch.slice_idx + 1 < pipeline_parallel_size - pipeline_parallel_rank and cnt_microbatches == num_microbatches or
                    batch.num_slices - batch.slice_idx <= pipeline_parallel_rank and cnt_microbatches == 1)
        nfwd = vol if fwd else 0
        nbwd = vol if bwd else 0

        # if msg == "fwd" and fwd or msg == "bwd" and bwd:
        #     torch.distributed.barrier(group=parallel_state.get_pipeline_model_parallel_group(), async_op=True)
        # if fwd or bwd:
        #     print(f"rank{pipeline_parallel_rank}: {msg}: {idx, vol, fwd, bwd} {cache_len}", flush=True)
        return CtxPair(idx, 0, nfwd, nbwd)

    # print messages for debug.
    DEBUG_P2PCOMM = False
    DEBUG_OFFLOAD = False
    def print_debug(flag, msg, value=None):
        if flag:
            if isinstance(value, torch.Tensor):
                value = (value.dtype, value.shape, value.abs().mean().item())
            print(f"rank{pipeline_parallel_rank}: {msg}: {value}", flush=True)

    # vocab in pp
    if vocab_in_pp:
        dummy_inputs = deque(); dummy_inputs.append([])
        dummy_outputs = deque(); dummy_outputs.append([])
        for cnt_pp_first in range(pipeline_parallel_rank):
            print_debug(DEBUG_P2PCOMM, "fwd pre+" , value=cnt_pp_first)
            encoder_input, dummy_input, _ = pre_process_forward(model, None, input_shape)
            print_debug(DEBUG_P2PCOMM, "fwd pre-" , value=cnt_pp_first)
            dummy_inputs[-1].append(dummy_input)
            if len(dummy_inputs[-1]) == num_slices:
                dummy_inputs.append([])
            del encoder_input, _

        cnt_pp_first = pipeline_parallel_rank
        cnt_pp_last = pipeline_parallel_rank - pipeline_parallel_size + 1

    # receive the first input_tensor.
    print_debug(DEBUG_P2PCOMM, "fwd recv+")
    input_tensor = p2p_communication.recv_forward(tensor_shape, dtype, batch_p2p_comm, timers)
    fwd_reqs = []
    bwd_reqs = []
    print_debug(DEBUG_P2PCOMM, "fwd recv-")

    while num_slices_flight or num_slices_target:
        cuda_sync_and_record(sync_level=1)
        """Forward"""
        if num_slices_flight < num_slices_target:
            if not batch_fwd:
                cnt_microbatches, batch_fwd = make_microbatch(cnt_microbatches)
                num_slices_target = max(num_slices_target, batch_fwd.num_slices_to_forward + num_slices_warmup)

        if vocab_in_pp and cnt_pp_first < num_microbatches * num_slices:
            if first_stage:
                data = batch_fwd.pop_data()
            else:
                data = None
            print_debug(DEBUG_P2PCOMM, "fwd pre+" , value=cnt_pp_first)
            encoder_input, dummy_input, data = pre_process_forward(model, data, input_shape)
            print_debug(DEBUG_P2PCOMM, "fwd pre-" , value=cnt_pp_first)
            dummy_inputs[-1].append(dummy_input)
            if len(dummy_inputs[-1]) == num_slices:
                dummy_inputs.append([])
            if first_stage:
                batch_fwd.append_data(data)
                assert input_tensor is None
                input_tensor = encoder_input.detach().requires_grad_()
            del encoder_input, data

        if num_slices_flight < num_slices_target:   # do forward to gain in-flight micro-batches.
            assert all(req.wait() for req in fwd_reqs); fwd_reqs = None
            print_debug(DEBUG_P2PCOMM, "fwd sendrecv-")
            batch_fwd.kv_cache.ctx_pair = calc_ctx_pair(batch_fwd, attn_balance)
            # if not (batch_fwd.slice_idx + 1 < pipeline_parallel_size - pipeline_parallel_rank and cnt_microbatches == 1 or
            #         batch_fwd.num_slices - batch_fwd.slice_idx <= pipeline_parallel_rank and cnt_microbatches == num_microbatches):
            #     torch.distributed.all_reduce(torch.empty((), dtype=dtype, device='cuda'), group=parallel_state.get_pipeline_model_parallel_group())
            # batch_fwd._bwd.append(not (batch_fwd.slice_idx + 1 < pipeline_parallel_size - pipeline_parallel_rank and cnt_microbatches == num_microbatches or
            #         batch_fwd.num_slices - batch_fwd.slice_idx <= pipeline_parallel_rank and cnt_microbatches == 1))
            print_debug(DEBUG_P2PCOMM or DEBUG_OFFLOAD, "forward+")
            output_tensor = batch_fwd.forward(input_tensor); input_tensor = None
            print_debug(DEBUG_P2PCOMM or DEBUG_OFFLOAD, "forward-")
            num_slices_flight += 1

            recv_prev = not first_stage and (batch_fwd.num_slices_to_forward or cnt_microbatches < num_microbatches)
            print_debug(DEBUG_P2PCOMM, "fwd sendrecv+", output_tensor)
            input_tensor, fwd_reqs = \
                p2p_communication.send_forward_recv_forward(output_tensor,
                                                            recv_prev,
                                                            tensor_shape,
                                                            dtype,
                                                            batch_p2p_comm,
                                                            overlap_p2p_comm,
                                                            timers)

        if vocab_in_pp and \
            cnt_pp_last >= 0 and cnt_pp_last < num_microbatches * num_slices:
            if last_stage:
                output = (batch_fwd or mb_queue[-1]).pop_output()
            else:
                output = None
            print_debug(DEBUG_P2PCOMM, "fwd post+" , value=cnt_pp_last)
            dummy_output, loss = post_process_forward(model, output, tensor_shape, input_shape, dtype)
            print_debug(DEBUG_P2PCOMM, "fwd post-" , value=cnt_pp_last)
            dummy_outputs[-1].append((dummy_output, loss))
            if len(dummy_outputs[-1]) == num_slices:
                dummy_outputs.append([])
            if last_stage:
                (batch_fwd or mb_queue[-1]).append_output(loss)
            del output

        if batch_fwd and batch_fwd.num_slices_to_forward == 0:
            forward_data_store.append(batch_fwd.compute_loss(loss_div))
            mb_queue.append(batch_fwd); batch_fwd = None

        if forward_only:
            num_slices_flight = 0
            continue

        if offload_req:
            offload_req.wait(); offload_req = None
            print_debug(DEBUG_OFFLOAD, "offload-")

        """Offload"""
        if num_slices_target: # before the cooldown
            prior_works = fwd_reqs
            mb = batch_fwd or mb_queue[-1]
            curr_om = mb.curr_om()
            if 2 * mb.num_slices_to_forward + num_slices_warmup > 2:
                # there are enough forward stages to overlap with offload and onload.
                print_debug(DEBUG_OFFLOAD, "offload+")
                offload_req = curr_om.offload(prior_works=prior_works)
            else:
                # never offload/onload acts. for this slice.
                curr_om.reset()
        else: # during the cooldown, no `fwd_reqs` are created.
            prior_works = bwd_reqs

        """Backward"""
        if num_slices_flight >= num_slices_target:  # do backward to consume in-flight micro-batches.
            assert all(req.wait() for req in (bwd_reqs or [])); bwd_reqs = None
            print_debug(DEBUG_P2PCOMM, "bwd sendrecv-")
            if onload_req:
                onload_req.wait(); onload_req = None
                print_debug(DEBUG_OFFLOAD, "onload-")
            batch_bwd = batch_bwd or mb_queue.popleft()

        if vocab_in_pp:
            cnt_pp_first += 1
            cnt_pp_last += 1

        if vocab_in_pp and \
            cnt_pp_last >= num_slices and \
            cnt_pp_last < (num_microbatches + 1) * num_slices:
            dummy_output, loss = dummy_outputs[0].pop()
            if len(dummy_outputs[0]) == 0:
                dummy_outputs.popleft()
            if last_stage:
                grad_loss = batch_bwd.pop_output_grad()
            else:
                grad_loss = None
            print_debug(DEBUG_P2PCOMM, "bwd post+" , value=cnt_pp_last)
            grad = post_process_backward(loss, grad_loss, dummy_output, tensor_shape)
            print_debug(DEBUG_P2PCOMM, "bwd post-" , value=cnt_pp_last)
            if last_stage:
                assert output_tensor_grad is None
                output_tensor_grad = grad
            del dummy_output, loss, grad_loss, grad

        if num_slices_flight >= num_slices_target:  # do backward to consume in-flight micro-batches.
            # if batch_bwd._bwd.pop():
            #     torch.distributed.all_reduce(torch.empty((), dtype=dtype, device='cuda'), group=parallel_state.get_pipeline_model_parallel_group())
            print_debug(DEBUG_P2PCOMM or DEBUG_OFFLOAD, "backward+")
            input_tensor_grad = batch_bwd.backward(output_tensor_grad); output_tensor_grad = None
            print_debug(DEBUG_P2PCOMM or DEBUG_OFFLOAD, "backward-")
            if not batch_bwd.num_slices_to_backward:
                batch_bwd = None
            num_slices_flight -= 1

        if vocab_in_pp and cnt_pp_first >= num_slices_preset + 2 * (pipeline_parallel_size - 1):
            dummy_input = dummy_inputs[0].pop()
            if len(dummy_inputs[0]) == 0:
                dummy_inputs.popleft()
            if first_stage:
                assert input_tensor_grad is not None
                encoder_input_grad = input_tensor_grad
                input_tensor_grad = None
            else:
                encoder_input_grad = None
            print_debug(DEBUG_P2PCOMM, "bwd pre+" , value=cnt_pp_first)
            pre_process_backward(dummy_input, encoder_input_grad)
            print_debug(DEBUG_P2PCOMM, "bwd pre-" , value=cnt_pp_first)
            del dummy_input, encoder_input_grad

        num_slices_target = (batch_fwd or cnt_microbatches < num_microbatches) and num_slices_target
        if num_slices_flight + 1 >= num_slices_target:
            recv_next = not last_stage and (mb_queue or batch_bwd or batch_fwd)
            print_debug(DEBUG_P2PCOMM, "bwd sendrecv+")
            output_tensor_grad, bwd_reqs = \
                p2p_communication.send_backward_recv_backward(input_tensor_grad,
                                                              recv_next,
                                                              tensor_shape,
                                                              dtype,
                                                              batch_p2p_comm,
                                                              overlap_p2p_comm,
                                                              timers=timers)

        """Onload"""
        # NOTE(lizhouyang): `onload` should start after the forward, **NOT** the backward.
        # `prior_works` are usually `fwd_reqs` except for cooldown.
        if num_slices_flight and num_slices_flight + 1 >= num_slices_target:  # after the second last round of warmup.
            next_om = (batch_bwd or (mb_queue[0] if mb_queue else batch_fwd)).curr_om()
            print_debug(DEBUG_OFFLOAD, "onload+", cnt_onload)
            onload_req = next_om.onload(prior_works=prior_works, buffer_name="onload", buffer_idx=cnt_onload % 2)
            cnt_onload += 1

    if vocab_in_pp:
        assert len(dummy_inputs[-1]) == 0
        dummy_inputs.pop()
        if pipeline_parallel_rank:
            assert len(dummy_inputs[0]) == pipeline_parallel_rank
        else:
            assert not dummy_inputs
        while dummy_inputs:
            dummy_input = dummy_inputs[0].pop()
            if len(dummy_inputs[0]) == 0:
                dummy_inputs.popleft()
            print_debug(DEBUG_P2PCOMM, "bwd pre+" , value=cnt_pp_first)
            pre_process_backward(dummy_input, None)
            print_debug(DEBUG_P2PCOMM, "bwd pre-" , value=cnt_pp_first)
    if not forward_only:
        assert all(req.wait() for req in (bwd_reqs or [])); bwd_reqs = None
    return forward_data_store


def pipelining_with_variable_slicing(**kwargs):
    """Run non-interleaved pipeline parallelism with variable chunk counts.

    This is a separate entry point from the original SlimPipe schedules so the
    baseline remains directly comparable. It reuses the non-interleaved slicing
    implementation with dynamic target sizing enabled.
    """
    return pipelining_with_slicing(_variable_slicing=True, **kwargs)


def pipelining_with_variable_slicing_slice_v(*,
                                         forward_step_func,
                                         get_batch_func,
                                         data_iterator: Union[Iterator, List[Iterator]],
                                         model: Union[torch.nn.Module, List[torch.nn.Module]],
                                         num_microbatches: int,
                                         micro_seq_length: int,
                                         kv_cache_class: Type[Cache],
                                         dtype: Optional[torch.dtype] = None,
                                         tensor_shape: Optional[Shape] = None,
                                         decoder_seq_length: Optional[int] = None,
                                         grad_scaler: Callable = None,
                                         sequence_parallel: bool = False,
                                         overlap_p2p_comm: bool = False,
                                         batch_p2p_comm: bool = True,
                                         attn_balance: int = 0,
                                         vocab_in_pp: bool = False,
                                         forward_only: bool = False,
                                         timers: Callable = None,
                                         collect_non_loss_data: bool = False,
                                         enable_autocast: bool = False,
                                         deallocate_pipeline_outputs: bool = False,
                                         no_sync_func: Optional[Callable] = None,
                                         grad_sync_func: Optional[Callable] = None,
                                         param_sync_func: Optional[Callable] = None,
                                         pre_p2p_func: Optional[Callable] = None,
                                         post_p2p_async_func: Optional[Callable] = None,
                                         offload_ratio: float = 0,
                                         offload_delay_to_next_stage: bool = False,
                                         ):
    """Run the SliceV schedule with variable-length slices and two model chunks.

    This is a separate experimental schedule. The existing variable-slicing
    1F1B schedule remains available through ``--variable-seq-schedule 1f1b``.
    """
    assert isinstance(model, list) and len(model) == 2, \
        "SliceV requires exactly two virtual pipeline model chunks"
    if isinstance(data_iterator, list):
        data_iterator = data_iterator[0]
    assert get_batch_func is not None
    assert not forward_only, "SliceV only supports backward training"
    assert not attn_balance, "SliceV does not support attention balancing yet"
    assert not vocab_in_pp, "SliceV does not support vocab-in-PP yet"
    assert not no_sync_func and not grad_sync_func and not param_sync_func, \
        "SliceV does not support custom sync hooks yet"
    assert not offload_ratio, "SliceV does not support activation offload yet"

    WeightGradStore.assert_supported()

    model_type = get_model_type(model[0])
    assert model_type != ModelType.encoder_and_decoder, \
        "SliceV does not support encoder-decoder models"

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    tensor_shape = get_actual_tensor_shape(tensor_shape, sequence_parallel, micro_seq_length)
    args = get_args()
    variable_seq_debug_limit = getattr(args, 'variable_seq_debug_num_batches', 0)

    forward_funcs = [
        partial(forward_step, forward_step_func=forward_step_func,
                model=model_chunk, timers=timers,
                enable_autocast=enable_autocast)
        for model_chunk in model
    ]
    backward_func = partial(backward_step, model_type=model_type,
                            timers=timers,
                            deallocate_pipeline_outputs=deallocate_pipeline_outputs)

    microbatches: Dict[Tuple[int, int], MicroBatch] = {}
    split_counts: List[int] = []
    forward_data_store = []
    loss_div = num_microbatches

    for microbatch_idx in range(num_microbatches):
        if timers is not None:
            record_context = timers.is_recording_active()
            if record_context:
                timers.set_record_context(mb=microbatch_idx)
            timers('batch-generator', log_level=2).start()
        sliced_batch, loss_func = get_batch_func(data_iterator)
        if timers is not None:
            timers('batch-generator').stop()
            if record_context:
                timers.clear_record_context()
        slices = sliced_batch(micro_seq_length)
        assert len(slices) >= 1, "variable sequence slicing produced no slices."
        split_counts.append(len(slices))
        for chunk in range(2):
            parallel_state.set_virtual_pipeline_model_parallel_rank(chunk)
            chunk_loss_func = loss_func if (
                chunk == 1 and pipeline_parallel_rank == 0
            ) else None
            microbatches[(microbatch_idx, chunk)] = MicroBatch(
                slices,
                kv_cache_class(),
                offload_ratio,
                forward_funcs[chunk],
                backward_func,
                chunk_loss_func,
                grad_scaler,
                batch_idx=microbatch_idx,
                timers=timers,
            )

    schedules, plan = build_slice_v_schedule(
        pipeline_parallel_size,
        num_microbatches,
        split_counts,
    )
    schedule = schedules[pipeline_parallel_rank]

    def comm_key(node):
        return (node.kind, node.chunk, node.microbatch, node.split)

    def incoming_messages(stage, stage_schedule):
        messages = []
        for node in stage_schedule:
            key = comm_key(node)
            if node.kind == 'F' and node.chunk == 0 and stage > 0:
                messages.append(('prev', key))
            elif node.kind == 'F' and node.chunk == 1 and stage < pipeline_parallel_size - 1:
                messages.append(('next', key))
            elif node.kind == 'B' and node.chunk == 1 and stage > 0:
                messages.append(('prev', key))
            elif node.kind == 'B' and node.chunk == 0 and stage < pipeline_parallel_size - 1:
                messages.append(('next', key))
        return messages

    def outgoing_messages(stage, stage_schedule):
        messages = []
        for node in stage_schedule:
            key = comm_key(node)
            if node.kind == 'F' and node.chunk == 0 and stage < pipeline_parallel_size - 1:
                messages.append(('next', key))
            elif node.kind == 'F' and node.chunk == 1 and stage > 0:
                messages.append(('prev', key))
            elif node.kind == 'B' and node.chunk == 1 and stage < pipeline_parallel_size - 1:
                messages.append(('next', key))
            elif node.kind == 'B' and node.chunk == 0 and stage > 0:
                messages.append(('prev', key))
        return messages

    def validate_p2p_order():
        for left in range(pipeline_parallel_size - 1):
            right = left + 1
            left_out = [key for direction, key in outgoing_messages(left, schedules[left])
                        if direction == 'next']
            right_in = [key for direction, key in incoming_messages(right, schedules[right])
                        if direction == 'prev']
            if left_out != right_in:
                raise RuntimeError(
                    "SliceV schedule has mismatched left-to-right P2P order "
                    f"between ranks {left}->{right}: send={left_out[:8]}, recv={right_in[:8]}"
                )
            right_out = [key for direction, key in outgoing_messages(right, schedules[right])
                         if direction == 'prev']
            left_in = [key for direction, key in incoming_messages(left, schedules[left])
                       if direction == 'next']
            if right_out != left_in:
                raise RuntimeError(
                    "SliceV schedule has mismatched right-to-left P2P order "
                    f"between ranks {right}->{left}: send={right_out[:8]}, recv={left_in[:8]}"
                )

    validate_p2p_order()

    if variable_seq_debug_limit:
        # DEBUG for variable length training.
        print(
            "[variable-seq][slice-v-schedule] "
            f"rank={pipeline_parallel_rank}/{pipeline_parallel_size}, "
            f"split_counts={split_counts}, events={len(schedule)}, "
            f"phase_repeats={plan.phase_repeats[pipeline_parallel_rank]}",
            flush=True,
        )
        for node in schedule[:variable_seq_debug_limit * 12]:
            # DEBUG for variable length training.
            print(
                "[variable-seq][slice-v-event] "
                f"rank={pipeline_parallel_rank}, kind={node.kind}{node.chunk}, "
                f"microbatch={node.microbatch}, split={node.split}, "
                f"phase={node.phase}, slot={node.slot}",
                flush=True,
            )

    local_forward_bridge: Dict[Tuple[int, int], torch.Tensor] = {}
    local_backward_bridge: Dict[Tuple[int, int], torch.Tensor] = {}
    trace_slice_v = variable_seq_debug_limit > 0
    pending_send = None
    outstanding_sends = []

    def node_desc(node):
        return (
            f"kind={node.kind}{node.chunk}, mb={node.microbatch}, "
            f"split={node.split}, slot={node.slot}"
        )

    def trace(action, node=None, peer=None, key=None):
        if not trace_slice_v:
            return
        details = [f"rank={pipeline_parallel_rank}", f"action={action}"]
        if node is not None:
            details.append(node_desc(node))
        if peer is not None:
            details.append(f"peer={peer}")
        if key is not None:
            details.append(f"key={key}")
        # DEBUG for variable length training.
        print("[variable-seq][slice-v-trace] " + ", ".join(details), flush=True)

    def wait_reqs(reqs, action="wait-reqs", node=None, peer=None, key=None):
        if reqs:
            trace(action + "-begin", node=node, peer=peer, key=key)
        for req in (reqs or []):
            req.wait()
        if reqs:
            trace(action + "-end", node=node, peer=peer, key=key)

    def allocate_recv_tensor():
        if dtype is None:
            raise RuntimeError("dtype must be provided for SliceV P2P receives")
        if tensor_shape is None:
            raise RuntimeError("tensor_shape must be provided for SliceV P2P receives")
        return torch.empty(tensor_shape,
                           requires_grad=True,
                           device=torch.cuda.current_device(),
                           dtype=dtype)

    pipeline_group = parallel_state.get_pipeline_model_parallel_group()
    prev_pipeline_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
    next_pipeline_rank = parallel_state.get_pipeline_model_parallel_next_rank()

    def ordered_p2p_ops(tensor_send_prev, tensor_recv_prev,
                        tensor_send_next, tensor_recv_next):
        def send_next():
            if tensor_send_next is None:
                return None
            return dist.isend(tensor_send_next, next_pipeline_rank, group=pipeline_group)

        def recv_prev():
            if tensor_recv_prev is None:
                return None
            return dist.irecv(tensor_recv_prev, prev_pipeline_rank, group=pipeline_group)

        def send_prev():
            if tensor_send_prev is None:
                return None
            return dist.isend(tensor_send_prev, prev_pipeline_rank, group=pipeline_group)

        def recv_next():
            if tensor_recv_next is None:
                return None
            return dist.irecv(tensor_recv_next, next_pipeline_rank, group=pipeline_group)

        if pipeline_parallel_rank % 2 == 0:
            ordered_ops = (
                ('send', send_next),
                ('recv', recv_prev),
                ('send', send_prev),
                ('recv', recv_next),
            )
        else:
            ordered_ops = (
                ('recv', recv_prev),
                ('send', send_next),
                ('recv', recv_next),
                ('send', send_prev),
            )
        requests = {'send': [], 'recv': []}
        for op_type, op in ordered_ops:
            req = op()
            if req is not None:
                requests[op_type].append(req)
        return requests

    def communicate(recv_direction, node=None):
        """Flush one pending send and receive the current task input together."""
        nonlocal pending_send
        tensor_send_prev = None
        tensor_send_next = None
        send_node = None
        if pending_send is not None:
            send_direction, send_tensor, send_node = pending_send
            if send_direction == 'prev':
                tensor_send_prev = send_tensor
            else:
                tensor_send_next = send_tensor

        tensor_recv_prev = allocate_recv_tensor() if recv_direction == 'prev' else None
        tensor_recv_next = allocate_recv_tensor() if recv_direction == 'next' else None
        if pending_send is None and recv_direction is None:
            return None

        trace(
            "p2p-begin",
            node=node,
            key={
                'send': comm_key(send_node) if send_node is not None else None,
                'send_direction': pending_send[0] if pending_send is not None else None,
                'recv': comm_key(node) if recv_direction is not None else None,
                'recv_direction': recv_direction,
            },
        )
        requests = ordered_p2p_ops(
            tensor_send_prev,
            tensor_recv_prev,
            tensor_send_next,
            tensor_recv_next,
        )
        if requests['send']:
            outstanding_sends.append((requests['send'], send_tensor, send_node))
        wait_reqs(requests['recv'], action="p2p-recv-wait", node=node)
        trace("p2p-end", node=node)
        pending_send = None
        return tensor_recv_prev if recv_direction == 'prev' else tensor_recv_next

    def defer_send(direction, tensor, node):
        nonlocal pending_send
        assert pending_send is None, "SliceV allows at most one pending send"
        pending_send = (direction, tensor, node)
        trace("p2p-defer-send", node=node, key=direction)

    def detach_pipeline_boundary(tensor):
        if tensor is None:
            return None
        detached = tensor.detach()
        detached.requires_grad_()
        return detached

    def run_forward(node):
        parallel_state.set_virtual_pipeline_model_parallel_rank(node.chunk)
        bridge_key = (node.microbatch, node.split)
        trace("forward-begin", node=node)
        if node.chunk == 0:
            recv_direction = None if pipeline_parallel_rank == 0 else 'prev'
            input_tensor = communicate(recv_direction, node)
        else:
            if pipeline_parallel_rank == pipeline_parallel_size - 1:
                communicate(None, node)
                trace("forward-local-bridge-pop", node=node, key=bridge_key)
                input_tensor = local_forward_bridge.pop(bridge_key)
            else:
                input_tensor = communicate('next', node)
        mb = microbatches[(node.microbatch, node.chunk)]
        trace("forward-compute-begin", node=node)
        output_tensor = mb.forward(input_tensor)
        trace("forward-compute-end", node=node)
        if node.chunk == 0:
            if pipeline_parallel_rank == pipeline_parallel_size - 1:
                trace("forward-local-bridge-push", node=node, key=bridge_key)
                local_forward_bridge[bridge_key] = detach_pipeline_boundary(output_tensor)
            else:
                defer_send('next', output_tensor, node)
        else:
            if pipeline_parallel_rank != 0:
                defer_send('prev', output_tensor, node)
        if node.chunk == 1 and mb.num_slices_to_forward == 0:
            trace("loss-begin", node=node)
            forward_data_store.append(mb.compute_loss(loss_div))
            trace("loss-end", node=node)
        trace("forward-end", node=node)

    def run_backward(node):
        parallel_state.set_virtual_pipeline_model_parallel_rank(node.chunk)
        bridge_key = (node.microbatch, node.split)
        trace("backward-begin", node=node)
        if node.chunk == 1:
            recv_direction = None if pipeline_parallel_rank == 0 else 'prev'
            output_tensor_grad = communicate(recv_direction, node)
        else:
            if pipeline_parallel_rank == pipeline_parallel_size - 1:
                communicate(None, node)
                trace("backward-local-bridge-pop", node=node, key=bridge_key)
                output_tensor_grad = local_backward_bridge.pop(bridge_key)
            else:
                output_tensor_grad = communicate('next', node)
        mb = microbatches[(node.microbatch, node.chunk)]
        trace("backward-compute-begin", node=node)
        input_tensor_grad = mb.backward_b(output_tensor_grad, chunk=node.chunk)
        trace("backward-compute-end", node=node)
        if node.chunk == 1:
            if pipeline_parallel_rank == pipeline_parallel_size - 1:
                trace("backward-local-bridge-push", node=node, key=bridge_key)
                local_backward_bridge[bridge_key] = input_tensor_grad
            else:
                defer_send('next', input_tensor_grad, node)
        else:
            if pipeline_parallel_rank != 0:
                defer_send('prev', input_tensor_grad, node)
        trace("backward-end", node=node)

    def run_weight(node):
        parallel_state.set_virtual_pipeline_model_parallel_rank(node.chunk)
        communicate(None, node)
        trace("weight-begin", node=node)
        microbatches[(node.microbatch, node.chunk)].weight_grad(chunk=node.chunk)
        trace("weight-end", node=node)

    for node in schedule:
        cuda_sync_and_record(sync_level=1)
        if node.kind == 'F':
            run_forward(node)
        elif node.kind == 'B':
            run_backward(node)
        elif node.kind == 'W':
            run_weight(node)
        else:
            raise RuntimeError(f"unknown SliceV schedule event: {node}")

    communicate(None)
    for send_reqs, _send_tensor, send_node in outstanding_sends:
        wait_reqs(send_reqs, action="p2p-send-final-wait", node=send_node)
    assert not local_forward_bridge, \
        f"Unconsumed local forward bridge tensors: {list(local_forward_bridge.keys())[:8]}"
    assert not local_backward_bridge, \
        f"Unconsumed local backward bridge tensors: {list(local_backward_bridge.keys())[:8]}"
    WeightGradStore.assert_empty()
    return forward_data_store


def pipelining_with_interleaved_slicing(*,
                                        forward_step_func,
                                        get_batch_func,
                                        data_iterator: Union[Iterator, List[Iterator]],
                                        model: Union[torch.nn.Module, List[torch.nn.Module]],
                                        num_microbatches: int,
                                        micro_seq_length: int,
                                        kv_cache_class: Type[Cache],
                                        dtype: Optional[torch.dtype] = None,
                                        tensor_shape: Optional[Shape] = None,
                                        decoder_seq_length: Optional[int] = None, # unused
                                        grad_scaler: Callable = None,
                                        sequence_parallel: bool = False,
                                        overlap_p2p_comm: bool = False,
                                        batch_p2p_comm: bool = True,
                                        attn_balance: int = 0,
                                        vocab_in_pp: bool = False,
                                        forward_only: bool = False,
                                        timers: Callable = None,
                                        collect_non_loss_data: bool = False, # unused
                                        enable_autocast: bool = False,
                                        deallocate_pipeline_outputs: bool = False,
                                        no_sync_func: Optional[Callable] = None, # unused
                                        grad_sync_func: Optional[Callable] = None, # unused
                                        param_sync_func: Optional[Callable] = None, # unused
                                        pre_p2p_func: Optional[Callable] = None,
                                        post_p2p_async_func: Optional[Callable] = None,
                                        offload_ratio: float = 0,
                                        offload_delay_to_next_stage: bool = False, # unused
                                        ):
    """Run interleaved 1F1B schedule (model split into model chunks), with batch and token level pipeline parallelism.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    assert isinstance(model, list), \
        "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), \
        "invalid model chunking"
    assert isinstance(data_iterator, list), \
        "interleaved pipeline parallelism expected each model chunk to have a data iterator"
    assert overlap_p2p_comm, \
        "Slicing pipeline parallelism only supports overlapping p2p communication"
    assert not batch_p2p_comm, \
        "Slicing pipeline parallelism does not support using batched p2p communication"

    assert not isinstance(model, torchDDP), "torchDDP is no longer supported."
    model_type = get_model_type(model[0])
    assert model_type != ModelType.encoder_and_decoder, \
        "Interleaving is not supported with an encoder and decoder model."

    assert not no_sync_func, "Not implemented!"
    assert not grad_sync_func, "Not implemented!"
    assert not param_sync_func, "Not implemented!"

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    first_stage = parallel_state.is_pipeline_first_stage(ignore_virtual=True)
    last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=True)

    num_slices = tensor_shape[0] // micro_seq_length
    # shape of input_ids.
    input_shape = (tensor_shape[1], micro_seq_length // parallel_state.get_context_parallel_world_size())
    tensor_shape = get_actual_tensor_shape(tensor_shape, sequence_parallel, micro_seq_length)

    cnt_microbatches = 0
    loss_div = num_microbatches
    forward_data_store = []
    num_stages = len(model)
    # find the minimum factor of `num_slices` that is greater than `pipeline_parallel_size`.
    assert num_slices >= pipeline_parallel_size
    group_size = pipeline_parallel_size
    while num_slices % group_size:
        group_size += 1
    # minimun number of slices to fullfill the pipeline.
    num_slices_preset = num_slices * num_stages
    # warm up to prefill the pipeline.
    num_slices_warmup = 2 * (pipeline_parallel_size - pipeline_parallel_rank - 1)
    # in-flight slices need to gain.
    num_slices_target = 4096 if forward_only else num_slices_preset + num_slices_warmup
    # current in-flight slices.
    num_slices_flight = 0
    # batches
    batch_queue: Deque[CycledBatch] = deque()
    batch_fwd: Optional[CycledBatch] = None
    batch_bwd: Optional[CycledBatch] = None
    # recv tensors in advance.
    recv_fwd_offset = first_stage * (group_size - pipeline_parallel_size)
    recv_bwd_offset = last_stage * (group_size - pipeline_parallel_size)
    input_tensors = deque()
    output_tensor_grads = deque()
    # send tensors in arrears.
    # send_fwd_offset = last_stage * (group_size - pipeline_parallel_size)
    # send_bwd_offset = first_stage * (group_size - pipeline_parallel_size)
    # output_tensors = deque()
    # input_tensor_grads = deque()
    recv_fwd_stage_idx = 0
    recv_bwd_stage_idx = -1 % num_stages
    input_tensor_grad = None
    # offload reqs
    offload_req = None
    onload_req = None
    vocab_offload_req = None
    vocab_onload_req = None
    cnt_onload = 0

    # attention balance
    attn_balancer = AttnBalancer(num_microbatches, pipeline_parallel_rank, pipeline_parallel_size, attn_balance)

    def make_microbatch(model, data_iterator, batch_idx=None):
        forward_func = partial(forward_step, forward_step_func=forward_step_func, model=model,
                               timers=timers, enable_autocast=enable_autocast)
        backward_func = partial(backward_step, model_type=model_type,
                                timers=timers, deallocate_pipeline_outputs=deallocate_pipeline_outputs) \
                        if not forward_only else None
        if timers is not None:
            record_context = timers.is_recording_active()
            if record_context:
                timers.set_record_context(mb=batch_idx)
            timers('batch-generator', log_level=2).start()
        sliced_batch, loss_func = get_batch_func(data_iterator)
        if timers is not None:
            timers('batch-generator').stop()
            if record_context:
                timers.clear_record_context()
        if not parallel_state.is_pipeline_last_stage():
            loss_func = None
        slices = sliced_batch(micro_seq_length)
        assert len(slices) >= group_size, "number of slices per microbatch is not enough."
        kv_cache = kv_cache_class()
        mb = MicroBatch(slices, kv_cache, offload_ratio, forward_func,
                        backward_func, loss_func, grad_scaler,
                        batch_idx=batch_idx,
                        timers=timers)
        return mb

    def make_cycled_batch(cnt_microbatches):
        assert cnt_microbatches < num_microbatches, "No more microbatches."
        mbatches = []
        for i in range(num_stages):
            parallel_state.set_virtual_pipeline_model_parallel_rank(i)
            mbatches.append(make_microbatch(model[i], data_iterator[i],
                                            cnt_microbatches))
        cbatch = CycledBatch(cnt_microbatches, mbatches, group_size)
        num_slices_total = num_slices_flight + (num_microbatches - cnt_microbatches) * cbatch.num_slices_to_forward
        assert num_slices_total >= num_slices_target, "number of total slices is not enough."
        return cnt_microbatches + 1, cbatch

    # print messages for debug.
    DEBUG_CTXPAIR = False
    DEBUG_P2PCOMM = False
    DEBUG_OFFLOAD = False
    cnt_fwd = 0
    cnt_bwd = 0
    def print_debug(flag, msg, value=None):
        if flag:
            stage = parallel_state.get_virtual_pipeline_model_parallel_rank()
            if isinstance(value, torch.Tensor):
                value = (value.dtype, value.shape, value.abs().mean().item())
            print(f"rank{pipeline_parallel_rank}: {msg}: {value}", flush=True)


    # vocab in pp
    if vocab_in_pp:
        dummy_inputs = deque(); dummy_inputs.append([])
        dummy_outputs = deque(); dummy_outputs.append([])
        for cnt_pp_first in range(pipeline_parallel_rank):
            print_debug(DEBUG_P2PCOMM, "fwd pre+" , value=cnt_pp_first)
            encoder_input, dummy_input, _ = pre_process_forward(model[0], None, input_shape)
            print_debug(DEBUG_P2PCOMM, "fwd pre-" , value=cnt_pp_first)
            dummy_inputs[-1].append(dummy_input)
            if len(dummy_inputs[-1]) == num_slices:
                dummy_inputs.append([])
            del encoder_input, _
        cnt_pp_first = pipeline_parallel_rank
        cnt_pp_last = pipeline_parallel_rank - pipeline_parallel_size + 1

    # receive the first input_tensor.
    parallel_state.set_virtual_pipeline_model_parallel_rank(recv_fwd_stage_idx)
    recv_prev = not parallel_state.is_pipeline_first_stage()
    print_debug(DEBUG_P2PCOMM, "fwd recv+" if recv_prev else "fwd noop+")
    if pre_p2p_func: pre_p2p_func()
    input_tensor = p2p_communication.recv_forward(tensor_shape, dtype, batch_p2p_comm, timers)
    if post_p2p_async_func: post_p2p_async_func([])
    fwd_recv = None
    input_tensors.append((input_tensor, fwd_recv))
    if len(input_tensors) > recv_fwd_offset:
        input_tensor, fwd_recv = input_tensors.popleft()

    while num_slices_flight or num_slices_target:
        cuda_sync_and_record(sync_level=1)
        """Forward"""
        if num_slices_flight < num_slices_target:   # do forward to gain in-flight micro-batches.
            if not batch_fwd:
                cnt_microbatches, batch_fwd = make_cycled_batch(cnt_microbatches)
                num_slices_target = max(num_slices_target, batch_fwd.num_slices_to_forward + num_slices_warmup)

        if vocab_in_pp and cnt_pp_first < num_microbatches * num_stages * num_slices and \
            cnt_pp_first % (num_stages * group_size) // group_size == 0:
            if first_stage:
                data = batch_fwd.curr_mbatch().pop_data()
            else:
                data = None
            print_debug(DEBUG_P2PCOMM, "fwd pre+" , value=cnt_pp_first)
            encoder_input, dummy_input, data = pre_process_forward(model[0], data, input_shape)
            print_debug(DEBUG_P2PCOMM, "fwd pre-" , value=cnt_pp_first)
            dummy_inputs[-1].append(dummy_input)
            if len(dummy_inputs[-1]) == num_slices:
                dummy_inputs.append([])
            if first_stage:
                batch_fwd.curr_mbatch().append_data(data)
                assert input_tensor is None
                input_tensor = encoder_input.detach().requires_grad_()
            del encoder_input, data

        if num_slices_flight < num_slices_target:
            if fwd_recv:
                fwd_recv.wait(); fwd_recv = None
            print_debug(DEBUG_P2PCOMM, "fwd sendrecv-" if recv_prev else "fwd send-", value=input_tensor)
            ctx_pair = attn_balancer.calc_ctx_pair(batch_fwd)
            print_debug(DEBUG_CTXPAIR, "fwd ctx_pair", value=ctx_pair)
            batch_fwd.curr_mbatch().update_kv_cache(ctx_pair)
            print_debug(DEBUG_P2PCOMM or DEBUG_OFFLOAD, "forward+", value=cnt_fwd)
            output_tensor = batch_fwd.forward(input_tensor); input_tensor = None
            recv_fwd_stage_idx = batch_fwd.stage_idx(offset=recv_fwd_offset)
            print_debug(DEBUG_P2PCOMM or DEBUG_OFFLOAD, "forward-", value=cnt_fwd); cnt_fwd += 1
            num_slices_flight += 1

            # output_tensors.append(output_tensor)
            # output_tensor = output_tensors.popleft() if len(output_tensors) > send_fwd_offset else None
            parallel_state.set_virtual_pipeline_model_parallel_rank(recv_fwd_stage_idx)
            if (batch_fwd and batch_fwd.num_slices_to_forward > recv_fwd_offset) or cnt_microbatches < num_microbatches:
                recv_prev = not parallel_state.is_pipeline_first_stage()
            else:
                recv_prev = False
            print_debug(DEBUG_P2PCOMM, "fwd sendrecv+" if recv_prev else "fwd send+", value=output_tensor)
            if pre_p2p_func: pre_p2p_func()
            input_tensor, fwd_reqs = \
                p2p_communication.send_forward_recv_forward(output_tensor,
                                                            recv_prev,
                                                            tensor_shape,
                                                            dtype,
                                                            batch_p2p_comm,
                                                            overlap_p2p_comm,
                                                            timers); output_tensor = None
            if post_p2p_async_func: post_p2p_async_func(fwd_reqs)
            fwd_recv = fwd_reqs[-1] if recv_prev else None
            input_tensors.append((input_tensor, fwd_recv)); input_tensor, fwd_recv = None, None
            input_tensor, fwd_recv = input_tensors.popleft() if len(input_tensors) > recv_fwd_offset else (None, None)

        if offload_req:
            offload_req.wait(); offload_req = None
            print_debug(DEBUG_OFFLOAD, "offload-")

        """Offload"""
        prior_works = [torch.cuda.current_stream().record_event()]  # offload/onload waits for forward.
        if num_slices_target: # before the cooldown
            prior_works += fwd_reqs
            curr_batch = batch_fwd or batch_queue[-1]
            curr_om = curr_batch.prev_mbatch().curr_om()
            if 2 * curr_batch.num_slices_to_forward + num_slices_warmup > 2:
                # there are enough forward stages to overlap with offload and onload.
                print_debug(DEBUG_OFFLOAD, "offload+")
                offload_req = curr_om.offload(prior_works=prior_works)
            else:
                # never offload/onload acts. for this slice.
                curr_om.reset()
        else: # during the cooldown, no `fwd_reqs` are created.
            prior_works += bwd_reqs

        if vocab_offload_req:
            vocab_offload_req.wait(); vocab_offload_req = None
            print_debug(DEBUG_OFFLOAD, "vocab offload-")
        if vocab_in_pp and \
            cnt_pp_last >= 0 and cnt_pp_last < num_microbatches * num_stages * num_slices and \
            cnt_pp_last % (num_stages * group_size) // group_size + 1 == num_stages:
            if last_stage:
                output = batch_fwd.prev_mbatch().pop_output()
            else:
                output = None
            print_debug(DEBUG_P2PCOMM, "fwd post+" , value=cnt_pp_last)
            dummy_output, loss, vocab_om = post_process_forward(
                model[0], output, tensor_shape, input_shape, dtype, offload_ratio)
            print_debug(DEBUG_P2PCOMM, "fwd post-" , value=cnt_pp_last)
            dummy_outputs[-1].append((dummy_output, loss, vocab_om))
            if len(dummy_outputs[-1]) + 2 <= num_slices:
                print_debug(DEBUG_OFFLOAD, "vocab offload+")
                vocab_offload_req = vocab_om.offload(
                    prior_works=[torch.cuda.current_stream().record_event()] + (fwd_reqs or bwd_reqs))
            else:
                vocab_om.reset()
            if len(dummy_outputs[-1]) == num_slices:
                dummy_outputs.append([])
            if last_stage:
                batch_fwd.prev_mbatch().append_output(loss)
            del output

        if batch_fwd and batch_fwd.num_slices_to_forward == 0:
            forward_data_store.append(batch_fwd.compute_loss(loss_div))
            batch_queue.append(batch_fwd); batch_fwd = None

        if forward_only:
            num_slices_flight = 0
            continue

        """Backward"""
        if num_slices_flight >= num_slices_target:  # do backward to consume in-flight micro-batches.
            batch_bwd = batch_bwd or batch_queue.popleft()

        if vocab_in_pp:
            cnt_pp_first += 1
            cnt_pp_last += 1

        if vocab_onload_req:
            vocab_onload_req.wait(); vocab_onload_req = None
            print_debug(DEBUG_OFFLOAD, "vocab onload-")
        if vocab_in_pp and \
            cnt_pp_last >= num_stages * num_slices and \
            cnt_pp_last < (num_microbatches + 1) * num_stages * num_slices and \
            cnt_pp_last % (num_stages * group_size) // group_size == 0:
            dummy_output, loss, vocab_om = dummy_outputs[0].pop()
            assert vocab_om.is_complete()
            if dummy_outputs[0]:
                vocab_om = dummy_outputs[0][-1][-1]
                vocab_onload_req = vocab_om.onload(
                    prior_works=prior_works, buffer_name="vocab", buffer_idx=cnt_pp_last % 2)
            if len(dummy_outputs[0]) == 0:
                dummy_outputs.popleft()
            if last_stage:
                grad_loss = batch_bwd.curr_mbatch().pop_output_grad()
            else:
                grad_loss = None
            print_debug(DEBUG_P2PCOMM, "bwd post+" , value=cnt_pp_last)
            grad = post_process_backward(loss, grad_loss, dummy_output, tensor_shape)
            print_debug(DEBUG_P2PCOMM, "bwd post-" , value=cnt_pp_last)
            if last_stage:
                assert output_tensor_grad is None
                output_tensor_grad = grad
            del dummy_output, loss, grad_loss, grad

        if num_slices_flight >= num_slices_target:
            if bwd_recv:
                bwd_recv.wait(); bwd_recv = None
            print_debug(DEBUG_P2PCOMM, "bwd sendrecv-" if recv_next else "bwd send-", value=output_tensor_grad)
            if onload_req:
                onload_req.wait(); onload_req = None
                print_debug(DEBUG_OFFLOAD, "onload-")
            print_debug(DEBUG_P2PCOMM or DEBUG_OFFLOAD, "backward+", value=cnt_bwd)
            input_tensor_grad = batch_bwd.backward(output_tensor_grad); output_tensor_grad = None
            recv_bwd_stage_idx = batch_bwd.stage_idx(offset=-recv_bwd_offset)
            print_debug(DEBUG_P2PCOMM or DEBUG_OFFLOAD, "backward-", value=cnt_bwd); cnt_bwd += 1
            if not batch_bwd.num_slices_to_backward:
                batch_bwd = None
            num_slices_flight -= 1
            # input_tensor_grads.append(input_tensor_grad)
            # input_tensor_grad = input_tensor_grads.popleft() if len(input_tensor_grads) > send_bwd_offset else None

        # enter cooldown phase if no more forward pass
        num_slices_target = (batch_fwd or cnt_microbatches < num_microbatches) and num_slices_target

        """Onload"""
        # NOTE(lizhouyang): `onload` should start after the forward, **NOT** the backward.
        # `prior_works` are usually `fwd_reqs` except for cooldown.
        if num_slices_flight and num_slices_flight + 1 >= num_slices_target:  # after the second last round of warmup.
            next_om = (batch_bwd or (batch_queue[0] if batch_queue else batch_fwd)).curr_mbatch().curr_om()
            print_debug(DEBUG_OFFLOAD, "onload+", cnt_onload)
            onload_req = next_om.onload(prior_works=prior_works, buffer_name="onload", buffer_idx=cnt_onload % 2)
            cnt_onload += 1

        if vocab_in_pp and cnt_pp_first >= num_slices_preset + 2 * (pipeline_parallel_size - 1) and \
            (cnt_pp_first - 2 * (pipeline_parallel_size - 1)) % (num_stages * group_size) \
                // group_size + 1 == num_stages:
            dummy_input = dummy_inputs[0].pop()
            if len(dummy_inputs[0]) == 0:
                dummy_inputs.popleft()
            if first_stage:
                assert input_tensor_grad is not None
                encoder_input_grad = input_tensor_grad
                input_tensor_grad = None
            else:
                encoder_input_grad = None
            flag_vocab = True
        else:
            flag_vocab = False

        if num_slices_flight + 1 >= num_slices_target:
            parallel_state.set_virtual_pipeline_model_parallel_rank(recv_bwd_stage_idx)
            if (batch_bwd and batch_bwd.num_slices_to_backward > recv_bwd_offset) or batch_queue or batch_fwd:
                recv_next = not parallel_state.is_pipeline_last_stage()
            else:
                recv_next = False
            print_debug(DEBUG_P2PCOMM, "bwd sendrecv+" if recv_next else "bwd send+", value=input_tensor_grad)
            if pre_p2p_func: pre_p2p_func()
            output_tensor_grad, bwd_reqs = \
                p2p_communication.send_backward_recv_backward(input_tensor_grad,
                                                              recv_next,
                                                              tensor_shape,
                                                              dtype,
                                                              batch_p2p_comm,
                                                              overlap_p2p_comm,
                                                              timers=timers); input_tensor_grad = None
            if post_p2p_async_func: post_p2p_async_func(bwd_reqs, num_slices_flight + pipeline_parallel_rank)
            bwd_recv = bwd_reqs[-1] if recv_next else None
            output_tensor_grads.append((output_tensor_grad, bwd_recv)); output_tensor_grad, bwd_recv = None, None
            output_tensor_grad, bwd_recv = output_tensor_grads.popleft() if len(output_tensor_grads) > recv_bwd_offset else (None, None)

        if flag_vocab:
            print_debug(DEBUG_P2PCOMM, "bwd pre+" , value=cnt_pp_first)
            pre_process_backward(dummy_input, encoder_input_grad)
            print_debug(DEBUG_P2PCOMM, "bwd pre-" , value=cnt_pp_first)
            del dummy_input, encoder_input_grad
            del flag_vocab

    if vocab_in_pp:
        assert len(dummy_inputs[-1]) == 0
        dummy_inputs.pop()
        if pipeline_parallel_rank:
            assert len(dummy_inputs[0]) == pipeline_parallel_rank
        else:
            assert not dummy_inputs
        while dummy_inputs:
            dummy_input = dummy_inputs[0].pop()
            if len(dummy_inputs[0]) == 0:
                dummy_inputs.popleft()
            print_debug(DEBUG_P2PCOMM, "bwd pre+" , value=cnt_pp_first)
            pre_process_backward(dummy_input, None)
            print_debug(DEBUG_P2PCOMM, "bwd pre-" , value=cnt_pp_first)
    cuda_sync_and_record(sync_level=1)
    assert all(req.wait() for req in fwd_reqs); fwd_reqs = None
    print_debug(DEBUG_P2PCOMM, "fwd send-")
    if not forward_only:
        assert all(req.wait() for req in bwd_reqs); bwd_reqs = None
        print_debug(DEBUG_P2PCOMM, "bwd send-")
    return forward_data_store


def forward_backward_pipelining_with_interleaving(*,
                                                  forward_step_func,
                                                  get_batch_func,
                                                  data_iterator: Union[Iterator, List[Iterator]],
                                                  model: Union[torch.nn.Module, List[torch.nn.Module]],
                                                  num_microbatches: int,
                                                  micro_seq_length: int,
                                                  dtype: torch.dtype,
                                                  kv_cache_class: Type[Cache],
                                                  tensor_shape: Shape,
                                                  decoder_seq_length: Optional[int] = None,
                                                  grad_scaler: Callable = None,
                                                  sequence_parallel: bool = False,
                                                  overlap_p2p_comm: bool = False,
                                                  batch_p2p_comm: bool = True,
                                                  attn_balance: int = 0, # unused
                                                  vocab_in_pp: bool = False, # unused
                                                  forward_only: bool = False,
                                                  timers: Callable = None,
                                                  collect_non_loss_data: bool = False,
                                                  enable_autocast: bool = False,
                                                  deallocate_pipeline_outputs: bool = False,
                                                  no_sync_func: Optional[Callable] = None,
                                                  grad_sync_func: Optional[Callable] = None,
                                                  param_sync_func: Optional[Callable] = None,
                                                  pre_p2p_func: Optional[Callable] = None,
                                                  post_p2p_async_func: Optional[Callable] = None,
                                                  offload_ratio: float = 0,
                                                  offload_delay_to_next_stage: bool = False,
                                                  ):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    assert isinstance(model, list), \
        "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), \
        "invalid model chunking"
    assert isinstance(data_iterator, list), \
        "interleaved pipeline parallelism expected each model chunk to have a data iterator"
    assert overlap_p2p_comm, \
        "Slicing pipeline parallelism only supports overlapping p2p communication"
    assert not batch_p2p_comm, \
        "Slicing pipeline parallelism does not support using batched p2p communication"

    assert not isinstance(model, torchDDP), "torchDDP is no longer supported."
    model_type = get_model_type(model[0])
    assert model_type != ModelType.encoder_and_decoder, \
        "Interleaving is not supported with an encoder and decoder model."

    assert not no_sync_func, "Not implemented!"
    assert not grad_sync_func, "Not implemented!"
    assert not param_sync_func, "Not implemented!"

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    first_stage = parallel_state.is_pipeline_first_stage(ignore_virtual=True)
    last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=True)

    tensor_shape = get_actual_tensor_shape(tensor_shape, sequence_parallel, micro_seq_length)

    assert num_microbatches % pipeline_parallel_size == 0,"num_microbatches must be divisible by pipeline_parallel_size"

    model_type = get_model_type(model[0])

    num_stages = len(model)
    loss_div = num_microbatches
    num_batches_warmup = \
        (pipeline_parallel_size -
         pipeline_parallel_rank) * 2 - 1
    num_batches_warmup += (num_stages - 1) * pipeline_parallel_size
    num_batches_warmup = min(
        num_batches_warmup,
        num_microbatches * num_stages + 1)    # one extra fake forward to delay backward recv.
    # in-flight slices need to gain.
    num_batches_target = 4096 if forward_only else num_batches_warmup
    # current in-flight slices.
    num_batches_flight = 0
    forward_data_store = []

    # batches
    batch_queue: Deque[GroupedBatch] = deque()
    batch_fwd: Optional[GroupedBatch] = None
    batch_bwd: Optional[GroupedBatch] = None

    # recv tensors
    fwd_stage_idx = 0
    bwd_stage_idx = -1 % num_stages
    input_tensor_grad = None
    fwd_recv = None
    # offload reqs
    offload_req = None
    onload_req = None
    cnt_onload = 0

    def make_microbatch(model, data_iterator, batch_idx=None):
        forward_func = partial(forward_step, forward_step_func=forward_step_func, model=model,
                               timers=timers, enable_autocast=enable_autocast)
        backward_func = partial(backward_step, model_type=model_type,
                                timers=timers, deallocate_pipeline_outputs=deallocate_pipeline_outputs) \
                        if not forward_only else None
        if timers is not None:
            record_context = timers.is_recording_active()
            if record_context:
                timers.set_record_context(mb=batch_idx)
            timers('batch-generator', log_level=2).start()
        sliced_batch, loss_func = get_batch_func(data_iterator)
        if timers is not None:
            timers('batch-generator').stop()
            if record_context:
                timers.clear_record_context()
        if not parallel_state.is_pipeline_last_stage():
            loss_func = None
        slices = sliced_batch(micro_seq_length)
        assert len(slices) <= 1, "this function only supports one slice per microbatch."
        kv_cache = kv_cache_class()
        mb = MicroBatch(slices, kv_cache, offload_ratio, forward_func,
                        backward_func, loss_func, grad_scaler,
                        batch_idx=batch_idx,
                        timers=timers)
        return mb

    total_num_microbatches = num_microbatches
    def make_grouped_batch(num_microbatches):
        assert num_microbatches, "No more microbatches."
        mbatches = []
        for i in range(num_stages):
            for j in range(pipeline_parallel_size):
                parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                mbatches.append(make_microbatch(model[i], data_iterator[i],
                                                total_num_microbatches - num_microbatches + j))
        cbatch = GroupedBatch(mbatches, num_stages, pipeline_parallel_size)
        return num_microbatches - pipeline_parallel_size, cbatch

    # print messages for debug.
    DEBUG_P2PCOMM = False
    DEBUG_OFFLOAD = False
    cnt_fwd = 0
    cnt_bwd = 0
    def print_debug(flag, msg, value=None):
        if flag:
            stage = parallel_state.get_virtual_pipeline_model_parallel_rank()
            if isinstance(value, torch.Tensor):
                value = (value.dtype, value.shape, value.abs().mean().item())
            print(f"rank{pipeline_parallel_rank}, stage{stage}: {msg}: {value}", flush=True)

    # receive the first input_tensor.
    parallel_state.set_virtual_pipeline_model_parallel_rank(fwd_stage_idx)
    recv_prev = not parallel_state.is_pipeline_first_stage()
    print_debug(DEBUG_P2PCOMM, "fwd recv+" if recv_prev else "fwd noop+")
    if pre_p2p_func: pre_p2p_func()
    input_tensor = p2p_communication.recv_forward(tensor_shape, dtype, batch_p2p_comm, timers)
    if post_p2p_async_func: post_p2p_async_func([])
    fwd_recv = None

    while num_batches_flight or num_batches_target:
        cuda_sync_and_record(sync_level=1)
        """Forward"""
        if num_batches_flight < num_batches_target:   # do forward to gain in-flight micro-batches.
            if fwd_recv:
                fwd_recv.wait(); fwd_recv = None
            if not batch_fwd:
                if num_microbatches:
                    num_microbatches, batch_fwd = make_grouped_batch(num_microbatches)
                    assert num_microbatches >= 0, "number of microbatches is not enough."
                    batch_queue.append(batch_fwd)
                    print_debug(DEBUG_P2PCOMM, "make grad batch")
                else:
                    num_batches_target = 0
                    continue
            print_debug(DEBUG_P2PCOMM or DEBUG_OFFLOAD, "forward+", value=cnt_fwd)
            output_tensor = batch_fwd.forward(input_tensor); input_tensor = None
            fwd_stage_idx = batch_fwd.forward_stage_idx()
            print_debug(DEBUG_P2PCOMM or DEBUG_OFFLOAD, "forward-", value=cnt_fwd); cnt_fwd += 1
            if batch_fwd.forward_stage_idx(-1) + 1 == num_stages: # at last chunk of model
                forward_data_store.append(batch_fwd.compute_loss(loss_div))
            if batch_fwd.num_batches_to_forward == 0:
                batch_fwd = None
            num_batches_flight += 1

            parallel_state.set_virtual_pipeline_model_parallel_rank(fwd_stage_idx)
            if (batch_fwd) or num_microbatches:
                recv_prev = not parallel_state.is_pipeline_first_stage()
            else:
                recv_prev = False
            print_debug(DEBUG_P2PCOMM, "fwd sendrecv+" if recv_prev else "fwd send+", value=output_tensor)
            if pre_p2p_func: pre_p2p_func()
            input_tensor, fwd_reqs = \
                p2p_communication.send_forward_recv_forward(output_tensor,
                                                            recv_prev,
                                                            tensor_shape,
                                                            dtype,
                                                            batch_p2p_comm,
                                                            overlap_p2p_comm,
                                                            timers); output_tensor = None
            if post_p2p_async_func: post_p2p_async_func(fwd_reqs)
            fwd_recv = fwd_reqs[-1] if recv_prev else None

        if forward_only:
            num_batches_flight = 0
            continue

        if offload_req:
            offload_req.wait(); offload_req = None
            print_debug(DEBUG_OFFLOAD, "offload-")

        """Offload"""
        prior_works = [torch.cuda.current_stream().record_event()]  # offload/onload waits for forward.
        if num_batches_target: # before the cooldown
            prior_works += fwd_reqs
            curr_batch = batch_fwd or (batch_queue[-1] if batch_queue else batch_bwd)
            curr_om = curr_batch.prev_mbatch().curr_om()
            if pipeline_parallel_size - pipeline_parallel_rank <= 2 and \
               curr_batch.forward_stage_idx(-1) + 1 == num_stages:
                print_debug(DEBUG_OFFLOAD, "offload noop")
                curr_om.reset()
            else:
                # there are enough forward stages to overlap with offload and onload.
                print_debug(DEBUG_OFFLOAD, "offload+")
                offload_req = curr_om.offload(prior_works=prior_works)
        else: # during the cooldown, no `fwd_reqs` are created.
            prior_works += bwd_reqs

        """Backward"""
        if num_batches_flight >= num_batches_target:  # do backward to consume in-flight micro-batches.
            if bwd_recv:
                bwd_recv.wait(); bwd_recv = None
            batch_bwd = batch_bwd or batch_queue.popleft()
            print_debug(DEBUG_P2PCOMM or DEBUG_OFFLOAD, "backward+", value=cnt_bwd)
            if onload_req:
                onload_req.wait(); onload_req = None
                print_debug(DEBUG_OFFLOAD, "onload-")
            input_tensor_grad = batch_bwd.backward(output_tensor_grad); output_tensor_grad = None
            bwd_stage_idx = batch_bwd.backward_stage_idx()
            print_debug(DEBUG_P2PCOMM or DEBUG_OFFLOAD, "backward-", value=cnt_bwd); cnt_bwd += 1
            if not batch_bwd.num_batches_to_backward:
                batch_bwd = None
            num_batches_flight -= 1

        if num_batches_flight + 1 >= num_batches_target:
            parallel_state.set_virtual_pipeline_model_parallel_rank(bwd_stage_idx)
            if batch_bwd or batch_queue:
                recv_next = not parallel_state.is_pipeline_last_stage()
            else:
                recv_next = False
            print_debug(DEBUG_P2PCOMM, "bwd sendrecv+" if recv_next else "bwd send+", value=input_tensor_grad)
            if pre_p2p_func: pre_p2p_func()
            output_tensor_grad, bwd_reqs = \
                p2p_communication.send_backward_recv_backward(input_tensor_grad,
                                                              recv_next,
                                                              tensor_shape,
                                                              dtype,
                                                              batch_p2p_comm,
                                                              overlap_p2p_comm,
                                                              timers=timers); input_tensor_grad = None
            if post_p2p_async_func:
                post_p2p_async_func(bwd_reqs, num_batches_flight + pipeline_parallel_rank)
            bwd_recv = bwd_reqs[-1] if recv_next else None

        """Onload"""
        # NOTE(lizhouyang): `onload` should start after the forward, **NOT** the backward.
        # `prior_works` are usually `fwd_reqs` except for cooldown.
        if num_batches_flight and num_batches_flight + 1 >= num_batches_target:  # after the second last round of warmup.
            mbatch = (batch_bwd or batch_queue[0]).curr_bwd_mbatch()
            if mbatch.num_slices_to_backward:
                print_debug(DEBUG_OFFLOAD, "onload+", cnt_onload)
                onload_req = mbatch.curr_om().onload(prior_works=prior_works, buffer_name="onload", buffer_idx=cnt_onload % 2)
                cnt_onload += 1
            else:
                print_debug(DEBUG_OFFLOAD, "noop")

    cuda_sync_and_record(sync_level=1)
    assert all(req.wait() for req in fwd_reqs); fwd_reqs = None
    print_debug(DEBUG_P2PCOMM, "fwd send-")
    if not forward_only:
        assert all(req.wait() for req in bwd_reqs); bwd_reqs = None
        print_debug(DEBUG_P2PCOMM, "bwd send-")
    assert not batch_fwd
    assert not batch_bwd
    assert not batch_queue
    assert cnt_fwd == cnt_bwd, (cnt_fwd, cnt_bwd)
    return forward_data_store

def get_tensor_shapes(*,
                      rank: int,
                      model_type: ModelType,
                      tensor_shape: Shape,
                      decoder_seq_length: int,
                      sequence_parallel: bool):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    tensor_shapes = []

    assert (
        len(tensor_shape) == 3
    ), f"`tensor_shape` should be [sequence_length, micro_batch_size, hidden_size] but {tensor_shape}"

    seq_length, micro_batch_size, hidden_size = tensor_shape

    if sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()

    if model_type == ModelType.encoder_and_decoder:
        if sequence_parallel:
            decoder_seq_length = decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()

        if parallel_state.is_pipeline_stage_before_split(rank):
            tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
    else:
        tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
    return tensor_shapes



def forward_backward_pipelining_without_interleaving(*,
                                                     forward_step_func,
                                                     get_batch_func,
                                                     data_iterator: Union[Iterator, List[Iterator]],
                                                     model: Union[torch.nn.Module, List[torch.nn.Module]],
                                                     num_microbatches: int,
                                                     micro_seq_length: int,
                                                     kv_cache_class: Type[Cache],
                                                     dtype: Optional[torch.dtype] = None,
                                                     tensor_shape: Optional[Shape] = None, # unused
                                                     decoder_seq_length: Optional[int] = None, # unused
                                                     grad_scaler: Callable = None,
                                                     sequence_parallel: bool = False, # unused
                                                     overlap_p2p_comm: bool = False, # unused
                                                     batch_p2p_comm: bool = True, # unused
                                                     attn_balance: int = 0, # unused
                                                     vocab_in_pp: bool = False, # unused
                                                     forward_only: bool = False,
                                                     timers: Callable = None,
                                                     collect_non_loss_data: bool = False,
                                                     enable_autocast: bool = False,
                                                     deallocate_pipeline_outputs: bool = False,
                                                     no_sync_func: Optional[Callable] = None,
                                                     grad_sync_func: Optional[Callable] = None, # unused
                                                     param_sync_func: Optional[Callable] = None, # unused
                                                     pre_p2p_func: Optional[Callable] = None, # unused
                                                     post_p2p_async_func: Optional[Callable] = None, # unused
                                                     offload_ratio: float = 0,
                                                     offload_delay_to_next_stage: bool = False, # unused
                                                     ):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    if isinstance(model, list):
        assert len(model) == 1, \
            "non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, \
            "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    if overlap_p2p_comm:
        raise ValueError("Non-interleaved pipeline parallelism does not support overlapping p2p communication")

    if not batch_p2p_comm:
        raise ValueError("Non-interleaved pipeline parallelism only supports using batched p2p communication")

    if offload_delay_to_next_stage:
        raise ValueError("Non-interleaved pipeline parallelism does not support offload_delay_to_next_stage")

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    first_stage = parallel_state.is_pipeline_first_stage()
    last_stage = parallel_state.is_pipeline_last_stage()
    tensor_shape = get_actual_tensor_shape(tensor_shape, sequence_parallel, micro_seq_length)

    model_type = get_model_type(model)

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []

    forward_func = partial(forward_step, forward_step_func=forward_step_func, model=model,
                           timers=timers, enable_autocast=enable_autocast)
    backward_func = None if forward_only else partial(backward_step, model_type=model_type,
                            timers=timers, deallocate_pipeline_outputs=deallocate_pipeline_outputs)

    loss_div = num_microbatches
    num_batches_preset = 0 # this method does not need preset batches
    num_batches_warmup = \
        (parallel_state.get_pipeline_model_parallel_world_size() -
         parallel_state.get_pipeline_model_parallel_rank())
    num_batches_warmup = min(
        num_batches_warmup,
        num_microbatches)
    # in-flight batches need to gain.
    num_batches_target = 4096 if forward_only else num_batches_preset + num_batches_warmup
    # current in-flight batches.
    num_batches_flight = 0
    mb_queue = deque()
    mb_fwd: Optional[MicroBatch] = None
    mb_bwd: Optional[MicroBatch] = None
    no_warmup = False
    forward_data_store = []
    offload_req = None
    onload_req = None

    def make_microbatch(num_mb):
        assert num_mb
        if timers is not None:
            record_context = timers.is_recording_active()
            if record_context:
                timers.set_record_context(mb=num_microbatches - num_mb)
            timers('batch-generator', log_level=2).start()
        sliced_batch, loss_func = get_batch_func(data_iterator)
        if timers is not None:
            timers('batch-generator').stop()
            if record_context:
                timers.clear_record_context()
        if not last_stage:
            loss_func = None
        slices = sliced_batch(micro_seq_length)
        num_total_batches = num_batches_flight + num_mb * len(slices) + 1
        assert  num_total_batches >= num_batches_target, "number of total slices is not enough."
        kv_cache = kv_cache_class()
        mb = MicroBatch(slices, kv_cache, offload_ratio, forward_func,
                        backward_func, loss_func, grad_scaler,
                        batch_idx=num_microbatches - num_mb,
                        timers=timers)
        return num_mb - 1, mb

    # print messages for debug.
    DEBUG_P2PCOMM = False
    DEBUG_OFFLOAD = False
    cnt_fwd = 0
    cnt_bwd = 0
    def print_debug(flag, msg, value=None):
        if flag:
            stage = parallel_state.get_virtual_pipeline_model_parallel_rank()
            if isinstance(value, torch.Tensor):
                value = (value.dtype, value.shape, value.abs().mean().item())
            print(f"rank{pipeline_parallel_rank}: {msg}: {value}", flush=True)

    while num_batches_flight or num_batches_target:
        cuda_sync_and_record(sync_level=1)
        """Forward"""
        if num_batches_flight < num_batches_target:   # do forward to gain in-flight micro-batches.
            if not mb_fwd:
                if num_microbatches:
                    print_debug(DEBUG_P2PCOMM, "num_microbatches", num_microbatches)
                    num_microbatches, mb_fwd = make_microbatch(num_microbatches)
                    num_batches_target = max(num_batches_target, num_batches_warmup)
                else:
                    print_debug(DEBUG_P2PCOMM, "over", num_batches_flight)
                    num_batches_target = 0
                    continue
            if num_batches_flight + 1 != num_batches_target or len(input_tensors) == 0:
                input_tensor = p2p_communication.recv_forward(tensor_shape, dtype, timers=timers)
            else:
                input_tensor = input_tensors.pop(0)
            output_tensor = mb_fwd.forward(input_tensor); input_tensor = None
            num_batches_flight += 1
            loss = mb_fwd.compute_loss(loss_div)
            forward_data_store.append(loss)
            mb_queue.append(mb_fwd); mb_fwd = None
            if last_stage:
                output_tensor = None
            if num_batches_flight == num_batches_target:
                output_tensor_grad = \
                    p2p_communication.send_forward_recv_backward(output_tensor,
                                            tensor_shape, dtype,
                                            timers=timers)
                output_tensors.append(output_tensor_grad); output_tensor_grad = None
            else:
                p2p_communication.send_forward(output_tensor, timers=timers)
            deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)
            output_tensor = None

        if forward_only:
            num_batches_flight = 0
            continue

        """Backward"""
        if num_batches_flight >= num_batches_target:  # do backward to consume in-flight micro-batches.
            print_debug(DEBUG_P2PCOMM, "num_batches_flight", num_batches_flight)
            print_debug(DEBUG_P2PCOMM, "num_batches_target", num_batches_target)
            mb_bwd = mb_bwd or mb_queue.popleft()
            bwd = False
            if num_batches_flight != num_batches_target or len(output_tensors) == 0:
                output_tensor_grad = p2p_communication.recv_backward(tensor_shape, dtype, timers=timers)
            else:
                output_tensor_grad = output_tensors.pop(0)
                bwd = True
            input_tensor_grad = mb_bwd.backward(output_tensor_grad); output_tensor_grad = None

            num_batches_flight -= 1
            if not mb_bwd.num_slices_to_backward:
                mb_bwd = None
            if first_stage:
                input_tensor_grad = None
            if bwd and num_microbatches:
                print_debug(DEBUG_P2PCOMM, "input_tensor_grad", input_tensor_grad)
                input_tensor = \
                    p2p_communication.send_backward_recv_forward(
                        input_tensor_grad, tensor_shape, dtype, timers=timers)
                input_tensors.append(input_tensor); input_tensor = None
            else:
                p2p_communication.send_backward(input_tensor_grad, timers=timers)
            input_tensor_grad = None
    return forward_data_store
