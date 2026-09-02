# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Pretrain utilities."""

from datetime import datetime
import gc
import math
import sys
import os
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_signal_handler
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_current_global_batch_size
from megatron import get_num_microbatches
from megatron import is_last_rank
from megatron import update_num_microbatches
from megatron.core import mpu, parallel_state, tensor_parallel
from megatron import print_rank_0
from megatron import print_rank_last
from megatron import reference
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.model import Float16Module
from megatron.model import GPTModel
from megatron.core.enums import ModelType
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.initialize import set_jit_fusion_options
from megatron.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.optimizer_param_scheduler import CustomOptimizerParamScheduler
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import unwrap_model
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.utils import calc_params_l2_norm
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.kv_cache import FakeCache, KVCache, Growth
from megatron.utils import report_memory
from megatron.profile_utils import annotate_range, enable_perf_model, perf_model_summary, perf_model_clear
from megatron.model.vision.knn_monitor import compute_feature_bank
from functools import partial


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             get_batch_func=None,
             no_wd_decay_cond=None,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME],
                                     dtype=torch.double,
                                     device='cuda')
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type, no_wd_decay_cond=no_wd_decay_cond)
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')

    # Data stuff.
    timers('train/valid/test-data-iterators-setup', log_level=0).start(
        barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
            for _ in range(len(model))
        ]
        train_data_iterator = [data_iterators[0]
                               for data_iterators in all_data_iterators]
        valid_data_iterator = [data_iterators[1]
                               for data_iterators in all_data_iterators]
        test_data_iterator = [data_iterators[2]
                              for data_iterators in all_data_iterators]
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')

    if args.train_golden_model:
        reference.adapter.setup(args, model, get_tensorboard_writer())

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup',
                'train/valid/test-data-iterators-setup'], barrier=True)
    print_rank_0('training ...')

    iteration = 0

    if args.dataloader_type == 'cyclic' and args.retro_add_retriever:
        args.train_iters = args.retro_cyclic_train_iters
        print_rank_0("retro cyclic train iters : %d" % args.train_iters)

    if args.do_train and args.train_iters > 0:
        iteration = train(forward_step_func, get_batch_func,
                          model, optimizer, opt_param_scheduler,
                          train_data_iterator, valid_data_iterator,
                          process_non_loss_data_func)
    print_datetime('after training is done')

    if args.do_valid:
        prefix = 'the end of training for val data'
        evaluate_and_print_results(prefix, forward_step_func, get_batch_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func,
                                   False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, opt_param_scheduler)

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, forward_step_func, get_batch_func,
                                   test_data_iterator, model,
                                   0, process_non_loss_data_func,
                                   True)

def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:
        assert model_type != ModelType.encoder_and_decoder, \
            "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert args.pipeline_model_parallel_split_rank is not None, \
                    "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (
                        rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder)
        else:
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # The two ends of a V-shaped pipeline live in different virtual chunks on
    # rank 0, so initialize their tied embedding copies locally.
    is_v_pipeline = getattr(args, 'dspp', False) or \
        args.variable_seq_schedule == 'slice-v'
    if is_v_pipeline and len(model) == 2 and \
            mpu.get_pipeline_model_parallel_rank() == 0:
        model[-1].word_embeddings_weight().data.copy_(
            model[0].word_embeddings_weight().data
        )

    # Disallow training and inference with Transformer Engine
    # for non-GPT models
    args.allow_transformer_engine = all([type(m) == GPTModel for m in model])
    assert args.allow_transformer_engine or args.transformer_impl == 'local', \
        'Transformer Engine is only approved for GPT models'

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if wrap_with_ddp:
        if args.DDP_impl == 'torch':
            i = torch.cuda.current_device()
            model = [torchDDP(model_module, device_ids=[i], output_device=i,
                              process_group=mpu.get_data_parallel_group())
                     for model_module in model]

        elif args.DDP_impl == 'local':
            model = [LocalDDP(model_module,
                              args.accumulate_allreduce_grads_in_fp32,
                              args.use_contiguous_buffers_in_local_ddp,
                              args.make_main_grad_addresss_divisible_by)
                     for model_module in model]
            # broad cast params from data parallel src rank to other data parallel ranks
            if args.data_parallel_random_init:
                for model_module in model:
                    model_module.broadcast_params()
        else:
            raise NotImplementedError('Unknown DDP implementation specified: '
                                      '{}. Exiting.'.format(args.DDP_impl))

    return model


def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size
        wd_incr_steps = args.train_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        lr_decay_steps = args.lr_decay_samples
        wd_incr_steps = args.train_samples
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        warmup_init_lr=args.warmup_init_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler)

    return opt_param_scheduler


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func, model_type)
    unwrapped_model = unwrap_model(model,
                                   (torchDDP, LocalDDP, Float16Module))

    optimizer = get_megatron_optimizer(model, no_wd_decay_cond,
                                       scale_lr_cond, lr_mult)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    if args.load is not None:
        timers = get_timers()
        timers('load-checkpoint', log_level=0).start(barrier=True)
        args.iteration = load_checkpoint(model, optimizer, opt_param_scheduler)
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0

    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    # get model without FP16 and/or TorchDDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, opt_param_scheduler


_OPTIMIZER_OVERLAP_STREAM = None


def get_optimizer_overlap_stream():
    global _OPTIMIZER_OVERLAP_STREAM
    if _OPTIMIZER_OVERLAP_STREAM is None:
        _OPTIMIZER_OVERLAP_STREAM = torch.cuda.Stream()
    return _OPTIMIZER_OVERLAP_STREAM


_OPTIMIZER_EXPERT_OVERLAP_STREAM = None


def get_optimizer_expert_overlap_stream():
    global _OPTIMIZER_EXPERT_OVERLAP_STREAM
    if _OPTIMIZER_EXPERT_OVERLAP_STREAM is None:
        _OPTIMIZER_EXPERT_OVERLAP_STREAM = torch.cuda.Stream()
    return _OPTIMIZER_EXPERT_OVERLAP_STREAM


def make_optimizer_overlap_func(models, optimizer, args, timers, skipped_last_iter, overlap_gather, overlap_reduce, gather_ratio, reduce_ratio):
    # 1. Partition optimizer jobs
    gather_fast_jobs, gather_slow_jobs, reduce_slow_jobs, reduce_fast_jobs = \
        optimizer.partition_into_sub_divisions(overlap_gather, overlap_reduce, gather_ratio, reduce_ratio)

    # 3. Schedule all jobs.
    opt_stream = get_optimizer_overlap_stream()
    opt_expert_stream = get_optimizer_expert_overlap_stream()
    opt_event = torch.cuda.Event()
    gather_event_pairs = list()
    reduce_event_pairs = list()

    def opt_prepare_func():
        if overlap_gather and skipped_last_iter:
            for model in models:
                model.zero_grad_buffer()
        if overlap_gather and not skipped_last_iter:
            for job in gather_fast_jobs:
                optimizer.gather_sub_division(args, timers, *job)
                optimizer.post_gather_sub_division(args, timers, *job)
            for _ in range(mpu.get_pipeline_model_parallel_rank()):
                if gather_slow_jobs:
                    job = gather_slow_jobs.pop(0)
                    optimizer.gather_sub_division(args, timers, *job)
                    optimizer.post_gather_sub_division(args, timers, *job)

    def pre_p2p_func():
        for event_pairs in [gather_event_pairs, reduce_event_pairs]:
            if event_pairs and event_pairs[-1][3] is None:
                t4 = torch.cuda.Event(enable_timing=True)
                t4.record()
                event_pairs[-1][3] = t4
        opt_event.wait()

    post_gather_slow_jobs = []

    def post_p2p_async_func(p2p_wait_handles, num_backward=0):
        while post_gather_slow_jobs:
            job = post_gather_slow_jobs.pop(0)
            optimizer.post_gather_sub_division(args, timers, *job)
        if overlap_gather and not skipped_last_iter and gather_slow_jobs:
            job = gather_slow_jobs.pop(0)
            tune = args.kaimm_overlap_tune_ratio and optimizer.is_largest_slow_gather(*job)
            opt_event.record()
            with torch.cuda.stream(opt_stream):
                opt_event.wait()
                if p2p_wait_handles:
                    for req in p2p_wait_handles:
                        req.wait()
                    if not args.kaimm_overlap_optimizer_no_barrier:
                        x = torch.empty(1, device="cuda")
                        torch.distributed.all_reduce(x, group=mpu.get_network_barrier_group())
                if tune:
                    t1 = torch.cuda.Event(enable_timing=True)
                    t1.record()
            if isinstance(job[0], list):
                optimizer.gather_sub_division(args, timers, *job, stream=opt_stream, ep_stream=opt_expert_stream)
            else:
                with torch.cuda.stream(opt_stream):
                    optimizer.gather_sub_division(args, timers, *job)
            with torch.cuda.stream(opt_stream):
                if tune:
                    t2 = torch.cuda.Event(enable_timing=True)
                    t2.record()
                opt_event.record()
            if tune:
                t3 = torch.cuda.Event(enable_timing=True)
                if p2p_wait_handles:
                    # event starts after p2p
                    for req in p2p_wait_handles:
                        req.wait()
                t3.record()
                gather_event_pairs.append([t1, t2, t3, None])
            post_gather_slow_jobs.append(job)
        if len(reduce_slow_jobs) >= num_backward > 0:
            job = reduce_slow_jobs.pop(0)
            tune = args.kaimm_overlap_tune_ratio and optimizer.is_largest_slow_reduce(*job)
            optimizer.pre_reduce_sub_division(args, timers, *job)
            opt_event.record()
            with torch.cuda.stream(opt_stream):
                opt_event.wait()
                if p2p_wait_handles:
                    for req in p2p_wait_handles:
                        req.wait()
                    if not args.kaimm_overlap_optimizer_no_barrier:
                        x = torch.empty(1, device="cuda")
                        torch.distributed.all_reduce(x, group=mpu.get_network_barrier_group())
                if tune:
                    t1 = torch.cuda.Event(enable_timing=True)
                    t1.record()
            if isinstance(job[0], list):
                optimizer.reduce_sub_division(args, timers, *job, stream=opt_stream, ep_stream=opt_expert_stream)
            else:
                with torch.cuda.stream(opt_stream):
                    optimizer.reduce_sub_division(args, timers, *job)
            with torch.cuda.stream(opt_stream):
                if tune:
                    t2 = torch.cuda.Event(enable_timing=True)
                    t2.record()
                opt_event.record()
            if tune:
                t3 = torch.cuda.Event(enable_timing=True)
                if p2p_wait_handles:
                    # event starts after p2p
                    for req in p2p_wait_handles:
                        req.wait()
                t3.record()
                reduce_event_pairs.append([t1, t2, t3, None])

    def opt_remained_func():
        opt_event.wait()
        for job in reduce_slow_jobs:
            optimizer.pre_reduce_sub_division(args, timers, *job)
            optimizer.reduce_sub_division(args, timers, *job)
        for job in reduce_fast_jobs:
            optimizer.pre_reduce_sub_division(args, timers, *job)
            optimizer.reduce_sub_division(args, timers, *job)

    def get_overlap_time_pairs():
        gather_time_pairs = [
            (max(t1.elapsed_time(t2), t3.elapsed_time(t2)), t3.elapsed_time(t4))
            for t1, t2, t3, t4 in gather_event_pairs
            if t4 is not None
        ]
        reduce_time_pairs = [
            (max(t1.elapsed_time(t2), t3.elapsed_time(t2)), t3.elapsed_time(t4))
            for t1, t2, t3, t4 in reduce_event_pairs
            if t4 is not None
        ]
        return gather_time_pairs, reduce_time_pairs

    return opt_prepare_func, pre_p2p_func, post_p2p_async_func, opt_remained_func, get_overlap_time_pairs


def optimizer_overlap_tuning_suggest_ratio(current_ratio, target_time_ratio, time_pairs_all, PP):
    time_pairs_all_nz = time_pairs_all[(time_pairs_all[:, 0] > 0).nonzero().flatten()]
    if len(time_pairs_all_nz) == 0:
        return None, None, None
    p90_comm_time = torch.quantile(time_pairs_all_nz[:, 0].sort().values, .90).item()
    min_calc_time = time_pairs_all_nz[:, 1].min().item()
    capacity_ratio = target_time_ratio / (p90_comm_time / min_calc_time) * current_ratio
    suggested_ratio = capacity_ratio
    if capacity_ratio * PP > 1:
        comm_times = math.ceil(1 / capacity_ratio)
        suggested_ratio = 1 / comm_times
    msg = f"current_ratio {current_ratio:.6f} " + \
        f"p90_comm_time {p90_comm_time:.3f} " + \
        f"min_calc_time {min_calc_time:.3f} " + \
        f"target_time_ratio {target_time_ratio:.6f} " + \
        f"capacity_ratio {capacity_ratio:.6f} " + \
        f"suggested_ratio {math.ceil(suggested_ratio * 10 ** 6) / 10 ** 6:.6f}"
    return capacity_ratio, suggested_ratio, msg


def get_kv_cache_class(args):
    dtype = args.params_dtype
    kv_length = args.micro_seq_length // parallel_state.get_context_parallel_world_size()
    num_heads = args.num_query_groups if args.group_query_attention else args.num_attention_heads
    num_heads //= mpu.get_tensor_model_parallel_world_size()
    shape = (kv_length, args.micro_batch_size, num_heads, args.kv_channels)
    if args.micro_seq_length:
        if args.kaimm_kv_cache_impl == 'chunked':
            growth = None   # chunked cache, no growth.
        elif args.kaimm_kv_cache_impl == 'extended':
            growth = Growth(growth_rate=2)    # geometry growth by 2.
            # growth = Growth(growth_seq_length=args.pipeline_model_parallel_size * args.micro_seq_length)  # linear growth by p.
        return partial(KVCache, dtype, shape, growth)
    else:
        return FakeCache


def train_step(forward_step_func, get_batch_func, data_iterator,
               model, optimizer, opt_param_scheduler, skipped_last_iter):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Register backward hooks
    overlap_optimizer_communication = args.kaimm_overlap_optimizer_communication and mpu.get_data_parallel_world_size() >= 2
    if overlap_optimizer_communication:
        if args.DDP_impl != "local" or not args.use_contiguous_buffers_in_local_ddp:
            raise ValueError("Can not use kaimm_overlap_optimizer_communication without use_contiguous_buffers_in_local_ddp")
        if not args.use_distributed_optimizer:
            raise ValueError("Can not use kaimm_overlap_optimizer_communication without use_distributed_optimizer")
        if args.virtual_pipeline_model_parallel_size is None:
            raise ValueError("Can not use kaimm_overlap_optimizer_communication without PP interleaving")
        if not args.untie_embeddings_and_output_weights and not args.kaimm_vocab_in_pipeline_parallel:
            raise ValueError("Can not use kaimm_overlap_optimizer_communication without untie_embeddings_and_output_weights")
        gather_ratio = args.kaimm_overlap_gather_ratio
        reduce_ratio = args.kaimm_overlap_reduce_ratio
        overlap_gather = bool(gather_ratio)
        overlap_reduce = bool(reduce_ratio)
        if not overlap_gather and not overlap_reduce:
            raise ValueError("both gather ratio and reduce ratio are zero")
        opt_prepare_func, pre_p2p_func, post_p2p_async_func, opt_remained_func, get_overlap_time_pairs = \
            make_optimizer_overlap_func(model, optimizer, args, timers, skipped_last_iter, overlap_gather, overlap_reduce, gather_ratio, reduce_ratio)
    else:
        pre_p2p_func = None
        post_p2p_async_func = None

    # Set grad to zero.
    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
        if not (overlap_optimizer_communication and overlap_gather):
            for partition in model:
                partition.zero_grad_buffer()
    optimizer.zero_grad()

    if overlap_optimizer_communication:
        opt_prepare_func()

    # Forward pass.
    timers('forward-backward', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    forward_backward_func = get_forward_backward_func(
        args.micro_seq_length,
        variable_slicing=args.variable_seq_slicing,
        dspp=args.dspp)
    fwd_bwd_timers = timers if args.timing_log_level > 1 else None
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        get_batch_func=get_batch_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        micro_seq_length=args.micro_seq_length,
        kv_cache_class=get_kv_cache_class(args),
        dtype=args.params_dtype,
        tensor_shape=(args.seq_length, args.micro_batch_size, args.hidden_size),
        grad_scaler=optimizer.scale_loss,
        sequence_parallel=args.sequence_parallel,
        overlap_p2p_comm=args.overlap_p2p_comm,
        batch_p2p_comm=not args.overlap_p2p_comm,
        attn_balance=args.kaimm_pipeline_attn_balance,
        vocab_in_pp=args.kaimm_vocab_in_pipeline_parallel,
        forward_only=False,
        timers=fwd_bwd_timers,
        pre_p2p_func=pre_p2p_func,
        post_p2p_async_func=post_p2p_async_func,
        offload_ratio=args.kaimm_offload_activation_ratio,
        offload_delay_to_next_stage=args.kaimm_offload_delay_to_next_stage)
    timers('forward-backward').stop()

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Reduce gradients.
    if overlap_optimizer_communication and overlap_reduce:
        opt_remained_func()
    else:
        optimizer.reduce_model_grads(args, timers)

    if args.train_golden_model:
        reference.adapter.step()

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0],
                                       (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
    timers('optimizer').stop()

    # Gather params.
    if update_successful and not (overlap_optimizer_communication and overlap_gather):
        optimizer.gather_model_params(args, timers)

    # Vision momentum.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0],
                                       (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size // \
                    args.context_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    if overlap_optimizer_communication and args.kaimm_overlap_tune_ratio:
        TP = mpu.get_tensor_model_parallel_world_size()
        PP = mpu.get_pipeline_model_parallel_world_size()
        DP = mpu.get_data_parallel_world_size()
        torch.cuda.synchronize()
        gather_time_pairs, reduce_time_pairs = get_overlap_time_pairs()
        len_padded_time_pairs = PP * len(model)
        for time_pairs, ratio_option_name in [
                (gather_time_pairs, "kaimm_overlap_gather_ratio"),
                (reduce_time_pairs, "kaimm_overlap_reduce_ratio"),]:
            time_pairs_cuda = torch.nn.functional.pad(torch.tensor(time_pairs, device="cuda").view(-1, 2),
                                                      [0, 0, 0, len_padded_time_pairs - len(time_pairs)])
            time_pairs_all = torch.empty((PP, DP, TP, *time_pairs_cuda.shape),
                                         dtype=time_pairs_cuda.dtype, device=time_pairs_cuda.device)
            torch.distributed.all_gather_into_tensor(time_pairs_all, time_pairs_cuda)
            next_ratio = 0.
            if torch.distributed.get_rank() == 0:
                current_ratio = getattr(args, ratio_option_name)
                target_time_ratio = args.kaimm_overlap_tuning_target_time_ratio
                time_pairs_all = time_pairs_all.reshape(-1, time_pairs_all.shape[-1])
                capacity_ratio, suggested_ratio, msg = \
                    optimizer_overlap_tuning_suggest_ratio(current_ratio, target_time_ratio, time_pairs_all, PP)
                if capacity_ratio is not None:
                    print(f"{ratio_option_name} {msg}")
                    next_ratio = min(1., current_ratio * .8 + capacity_ratio * .2)
            next_ratio_cuda = torch.tensor(next_ratio, device="cuda")
            torch.distributed.broadcast(next_ratio_cuda, src=0)
            next_ratio = next_ratio_cuda.item()
            if next_ratio > 0.:
                setattr(args, ratio_option_name, next_ratio)

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    is_loss_stage = (
        mpu.get_pipeline_model_parallel_rank() == 0
        if args.dspp or args.variable_seq_schedule == 'slice-v'
        else mpu.is_pipeline_last_stage(ignore_virtual=True)
    )
    if is_loss_stage:
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'grads-all-reduce',
        'grads-reduce-scatter',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * \
        (args.data_parallel_size // args.context_parallel_size) * \
        get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size', batch_size, iteration)
            writer.add_scalar('batch-size vs samples', batch_size,
                              args.consumed_train_samples)
        for key in loss_dict:
            writer.add_scalar(key , loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size', args.world_size, iteration)
            writer.add_scalar('world-size vs samples', args.world_size,
                              args.consumed_train_samples)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm,
                              args.consumed_train_samples)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )
    if args.num_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        moe_loss_dict = {}
        track_moe_metrics(moe_loss_scale, args.num_layers, args.moe_layer_interval, args.moe_first, iteration, writer, wandb_writer, moe_loss_dict, args.moe_per_layer_logging)
        loss_dict.update(moe_loss_dict)
        for key, value in moe_loss_dict.items():
            if key in total_loss_dict:
                total_loss_dict[key] += value
            else:
                total_loss_dict[key] = value

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations
        if writer:
            if args.log_timers_to_tensorboard:
                writer.add_scalar('iteration-time',
                                  elapsed_time_per_iteration, iteration)

        samples_per_second = args.global_batch_size / elapsed_time_per_iteration
        tokens_per_sec_per_replica = samples_per_second * args.seq_length  / args.world_size

        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += ' samples per second: {:.3f} |'.format(samples_per_second)
        log_string += ' tokens per sec per replica: {:.5f} |'.format(tokens_per_sec_per_replica)
        log_string += ' learning rate: {:.8E} |'.format(learning_rate)
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        if args.dspp:
            print_rank_0(log_string)
        else:
            print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag -= 1
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler):
    args = get_args()
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    timers('save-checkpoint', log_level=0).start(barrier=True)
    # Copy optimizer params to model params in need when `overlap_gather`
    optimizer.gather_model_params(args, timers)
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler, train_data_iterator)
    timers('save-checkpoint').stop(barrier=True)
    timers.log(['save-checkpoint'])
    if args.kaimm_gc_interval > 0:
        gc.collect()


def train(forward_step_func, get_batch_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # The flag to mark whether update_successful or not in the last iter.
    # If true, skip all_gather before forward.
    # Only used in --kaimm-overlap-optimizer-communication.
    skipped_iter = True

    # Manual GC
    if args.kaimm_gc_interval > 0:
        gc.disable()
        gc.collect()

    # Iterations.
    iteration = args.iteration

    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('before the start of training step')
    report_memory_flag = 2
    while iteration < args.train_iters:
        update_num_microbatches(args.consumed_train_samples)
        args.curr_iteration = iteration
        timers.set_recording_iteration(iteration)
        if args.kaimm_gc_interval > 0:
            if iteration % args.kaimm_gc_interval == 0:
                gc.collect(0)
        if args.kaimm_profile_analysis:
            enable_perf_model()
        with annotate_range(f"train_step rank_{torch.distributed.get_rank()} it_{iteration}"):
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                train_step(forward_step_func,
                           get_batch_func,
                           train_data_iterator,
                           model,
                           optimizer,
                           opt_param_scheduler,
                           skipped_iter)
        iteration += 1
        args.consumed_train_samples += mpu.get_data_parallel_for_sample_world_size() * \
                                       args.micro_batch_size * \
                                       get_num_microbatches()

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad)
        timers.record()

        if args.kaimm_profile_analysis:
            perf_model_summary()
            perf_model_clear()

        # Autoresume
        if args.adlr_autoresume and \
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              opt_param_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
           args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, forward_step_func, get_batch_func,
                                       valid_data_iterator, model,
                                       iteration, process_non_loss_data_func,
                                       False)

        # Checkpointing
        saved_checkpoint = False
        if args.exit_signal_handler:
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler)
                print_datetime('exiting program after receiving SIGTERM.')
                sys.exit()

        if args.save and args.save_interval and \
           iteration % args.save_interval == 0:
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler)
            saved_checkpoint = True

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             opt_param_scheduler)
                print_datetime('exiting program after {} minutes'.format(train_time))
                sys.exit()

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            sys.exit()

    if args.kaimm_gc_interval > 0:
        gc.enable()

    # Copy optimizer params to model params in need when `overlap_gather`
    optimizer.gather_model_params(args, timers)

    return iteration


def evaluate(forward_step_func,
             get_batch_func,
             data_iterator,
             model,
             process_non_loss_data_func,
             verbose=False):
    """Evaluation."""
    args = get_args()

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))

            forward_backward_func = get_forward_backward_func(
                args.micro_seq_length,
                variable_slicing=args.variable_seq_slicing,
                dspp=args.dspp)
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                get_batch_func=get_batch_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                micro_seq_length=args.micro_seq_length,
                kv_cache_class=get_kv_cache_class(args),
                dtype=args.params_dtype,
                tensor_shape=(args.seq_length, args.micro_batch_size, args.hidden_size),
                sequence_parallel=args.sequence_parallel,
                attn_balance=args.kaimm_pipeline_attn_balance,
                vocab_in_pp=args.kaimm_vocab_in_pipeline_parallel,
                forward_only=True,
                timers=None)

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            is_loss_stage = (
                mpu.get_pipeline_model_parallel_rank() == 0
                if args.dspp or args.variable_seq_schedule == 'slice-v'
                else mpu.is_pipeline_last_stage(ignore_virtual=True)
            )
            if is_loss_stage:
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = total_loss_dict.get(
                            key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]

            args.consumed_valid_samples += mpu.get_data_parallel_for_sample_world_size() \
                                           * args.micro_batch_size \
                                           * get_num_microbatches()
        collected_non_loss_data = None
        if process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func, data_iterator, model, optimizer=None,
                timers=None, forward_only=True, collect_non_loss_data=True)

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()

    return total_loss_dict, collected_non_loss_data

def evaluate_and_print_results(prefix, forward_step_func, get_batch_func,
                               data_iterator, model,
                               iteration, process_non_loss_data_func,
                               verbose=False):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    writer = get_tensorboard_writer()

    total_loss_dict, collected_non_loss_data = evaluate(
        forward_step_func, get_batch_func, data_iterator, model,
        process_non_loss_data_func, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer:
            writer.add_scalar('{} validation'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} validation vs samples'.format(key),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation ppl'.format(key), ppl,
                                  iteration)
                writer.add_scalar('{} validation ppl vs samples'.format(key),
                                  ppl, args.consumed_train_samples)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_loss_rank = print_rank_0 if args.dspp else print_rank_last
    print_loss_rank('-' * length)
    print_loss_rank(string)
    print_loss_rank('-' * length)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider):
    """XXX"""
    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_tensor_model_parallel_rank() == 0:

        # Number of train/valid/test samples.
        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = args.train_iters * args.global_batch_size
        eval_iters = (args.train_iters // args.eval_interval + 1) * \
                     args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_samples,
                                      eval_iters * args.global_batch_size,
                                      test_iters * args.global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            train_val_test_num_samples)

        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples)
        valid_dataloader = build_pretraining_data_loader(
            valid_ds, args.consumed_valid_samples)
        test_dataloader = build_pretraining_data_loader(test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    return train_dataloader, valid_dataloader, test_dataloader


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):

    args = get_args()

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = \
        build_train_valid_test_data_loaders(
            build_train_valid_test_datasets_provider)

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']

    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(train_dataloader))
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(valid_dataloader))
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader) if dl_type == 'single' \
                             else iter(cyclic_iter(test_dataloader))
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
