# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD

from megatron import get_args

from .distrib_optimizer import DistributedOptimizer
from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import ChainedOptimizer, Float16OptimizerWithFloat16Params, FP32Optimizer
from .sophia import SophiaG


def get_param_groups(model_chunks, no_weight_decay_cond, scale_lr_cond, lr_mult):
    """Create parameter groups for optimizer.

    Creates parameter groups based on weight decay condition (regularized vs
    non regularized), learning rate scale condition (lr vs lr_mult * lr),
    and whether it is expert parameters. scale_lr_cond is used during finetuning
    where head of the network requires a scaled version of the base learning rate.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        no_weight_decay_cond (func): function to determine whether a parameter
            should not perform weight decay.
        scale_lr_cond (func): function to determine whether a parameter
            should have a scaled learning rate.
        lr_mult (float): learning rate multiplier for parameters that
            satisfy scale_lr_cond.
    """
    # map (wd_mult, lr_mult, is_expert_parallel) to params
    params_map = {
        (1.0, 1.0, False): [],
        (1.0, 1.0, True): [],
        (1.0, lr_mult, False): [],
        (1.0, lr_mult, True): [],
        (0.0, 1.0, False): [],
        (0.0, 1.0, True): [],
        (0.0, lr_mult, False): [],
        (0.0, lr_mult, True): [],
    }

    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)

            if no_weight_decay_cond is not None:
                no_wd = no_weight_decay_cond(name, param)
            else:
                # do not regularize biases nor Norm parameters
                no_wd = name.endswith(".bias") or len(param.shape) == 1

            if scale_lr_cond is not None:
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_mult, _lr_mult = 1.0, 1.0
            elif not no_wd and scale_lr:
                wd_mult, _lr_mult = 1.0, lr_mult
            elif no_wd and not scale_lr:
                wd_mult, _lr_mult = 0.0, 1.0
            else:
                wd_mult, _lr_mult = 0.0, lr_mult

            params_map[(wd_mult, _lr_mult, is_expert_parallel)].append(param)

    param_groups = []
    for (wd_mult, _lr_mult, is_expert_parallel), params in params_map.items():
        if len(params) == 0:
            continue
        param_groups.append(
            {
                'params': params,
                'wd_mult': wd_mult,
                'lr_mult': _lr_mult,
                'is_expert_parallel': is_expert_parallel,
            }
        )

    # Add a dummy param_group since apex FusedAdam and FusedSGD do not allow empty param_groups
    if len(param_groups) == 0:
        param_groups.append({'params': [], 'wd_mult': 1.0, 'lr_mult': 1.0})

    return param_groups


def get_megatron_optimizer_switch_moe(model,
                                      is_expert_parallel,
                                      no_weight_decay_cond=None,
                                      scale_lr_cond=None,
                                      lr_mult=1.0):
    args = get_args()

    # Base optimizer.
    param_groups = get_param_groups(model,
                                    no_weight_decay_cond,
                                    scale_lr_cond,
                                    lr_mult)
    param_groups = [
        param
        for param in param_groups
        if param["is_expert_parallel"] == is_expert_parallel
    ]

    if args.optimizer == 'adam':
        optimizer = Adam(param_groups,
                         lr=args.lr,
                         weight_decay=args.weight_decay,
                         betas=(args.adam_beta1, args.adam_beta2),
                         eps=args.adam_eps)
    elif args.optimizer == 'sgd':
        optimizer = SGD(param_groups,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        momentum=args.sgd_momentum)
    elif args.optimizer == 'sophia':
        optimizer = SophiaG(param_groups,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        betas=(args.sophia_beta1, args.sophia_beta2),
                        rho=args.sophia_rho)
    else:
        raise Exception('{} optimizer is not supported.'.format(
            args.optimizer))

    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    if args.DDP_impl == 'local':
        params_have_main_grad = True

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if args.fp16 or args.bf16 or args.use_distributed_optimizer:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None

        # Constant loss scale.
        if args.loss_scale:
            grad_scaler = ConstantGradScaler(args.loss_scale)

        # Dynamic loss scale.
        else:
            if args.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=args.initial_loss_scale,
                    min_scale=args.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=args.loss_scale_window,
                    hysteresis=args.hysteresis)

        # Megatron optimizer.
        opt_ty = DistributedOptimizer \
            if args.use_distributed_optimizer else \
            Float16OptimizerWithFloat16Params
        return opt_ty(optimizer,
                      args.clip_grad,
                      args.log_num_zeros_in_grad,
                      params_have_main_grad,
                      args.use_contiguous_buffers_in_local_ddp,
                      args.fp16,
                      args.bf16,
                      args.params_dtype,
                      grad_scaler,
                      is_expert_parallel,
                      model)

    # FP32.
    return FP32Optimizer(optimizer, args.clip_grad,
                         args.log_num_zeros_in_grad,
                         params_have_main_grad,
                         args.use_contiguous_buffers_in_local_ddp,
                         is_expert_parallel,
                         model)


def get_megatron_optimizer(model,
                           no_weight_decay_cond=None,
                           scale_lr_cond=None,
                           lr_mult=1.0):
    args = get_args()
    dense_optimizer = get_megatron_optimizer_switch_moe(model, is_expert_parallel=False, no_weight_decay_cond=no_weight_decay_cond, scale_lr_cond=scale_lr_cond, lr_mult=lr_mult)
    if args.num_experts == 1 or args.expert_model_parallel_size == 1:
        return dense_optimizer
    model_ep = [m.model_ep for m in model]
    moe_optimizer = get_megatron_optimizer_switch_moe(model_ep, is_expert_parallel=True, no_weight_decay_cond=no_weight_decay_cond, scale_lr_cond=scale_lr_cond, lr_mult=lr_mult)
    return ChainedOptimizer([dense_optimizer, moe_optimizer])
