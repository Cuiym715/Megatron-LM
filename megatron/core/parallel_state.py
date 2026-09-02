# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import os
import torch
from typing import Optional

from .utils import GlobalMemoryBuffer, GlobalTEUserBuffer

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_GROUP_GLOO = None
# DSPP uses two NCCL communicators per physical edge so traffic in one
# direction cannot head-of-line block traffic in the other direction.
_PIPELINE_MODEL_PARALLEL_NEXT_GROUP = None
_PIPELINE_MODEL_PARALLEL_PREV_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Network barrier group that the current rank belongs to. Used in --kaimm-overlap-optimizer-communication.
_NETWORK_BARRIER_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_SLOW = None
_DATA_PARALLEL_GROUP_GLOO = None
# FP8 amax reduction group.
_AMAX_REDUCTION_GROUP = None
# Expert parallel group that the current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None
_TENSOR_AND_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP_SLOW = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = None

# Context parallel
#   DP groups collaborating on the same context (token sequence) are put into a context parallel group.
#   Data parallel is the Cartesian product of the following factors:
#   1. Data parallel for context, i.e. context parallel.
#   2. Data parallel for sample.
#   In the case of context parallel, be careful on the following matters:
#   1. The dataloader should load contiguous tokens along the context parallel group.
#   2. Attention is performed across the context parallel group.
#   3. Loss and learning rate should be carefully scaled.
_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_GROUP_SLOW = None
_CONTEXT_PARALLEL_GROUP_LOCAL = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_DATA_PARALLEL_WORLD_SIZE = None
_MPU_CONTEXT_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None
_MPU_DATA_PARALLEL_RANK = None
_MPU_CONTEXT_PARALLEL_RANK = None
_MPU_EXPERT_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None
_GLOBAL_TE_USER_BUFFER = None

# MOE logging
_MOE_AUX_LOSSES_LOGGING_TRACKER = {}


def get_nccl_options(pg_name, nccl_comm_cfgs):
    """Set the NCCL process group options.

    Arguments:
        pg_name (str): process group name
        nccl_comm_cfgs (dict): nccl communicator configurations

    When an option (e.g., max_ctas) is not found in the config, use the NCCL default setting.
    """
    if pg_name in nccl_comm_cfgs:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get('cga_cluster_size', 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get('max_ctas', 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get('min_ctas', 1)
        return nccl_options
    else:
        return None


def check_ctas_settings_are_effective():
    assert not os.environ.get("NCCL_MIN_NRINGS"), "NCCL_MIN_NRINGS overrides torch.distributed.ProcessGroupNCCL.Options.min_ctas"
    assert not os.environ.get("NCCL_MAX_NRINGS"), "NCCL_MAX_NRINGS overrides torch.distributed.ProcessGroupNCCL.Options.max_ctas"
    assert not os.environ.get("NCCL_MIN_NCHANNELS"), "NCCL_MIN_CHANNELS overrides torch.distributed.ProcessGroupNCCL.Options.min_ctas"
    assert not os.environ.get("NCCL_MAX_NCHANNELS"), "NCCL_MAX_NCHANNELS overrides torch.distributed.ProcessGroupNCCL.Options.max_ctas"
    assert not os.environ.get("NCCL_MIN_CTAS"), "NCCL_MIN_CTAS overrides torch.distributed.ProcessGroupNCCL.Options.min_ctas"
    assert not os.environ.get("NCCL_MAX_CTAS"), "NCCL_MAX_CTAS overrides torch.distributed.ProcessGroupNCCL.Options.max_ctas"


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_fp8: bool = False,
    *,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    kaimm_overlap_cp_slow_ctas = None,
    kaimm_overlap_optimizer_communication = False,
    kaimm_overlap_optimizer_slow_ctas = None,
    overlap_sp_ag = False,
    overlap_sp_rs = False
) -> None:
    """Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        use_fp8 (bool, default = False):
            Construct GPU groups needed for FP8 training, namely for
            amax reduction across the product of the data-parallel and
            tensor-parallel groups.

        nccl_communicator_config_path (str, default = None):
            Path to the yaml file of NCCL communicator configurations.
            `min_ctas`, `max_ctas`, and `cga_cluster_size` can be set
            for each communicator.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size) != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    data_parallel_size: int = world_size // (tensor_model_parallel_size *
                                             pipeline_model_parallel_size)
    assert data_parallel_size % context_parallel_size == 0, f"{data_parallel_size} % {context_parallel_size} != 0"

    if data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    num_tensor_model_parallel_groups: int  = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    num_data_parallel_groups: int = world_size // data_parallel_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size >= 2:
            raise RuntimeError("pipeline-model-parallel size should be greater than or equal to 2 with "
                               "interleaved schedule")
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_SLOW
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GROUP_SLOW
    global _CONTEXT_PARALLEL_GROUP_LOCAL
    assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))
            group = torch.distributed.new_group(ranks)
            group_gloo = torch.distributed.new_group(ranks, backend="gloo")
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
                _DATA_PARALLEL_GROUP_GLOO = group_gloo
                _DATA_PARALLEL_GLOBAL_RANKS = ranks
            if kaimm_overlap_optimizer_communication and data_parallel_size >= 2:
                assert kaimm_overlap_optimizer_slow_ctas is not None
                check_ctas_settings_are_effective()
                opt_slow = torch.distributed.ProcessGroupNCCL.Options()
                opt_slow.config.min_ctas = opt_slow.config.max_ctas = kaimm_overlap_optimizer_slow_ctas
                group_slow = torch.distributed.new_group(ranks, pg_options=opt_slow)
                if rank in ranks:
                    _DATA_PARALLEL_GROUP_SLOW = group_slow
            for k in range(data_parallel_size // context_parallel_size):
                ranks = range(
                    start_rank + j + k * (tensor_model_parallel_size * context_parallel_size),
                    start_rank + j + (k + 1) * (tensor_model_parallel_size * context_parallel_size),
                    tensor_model_parallel_size,
                )
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    _CONTEXT_PARALLEL_GROUP = group
                if context_parallel_size >= 2:
                    check_ctas_settings_are_effective()
                    opt_slow = torch.distributed.ProcessGroupNCCL.Options()
                    if kaimm_overlap_cp_slow_ctas is not None:
                        opt_slow.config.min_ctas = opt_slow.config.max_ctas = kaimm_overlap_cp_slow_ctas
                    group_slow = torch.distributed.new_group(ranks, pg_options=opt_slow)
                    if rank in ranks:
                        _CONTEXT_PARALLEL_GROUP_SLOW = group_slow

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for i in range(data_parallel_size):
        ranks = [data_parallel_group_ranks[i]
                 for data_parallel_group_ranks in all_data_parallel_group_ranks]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        'tensor model parallel group is already initialized'
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size,
                      (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    if kaimm_overlap_optimizer_communication and data_parallel_size >= 2:
        global _NETWORK_BARRIER_GROUP
        assert _NETWORK_BARRIER_GROUP is None, \
            'network barrier group is already initialized'
        group_size = tensor_model_parallel_size * data_parallel_size
        for i in range(0, world_size, group_size):
            ranks = range(i, i + group_size)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _NETWORK_BARRIER_GROUP = group

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_GROUP_GLOO
    global _PIPELINE_MODEL_PARALLEL_NEXT_GROUP
    global _PIPELINE_MODEL_PARALLEL_PREV_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, \
        'pipeline model parallel group is already initialized'
    assert _PIPELINE_MODEL_PARALLEL_NEXT_GROUP is None
    assert _PIPELINE_MODEL_PARALLEL_PREV_GROUP is None
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, \
        'position embedding group is already initialized'
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks)
        group_gloo = torch.distributed.new_group(ranks, backend="gloo")
        from megatron import get_args
        if getattr(get_args(), 'dspp', False):
            edge_groups = []
            for edge in range(len(ranks) - 1):
                edge_ranks = [ranks[edge], ranks[edge + 1]]
                next_group = torch.distributed.new_group(edge_ranks)
                prev_group = torch.distributed.new_group(edge_ranks)
                edge_groups.append((edge_ranks, next_group, prev_group))
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_MODEL_PARALLEL_GROUP_GLOO = group_gloo
            if getattr(get_args(), 'dspp', False):
                _PIPELINE_MODEL_PARALLEL_NEXT_GROUP = {}
                _PIPELINE_MODEL_PARALLEL_PREV_GROUP = {}
                for edge, (edge_ranks, next_group, prev_group) in enumerate(edge_groups):
                    if rank in edge_ranks:
                        _PIPELINE_MODEL_PARALLEL_NEXT_GROUP[edge] = next_group
                        _PIPELINE_MODEL_PARALLEL_PREV_GROUP[edge] = prev_group
                        lane_warmup = torch.empty((), dtype=torch.float, device='cuda')
                        low_rank, high_rank = edge_ranks
                        next_op = torch.distributed.P2POp(
                            torch.distributed.isend if rank == low_rank else torch.distributed.irecv,
                            lane_warmup,
                            high_rank if rank == low_rank else low_rank,
                            next_group,
                        )
                        prev_op = torch.distributed.P2POp(
                            torch.distributed.isend if rank == high_rank else torch.distributed.irecv,
                            lane_warmup,
                            low_rank if rank == high_rank else high_rank,
                            prev_group,
                        )
                        for request in torch.distributed.batch_isend_irecv([next_op]):
                            request.wait()
                        for request in torch.distributed.batch_isend_irecv([prev_op]):
                            request.wait()
            _PIPELINE_GLOBAL_RANKS = ranks
            # warmup collective comm for `batch_isend_irecv`,
            # refer to https://pytorch.org/docs/stable/distributed.html#torch.distributed.batch_isend_irecv
            tensor = torch.empty((), dtype=torch.float, device='cuda')
            torch.distributed.all_reduce(tensor, group=group)
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            from megatron import get_args
            is_slice_v = (
                getattr(get_args(), 'variable_seq_schedule', '1f1b') == 'slice-v'
                or getattr(get_args(), 'dspp', False)
            )
            if is_slice_v:
                embedding_ranks = [ranks[0]]
            else:
                embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank is not None and not is_slice_v:
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [ranks[0],
                                       ranks[pipeline_model_parallel_split_rank],
                                       ranks[-1]]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0],
                                       ranks[pipeline_model_parallel_split_rank]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(embedding_ranks)
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(position_embedding_ranks)
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the FP8 groups.
    global _AMAX_REDUCTION_GROUP
    assert _AMAX_REDUCTION_GROUP is None, \
        'FP8 amax reduction group is already initialized'
    if use_fp8:
        amax_group_size: int = tensor_model_parallel_size * data_parallel_size
        num_amax_groups: int = world_size // amax_group_size
        for i in range(num_amax_groups):
            start_rank = i * amax_group_size
            end_rank = (i + 1) * amax_group_size
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _AMAX_REDUCTION_GROUP = group

    # Build the tensor + expert parallel groups
    global _EXPERT_MODEL_PARALLEL_GROUP
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is None
    ), 'Tensor + expert parallel group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is None
    ), 'Data modulo expert group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP_SLOW
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO
    tensor_and_data_group_size: int = tensor_model_parallel_size * data_parallel_size
    num_tensor_and_data_groups: int = world_size // tensor_and_data_group_size
    tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
    num_expert_groups: int = data_parallel_size // expert_model_parallel_size
    for i in range(num_tensor_and_data_groups):
        for j in range(num_expert_groups):
            # TPxEP Group
            start_rank = i * tensor_and_data_group_size + j * tensor_and_expert_group_size
            end_rank = i * tensor_and_data_group_size + (j + 1) * tensor_and_expert_group_size
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(
                ranks, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_AND_EXPERT_PARALLEL_GROUP = group
            for k in range(tensor_model_parallel_size):
                ranks = range(
                    start_rank + k, end_rank, tensor_model_parallel_size
                )
                group = torch.distributed.new_group(
                    ranks, pg_options=get_nccl_options('exp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    _EXPERT_MODEL_PARALLEL_GROUP = group

    for i in range(num_tensor_and_data_groups):
        start_rank = i * tensor_and_data_group_size
        end_rank = (i + 1) * tensor_and_data_group_size
        for j in range(tensor_and_expert_group_size):
            ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
            group = torch.distributed.new_group(
                ranks, pg_options=get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
            )
            group_gloo = torch.distributed.new_group(ranks, backend="gloo")
            if rank in ranks:
                _DATA_MODULO_EXPERT_PARALLEL_GROUP = group
                _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo
            if kaimm_overlap_optimizer_communication and data_parallel_size >= 2:
                assert kaimm_overlap_optimizer_slow_ctas is not None
                check_ctas_settings_are_effective()
                opt_slow = torch.distributed.ProcessGroupNCCL.Options()
                opt_slow.config.min_ctas = opt_slow.config.max_ctas = kaimm_overlap_optimizer_slow_ctas
                group_slow = torch.distributed.new_group(ranks, pg_options=opt_slow)
                if rank in ranks:
                    _DATA_MODULO_EXPERT_PARALLEL_GROUP_SLOW = group_slow

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()
    if(overlap_sp_ag or overlap_sp_rs):
        _set_global_te_user_buffer()

def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or \
        _PIPELINE_MODEL_PARALLEL_GROUP is None or \
        _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group_gloo():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP_GLOO is not None, \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP_GLOO


def get_pipeline_model_parallel_next_group(edge):
    """Return the DSPP communicator for traffic to the next physical stage."""
    assert _PIPELINE_MODEL_PARALLEL_NEXT_GROUP is not None, \
        'pipeline next-direction group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_NEXT_GROUP[edge]


def get_pipeline_model_parallel_prev_group(edge):
    """Return the DSPP communicator for traffic to the previous physical stage."""
    assert _PIPELINE_MODEL_PARALLEL_PREV_GROUP is not None, \
        'pipeline prev-direction group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_PREV_GROUP[edge]


def get_context_parallel_group():
    """Get the context parallel group the caller rank belongs to."""
    assert _CONTEXT_PARALLEL_GROUP is not None, \
        'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_group_slow():
    """Get the context parallel group-slow the caller rank belongs to."""
    assert _CONTEXT_PARALLEL_GROUP_SLOW is not None, \
        'context parallel group-slow is not initialized'
    return _CONTEXT_PARALLEL_GROUP_SLOW


def get_context_parallel_group_local():
    """Get the context parallel group-local the caller rank belongs to."""
    assert _CONTEXT_PARALLEL_GROUP_LOCAL is not None, \
        'context parallel group-local is not initialized'
    return _CONTEXT_PARALLEL_GROUP_LOCAL


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_data_parallel_group_slow():
    """Get the data parallel group-slow the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP_SLOW is not None, \
        'data parallel group-slow is not initialized'
    return _DATA_PARALLEL_GROUP_SLOW


def get_data_parallel_group_gloo():
    """Get the data parallel group-gloo the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP_GLOO is not None, \
        'data parallel group-gloo is not initialized'
    return _DATA_PARALLEL_GROUP_GLOO


def get_network_barrier_group():
    """Get the network barrier group the caller rank belongs to."""
    assert _NETWORK_BARRIER_GROUP is not None, \
        'network barrier group is not initialized'
    return _NETWORK_BARRIER_GROUP


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, \
        'embedding group is not initialized'
    return _EMBEDDING_GROUP


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert _POSITION_EMBEDDING_GROUP is not None, \
        'position embedding group is not initialized'
    return _POSITION_EMBEDDING_GROUP


def get_amax_reduction_group():
    """Get the FP8 amax reduction group the caller rank belongs to."""
    assert _AMAX_REDUCTION_GROUP is not None, \
        'FP8 amax reduction group is not initialized'
    return _AMAX_REDUCTION_GROUP


def get_expert_model_parallel_group():
    assert (
        _EXPERT_MODEL_PARALLEL_GROUP is not None
    ), 'expert model parallel group is not initialized'
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_tensor_and_expert_parallel_group():
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is not None
    ), 'tensor and expert parallel group is not initialized'
    return _TENSOR_AND_EXPERT_PARALLEL_GROUP


def get_data_modulo_expert_parallel_group():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is not None
    ), 'data modulo expert parallel group is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP


def get_data_modulo_expert_parallel_group_slow():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP_SLOW is not None
    ), 'data modulo expert parallel group-slow is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP_SLOW


def get_data_modulo_expert_parallel_group_gloo():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO is not None
    ), 'data modulo expert parallel group-gloo is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO


def set_expert_model_parallel_world_size(world_size):
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_data_parallel_world_size(world_size):
    """Set the data parallel size"""
    global _MPU_DATA_PARALLEL_WORLD_SIZE
    _MPU_DATA_PARALLEL_WORLD_SIZE = world_size


def set_context_parallel_world_size(world_size):
    """Set the context parallel size"""
    global _MPU_CONTEXT_PARALLEL_WORLD_SIZE
    _MPU_CONTEXT_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    global _MPU_CONTEXT_PARALLEL_WORLD_SIZE
    if _MPU_CONTEXT_PARALLEL_WORLD_SIZE is not None:
        return _MPU_CONTEXT_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_context_parallel_group())


def set_expert_model_parallel_rank(rank):
    """Set expert model parallel rank."""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = rank


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def set_data_parallel_rank(rank):
    """Set data parallel rank."""
    global _MPU_DATA_PARALLEL_RANK
    _MPU_DATA_PARALLEL_RANK = rank


def set_context_parallel_rank(rank):
    """Set context parallel rank."""
    global _MPU_CONTEXT_PARALLEL_RANK
    _MPU_CONTEXT_PARALLEL_RANK = rank


def set_pipeline_model_parallel_split_rank(rank):
    """Set pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def get_context_parallel_rank():
    """Return my rank for the context parallel group."""
    global _MPU_CONTEXT_PARALLEL_RANK
    if _MPU_CONTEXT_PARALLEL_RANK is not None:
        return _MPU_CONTEXT_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_context_parallel_group())


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if get_virtual_pipeline_model_parallel_world_size() is not None and \
            get_virtual_pipeline_model_parallel_rank() != 0:
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = \
            get_virtual_pipeline_model_parallel_world_size()
        if virtual_pipeline_model_parallel_world_size is not None:
            from megatron import get_args
            args = get_args()
            if (
                getattr(args, 'variable_seq_schedule', '1f1b') == 'slice-v'
                or getattr(args, 'dspp', False)
            ):
                assert virtual_pipeline_model_parallel_world_size == 2
                return get_pipeline_model_parallel_rank() == 0 and \
                    get_virtual_pipeline_model_parallel_rank() == 1
        if virtual_pipeline_model_parallel_world_size is not None and \
            get_virtual_pipeline_model_parallel_rank() != (
                virtual_pipeline_model_parallel_world_size - 1):
            return False
    return get_pipeline_model_parallel_rank() == (
        get_pipeline_model_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    from megatron import get_args
    args = get_args()
    if (
        getattr(args, 'variable_seq_schedule', '1f1b') == 'slice-v'
        or getattr(args, 'dspp', False)
    ):
        return rank in _EMBEDDING_GLOBAL_RANKS and (
            is_pipeline_first_stage(ignore_virtual=False)
            or is_pipeline_last_stage(ignore_virtual=False)
        )
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS


def is_pipeline_stage_before_split(rank=None):
    """Return True if pipeline stage executes encoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank < _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_after_split(rank=None):
    """Return True if pipeline stage executes decoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank >= _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_at_split():
    """Return true if pipeline stage executes decoder block and next
    stage executes encoder block for a model with both encoder and
    decoder."""
    rank = get_pipeline_model_parallel_rank()
    return is_pipeline_stage_before_split(rank) and \
            is_pipeline_stage_after_split(rank+1)


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def set_virtual_pipeline_model_parallel_world_size(world_size):
    """Set the virtual pipeline-parallel world size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    assert _DATA_PARALLEL_GLOBAL_RANKS is not None, \
        "Data parallel group is not initialized"
    return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]

def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_global_rank(rank):
    """Return the global rank from relative rank in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[rank]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    global _MPU_DATA_PARALLEL_WORLD_SIZE
    if _MPU_DATA_PARALLEL_WORLD_SIZE is not None:
        return _MPU_DATA_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    global _MPU_DATA_PARALLEL_RANK
    if _MPU_DATA_PARALLEL_RANK is not None:
        return _MPU_DATA_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_data_parallel_group())


def get_data_parallel_for_sample_world_size():
    return get_data_parallel_world_size() // get_context_parallel_world_size()


def get_data_parallel_for_sample_rank():
    return get_data_parallel_rank() // get_context_parallel_world_size()


def get_expert_model_parallel_world_size():
    """Return world size for the expert model parallel group"""
    if _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE:
        return _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_world_size // get_tensor_model_parallel_world_size()
    else:
        return 0


def get_tensor_and_expert_parallel_world_size():
    """Return world size for the expert model parallel group times model parallel group.
       Currently, each expert will also be distributed across TP group by default.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_world_size
    else:
        return 0


def get_expert_model_parallel_rank():
    """Return my rank for the expert parallel group"""
    if _MPU_EXPERT_MODEL_PARALLEL_RANK:
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_rank = torch.distributed.get_rank(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_rank // get_tensor_model_parallel_world_size()
    else:
        return 0


def get_data_modulo_expert_parallel_world_size():
    """Return world_size for the data modulo expert parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_data_modulo_expert_parallel_group())
    else:
        return 0


def get_data_modulo_expert_parallel_rank():
    """Return my rank for the data modulo expert parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_data_modulo_expert_parallel_group())
    else:
        return 0


def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()

def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'
    return _GLOBAL_MEMORY_BUFFER

def _set_global_te_user_buffer():
    """Initialize global TE userbuffer"""
    global _GLOBAL_TE_USER_BUFFER
    assert _GLOBAL_TE_USER_BUFFER is None, 'global TE userbuffer is already initialized'
    _GLOBAL_TE_USER_BUFFER = GlobalTEUserBuffer()

def get_global_te_user_buffer(name, shape, dtype, ag):
    """Return the GlobalBuffer object"""
    assert _GLOBAL_TE_USER_BUFFER is not None, 'global TE userbuffer is not initialized'
    rank = torch.distributed.get_rank()
    tp_world_size = get_tensor_model_parallel_world_size()
    return _GLOBAL_TE_USER_BUFFER.get_ub(name, shape, dtype,
                                         tp_world_size, rank, ag)

def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP_GLOO
    _PIPELINE_MODEL_PARALLEL_GROUP_GLOO = None
    global _PIPELINE_MODEL_PARALLEL_NEXT_GROUP
    _PIPELINE_MODEL_PARALLEL_NEXT_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_PREV_GROUP
    _PIPELINE_MODEL_PARALLEL_PREV_GROUP = None
    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None
    global _CONTEXT_PARALLEL_GROUP_SLOW
    _CONTEXT_PARALLEL_GROUP_SLOW = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP_SLOW
    _DATA_PARALLEL_GROUP_SLOW = None
    global _NETWORK_BARRIER_GROUP
    _NETWORK_BARRIER_GROUP = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
    global _AMAX_REDUCTION_GROUP
    _AMAX_REDUCTION_GROUP = None
    global _EXPERT_MODEL_PARALLEL_GROUP
    _EXPERT_MODEL_PARALLEL_GROUP = None
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    _TENSOR_AND_EXPERT_PARALLEL_GROUP = None
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    _DATA_MODULO_EXPERT_PARALLEL_GROUP = None
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP_SLOW
    _DATA_MODULO_EXPERT_PARALLEL_GROUP_SLOW = None
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO
    _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_DATA_PARALLEL_WORLD_SIZE
    _MPU_DATA_PARALLEL_WORLD_SIZE = None
    global _MPU_CONTEXT_PARALLEL_WORLD_SIZE
    _MPU_CONTEXT_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None
    global _MPU_DATA_PARALLEL_RANK
    _MPU_DATA_PARALLEL_RANK = None
    global _MPU_CONTEXT_PARALLEL_RANK
    _MPU_CONTEXT_PARALLEL_RANK = None
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None
    global _GLOBAL_TE_USER_BUFFER
    _GLOBAL_TE_USER_BUFFER = None
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = None
