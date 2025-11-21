# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import Callable

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP, GroupedMLP_ReCompute
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"
        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.shared_expert = None
        self.coefficient = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        pass

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, layer_number: int = None
    ):
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        if self.config.moe_grouped_gemm:
            if config.kaimm_recompute_token_dispatcher:
                self.experts = GroupedMLP_ReCompute(self.num_local_experts, self.config)
            else:
                self.experts = GroupedMLP(self.num_local_experts, self.config)
        elif self.config.moe_te_grouped_gemm:
            self.experts = TEGroupedMLP(self.num_local_experts, self.config)
        else:
            self.experts = SequentialMLP(self.num_local_experts, self.config)
        if config.shared_expert_hidden_size:
            from megatron.model.transformer import ParallelMLP
            self.shared_expert = ParallelMLP(config.shared_expert_hidden_size,
                                             config.init_method, config.output_layer_init_method,
                                             ub_fc1_fw_obj=config.ub_fc1_fw_obj,
                                             ub_fc2_fw_obj=config.ub_fc2_fw_obj,
                                             ub_fc2_bw_obj=config.ub_fc2_bw_obj,
                                             recompute_mlp_activation_func=config.kaimm_recompute_mlp_activation_func,
                                             recompute_mlp_fc1=config.kaimm_recompute_mlp_fc1)
            if config.shared_expert_combine_method == "softmax":
                self.coefficient = torch.nn.Linear(config.hidden_size, 2)
                if config.perform_initialization:
                    with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
                        config.init_method(self.coefficient.weight)
                    with torch.no_grad():
                        self.coefficient.bias.zero_()
                setattr(self.coefficient.weight, 'sequence_parallel', config.sequence_parallel)
                setattr(self.coefficient.bias, 'sequence_parallel', config.sequence_parallel)
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )
        self.moe_layer_recompute = config.moe_layer_recompute

    def forward(self, hidden_states: torch.Tensor, cp_data_to_save=None):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states, cp_data_to_save):
            probs, indices = self.router(hidden_states)
            if self.config.kaimm_recompute_token_dispatcher:
                expert_output, mlp_bias = self.experts(hidden_states, probs, indices, self.token_dispatcher)
            else:
                (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                    hidden_states, probs, indices
                )
                expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            if isinstance(cp_data_to_save, Callable):
                expert_output = cp_data_to_save(expert_output)
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            if self.config.shared_expert_hidden_size:
                shared_output, shared_mlp_bias = self.shared_expert(hidden_states)
                if shared_mlp_bias is not None:
                    raise NotImplementedError("mlp bias fusion for shared expert is not supported")
                if self.config.shared_expert_combine_method == "add":
                    output = output + shared_output
                elif self.config.shared_expert_combine_method == "softmax":
                    coef = self.coefficient(hidden_states).softmax(dim=-1)
                    output = output * coef[..., :1] + shared_output * coef[..., 1:]
                else:
                    raise NotImplementedError(f"not implemented {self.config.shared_expert_combine_method=}")
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, None, hidden_states, cp_data_to_save)
        else:
            output, mlp_bias = custom_forward(hidden_states, cp_data_to_save)

        return output, mlp_bias
