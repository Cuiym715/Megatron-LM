# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Optional, Tuple

import inspect
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.model.utils import swiglu

try:
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


class GroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using CUTLASS GroupedGEMM.
    
    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """

    def __init__(self, num_local_experts: int, config: TransformerConfig):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert (
            config.add_bias_linear == False
        ), "bias in the expert layer is not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."

        self.expert_parallel = config.expert_model_parallel_size > 1
        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError("Activation function must be silu or gelu when using GroupedMLP.")

            if self.config.activation_func == F.silu:
                self.activation_func = swiglu
            else:
                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

                self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        fc1_output_size = self.config.expert_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.expert_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight1,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    partition_dim=1,
                    init_method=config.init_method,
                    params_dtype=config.params_dtype,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    partition_dim=0,
                    init_method=config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                )
        else:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight1,
                    config.init_method,
                    partition_dim=1,
                    expert_parallel=self.expert_parallel,
                )
                _initialize_affine_weight_gpu(
                    self.weight2,
                    config.output_layer_init_method,
                    partition_dim=0,
                    expert_parallel=self.expert_parallel,
                )
        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

        self.gradient_accumulation_fusion = (self.config.gradient_accumulation_fusion and
                                             "b_main_grad" in inspect.signature(gg.ops.gmm).parameters)
        self.views_created = False
        self.recompute_mlp_activation_func = self.config.kaimm_recompute_mlp_activation_func
        self.recompute_mlp_fc1 = self.config.kaimm_recompute_mlp_fc1
        self.w1_main_grad = None
        self.w2_main_grad = None

    def forward(self, permuted_local_hidden_states, tokens_per_expert):
        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            # We cache the view results to save CPU time.
            if not self.views_created:
                if self.gradient_accumulation_fusion:
                    self.w1_main_grad = self.weight1.main_grad.view(w1.shape)
                    self.w2_main_grad = self.weight2.main_grad.view(w2.shape)
                self.views_created = True

            if self.gradient_accumulation_fusion:
                w1_main_grad = self.w1_main_grad
                w2_main_grad = self.w2_main_grad
            else:
                w1_main_grad = None
                w2_main_grad = None
            if self.recompute_mlp_fc1:
                assert self.recompute_mlp_activation_func
                def recompute_func(permuted_local_hidden_state):
                    fc1_output = gg.gmm_recompute(
                        permuted_local_hidden_state, w1, tokens_per_expert, trans_b=False, b_main_grad=w1_main_grad
                    )
                    intermediate_parallel = self.activation_func(fc1_output)
                    return intermediate_parallel
                intermediate_parallel = permuted_local_hidden_states
            elif self.recompute_mlp_activation_func:
                intermediate_parallel = gg.gmm_recompute(
                    permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False, b_main_grad=w1_main_grad
                )
                recompute_func = self.activation_func
            else:
                fc1_output = gg.gmm_recompute(
                    permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False, b_main_grad=w1_main_grad
                )
                intermediate_parallel = self.activation_func(fc1_output)
                recompute_func = None
            fc2_output = gg.gmm_recompute(intermediate_parallel, w2, tokens_per_expert, trans_b=False, b_main_grad=w2_main_grad, recompute_func=recompute_func)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure parameters still have gradients when no tokens are routed to this set of experts.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = self.activation_func(h)
            h = torch.matmul(h, w2)

            fc2_output = h

        return fc2_output, None

    def sharded_state_dict(self, prefix='', sharded_offsets=()):
        raise NotImplementedError(
            'Currently distributed checkpointing is not supported for GroupedMLP'
        )


class GroupedMLP_ReCompute(GroupedMLP):
    def __init__(self, num_local_experts: int, config: TransformerConfig):
        super().__init__(num_local_experts=num_local_experts, config=config)

    def forward(self, hidden_states, probs, indices, token_dispatcher):
        w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
        w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)
        # We cache the view results to save CPU time.
        if not self.views_created:
            if self.gradient_accumulation_fusion:
                self.w1_main_grad = self.weight1.main_grad.view(w1.shape)
                self.w2_main_grad = self.weight2.main_grad.view(w2.shape)
            else:
                self.w1_main_grad = None
                self.w2_main_grad = None
            self.views_created = True
        def recompute_func(input_hidden_state):
            (dispatched_input, tokens_per_expert) = token_dispatcher.token_permutation(
                input_hidden_state, probs, indices
            )
            if dispatched_input.nelement() != 0:
                fc1_output = gg.gmm_recompute(
                    dispatched_input, w1, tokens_per_expert, trans_b=False, b_main_grad=self.w1_main_grad
                )
                intermediate_parallel = self.activation_func(fc1_output)
            else:
                weight1 = w1.view(self.config.hidden_size, -1)
                h = torch.matmul(dispatched_input, weight1)
                intermediate_parallel = self.activation_func(h)
            return intermediate_parallel, tokens_per_expert
        
        fc2_output = gg.gmm_recompute_more(hidden_states, w2, trans_b=False, b_main_grad=self.w2_main_grad, recompute_func=recompute_func)

        return fc2_output, None

class TEGroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using TE's GroupedLinear.

    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """

    def __init__(self, num_local_experts, config: TransformerConfig):
        super().__init__(config=config)
        self.moe_extended_tp = config.moe_extended_tp
        self.num_local_experts = num_local_experts
        self.input_size = self.config.hidden_size

        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        expert_hidden_size = self.config.expert_hidden_size
        if self.config.gated_linear_unit:
            expert_hidden_size *= 2

        self.linear_fc1 = build_module(
            TEColumnParallelGroupedLinear,
            self.num_local_experts,
            self.input_size,
            expert_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name='fc1',
        )

        self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            TERowParallelGroupedLinear,
            self.num_local_experts,
            self.config.expert_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name='fc2',
        )

    def forward(
        self, permuted_local_hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of TEGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        tokens_per_expert = tokens_per_expert.tolist()
        intermediate_parallel, bias_parallel = self.linear_fc1(
            permuted_local_hidden_states, tokens_per_expert
        )

        if self.config.bias_activation_fusion and bias_parallel is not None:
            raise NotImplementedError("not implemented bias_activation_fusion")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                if self.config.activation_func == F.silu:
                    glu = swiglu
                else:
                    def glu(x):
                        x = torch.chunk(x, 2, dim=-1)
                        return self.config.activation_func(x[0]) * x[1]

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

        output, output_bias = self.linear_fc2(intermediate_parallel, tokens_per_expert)

        return output, output_bias


class SequentialMLP(MegatronModule):
    """An implementation of the Experts layer using a sequence of MLP layers.
    
    This class executes each expert sequentially.
    """

    def __init__(self, num_local_experts, config: TransformerConfig):
        super().__init__(config=config)
        self.add_bias = config.add_bias_linear
        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()
        for _ in range(self.num_local_experts):
            from megatron.model.transformer import ParallelMLP
            expert = ParallelMLP(config.expert_hidden_size,
                                 config.init_method, config.output_layer_init_method,
                                 is_expert=True,
                                 ub_fc1_fw_obj=config.ub_fc1_fw_obj,
                                 ub_fc2_fw_obj=config.ub_fc2_fw_obj,
                                 ub_fc2_bw_obj=config.ub_fc2_bw_obj,
                                 recompute_mlp_activation_func=config.kaimm_recompute_mlp_activation_func,
                                 recompute_mlp_fc1=config.kaimm_recompute_mlp_fc1)
            self.local_experts.append(expert)

    def forward(self, permuted_local_hidden_states, tokens_per_expert):
        output_local = torch.zeros_like(permuted_local_hidden_states)
        output_bias_local = None
        if self.add_bias:
            output_bias_local = torch.zeros_like(permuted_local_hidden_states)

        cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
        # Insert zero at the begining for offset index's convenience
        zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
        cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))
        for expert_num, expert in enumerate(self.local_experts):
            start = cumsum_num_tokens[expert_num]
            end = cumsum_num_tokens[expert_num + 1]
            hidden = permuted_local_hidden_states[start:end]

            # Transform to ParallelMLP layout
            s, h = hidden.shape
            output, output_bias = expert(hidden.view(s, 1, h))
            output = output.view(s, h)
            if output_bias is not None:
                raise NotImplementedError("not implemented transform output_bias")

            output_local[start:end] = output
            if self.add_bias:
                output_bias = output_bias.expand_as(output)
                output_bias_local[start:end, :] = output_bias

        return output_local, output_bias_local
