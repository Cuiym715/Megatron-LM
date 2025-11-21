import torch
import torch.nn as nn


# TODO not able to build apex cpp extention for Fused cuda kernel RMSNorm
"""
# Steps performed, 
# 1. copy 
#   https://github.com/NVIDIA/apex/blob/master/apex/normalization/fused_layer_norm.py, 
#   https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda.cpp, 
#   https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda_kernel.cu 
# to ./megatron/model/fused_layer_norm.py, 
#   ./megatron/fused_kernels/layer_norm_cuda.cpp, 
#   ./megatron/fused_kernels/layer_norm_cuda_kernel.cu, 
#   and update ./megatron/fused_kernels/__init__.py accordingly 
# 
# 2. use below line to import MixedFusedRMSNorm
#    torch.nn.LayerNorm is slower than apex.FusedLayerNorm for shapes typical in NLP models. 
#    For example: (512, 16, 1024) with normalization over the last dimension is slower using torch.nn.LayerNorm
# from megatron.model.fused_layer_norm import MixedFusedRMSNorm as RMSNorm # for cuda
"""
class RMSLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, sequence_parallel=False):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super(RMSLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.sequence_parallel = sequence_parallel
        # set sequence parallelism flag on weight parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states
