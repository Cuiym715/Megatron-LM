import math
import torch
from torch.nn import LayerNorm
from functools import partial
from collections import OrderedDict
from megatron.model.enums import AttnMaskType
from megatron.model.fused_layer_norm import MixedFusedLayerNorm, MixedFusedRMSNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.utils import attention_mask_func
from megatron.fused_kernels import load
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from enum import Enum

FWD_HOOK = []
BWD_HOOK = []
TENSOR_HOOK = []


try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN
    from apex.contrib.layer_norm.layer_norm import FastRMSNormFN
    HAVE_PERSIST_LAYER_NORM = True
except:
    HAVE_PERSIST_LAYER_NORM = False

from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction, FusedRMSNormAffineFunction


class HookTypeEnum(int, Enum):
    """ Enum class for hook types """
    FWD = 0
    BWD = 1
    TENSOR = 2


def to_cpu_deatch(tuple_tensor):
    if isinstance(tuple_tensor, tuple):
        res = []
        for tensor in tuple_tensor:
            if tensor is None:
                res.append(None)
            else:
                res.append(tensor.cpu().detach().data)
        return tuple(res) if len(res) > 1 else res[0]
    elif isinstance(tuple_tensor, torch.Tensor):
        return tuple_tensor.cpu().detach().data
    else:
        raise NotImplementedError



class GPUTimer:
    def __init__(self, stream):
        self.start_ = torch.cuda.Event(enable_timing=True)
        self.stop_ = torch.cuda.Event(enable_timing=True)
        self.stream_ = stream

    def start(self):
        self.stream_.record_event(self.start_)

    def stop(self):
        self.stream_.record_event(self.stop_)

    def sync(self):
        self.stream_.synchronize()

    def millis(self):
        return self.start_.elapsed_time(self.stop_)


def fwd_fn(self, input, output):
    # print(f"FWD_HOOK_FUNCTION for {self} !!!")
    self.activation = to_cpu_deatch(output)


def bwd_fn(self, grad_input, grad_output):
    # print(f"BWD_HOOK_FUNCTION for {self} !!!")
    self.activation_grad = to_cpu_deatch(grad_input)


def tensor_fn(param, grad):
    # print(f"TENSOR_HOOK_FUNCTION!!!")
    param.numel_data = to_cpu_deatch(grad)


def post_grad_compute_fn(param, none_tuple, grad):
    # print(f"TENSOR_HOOK_FUNCTION!!!")
    # print("-"* 10, "Rank", torch.distributed.get_rank(), param, grad[0].shape)
    param.numel_data = to_cpu_deatch(grad)


def register_fwd_hook(model, fwd_fn):
    for child_name, module in model.named_children():
        try:
            next(module.named_children())
            is_leaf_module = False
        except:
            is_leaf_module = True
        if is_leaf_module:
            print("Register fwd hook for ", child_name, module)
            FWD_HOOK.append(module.register_forward_hook(fwd_fn))
        else:
            register_fwd_hook(module, fwd_fn)


def register_bwd_hook(model, bwd_fn):
    # print(model)
    
    # for child_name, module in model.named_children():
    #     try:
    #         next(module.named_children())
    #         is_leaf_module = False
    #     except:
    #         is_leaf_module = True
        # if is_leaf_module:
            # print("Register bwd hook for ", child_name, module)
            BWD_HOOK.append(model.register_full_backward_hook(bwd_fn))
        # else:
            # register_bwd_hook(module, bwd_fn)


def register_tensor_hook(model, tensor_fn):
    for n, param in model.named_parameters():
        if param.requires_grad:
            print("Register tensor hook for ", n, param.shape)
            TENSOR_HOOK.append(param.register_hook(partial(tensor_fn, param)))


def register_post_grad_compute_hook(model, tensor_fn):
    """
    https://github.com/pytorch/pytorch/blob/00cb184512f3a636d87793f46d3f9c7fea406b25/torch/distributed/fsdp/fully_sharded_data_parallel.py#L2825-L2835
    """
    for n, param in model.named_parameters():
        if param.requires_grad:
            print("Register post-grad-compute hook for ", n, param.shape)
            # Get a grad_fn on p_tmp.
            param_tmp = param.expand_as(param)
            assert (
                param_tmp.grad_fn is not None
            ), "p_tmp grad_fn should not be None, it is used to access \
                p's AccumulateGrad object and register post hook on it."
            # Gets its AccumulateGrad object.
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            # handle = grad_acc.register_hook(functools.partial(self._post_backward_hook, p))
            TENSOR_HOOK.append(grad_acc.register_hook(partial(tensor_fn, n)))


register_fwd_hook_fn = partial(register_fwd_hook, fwd_fn=fwd_fn)
register_bwd_hook_fn = partial(register_bwd_hook, bwd_fn=bwd_fn)
register_tensor_hook_fn = partial(register_tensor_hook, tensor_fn=tensor_fn)
register_post_grad_compute_hook_fn = partial(register_post_grad_compute_hook,
                                             tensor_fn=post_grad_compute_fn)


def wrap_model_with_hook(*hook_type, model):
    if HookTypeEnum.FWD in hook_type:
        register_fwd_hook_fn(model=model)
    if HookTypeEnum.BWD in hook_type:
        register_bwd_hook_fn(model=model)
    if HookTypeEnum.TENSOR in hook_type:
        register_tensor_hook_fn(model=model)



def test_rms_norm_forward(hidden_dim = 8192, 
                          no_persist_rms_norm = True,
                          compute_precision = torch.float16):
    bs_test_cases = [1, 2, 4, 8]
    seq_len_test_cases = [1024, 2048, 4096, 8192]

    for bs in bs_test_cases:
        for seq_len in seq_len_test_cases:
            embedding_output = torch.rand(bs, seq_len, hidden_dim).cuda().to(compute_precision)

            fused_rmsnorm_layer = (
                MixedFusedRMSNorm(normalized_shape=embedding_output.size(-1), no_persist_rms_norm = no_persist_rms_norm).cuda().to(compute_precision)
            )

            torch_rmsnorm_layer = (
                LlamaRMSNorm(hidden_size=embedding_output.size(-1)).cuda().to(compute_precision)
            )
            weight = torch.rand_like(fused_rmsnorm_layer.weight)
            
            fused_rmsnorm_layer.weight.data.copy_(weight.data)
            torch_rmsnorm_layer.weight.data.copy_(weight.data)

            fused_output = fused_rmsnorm_layer(embedding_output)
            torch_output = torch_rmsnorm_layer(embedding_output)
            test_result = (fused_output.float() - torch_output.float()).abs()

            while test_result.dim() != 1:
                test_result = test_result.mean(dim=-1)

            abs_diff = test_result.mean(dim=-1)
            max_diff = test_result.max()
            print(f"{bs}, {seq_len}, {hidden_dim}, {abs_diff <= 1e-3}, {abs_diff}, {compute_precision}, {max_diff}")


def test_rms_norm_backward(hidden_dim = 8192, 
                          no_persist_rms_norm = True,
                          compute_precision = torch.float16):
    
    bs_test_cases = [1, 2, 4, 8]
    seq_len_test_cases = [1024, 2048, 4096, 8192]

    for bs in bs_test_cases:
        for seq_len in seq_len_test_cases:
            idt1 = torch.nn.Linear(hidden_dim, hidden_dim).cuda().to(compute_precision)
            idt2 = torch.nn.Linear(hidden_dim, hidden_dim).cuda().to(compute_precision)
            embedding_output_fused = torch.rand(bs, seq_len, hidden_dim, requires_grad = True).cuda().to(compute_precision)
            embedding_output_torch = embedding_output_fused.clone()


            embedding_output_fused_identity = idt1(embedding_output_fused)
            embedding_output_torch_identity = idt2(embedding_output_torch)

            fused_rmsnorm_layer = (
                MixedFusedRMSNorm(normalized_shape=embedding_output_fused.size(-1), no_persist_rms_norm = no_persist_rms_norm).cuda().to(compute_precision)
            )

            torch_rmsnorm_layer = (
                LlamaRMSNorm(hidden_size=embedding_output_torch.size(-1)).cuda().to(compute_precision)
            )
            weight = torch.rand_like(fused_rmsnorm_layer.weight)
            
            fused_rmsnorm_layer.weight.data.copy_(weight.data)
            torch_rmsnorm_layer.weight.data.copy_(weight.data)

            register_bwd_hook(fused_rmsnorm_layer, bwd_fn=bwd_fn)
            register_bwd_hook(torch_rmsnorm_layer, bwd_fn=bwd_fn)

            fused_output = fused_rmsnorm_layer(embedding_output_fused_identity)
            torch_output = torch_rmsnorm_layer(embedding_output_torch_identity)
            
            
            l1loss = torch.nn.L1Loss()
            fake_label = torch.randn_like(fused_output)

            l1_loss_fused = l1loss(fused_output, fake_label)
            l1_loss_torch = l1loss(torch_output, fake_label)
            
            l1_loss_fused.backward()
            l1_loss_torch.backward()

            test_result = (fused_rmsnorm_layer.activation_grad.float() - torch_rmsnorm_layer.activation_grad.float()).abs()
            abs_diff = test_result.mean()
            max_diff = test_result.max()
            print(f"act, {bs}, {seq_len}, {hidden_dim}, {abs_diff <= 1e-3}, {abs_diff}, {compute_precision}, {max_diff}")


            test_result = (torch_rmsnorm_layer.weight.grad.float() - fused_rmsnorm_layer.weight.grad.float()).abs()
            abs_diff = test_result.mean()
            max_diff = test_result.max()
            print(f"weight, {bs}, {seq_len}, {hidden_dim}, {abs_diff <= 1e-3}, {abs_diff}, {compute_precision}, {max_diff}")

def benchmark_rms_norm_forward(hidden_dim = 8192, 
                          no_persist_rms_norm = True,
                          compute_precision = torch.float16, 
                          warmup_iter = 50, benchmark_iter = 50):


    # print(embedding_output.shape)
    
    
    stream = torch.cuda.Stream()

    bs_test_cases = [1, 2, 4, 8]
    seq_len_test_cases = [1024, 2048, 4096, 8192]
    with torch.cuda.stream(stream):
        timer = GPUTimer(stream)
        for bs in bs_test_cases:
            for seq_len in seq_len_test_cases:
                embedding_output = torch.rand(bs, seq_len, hidden_dim).cuda().to(compute_precision)
                fused_rmsnorm_layer = (
                    MixedFusedRMSNorm(normalized_shape=embedding_output.size(-1), 
                    no_persist_rms_norm = no_persist_rms_norm).cuda().to(compute_precision)
                )
                weight = torch.rand_like(fused_rmsnorm_layer.weight)
                fused_rmsnorm_layer.weight.data.copy_(weight.data)

                for _ in range(0, warmup_iter):
                    fused_rmsnorm_layer(embedding_output)

                timer.start()    
                for _ in range(0, benchmark_iter):
                    fused_rmsnorm_layer(embedding_output)

                timer.stop()
                timer.sync()
                ms_fwd = timer.millis() / benchmark_iter
                print(f"{no_persist_rms_norm}, {bs}, {seq_len}, {compute_precision}, {ms_fwd}")


def benchmark_rms_norm_backward(hidden_dim = 8192, 
                          no_persist_rms_norm = True,
                          compute_precision = torch.float16, 
                          warmup_iter = 50, benchmark_iter = 50):


    # print(embedding_output.shape)
    
    
    stream = torch.cuda.Stream()

    bs_test_cases = [1, 2, 4, 8]
    seq_len_test_cases = [1024, 2048, 4096, 8192]
    with torch.cuda.stream(stream):
        timer = GPUTimer(stream)
        for bs in bs_test_cases:
            for seq_len in seq_len_test_cases:
                embedding_output = torch.rand(bs, seq_len, hidden_dim).cuda().to(compute_precision)
                
                fused_rmsnorm_layer = (
                    MixedFusedRMSNorm(normalized_shape=embedding_output.size(-1), 
                    no_persist_rms_norm = no_persist_rms_norm).cuda().to(compute_precision)
                )
                weight = torch.rand_like(fused_rmsnorm_layer.weight)
                fused_rmsnorm_layer.weight.data.copy_(weight.data)

                out = fused_rmsnorm_layer(embedding_output)
                fake_grad = torch.randn_like(out)
                for _ in range(0, warmup_iter):
                    out = fused_rmsnorm_layer(embedding_output)
                    fake_grad = torch.randn_like(out)
                    out.backward(fake_grad)

                timer.start()   
                times = [] 
                for _ in range(0, benchmark_iter):
                    out = fused_rmsnorm_layer(embedding_output)
                    fake_grad = torch.randn_like(out)
                    timer.start()
                    out.backward(fake_grad)
                    timer.stop()
                    timer.sync()
                    ms_bwd = timer.millis() 
                    times.append(ms_bwd)
                ms_bwd_avg = sum(times) / len(times)
                print(f"{no_persist_rms_norm}, {bs}, {seq_len}, {compute_precision}, {ms_bwd_avg}")




if __name__ == "__main__":
    test_rms_norm_forward(no_persist_rms_norm = False, compute_precision = torch.float16)
    test_rms_norm_forward(no_persist_rms_norm = False, compute_precision = torch.bfloat16)

    test_rms_norm_backward(no_persist_rms_norm = False, compute_precision = torch.float16)
    test_rms_norm_backward(no_persist_rms_norm = False, compute_precision = torch.bfloat16)

    benchmark_rms_norm_forward(no_persist_rms_norm = True)
    benchmark_rms_norm_forward(no_persist_rms_norm = True, compute_precision = torch.bfloat16)
    benchmark_rms_norm_forward(no_persist_rms_norm = False)
    benchmark_rms_norm_forward(no_persist_rms_norm = False, compute_precision = torch.bfloat16)

    benchmark_rms_norm_backward(no_persist_rms_norm = True)
    benchmark_rms_norm_backward(no_persist_rms_norm = True, compute_precision = torch.bfloat16)
    benchmark_rms_norm_backward(no_persist_rms_norm = False)
    benchmark_rms_norm_backward(no_persist_rms_norm = False, compute_precision = torch.bfloat16)


