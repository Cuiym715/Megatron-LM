# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import torch

try:
    import grouped_gemm
    from grouped_gemm import backend
except ImportError:
    grouped_gemm = None
    backend = None


def grouped_gemm_is_available():
    return grouped_gemm is not None


def assert_grouped_gemm_is_available():
    assert grouped_gemm_is_available(), (
        "Grouped GEMM is not available. Please run "
        "`pip install git+https://github.com/fanshiqing/grouped_gemm@main`."
    )


ops = grouped_gemm.ops if grouped_gemm_is_available() else None


class GroupedGemmRecompute(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, batch_sizes, trans_b, b_main_grad, recompute_func):
        assert torch.count_nonzero(batch_sizes) != 0, "Input batch_sizes should not be all zeros!"
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.recompute_func = recompute_func
        ctx.trans_b = trans_b
        ctx.b_main_grad = b_main_grad
        if recompute_func is not None:
            with torch.no_grad():
                a = recompute_func(a)
        return backend.gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        recompute_func = ctx.recompute_func
        if recompute_func is not None:
            recompute_input, b, batch_sizes = ctx.saved_tensors
            detach_inputs = recompute_input.detach()
            detach_inputs.requires_grad = recompute_input.requires_grad
            with torch.enable_grad():
                recompute_out = recompute_func(detach_inputs)
            a = recompute_out.detach()
        else:
            a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b
        b_main_grad = ctx.b_main_grad

        agrad = None
        if ctx.needs_input_grad[0]:
            agrad = backend.gmm(
                grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

        bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            if b_main_grad is None:
                bgrad = backend.gmm(
                    lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
            else:
                backend.gmm(
                    lhs, rhs, batch_sizes, trans_a=True, trans_b=False, c=b_main_grad, accumulate=True)
                bgrad = None

        if recompute_func is not None:
            torch.autograd.backward(recompute_out, agrad)
            agrad = detach_inputs.grad

        return agrad, bgrad, None, None, None, None

def gmm_recompute(a, b, batch_sizes, trans_b=False, b_main_grad=None, recompute_func=None):
    return GroupedGemmRecompute.apply(a, b, batch_sizes, trans_b, b_main_grad, recompute_func)


class GroupedGemmRecompute_More(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, trans_b, b_main_grad, recompute_func):
        ctx.save_for_backward(a, b)
        ctx.recompute_func = recompute_func
        ctx.trans_b = trans_b
        ctx.b_main_grad = b_main_grad
        ctx.b_shape = b.shape
        if recompute_func is not None:
            with torch.no_grad():
                a, batch_sizes = recompute_func(a)
        if a.nelement() == 0:
            b = b.view(-1, b.shape[-1])
            return torch.matmul(a, b)
        return backend.gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        recompute_func = ctx.recompute_func
        if recompute_func is not None:
            recompute_input, b = ctx.saved_tensors
            detach_inputs = recompute_input.detach()
            detach_inputs.requires_grad = recompute_input.requires_grad
            with torch.enable_grad():
                recompute_out, batch_sizes = recompute_func(detach_inputs)
            a = recompute_out.detach()
        else:
            assert False
        trans_b = ctx.trans_b
        b_main_grad = ctx.b_main_grad

        if a.nelement() == 0:
            agrad = torch.matmul(grad, b.view(-1, b.shape[-1]).transpose(0, 1))
            bgrad = torch.matmul(a.transpose(0, 1), grad)
            bgrad = bgrad.view(ctx.b_shape)
            torch.autograd.backward(recompute_out, agrad)
            agrad = detach_inputs.grad
            return agrad, bgrad, None, None, None

        agrad = None
        if ctx.needs_input_grad[0]:
            agrad = backend.gmm(
                grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

        bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            if b_main_grad is None:
                bgrad = backend.gmm(
                    lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
            else:
                backend.gmm(
                    lhs, rhs, batch_sizes, trans_a=True, trans_b=False, c=b_main_grad, accumulate=True)
                bgrad = None

        if recompute_func is not None:
            torch.autograd.backward(recompute_out, agrad)
            agrad = detach_inputs.grad

        return agrad, bgrad, None, None, None

def gmm_recompute_more(a, b, trans_b=False, b_main_grad=None, recompute_func=None):
    return GroupedGemmRecompute_More.apply(a, b, trans_b, b_main_grad, recompute_func)