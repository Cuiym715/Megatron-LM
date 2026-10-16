"""Stage-A GPU correctness, not performance measurements or scheduling tests.

Nominal-input acceptance criteria (declared before running): split vs fused FA uses
atol=4e-3/rtol=1e-2; vs independent FP64 math uses atol=4e-2/rtol=4e-2
plus relative L2 <=1.2e-2. Both fused and split must pass the math criteria.
The amplitude-2 stress test requires exact equality with original FA plus the
same FP64 L2 bound: original FA itself fails the nominal elementwise criterion
on that input. The observed failure is retained in the implementation report.
"""
import json

import pytest
import torch

from megatron.core.kv_cache.fa_split import backward_dkv, backward_dq, prepare_backward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def inputs(length, heads, seed=1234, amplitude=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        (torch.randn(length, heads, 128, device="cuda", generator=gen) * amplitude)
        .to(torch.bfloat16)
        for _ in range(4)
    )


def fa_forward(q, k, v, scale):
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_forward

    out, lse, _, rng = _wrapped_flash_attn_forward(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        0.0, scale, True, -1, -1, 0.0, None, False,
    )
    return out.squeeze(0), lse.squeeze(0), rng


def fa_backward(q, k, v, out, dout, lse, rng, scale):
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_backward

    grads = tuple(torch.empty_like(t).unsqueeze(0) for t in (q, k, v))
    _wrapped_flash_attn_backward(
        dout.unsqueeze(0), q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        out.unsqueeze(0), lse.unsqueeze(0), *grads,
        0.0, scale, True, -1, -1, 0.0, None, False, rng,
    )
    return tuple(t.squeeze(0) for t in grads)


def math_reference(q, k, v, dout, scale):
    # Independent full attention, only for small correctness tests. No reuse of
    # the split backward's formulas, saved O/LSE, or its native kernels.
    leaves = [t.detach().double().requires_grad_(True) for t in (q, k, v)]
    qt, kt, vt = [t.transpose(0, 1) for t in leaves]
    scores = qt @ kt.transpose(-2, -1) * scale
    mask = torch.ones(q.size(0), q.size(0), device=q.device, dtype=torch.bool).tril()
    probs = scores.masked_fill(~mask, -torch.inf).softmax(dim=-1)
    out = (probs @ vt).transpose(0, 1)
    grads = torch.autograd.grad(out, leaves, dout.double())
    return out.detach(), grads


def errors(actual, expected):
    diff = actual.double() - expected.double()
    norm = torch.linalg.vector_norm(expected.double()).item()
    return {
        "max_abs": diff.abs().max().item(),
        "relative_l2": torch.linalg.vector_norm(diff).item() / norm if norm > 1e-8 else None,
    }


@pytest.mark.parametrize("length,heads,scale,amplitude", [
    (1, 1, 128**-0.5, 1.0), (17, 2, 128**-0.5, 1.0),
    (63, 4, 128**-0.5, 1.0), (64, 4, 128**-0.5, 1.0),
    (65, 4, 128**-0.5, 1.0), (127, 2, 128**-0.5, 1.0),
    (128, 4, 128**-0.5, 1.0), (129, 4, 128**-0.5, 1.0),
    (257, 16, 128**-0.5, 1.0), (513, 2, 128**-0.5, 1.0),
    (129, 2, 0.037, 1.0),
])
def test_split_matches_fused_and_fp64(length, heads, scale, amplitude):
    q, k, v, dout = inputs(length, heads, amplitude=amplitude)
    out, lse, rng = fa_forward(q, k, v, scale)
    fused = fa_backward(q, k, v, out, dout, lse, rng, scale)
    state = prepare_backward(q, k, v, out, dout, lse, softmax_scale=scale)
    # Calling dKV first proves it does not rely on dQ execution/workspace.
    dk, dv = backward_dkv(state)
    dq = backward_dq(state)
    split = dq, dk, dv
    ref_out, ref_grads = math_reference(q, k, v, dout, scale)
    torch.testing.assert_close(out.double(), ref_out, atol=0.04, rtol=0.04)
    torch.testing.assert_close(
        state.delta[:, :length], (out.float() * dout.float()).sum(-1).T,
        atol=1e-4, rtol=1e-5,
    )
    record = {"length": length, "heads": heads, "head_dim": 128,
              "scale": scale, "amplitude": amplitude, "gradients": {}}
    for name, got, base, ref in zip(("dq", "dk", "dv"), split, fused, ref_grads):
        assert got.dtype == q.dtype and got.shape == q.shape
        assert not got.requires_grad
        torch.testing.assert_close(got, base, atol=0.004, rtol=0.01)
        for value in (got, base):
            torch.testing.assert_close(value.double(), ref, atol=0.04, rtol=0.04)
            # Length=1 has mathematically zero dQ/dK; relative error is undefined.
            if ref.norm() > 1e-8:
                assert errors(value, ref)["relative_l2"] <= 0.012
            else:
                assert errors(value, ref)["max_abs"] <= 1e-5
        record["gradients"][name] = {
            "split_vs_fused": errors(got, base),
            "split_vs_fp64": errors(got, ref),
            "fused_vs_fp64": errors(base, ref),
        }
    print("SPLIT_CHECK " + json.dumps(record))


@pytest.mark.parametrize("seed", [1234, 2026, 17])
def test_stress_preserves_original_fa_accuracy(seed):
    q, k, v, dout = inputs(65, 2, seed=seed, amplitude=2.0)
    scale = 128**-0.5
    out, lse, rng = fa_forward(q, k, v, scale)
    fused = fa_backward(q, k, v, out, dout, lse, rng, scale)
    state = prepare_backward(q, k, v, out, dout, lse)
    split = backward_dq(state), *backward_dkv(state)
    _, refs = math_reference(q, k, v, dout, scale)
    record = {"length": 65, "heads": 2, "amplitude": 2.0, "seed": seed,
              "criterion": "exact original FA equality and FP64 relative L2 <= 0.012",
              "gradients": {}}
    for name, got, base, ref in zip(("dq", "dk", "dv"), split, fused, refs):
        torch.testing.assert_close(got, base, atol=0, rtol=0)
        assert errors(got, ref)["relative_l2"] <= 0.012
        record["gradients"][name] = {
            "split_vs_fused": errors(got, base), "split_vs_fp64": errors(got, ref),
            "fused_vs_fp64": errors(base, ref),
            "original_passes_nominal_elementwise_criterion": bool(
                ((base.double() - ref).abs() <= 0.04 + 0.04 * ref.abs()).all()),
        }
    print("SPLIT_CHECK " + json.dumps(record))


def test_repeat_zero_gradient_and_state_reuse():
    q, k, v, dout = inputs(129, 4)
    scale = 128**-0.5
    out, lse, _ = fa_forward(q, k, v, scale)
    state = prepare_backward(q, k, v, out, dout, lse)
    before = [t.clone() for t in state.tensors()]
    first = backward_dq(state), *backward_dkv(state)
    for _ in range(3):
        # Alternate order; each call must use independent output/accumulation.
        dk, dv = backward_dkv(state)
        dq = backward_dq(state)
        for got, expected in zip((dq, dk, dv), first):
            torch.testing.assert_close(got, expected, atol=0.001, rtol=0.001)
    for t, snapshot in zip(state.tensors(), before):
        torch.testing.assert_close(t, snapshot, atol=0, rtol=0)
    zero = prepare_backward(q, k, v, out, torch.zeros_like(dout), lse)
    for grad in (backward_dq(zero), *backward_dkv(zero)):
        assert torch.count_nonzero(grad).item() == 0
    # A second state with different upstream gradients must not overwrite state 1.
    for got, expected in zip((backward_dq(state), *backward_dkv(state)), first):
        torch.testing.assert_close(got, expected, atol=0.001, rtol=0.001)


def test_nondefault_stream_and_stream_contract():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        q, k, v, dout = inputs(65, 2)
        out, lse, rng = fa_forward(q, k, v, 128**-0.5)
        state = prepare_backward(q, k, v, out, dout, lse)
        got = backward_dq(state), *backward_dkv(state)
        expected = fa_backward(q, k, v, out, dout, lse, rng, 128**-0.5)
        for a, b in zip(got, expected):
            torch.testing.assert_close(a, b, atol=0.004, rtol=0.01)
    with pytest.raises(RuntimeError, match="same CUDA stream"):
        backward_dq(state)
    stream.synchronize()


@pytest.mark.parametrize("case", ["invalid_gqa", "unequal_length", "head_dim", "dtype", "layout",
                                  "lse", "scale", "causal", "dropout", "empty", "cpu"])
def test_reject_unsupported_contract(case):
    q, k, v, dout = inputs(17, 4)
    out, lse, _ = fa_forward(q, k, v, 128**-0.5)
    kwargs = {}
    if case == "invalid_gqa":
        k, v = k[:, :3].contiguous(), v[:, :3].contiguous()
    elif case == "unequal_length":
        k, v = k[:-1], v[:-1]
    elif case == "head_dim":
        q, k, v, out, dout = [t[..., :64].contiguous() for t in (q, k, v, out, dout)]
    elif case == "dtype":
        q, k, v, out, dout = [t.half() for t in (q, k, v, out, dout)]
    elif case == "layout":
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
    elif case == "lse":
        lse = lse.T.contiguous()
    elif case == "scale":
        kwargs["softmax_scale"] = float("nan")
    elif case == "causal":
        kwargs["causal"] = False
    elif case == "dropout":
        kwargs["dropout_p"] = 0.1
    elif case == "empty":
        q, k, v, out, dout = [t[:0] for t in (q, k, v, out, dout)]
    elif case == "cpu":
        q, k, v, out, dout, lse = [t.cpu() for t in (q, k, v, out, dout, lse)]
    with pytest.raises((RuntimeError, ValueError)):
        prepare_backward(q, k, v, out, dout, lse, **kwargs)


def test_reject_mutated_state():
    q, k, v, dout = inputs(17, 2)
    out, lse, _ = fa_forward(q, k, v, 128**-0.5)
    state = prepare_backward(q, k, v, out, dout, lse)
    state.delta.add_(1)
    with pytest.raises(RuntimeError, match="modified"):
        backward_dkv(state)


def test_multiple_devices_do_not_use_default_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires a second GPU")
    with torch.cuda.device(1):
        q, k, v, dout = inputs(65, 2)
        out, lse, rng = fa_forward(q, k, v, 128**-0.5)
        state = prepare_backward(q, k, v, out, dout, lse)
    # Current device differs from state device: CUDAGuard must select state GPU.
    with torch.cuda.device(0):
        actual = backward_dq(state), *backward_dkv(state)
        expected = fa_backward(q, k, v, out, dout, lse, rng, 128**-0.5)
        for a, b in zip(actual, expected):
            assert a.device.index == 1
            torch.testing.assert_close(a, b, atol=0.004, rtol=0.01)
