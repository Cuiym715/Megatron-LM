import flash_attn
import inspect


version_major, version_minor = [int(x) for x in flash_attn.__version__.split(".")[:2]]
support_alibi = "alibi_bias_max" in inspect.signature(flash_attn.flash_attn_func).parameters


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    return_attn_probs=False,
    *,
    alibi_bias_max,
    tp_world_size,
    tp_rank,
):
    if version_major == 2 and version_minor == 3 and not support_alibi:
        assert not alibi_bias_max, "this version of flash-attn does not support alibi"
        return flash_attn.flash_attn_interface.flash_attn_func(
            q, k, v, dropout_p, softmax_scale, causal, window_size, return_attn_probs)
    elif version_major == 2 and version_minor == 2 and not support_alibi:
        assert window_size == (-1, -1), "flash-attn 2.2 does not support window_size"
        assert not alibi_bias_max, "this version of flash-attn does not support alibi"
        return flash_attn.flash_attn_interface.flash_attn_func(
            q, k, v, dropout_p, softmax_scale, causal, return_attn_probs)
    elif version_major == 2 and version_minor == 2 and support_alibi:
        assert window_size == (-1, -1), "flash-attn 2.2 does not support window_size"
        return flash_attn.flash_attn_interface.flash_attn_func(
            q, k, v, dropout_p, softmax_scale, causal, return_attn_probs,
            alibi_bias_max=alibi_bias_max, tp_world_size=tp_world_size, tp_rank=tp_rank)
    else:
        raise NotImplementedError("cannot dispatch flash_attn_func")


def _flash_attn_forward(q, k, v, dropout_p, softmax_scale, causal, window_size, return_softmax,
                        *, alibi_bias_max, tp_world_size, tp_rank):
    if version_major == 2 and version_minor == 3 and not support_alibi:
        assert not alibi_bias_max, "this version of flash-attn does not support alibi"
        return flash_attn.flash_attn_interface._flash_attn_forward(
            q, k, v, dropout_p, softmax_scale, causal, window_size, return_softmax)
    elif version_major == 2 and version_minor == 2 and not support_alibi:
        assert window_size == (-1, -1), "flash-attn 2.2 does not support window_size"
        assert not alibi_bias_max, "this version of flash-attn does not support alibi"
        return flash_attn.flash_attn_interface._flash_attn_forward(
            q, k, v, dropout_p, softmax_scale, causal, return_softmax)
    elif version_major == 2 and version_minor == 2 and support_alibi:
        assert window_size == (-1, -1), "flash-attn 2.2 does not support window_size"
        return flash_attn.flash_attn_interface._flash_attn_forward(
            q, k, v, dropout_p, softmax_scale, causal, return_softmax,
            alibi_bias_max=alibi_bias_max, tp_world_size=tp_world_size, tp_rank=tp_rank)
    else:
        raise NotImplementedError(f"cannot dispatch _flash_attn_forward to "
                                  f"flash-attn=={flash_attn.__version__} support_alibi={support_alibi}")


def _flash_attn_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    rng_state=None,
    *,
    alibi_bias_max,
    tp_world_size,
    tp_rank,
):
    if version_major == 2 and version_minor == 3 and not support_alibi:
        assert not alibi_bias_max, "this version of flash-attn does not support alibi"
        return flash_attn.flash_attn_interface._flash_attn_backward(
            dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal, window_size, rng_state=rng_state)
    elif version_major == 2 and version_minor == 2 and not support_alibi:
        assert window_size == (-1, -1), "flash-attn 2.2 does not support window_size"
        assert not alibi_bias_max, "this version of flash-attn does not support alibi"
        return flash_attn.flash_attn_interface._flash_attn_backward(
            dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal, rng_state=rng_state)
    elif version_major == 2 and version_minor == 2 and support_alibi:
        assert window_size == (-1, -1), "flash-attn 2.2 does not support window_size"
        return flash_attn.flash_attn_interface._flash_attn_backward(
            dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal=causal, rng_state=rng_state,
            alibi_bias_max=alibi_bias_max, tp_world_size=tp_world_size, tp_rank=tp_rank)
    else:
        raise NotImplementedError(f"cannot dispatch _flash_attn_backward to "
                                  f"flash-attn=={flash_attn.__version__} support_alibi={support_alibi}")
