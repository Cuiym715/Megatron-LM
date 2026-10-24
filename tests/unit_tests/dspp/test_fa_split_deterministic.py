"""Opt-in deterministic dQ diagnostics. Normal training still defaults to False."""
import json

import pytest
import torch

from megatron.core.kv_cache.fa_split import (
    backward_dkv, backward_dq, backward_dq_current, prepare_backward,
)
from .test_fa_split import fa_forward, fa_backward
from .test_fa_split_regions import make_inputs, packed_forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("nq,nk,hq,hk", [
    (2048, 4096, 16, 16),  # Shape of the observed training dQ repeat mismatch.
    (65, 194, 8, 2),       # Nonaligned current/history boundary and GQA.
    (17, 17, 4, 1),        # Empty history and MQA; many idle dQ partitions.
])
def test_deterministic_dq_repeat_and_current(nq, nk, hq, hk):
    q, k, v, dout = make_inputs(nq, nk, hq, hk)
    scale = 128**-0.5
    out, lse, rng = fa_forward(q, k, v, scale)
    state = prepare_backward(q, k, v, out, dout, lse)
    before = [t.clone() for t in state.tensors()]
    first = backward_dq_current(state, deterministic=True)
    for _ in range(2):
        again = backward_dq_current(state, deterministic=True)
        for a, b in zip(first, again):
            assert torch.equal(a, b)
    assert torch.equal(backward_dq(state, deterministic=True), first[0])
    current = backward_dkv(state, "current", compact=True)
    assert all(torch.equal(a, b) for a, b in zip(first[1:], current))
    for got, base in zip((first[0], *backward_dkv(state)),
                         fa_backward(q, k, v, out, dout, lse, rng, scale)):
        torch.testing.assert_close(got, base, atol=.004, rtol=.01)
    for a, b in zip(state.tensors(), before):
        assert torch.equal(a, b)
    # Exercise omitted and explicit False without making a bitwise promise for
    # this intentionally nondeterministic path.
    for default in (backward_dq_current(state),
                    backward_dq_current(state, deterministic=False)):
        for got, expected in zip(default, first):
            torch.testing.assert_close(got, expected, atol=.004, rtol=.01)
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    splits = (sms + hq - 1) // hq
    print("DETERMINISTIC_CHECK " + json.dumps(dict(
        shape=[nq, nk, hq, hk], repeat_bitwise_equal=True, partitions=splits,
        extra_dq_workspace_bytes=(splits-1)*((nq+127)//128*128)*hq*128*4)))


def test_deterministic_varlen_dispatch():
    q, k, v, dout = make_inputs(82, 274, 8, 2)
    cq, ck = [0, 65, 82], [0, 194, 274]
    out, lse, _, _ = packed_forward(
        q, k, v, torch.tensor(cq, device='cuda', dtype=torch.int32),
        torch.tensor(ck, device='cuda', dtype=torch.int32), 128**-0.5)
    state = prepare_backward(q, k, v, out, dout, lse,
                             cu_seqlens_q=cq, cu_seqlens_k=ck)
    a = backward_dq_current(state, deterministic=True)
    b = backward_dq_current(state, deterministic=True)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert torch.equal(a[0], backward_dq(state, deterministic=True))
    children = [backward_dq_current(s, deterministic=True) for s in state.sequences]
    for i in range(3):
        assert torch.equal(a[i], torch.cat([p[i] for p in children]))
