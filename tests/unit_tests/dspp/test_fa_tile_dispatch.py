"""Per-device dispatch cache; no timing assertions under shared GPU load."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from megatron.core.kv_cache.fa_split import (
    prepare_backward, backward_dq, backward_dq_current, backward_dkv,
)
from megatron.core.kv_cache.fa_split_build import load_extension
from .test_fa_split import fa_forward, fa_backward
from .test_fa_split_regions import make_inputs, SCALE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_resource_queries_and_attributes_are_initialized_once():
    ext = load_extension()
    before = ext.dispatch_counters()
    q, k, v, dout = make_inputs(65, 194, 4, 2)
    out, lse, rng = fa_forward(q, k, v, SCALE)
    state = prepare_backward(q, k, v, out, dout, lse)
    first = ext.dispatch_counters()
    # Three additional kernels are initialized lazily only if CP-QO partials ran.
    assert first in ([1, 7], [1, 10])
    assert before in ([0, 0], first)
    info = ext.dispatch_info(q.get_device())
    expected_n = 128 if info[0] >= 144 * 1024 else 64
    assert info[2:] == [64, expected_n, 64, 64]
    ref = fa_backward(q, k, v, out, dout, lse, rng, SCALE)
    for _ in range(32):
        # Include repeated preparation, both D paths, H and deterministic D.
        state = prepare_backward(q, k, v, out, dout, lse)
        dq, ck, cv = backward_dq_current(state)
        hk, hv = backward_dkv(state, 'history', compact=True)
        for got, expected in zip((dq, torch.cat((hk, ck)), torch.cat((hv, cv))), ref):
            torch.testing.assert_close(got, expected, atol=.004, rtol=.01)
        for deterministic in (False, True):
            torch.testing.assert_close(backward_dq(state, deterministic=deterministic),
                                       ref[0], atol=.004, rtol=.01)
    torch.cuda.synchronize()
    # A different host thread must share the same device initialization.
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(ext.dispatch_info, [q.get_device()] * 8)) == [info] * 8
    assert ext.dispatch_counters() == first
