"""Experimental FA2 backward split with current/history regions and GQA.

BF16 [tokens, heads, 128], full bottom-right causal attention, dropout=0.
LSE is natural-log FP32 [Hq, total_q] from the matching full-context forward.
Optional cumulative sequence lengths dispatch each independent sequence to FA2
kernels; this is not a fused varlen launcher. Packed autograd uses this module
when explicitly enabled. No higher-order autograd or cross-stream use.
"""
from dataclasses import dataclass, field
import operator

import torch

from .fa_split_build import load_extension


class _StateContract:
    def _check(self):
        if torch.cuda.current_stream(self.q.device).cuda_stream != self._stream:
            raise RuntimeError("Split backward requires prepare/dQ/dKV on the same CUDA stream")
        if tuple(t._version for t in self.tensors()) != self._versions:
            raise RuntimeError("Backward state tensors were modified after prepare")


@dataclass(frozen=True)
class SplitBackwardState(_StateContract):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    dout: torch.Tensor
    lse: torch.Tensor
    delta: torch.Tensor
    softmax_scale: float
    _stream: int = field(repr=False)
    _versions: tuple[int, ...] = field(repr=False)

    def tensors(self):
        return self.q, self.k, self.v, self.out, self.dout, self.lse, self.delta


@dataclass(frozen=True)
class VarlenSplitBackwardState(_StateContract):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    dout: torch.Tensor
    lse: torch.Tensor
    sequences: tuple[SplitBackwardState, ...]
    cu_seqlens_q: tuple[int, ...]
    cu_seqlens_k: tuple[int, ...]
    _metadata_tensors: tuple[torch.Tensor, ...] = field(repr=False)
    _stream: int = field(repr=False)
    _versions: tuple[int, ...] = field(repr=False)

    def tensors(self):
        return (self.q, self.k, self.v, self.out, self.dout, self.lse,
                *self._metadata_tensors)

    def _check_metadata(self):
        # Q/K/V/O/dO views share version counters with the child states. Packed
        # LSE can be copied into contiguous child tensors, so retain this guard.
        if (self.lse._version != self._versions[5]
                or tuple(t._version for t in self._metadata_tensors) != self._versions[6:]):
            raise RuntimeError("Backward state tensors were modified after prepare")


def _cu_values(cu, total, device):
    if isinstance(cu, torch.Tensor):
        if (cu.ndim != 1 or cu.dtype not in (torch.int32, torch.int64)
                or not cu.is_contiguous() or cu.is_inference()
                or (cu.device.type != "cpu" and cu.device != device)):
            raise ValueError("cu_seqlens must be contiguous versioned int32/int64 vectors on CPU or Q device")
        # One explicit setup synchronization for CUDA metadata. Store immutable
        # host coordinates; no mutable global batch metadata is read at launch.
        values = cu.tolist()
    elif isinstance(cu, (tuple, list)):
        values = cu
    else:
        raise ValueError("cu_seqlens must be an integer tuple/list or tensor")
    try:
        if any(isinstance(x, bool) for x in values):
            raise TypeError("boolean offset")
        values = tuple(operator.index(x) for x in values)
    except TypeError as exc:
        raise ValueError("cu_seqlens offsets must be integers") from exc
    if (len(values) < 2 or values[0] != 0 or values[-1] != total
            or any(a >= b for a, b in zip(values, values[1:]))):
        raise ValueError("cu_seqlens must start at zero, strictly increase and end at total tokens")
    return values


@torch.no_grad()
def prepare_backward(q, k, v, out, dout, lse, *, softmax_scale=None,
                     causal=True, dropout_p=0.0, cu_seqlens_q=None,
                     cu_seqlens_k=None) -> SplitBackwardState | VarlenSplitBackwardState:
    """Prepare shared delta, retaining tensors and immutable sequence boundaries.

    Caller supplies matching O/LSE/scale from full-context attention, without
    softcap, ALiBi or local windows. Each sequence's Q is the suffix of its KV;
    history length is Sk-Sq >= 0. Empty sequences are rejected (empty history
    is supported). CUDA cu_seqlens cause a setup synchronization. No tensor may
    be modified until the state is no longer used.
    """
    if causal is not True or dropout_p != 0.0:
        raise ValueError("Split backward requires causal=True and dropout_p=0")
    inputs = (q, k, v, out, dout, lse)
    if any(t.is_inference() for t in inputs):
        raise ValueError("Split backward requires versioned tensors, not inference-mode tensors")
    scale = 128**-0.5 if softmax_scale is None else float(softmax_scale)
    if cu_seqlens_q is None and cu_seqlens_k is None:
        delta = load_extension().prepare(*inputs, scale)
        tensors = (*inputs, delta)
        return SplitBackwardState(
            *tensors, scale, torch.cuda.current_stream(q.device).cuda_stream,
            tuple(t._version for t in tensors),
        )
    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError("provide both cu_seqlens_q and cu_seqlens_k")
    if (not q.is_cuda or any(t.device != q.device for t in inputs)
            or any(t.ndim != 3 or not t.is_contiguous() for t in inputs[:5])
            or q.dtype != torch.bfloat16 or q.size(1) <= 0 or q.size(2) != 128
            or any(t.dtype != q.dtype for t in inputs[:5])
            or out.shape != q.shape or dout.shape != q.shape
            or k.shape != v.shape or k.size(1) <= 0 or k.size(2) != 128
            or q.size(1) % k.size(1) != 0
            or lse.shape != (q.size(1), q.size(0)) or lse.dtype != torch.float32
            or not lse.is_contiguous()):
        raise ValueError("invalid packed Q/K/V/O/dO or LSE shapes, layout, device or dtype")
    cq = _cu_values(cu_seqlens_q, q.size(0), q.device)
    ck = _cu_values(cu_seqlens_k, k.size(0), q.device)
    if len(cq) != len(ck) or any(ck[i+1]-ck[i] < cq[i+1]-cq[i] for i in range(len(cq)-1)):
        raise ValueError("require equal sequence counts and Sk >= Sq in each sequence")
    sequences = tuple(
        prepare_backward(q[qa:qb], k[ka:kb], v[ka:kb], out[qa:qb], dout[qa:qb],
                         lse[:, qa:qb].contiguous(), softmax_scale=scale)
        for qa, qb, ka, kb in zip(cq, cq[1:], ck, ck[1:])
    )
    metadata = tuple(t for t in (cu_seqlens_q, cu_seqlens_k) if isinstance(t, torch.Tensor))
    return VarlenSplitBackwardState(
        *inputs, sequences, cq, ck, metadata,
        torch.cuda.current_stream(q.device).cuda_stream,
        tuple(t._version for t in (*inputs, *metadata)),
    )


@torch.no_grad()
def backward_dq(state: SplitBackwardState | VarlenSplitBackwardState) -> torch.Tensor:
    """Complete dQ over ALL legal KV positions, independent of dKV selection."""
    if isinstance(state, VarlenSplitBackwardState):
        # Child checks cover all shared tensor versions and copied LSE/delta.
        # Only CUDA/CPU metadata tensors need an additional outer check.
        state._check_metadata()
        parts = [backward_dq(s) for s in state.sequences]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    state._check()
    return load_extension().dq(*state.tensors(), state.softmax_scale)


@torch.no_grad()
def backward_dq_current(state: SplitBackwardState | VarlenSplitBackwardState
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One scan: all-context dQ and compact current-only dK/dV.

    Each independent packed sequence has its own bottom-right causal boundary.
    Nonaligned boundary tiles can compute discarded history partials; no
    history output is returned or routed. Saved state remains usable by H.
    """
    if isinstance(state, VarlenSplitBackwardState):
        state._check_metadata()
        parts = [backward_dq_current(s) for s in state.sequences]
        if len(parts) == 1:
            return parts[0]
        return tuple(torch.cat([p[i] for p in parts], dim=0) for i in range(3))
    state._check()
    return tuple(load_extension().dq_current(*state.tensors(), state.softmax_scale))


@torch.no_grad()
def backward_dkv(state: SplitBackwardState | VarlenSplitBackwardState,
                 region: str = "all", *, compact: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 'all', 'current' or 'history' dKV; return compact KV-head tensors.

    Results have the full input KV shape, with exact zeros outside the selected
    region. Thus current+history reconstructs all, with no uninitialized rows.
    Only intersecting 64-token tiles execute; a nonaligned boundary can incur
    surplus tile computation whose results are discarded. K/V are not expanded
    by head; GQA partial gradients are summed once, following upstream FA2.

    compact=True returns only the selected rows, concatenated by sequence in
    varlen mode. It avoids the full-length zero output, but preserves the same
    tile computation, partial-gradient workspace and reduction precision.
    """
    if region not in ("all", "current", "history"):
        raise ValueError("region must be 'all', 'current' or 'history'")
    if isinstance(state, VarlenSplitBackwardState):
        state._check_metadata()
        pairs = [backward_dkv(s, region, compact=compact) for s in state.sequences]
        if compact:
            pairs = [p for p in pairs if p[0].size(0)] or pairs[:1]
        if len(pairs) == 1:
            return pairs[0]
        return tuple(torch.cat([p[i] for p in pairs], dim=0) for i in (0, 1))
    state._check()
    history = state.k.size(0) - state.q.size(0)
    begin = history if region == "current" else 0
    end = history if region == "history" else state.k.size(0)
    return tuple(load_extension().dkv(*state.tensors(), state.softmax_scale, begin, end, compact))
