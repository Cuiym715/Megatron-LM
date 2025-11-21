import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple
from typing_extensions import Self

import torch
import torch.distributed as dist
from megatron.core import parallel_state


class Cache(ABC):
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """
    @abstractmethod
    def update(self, layer_idx: int, key_value: List[torch.Tensor]) -> List[torch.Tensor]:
        """Updates the cache with the new `key_value` for the layer `layer_idx`."""
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    # @abstractmethod
    # def get_seq_length(self, layer_idx: Optional[int]=None) -> int:
    #     """Returns the sequence length of the cached states. A layer index can be optionally passed."""
    #     raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    @abstractmethod
    def detach(self) -> Self:
        """Detach key and value states for future autograd."""
        raise NotImplementedError("Make sure to implement `detach` in a subclass.")

    @abstractmethod
    def dump(self, layer_idx: Optional[int]=None) -> List[torch.Tensor]:
        """Dump key and value states as a list of tensors."""
        raise NotImplementedError("Make sure to implement `dump` in a subclass.")


class FakeCache(Cache):
    def __bool__(self) -> bool:
        return False

    def update(self, layer_idx: int, key_value: List[torch.Tensor]) -> List[torch.Tensor]:
        """Updates the cache for the layer `layer_idx`."""
        return key_value

    # def get_seq_length(self, layer_idx: Optional[int]=None) -> int:
    #     """Returns the sequence length of the cached states. A layer index can be optionally passed."""
    #     return 0

    def detach(self) -> Self:
        """Detach key and value states for future autograd."""
        return self

    def dump(self, layer_idx: Optional[int]=None) -> List[torch.Tensor]:
        """Dump key and value states as a list of tensors."""
        return []


@dataclass
class Growth:
    growth_rate: float = 0
    growth_seq_length: int = 0

    def extend(self, past: torch.Tensor, req_size: int) -> None:
        cur_size = past.untyped_storage().nbytes()
        if cur_size < req_size:
            tgt_size = max(req_size,
                           math.ceil(cur_size * self.growth_rate),
                           cur_size + self.growth_seq_length * past.shape[1:].numel() * past.element_size())
            past.untyped_storage().resize_(tgt_size)
            def shrink(param):
                param.untyped_storage().resize_(cur_size)
            past.register_post_accumulate_grad_hook(shrink)
        return past


class ConcatInplace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, past: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        ctx.length = past.shape[0]
        ctx.stride = None if curr.is_contiguous() else curr.stride()
        storage = past.untyped_storage()
        shape = list(curr.shape)
        shape[0] += ctx.length
        concat = past.new_empty(0).set_(storage, 0, shape)
        concat[ctx.length:] = curr
        return concat

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_past, grad_curr = grad[:ctx.length], grad[ctx.length:]
        if ctx.stride is not None:
            grad_curr = grad_curr.new_empty_strided(grad_curr.shape, ctx.stride).copy_(grad_curr)
        return grad_past, grad_curr


def concat_inplace(past: Optional[torch.Tensor], curr: torch.Tensor, growth: Growth) -> torch.Tensor:
    """Concatenate two Tensors in-place with first Tensor's Storage. The first Tensor can be None."""
    if past is None:
        past = curr.new_empty(0, *curr.shape[1:])
    past.requires_grad = True
    concat = ConcatInplace.apply(growth.extend(past, past.nbytes + curr.nbytes), curr)
    setattr(concat, "_kv_cache_impl", "extended")
    return concat


class ChunkedTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, tensors: List[torch.Tensor], dim: int=0):
        assert tensors
        for t in tensors:
            if t.size(dim):
                break
            tensors.pop(0)
        size = list(t.size())
        size[dim] = sum(t.size(dim) for t in tensors)
        strides = list(t.stride())
        for i in range(len(strides)):
            if strides[i] > strides[dim]:
                strides[i] = sum(t.stride(i) for t in tensors)
        kwargs = {}
        kwargs["dtype"] = t.dtype
        kwargs["layout"] = t.layout
        kwargs["device"] = t.device
        return torch.Tensor._make_wrapper_subclass(cls, size, strides, **kwargs)

    def __init__(self, tensors: List[torch.Tensor], dim: int=0):
        self._tensors = tensors
        self._dim = dim

    def __repr__(self):
        return super().__repr__(tensor_contents=f"tensors={self._tensors}, dim={self._dim}")

    def __getitem__(self, slice) -> Self:
        return ChunkedTensor(self._tensors[slice], self._dim)

    def prefix(self, n=None) -> Self:
        return ChunkedTensor(self._tensors[:n], self._dim)

    def suffix(self, n=None) -> Self:
        return ChunkedTensor(self._tensors[-n:], self._dim)

    def concat(self) -> torch.Tensor:
        if len(self._tensors) == 1:
            return self._tensors[0]
        # return torch.cat(self._tensors, self._dim)
        import fast_cat_cuda
        concat = torch.empty_like(self)
        t = self._tensors[0]
        inner = t.stride(self._dim) * t.size(self._dim)
        outer = t.numel() // inner
        fast_cat_cuda.cat([t.data_ptr() for t in self._tensors], concat.data_ptr(),
                          outer, inner * self.element_size(), torch.cuda.current_stream().cuda_stream)
        return concat

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is torch.empty_like:
            assert not kwargs
            self = args[0]
            return torch.empty_strided(self.size(), self.stride(), dtype=self.dtype, device=self.device)
        return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.detach.default:
            assert not kwargs
            self = args[0]
            return cls(self._tensors, self._dim)
        elif func is torch.ops.aten.transpose.int:
            assert not kwargs
            self, dim0, dim1 = args
            tensors = [t.transpose(dim0, dim1) for t in self._tensors]
            dim = self._dim
            if dim in (dim0, dim1):
                dim ^= dim0 ^ dim1
            return cls(tensors, dim)
        elif func is torch.ops.aten.cat.default:
            assert not kwargs
            past, curr = args[0]
            return cls(past._tensors + [curr], past._dim)
        raise NotImplementedError(func, types, args, kwargs)
        # return super().__torch_dispatch__(func, types, args, kwargs)
        # kwargs = {} if kwargs is None else kwargs
        # def unwrap(t):
        #     return t.concat() if isinstance(t, cls) else t
        # return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))


def concat_chunked(past: Optional[ChunkedTensor], curr: torch.Tensor) -> ChunkedTensor:
    """Concatenate two Tensors into a ChunkedTensor. The first Tensor can be None."""
    if past is None:
        past = ChunkedTensor([curr.new_empty(0, *curr.shape[1:])])
    past.requires_grad = True
    # grad_past has shared storage with grad_concat, clone it to release the storage.
    past.register_hook(lambda grad: grad.clone())
    concat = torch.cat([past, curr])
    setattr(concat, "_kv_cache_impl", "chunked")
    return concat


class KVCache(Cache):
    """
    A cache that stores the Key and Value states as a list of tensors, one for each layer.
    The expected shape for each tensor is `[seq_length, batch_size, num_heads, head_dim]`.
    """
    def __init__(self,
                 dtype: torch.dtype,
                 shape: Optional[Tuple[int, int, int]],
                 growth: Optional[Growth]=None) -> None:
        self.dtype = dtype
        self.shape = shape
        self.cached_kv: Dict[int, List[torch.Tensor]] = {}
        self.growth = growth
        self.concat = partial(concat_inplace, growth=growth) if growth else concat_chunked
        self.ctx_pair = None

    def update(self, layer_idx: int, key_value: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Updates the cache with the new `key_value` for the layer `layer_idx`.

        Parameters:
            layer_idx (`int`):
                The index of the layer to cache the states for.
            key_value (`List[torch.Tensor]`):
                The new key and value states to cache.

        Return:
            A list containing the updated key and value states.
        """
        assert all(self.shape == t.shape for t in key_value), (self.shape, key_value[0].shape)
        past_key_value = self.cached_kv.get(layer_idx, [None] * len(key_value))
        key_value = [self.concat(past, curr) for past, curr in zip(past_key_value, key_value, strict=True)]
        self.cached_kv[layer_idx] = key_value
        return key_value

    # def get_seq_length(self, layer_idx: Optional[int]=None) -> int:
    #     """Returns the sequence length of the cached states. A layer index can be optionally passed."""
    #     if layer_idx is None:
    #         layer_idx = next(iter(self.cached_kv.keys()), None)
    #     if layer_idx in self.cached_kv:
    #         return self.cached_kv[layer_idx][0].shape[0]
    #     else:
    #         return 0

    def detach(self) -> Self:
        """Detach key and value states for future autograd."""
        for layer_idx, kv in self.cached_kv.items():
            self.cached_kv[layer_idx] = [x.detach() for x in kv]
        return self

    def dump(self, layer_idx: Optional[int]=None) -> List[torch.Tensor]:
        """Dump key and value states as a list of tensors."""
        if layer_idx is None:
            return sum(self.cached_kv.values(), [])
        else:
            return self.cached_kv[layer_idx]

    def get_shape(self, num: int) -> Tuple[int, int, int, int]:
        shape = list(self.shape)
        shape[0] *= num
        return tuple(shape)

    def copy(self, retain_kv=True, ctx_pair=None) -> Self:
        copy = KVCache(self.dtype, self.shape, self.growth)
        if retain_kv:
            copy.cached_kv = self.cached_kv.copy()
        copy.ctx_pair = ctx_pair or self.ctx_pair
        return copy

    def exchange_ctx(self, layer_idx: int, is_backward=False, for_kv=True, for_qo=False):
        ctx_pair = self.ctx_pair
        if ctx_pair is None:
            return
        num = ctx_pair.nbwd if is_backward else ctx_pair.nfwd
        if not for_kv and not for_qo or num == 0:
            return

        peer = ctx_pair.peer
        group = parallel_state.get_pipeline_model_parallel_group()
        # group = torch.distributed.group.WORLD
        grank = parallel_state.get_pipeline_model_parallel_global_rank(peer)
        if num < 0:
            other = []
            if for_kv:
                assert not ctx_pair.other_kv
                other += [t.prefix(-num).concat() for t in self.dump(layer_idx)]
                ctx_pair.other_kv.append([num, num])
            if for_qo:
                qo = ctx_pair.local_qo.pop()
                assert not ctx_pair.other_qo
                other += qo
                ctx_pair.other_qo.append([num for _ in qo])
            # print_debug(DEBUG_CTXPAIR, "send_other+", value=(peer, num))
            sends = [dist.P2POp(dist.isend, t, grank, group) for t in other]
            reqs = dist.batch_isend_irecv(sends); del sends
            # reqs = [dist.isend(t, grank, group) for t in other]
            assert len(ctx_pair.reqs) == 0
            ctx_pair.reqs.append(reqs)
            # print_debug(DEBUG_CTXPAIR, "send_other-", value=(peer, num))
        elif num > 0:
            other = []
            if for_kv:
                assert not ctx_pair.other_kv
                k = torch.empty(self.get_shape(num), dtype=self.dtype, device='cuda')
                other.append(k)
                v = torch.empty(self.get_shape(num), dtype=self.dtype, device='cuda')
                other.append(v)
                ctx_pair.other_kv.append([k, v])
            if for_qo:
                qo = ctx_pair.local_qo.pop()
                assert not ctx_pair.other_qo
                qo = [torch.empty_like(t) for t in qo]
                other += qo
                ctx_pair.other_qo.append(qo)
            # print_debug(DEBUG_CTXPAIR, "recv_other+", value=(peer, num))
            recvs = [dist.P2POp(dist.irecv, t, grank, group) for t in other]
            reqs = dist.batch_isend_irecv(recvs); del recvs
            # reqs = [dist.irecv(t, grank, group) for t in other]
            assert len(ctx_pair.reqs) == 0
            ctx_pair.reqs.append(reqs)
            # print_debug(DEBUG_CTXPAIR, "recv_other-", value=(peer, num))

    def exchange_ctx_for_backward(self, x, layer_idx):
        return ExchangeCtxForBackwardFunc.apply(x, self, layer_idx)


class ExchangeCtxForBackwardFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kv_cache, layer_idx):
        ctx.kv_cache = kv_cache.copy()
        ctx.layer_idx = layer_idx
        return x

    @staticmethod
    def backward(ctx, grad_x):
        ctx.kv_cache.exchange_ctx(ctx.layer_idx, is_backward=True, for_qo=True)
        return grad_x, None, None
