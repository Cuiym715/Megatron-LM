import contextlib
import math
import torch
import warnings
import weakref


from megatron.profile_utils import annotate_range


_MEMCPY_STREAM = dict()
_GPU_BUFFER_POOL = dict()
_CPU_BUFFER_POOL = list()


def set_ideal_affinity_for_current_gpu():
    import cuda.cuda
    import cuda.cudart
    import pynvml
    import uuid
    err, device_id = cuda.cudart.cudaGetDevice()
    assert err == cuda.cudart.cudaError_t.cudaSuccess
    err, device_uuid = cuda.cuda.cuDeviceGetUuid(device_id)
    assert err == cuda.cuda.CUresult.CUDA_SUCCESS
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(uuid.UUID(bytes=device_uuid.bytes)))
    pynvml.nvmlDeviceSetCpuAffinity(handle)


def get_memcpy_stream(key):
    if key not in _MEMCPY_STREAM:
        _MEMCPY_STREAM[key] = torch.cuda.Stream()
    return _MEMCPY_STREAM[key]


def get_persistent_gpu_buffer(key, size):
    if key in _GPU_BUFFER_POOL and _GPU_BUFFER_POOL[key].numel() < size:
        assert _GPU_BUFFER_POOL[key].ref_cnt == 0, "last onload tensors are not fully deleted"
        wref = weakref.ref(_GPU_BUFFER_POOL[key])
        del _GPU_BUFFER_POOL[key]
        assert wref() is None, "the gpu buffer is not deleted."
    if key not in _GPU_BUFFER_POOL:
        _GPU_BUFFER_POOL[key] = torch.empty(size, dtype=torch.uint8, device="cuda")
        _GPU_BUFFER_POOL[key].ref_cnt = 0
    return _GPU_BUFFER_POOL[key][:size]


def get_cpu_buffer(cal_size):
    best = -1
    for i, (buffer, _) in enumerate(_CPU_BUFFER_POOL):
        if buffer.numel() >= cal_size:
            if best == -1 or buffer.numel() < _CPU_BUFFER_POOL[best][0].numel():
                best = i
    if _CPU_BUFFER_POOL:
        buffer, event = _CPU_BUFFER_POOL.pop(best)
        event.wait()
        if buffer.numel() < cal_size:
            buffer = None   # release before allocate
        else:
            return buffer
    set_ideal_affinity_for_current_gpu()
    import wrap_gemm_cuda  # TODO: move to another libraray
    buffer = wrap_gemm_cuda.wrap_cuda_malloc_host(cal_size)
    return buffer


def recycle_cpu_buffer(buffer, event):
    _CPU_BUFFER_POOL.append((buffer, event))


def copy2d_(dst, src):
    assert dst.dtype == src.dtype, "dtype mismatch"
    if not dst.is_contiguous():
        raise NotImplementedError(f"unsupported dst shape {dst.shape} stride {dst.stride()}")
    shape = src.shape
    stride = src.stride()
    if stride[-1] == 1 and all(stride[i] == shape[i + 1] * stride[i + 1] for i in range(0, len(shape) - 2)):
        import wrap_gemm_cuda  # TODO: move to another libraray
        dw = src.dtype.itemsize
        cudaMemcpyDefault = 4
        wrap_gemm_cuda.wrap_cuda_memcpy_2d_async(dst.data_ptr(), shape[-1] * dw, src.data_ptr(), stride[-2] * dw,
                                                 shape[-1] * dw, shape[:-1].numel(), cudaMemcpyDefault,
                                                 torch.cuda.current_stream().cuda_stream)
    else:
        raise NotImplementedError(f"unsupported src shape {shape} stride {stride}")


def fast_contiguous(x):
    if x.is_contiguous():
        return x
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    copy2d_(out, x)
    return out


class TensorWrap:
    def __init__(self, x):
        self.x = x
        self.shape = x.shape
        self.dtype = x.dtype
        self.strides = x.stride()
        self.device = x.device
        self.base = None


class TensorPack:
    def __init__(self, tensor_wrap):
        self.tensor_wrap = tensor_wrap

    def get(self):
        assert self.tensor_wrap.x is not None
        return self.tensor_wrap.x

    def __del__(self):
        self.tensor_wrap.x = None
        if self.tensor_wrap.base is not None:
            self.tensor_wrap.base.ref_cnt -= 1
            self.tensor_wrap.base = None


class PackSolver:
    """Find the subset of `ws` whose sum is closest to and less than `cap`."""

    def __init__(self, ws):
        self.ws = ws

    def solve(self, cap):
        n = len(self.ws)
        self.memo = {}
        return self.dps(n - 1, cap)

    def dps(self, i, cap):
        if i == -1:
            return (0, [])
        if (i, cap) in self.memo:
            return self.memo[(i, cap)]
        va, a = self.dps(i - 1, cap)
        if cap < self.ws[i] or self.ws[i] == 0:
             self.memo[(i, cap)] = (va, a)
        else:
            vb, b = self.dps(i - 1, cap - self.ws[i])
            vb += self.ws[i]
            self.memo[(i, cap)] = (va, a) if va >= vb else (vb, b + [i])
        return self.memo[(i, cap)]


class OffloadManager:
    warned = False

    def __init__(self, offload_ratio):
        self.offload_ratio = offload_ratio
        self.reset()

    def reset(self, tensors=[]):
        self.tensors = sorted(tensors, key=lambda t: t.shape.numel() * t.dtype.itemsize)

    def is_complete(self):
        return not self.tensors

    def offload(self, *, prior_works=[], use_bucket=False):
        if self.is_complete():
            return None
        top = 0
        ws = [0] * len(self.tensors)
        dup = [-1] * len(self.tensors)
        for i, tensor in enumerate(self.tensors):
            if tensor.x.is_contiguous():
                for j, prev_tensor in enumerate(self.tensors[:i]):
                    if tensor.x.data_ptr() == prev_tensor.x.data_ptr() and prev_tensor.x.is_contiguous() and tensor.device == prev_tensor.device and tensor.shape.numel() == prev_tensor.shape.numel():
                        dup[i] = j
                        break
            if dup[i] == -1:
                n = tensor.shape.numel() * tensor.dtype.itemsize
                top += n
                ws[i] = n
        # print(f'{top=}')
        # print(f'{ws=}')
        res = top / 1024  # minimum resolution: 1/1024
        ws = [int((w + res - 1/1024) / res) for w in ws]
        pks = PackSolver(ws)
        cap = int(top * (1 - self.offload_ratio) / res)
        cap2, rem = pks.solve(cap)
        assert cap2 <= cap, (cap2, cap)
        # print(f'{cap2=}')
        offload_size = top
        for i in rem:
            if dup[i] == -1:
                tensor = self.tensors[i]
                offload_size -= tensor.shape.numel() * tensor.dtype.itemsize
        self.onload_size = offload_size # onload all complete tensors to buffer.
        expected = math.ceil(top * self.offload_ratio)
        if offload_size >= expected + res:
            expected = int(expected)
            if not __class__.warned:
                warnings.warn(f'cut offload size from {offload_size} to {expected}')
                __class__.warned = True
            offload_size = expected
        # print(f'{offload_size=}')
        del pks
        if use_bucket:
            buffer = get_persistent_gpu_buffer("offload", offload_size)
        else:
            buffer = None
        copy_tasks = []
        partially_offloaded_bases = set()
        offset = 0
        for i, tensor in enumerate(self.tensors):
            assert tensor.x is not None, "A saved tensor is released before offload."
            assert tensor.x.device.type == "cuda"
            if dup[i] == -1 and i not in rem:
                if tensor.x._base is not None:
                    partially_offloaded_bases.add(tensor.x._base)
                size = tensor.shape.numel() * tensor.dtype.itemsize
                if offset + size <= offload_size:   # whole tensor
                    if use_bucket:
                        buffer[offset:offset + size].view(tensor.dtype).as_strided(tensor.shape, tensor.strides).copy_(tensor.x)
                    else:
                        copy_tasks.append((offset, offset + size, tensor.x))
                else:   #   portion
                    size = offload_size - offset
                    tensor.x = fast_contiguous(tensor.x)
                    tensor.strides = tensor.x.stride()
                    linear_data = tensor.x.view(-1).view(torch.uint8)
                    if use_bucket:
                        buffer[offset:].copy_(linear_data[:size])
                    else:
                        copy_tasks.append((offset, offload_size, linear_data[:size]))
                    self.remained_not_offloaded = linear_data[size:].clone()
                offset += size
                tensor.x = None
            elif dup[i] != -1 and dup[i] not in rem:  # duplicate
                tensor.x = dup[i]
            elif tensor.x._base in partially_offloaded_bases:
                if dup[i] != -1:
                    raise NotImplementedError("does not support partially offload duplicate tensors")
                tensor.x = tensor.x.clone()
        assert offset == offload_size, (offset, offload_size)
        assert copy_tasks, (offload_size, copy_tasks)
        cal_size = offload_size
        assert offload_size <= cal_size
        stream = get_memcpy_stream("offload")
        if use_bucket:  # wait for copy_ to bucket
            stream.wait_stream(torch.cuda.current_stream())
        with annotate_range("offload"), torch.cuda.stream(stream):
            # wait for prior works (e.g. distributed communication).
            for work in prior_works:
                work.wait()
            self.buffer_cpu = get_cpu_buffer(cal_size)[:offload_size]
            if use_bucket:
                self.buffer_cpu.copy_(buffer, non_blocking=True)
                # buffer.record_stream(stream)
            else:
                # Pop to release the tensor immediately after copy, and `record_stream` to prevent it from being deallocated.
                # while copy_tasks:
                #     begin_idx, end_idx, x = copy_tasks.pop()
                #     x.record_stream(stream)
                for begin_idx, end_idx, x in copy_tasks:
                    dst = self.buffer_cpu[begin_idx:end_idx].view(x.dtype).as_strided(x.shape, x.stride())
                    dst.copy_(x, non_blocking=True)
        event = stream.record_event()
        # XXX(lizhouyang): Hold the bucket or tensors to prevent them from being deallocated before sync to the stream.
        # This trick can be used to avoid `record_stream` on these tensors.
        setattr(event, "_possession", buffer if use_bucket else copy_tasks)
        return event

    def onload(self, *, prior_works=[], overlap_d2h_h2d=True, ping_pong_onload=True,
               buffer_name="onload", buffer_idx=0):
        if self.is_complete():
            return None
        stream_key = "onload" if overlap_d2h_h2d else "offload"
        if ping_pong_onload:
            buffer_key = buffer_name + ":" + str(buffer_idx)
        else:
            buffer_key = buffer_name
        stream = get_memcpy_stream(stream_key)
        with annotate_range("onload"), torch.cuda.stream(stream):
            # wait for prior works (e.g. distributed communication).
            for work in prior_works:
                work.wait()
            buffer = get_persistent_gpu_buffer(buffer_key, self.onload_size)
            assert buffer._base.ref_cnt == 0, "last onload tensors are not fully deleted"
            offload_size = self.buffer_cpu.numel()
            buffer[:offload_size].copy_(self.buffer_cpu, non_blocking=True)
            offset = 0
            for tensor in self.tensors:
                if tensor.x is None:
                    size = tensor.shape.numel() * tensor.dtype.itemsize
                    if offset + size <= self.onload_size:
                        tensor.x = buffer[offset:offset+size].view(tensor.dtype).as_strided(tensor.shape, tensor.strides)
                    if ping_pong_onload:
                        tensor.base = buffer._base
                        tensor.base.ref_cnt += 1
                    else:
                        tensor.x = tensor.x.clone()
                    offset += size
                elif isinstance(tensor.x, int): # duplicate
                    t = self.tensors[tensor.x].x
                    assert t is not None
                    tensor.x = t.view(tensor.dtype).as_strided(tensor.shape, tensor.strides)
        assert offset == self.onload_size, (offset, self.onload_size)
        event = stream.record_event()
        recycle_cpu_buffer(self.buffer_cpu._base, event)
        del self.buffer_cpu
        if offload_size != self.onload_size:
            buffer[offload_size:].copy_(self.remained_not_offloaded)
            del self.remained_not_offloaded
        self.refs = [weakref.ref(t) for t in self.tensors]
        self.tensors = []
        return event

    def check_ref(self):
        if not hasattr(self, "refs"):
            return
        for r in self.refs:
            assert r() is None, r()
        del self.refs


class ForwardEmptyBackwardIdentityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.empty((), dtype=x.dtype, device=x.device).expand_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


class ForwardLeftBackwardRightFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right):
        return left

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output


def get_forward_tensor_and_backward_handle(x):
    backward_handle = torch.empty((), dtype=x.dtype, device=x.device).expand_as(x)
    backward_handle.requires_grad_(x.requires_grad)
    x.requires_grad_(False)
    x = ForwardLeftBackwardRightFunction.apply(x, backward_handle)
    return x, backward_handle


def forward_empty_backward_identity(x):
    return ForwardEmptyBackwardIdentityFunction.apply(x)


@contextlib.contextmanager
def offload_manager(om: OffloadManager):
    tensors = []
    def pack_hook(x):
        tensor_wrap = TensorWrap(x)
        base = x._base if x._is_view() else x
        is_parameter = isinstance(base, torch.nn.Parameter)
        is_too_small = x.numel() * x.element_size() < 1024 * 1024
        if is_parameter or is_too_small:
            return TensorPack(tensor_wrap)
        is_misaligned = x.data_ptr() % 32 != 0 or x.numel() * x.element_size() % 32 != 0
        if is_misaligned:
            raise NotImplementedError("not implemented offload misaligned tensor size")
        tensors.append(tensor_wrap)
        return TensorPack(tensor_wrap)

    def unpack_hook(tensor_pack):
        x = tensor_pack.get()
        return x

    if om.offload_ratio:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            yield
    else:   # skip recording activation tensors if offload_ratio is 0.
        yield

    om.reset(tensors)

    # must manually delete the `tensors`, otherwise, it will be brought out by the `pack_hook`
    # through the `torch.autograd.graph.saved_tensors_hooks`.
    del tensors
    return
