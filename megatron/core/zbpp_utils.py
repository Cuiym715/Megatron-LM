from collections import deque
from contextlib import contextmanager

from megatron import get_args, get_timers
from megatron.core import parallel_state


class WeightGradStore:
    """Queue deferred weight-gradient work for zero-bubble schedules."""

    should_split_bw = False
    cache = []
    weight_grad_queue = None

    @classmethod
    def lazy_init(cls):
        if cls.weight_grad_queue is not None:
            return
        num_chunks = parallel_state.get_virtual_pipeline_model_parallel_world_size() or 1
        cls.weight_grad_queue = [deque() for _ in range(num_chunks)]

    @classmethod
    def is_supported(cls):
        args = get_args()
        if args.pipeline_model_parallel_size <= 1:
            return False
        if getattr(args, 'overlap_grad_reduce', False):
            return False
        if not getattr(args, 'gradient_accumulation_fusion', False):
            return False
        if getattr(args, 'transformer_impl', None) == 'transformer_engine':
            return False
        return True

    @classmethod
    def assert_supported(cls):
        args = get_args()
        assert args.pipeline_model_parallel_size > 1, \
            'B/W split requires pipeline model parallelism'
        assert not getattr(args, 'overlap_grad_reduce', False), \
            'B/W split does not support overlap_grad_reduce yet'
        assert getattr(args, 'gradient_accumulation_fusion', False), \
            'B/W split requires --gradient-accumulation-fusion'
        assert getattr(args, 'transformer_impl', None) != 'transformer_engine', \
            'B/W split currently supports the local transformer implementation only'

    @classmethod
    def split_bw(cls):
        return cls.should_split_bw and cls.is_supported()

    @classmethod
    @contextmanager
    def set_split_bw(cls, enabled):
        prev = cls.should_split_bw
        cls.should_split_bw = enabled
        try:
            yield
        finally:
            cls.should_split_bw = prev

    @classmethod
    def put(cls, pre_func, func):
        assert cls.split_bw()
        cls.cache.append((pre_func, func))

    @classmethod
    def flush(cls, chunk=0):
        cls.lazy_init()
        if not cls.split_bw():
            assert len(cls.cache) == 0
            return
        cls.weight_grad_queue[chunk].append(cls.cache)
        cls.cache = []

    @classmethod
    def queue_size(cls, chunk=0):
        cls.lazy_init()
        return len(cls.weight_grad_queue[chunk])

    @classmethod
    def pop(cls, chunk=0, pop_num=None, timers=None):
        cls.lazy_init()
        if timers is None:
            try:
                timers = get_timers()
            except Exception:
                timers = None
        timer = timers('w-compute', log_level=2) if timers is not None else None
        if timer is not None:
            timer.start()
        if pop_num is None:
            pop_num = 1
        pop_num = min(pop_num, len(cls.weight_grad_queue[chunk]))
        for _ in range(pop_num):
            stored_grads = cls.weight_grad_queue[chunk].popleft()
            for pre_func, func in stored_grads:
                func(*pre_func(async_op=False))
        if timer is not None:
            timer.stop()
        return pop_num

    @classmethod
    def clear(cls, chunk=0, timers=None):
        cls.lazy_init()
        cleared = 0
        while cls.weight_grad_queue[chunk]:
            cleared += cls.pop(chunk=chunk, pop_num=1, timers=timers)
        return cleared

    @classmethod
    def assert_empty(cls):
        assert len(cls.cache) == 0, "deferred weight-gradient cache is not empty"
        if cls.weight_grad_queue is None:
            return
        rank = parallel_state.get_pipeline_model_parallel_rank()
        for chunk, queue in enumerate(cls.weight_grad_queue):
            assert not queue, (
                f"deferred weight-gradient queue is not empty: "
                f"rank={rank}, chunk={chunk}, size={len(queue)}"
            )
