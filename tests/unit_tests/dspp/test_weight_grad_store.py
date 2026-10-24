from collections import deque

import pytest

from megatron.core.zbpp_utils import WeightGradStore


@pytest.fixture
def empty_store(monkeypatch):
    previous_cache = WeightGradStore.cache
    previous_queue = WeightGradStore.weight_grad_queue
    previous_keyed = WeightGradStore.keyed_weight_grads
    previous_keys = WeightGradStore.keyed_task_keys
    WeightGradStore.cache = []
    WeightGradStore.weight_grad_queue = [deque(), deque()]
    WeightGradStore.keyed_weight_grads = {}
    WeightGradStore.keyed_task_keys = set()
    monkeypatch.setattr(WeightGradStore, "split_bw", classmethod(lambda cls: True))
    yield
    WeightGradStore.cache = previous_cache
    WeightGradStore.weight_grad_queue = previous_queue
    WeightGradStore.keyed_weight_grads = previous_keyed
    WeightGradStore.keyed_task_keys = previous_keys


def test_weight_grad_store_keeps_heterogeneous_physical_tasks_fifo(empty_store):
    executed = []

    for shape in ((256, 2048), (73, 2048)):
        WeightGradStore.put(
            lambda async_op=False, shape=shape: (shape,),
            lambda value: executed.append(value),
        )
        WeightGradStore.flush(chunk=1)

    assert WeightGradStore.pop(chunk=1) == 1
    assert WeightGradStore.pop(chunk=1) == 1
    assert executed == [(256, 2048), (73, 2048)]


def test_weight_grad_store_rejects_missing_w_task(empty_store, monkeypatch):
    monkeypatch.setattr(
        "megatron.core.zbpp_utils.parallel_state.get_pipeline_model_parallel_rank",
        lambda: 2,
    )
    with pytest.raises(RuntimeError, match=r"rank=2.*chunk=0.*available=0"):
        WeightGradStore.pop(chunk=0)


def test_keyed_w_executes_the_requested_physical_task(empty_store, monkeypatch):
    monkeypatch.setattr(
        'megatron.core.zbpp_utils.parallel_state.get_pipeline_model_parallel_rank', lambda: 0)
    executed = []
    for key, shapes in [('a', [(3, 7), (5, 9)]), ('empty', []), ('b', [(17, 2)])]:
        for shape in shapes:
            WeightGradStore.put(lambda async_op=False, shape=shape: (shape,), executed.append)
        WeightGradStore.flush(chunk=1, task_key=key)
    assert WeightGradStore.queue_size(1) == 3
    WeightGradStore.pop(chunk=1, task_key='b')
    WeightGradStore.pop(chunk=1, task_key='empty')
    WeightGradStore.pop(chunk=1, task_key='a')
    assert executed == [(17, 2), (3, 7), (5, 9)]
    with pytest.raises(RuntimeError, match='already consumed'):
        WeightGradStore.pop(chunk=1, task_key='b')
    with pytest.raises(RuntimeError, match='duplicate'):
        WeightGradStore.flush(chunk=1, task_key='b')
    WeightGradStore.assert_empty()


def test_keyed_w_rejects_wrong_chunk_and_duplicate_b(empty_store):
    WeightGradStore.flush(chunk=0, task_key=(5, 1, 9, 0, 2))
    with pytest.raises(RuntimeError, match='missing'):
        WeightGradStore.pop(chunk=1, task_key=(5, 1, 9, 0, 2))
    with pytest.raises(RuntimeError, match='duplicate'):
        WeightGradStore.flush(chunk=0, task_key=(5, 1, 9, 0, 2))
    with pytest.raises(AssertionError, match='keyed'):
        WeightGradStore.assert_empty()
