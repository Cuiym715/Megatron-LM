from collections import deque

import pytest

from megatron.core.zbpp_utils import WeightGradStore


@pytest.fixture
def empty_store(monkeypatch):
    previous_cache = WeightGradStore.cache
    previous_queue = WeightGradStore.weight_grad_queue
    WeightGradStore.cache = []
    WeightGradStore.weight_grad_queue = [deque(), deque()]
    monkeypatch.setattr(WeightGradStore, "split_bw", classmethod(lambda cls: True))
    yield
    WeightGradStore.cache = previous_cache
    WeightGradStore.weight_grad_queue = previous_queue


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
