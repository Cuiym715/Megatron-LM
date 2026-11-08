"""Layout invariants independent of FA kernels and distributed execution."""
from types import SimpleNamespace

import pytest

from megatron.core.datasets.dspp_cp_layout import cp_query_runs, cp_token_ranges


def test_cross_bin_short_relocation_reduces_shared_tokens_without_extra_tasks():
    from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
    from megatron.core.datasets.dspp_cp_layout import hybrid_cp_ownership
    lengths=(2464,768,3520,3936,864,3328,4384)
    baseline=build_dspp_batch_plan(lengths,2048)
    candidate=build_dspp_batch_plan(lengths,2048,packing_strategy='cp-aware')
    def traffic(plan):
        return sum(m.items[i].token_length for m in plan.microbatches
                   for i in hybrid_cp_ownership(m,2).shared_items
                   if m.items[i].segment_count==1)
    assert traffic(baseline)==864 and traffic(candidate)==0
    assert len(baseline.microbatches)==len(candidate.microbatches)
    def segments(plan):
        return sorted((i.sequence_id,i.segment_id,i.token_offset,i.token_length)
                      for m in plan.microbatches for i in m.items)
    assert segments(baseline)==segments(candidate)


@pytest.mark.parametrize('lengths', ((800,224,600,424), (1500,548), (700,700,648),
                                    (512,768,768), (1,)))
def test_hybrid_ownership_preserves_tokens_and_capacity(lengths):
    from megatron.core.datasets.dspp_cp_layout import hybrid_cp_ownership, owned_query_runs
    items = tuple(SimpleNamespace(token_offset=4096 if i == 0 and lengths[0] == 512 else 0,
        token_length=n, segment_count=3 if i == 0 and n == 512 else 1)
        for i, n in enumerate(lengths))
    meta = SimpleNamespace(chunk_size=2048, items=items)
    layout = hybrid_cp_ownership(meta, 2)
    assert sorted(i for row in layout.rank_indices for i in row) == list(range(sum(lengths)))
    assert all(len(row) <= 1024 for row in layout.rank_indices)
    for row in layout.rank_indices:
        runs = owned_query_runs(meta, row)
        assert sum(r.local_end-r.local_begin for r in runs) == len(row)
    if lengths == (800,224,600,424):
        assert layout.shared_items == ()
    if lengths == (512,768,768):
        assert layout.shared_items == (0,)
        assert all(len(row) == 1024 for row in layout.rank_indices)


@pytest.mark.parametrize('size', (2, 3, 4, 8))
def test_cp_ranges_partition_capacity_in_existing_rank_order(size):
    capacity = 16 * size
    positions = [i for r in range(size) for a, b in cp_token_ranges(capacity, size, r)
                 for i in range(a, b)]
    assert sorted(positions) == list(range(capacity))
    assert cp_token_ranges(capacity, size, 0) == ((0, 8), (capacity - 8, capacity))


def test_runs_keep_history_separate_from_causal_prefix():
    meta = SimpleNamespace(chunk_size=64, items=(
        SimpleNamespace(token_offset=128, token_length=20),
        SimpleNamespace(token_offset=0, token_length=23),
    ))
    # Rank 1 takes [32:48] then [16:32], so its second run crosses items.
    runs = cp_query_runs(meta, 2, 1)
    assert [(r.item_index, r.local_begin, r.local_end, r.key_begin, r.key_end,
             r.history_length, r.position_begin) for r in runs] == [
        (1, 0, 11, 148, 171, 0, 12),
        (0, 16, 20, 0, 148, 128, 144),
        (1, 20, 32, 148, 160, 0, 0),
    ]
    assert runs[1].key_end - runs[1].key_begin - 4 != runs[1].history_length


def test_empty_local_query_is_valid():
    meta = SimpleNamespace(chunk_size=64, items=(
        SimpleNamespace(token_offset=0, token_length=1),))
    assert cp_query_runs(meta, 2, 1) == ()


@pytest.mark.parametrize('capacity,size,rank', ((63, 2, 0), (64, 0, 0), (64, 2, 2)))
def test_invalid_partition_rejected(capacity, size, rank):
    with pytest.raises(ValueError):
        cp_token_ranges(capacity, size, rank)
