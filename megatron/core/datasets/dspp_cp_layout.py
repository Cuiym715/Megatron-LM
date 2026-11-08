"""CPU metadata for DSPP's existing paired CP token permutation.

Q intervals are intersected with packed items, never with concatenated KV:
the causal prefix end and the actual history boundary are distinct values.
"""
from dataclasses import dataclass

CP_IMPLEMENTATION = 'owner-partials-send-progress-fused-kv-v1'


def cp_token_ranges(capacity, size, rank):
    if size < 1 or not 0 <= rank < size:
        raise ValueError('invalid CP size/rank')
    if size == 1:
        return ((0, capacity),)
    if capacity <= 0 or capacity % (2 * size):
        raise ValueError('DSPP CP capacity must be divisible by 2*CP')
    width = capacity // (2 * size)
    return tuple((i * width, (i + 1) * width)
                 for i in (2 * rank, 2 * size - 1 - 2 * rank))


@dataclass(frozen=True)
class CpQueryRun:
    item_index: int
    local_begin: int
    local_end: int
    key_begin: int
    key_end: int
    history_length: int
    position_begin: int


@dataclass(frozen=True)
class CpOwnership:
    rank_indices: tuple
    shared_items: tuple


def hybrid_cp_ownership(meta, size):
    """Whole independent sequences first; split only unavoidable remainders.

    Long-chain pieces reserve their paired token intervals first. No global
    bin search or device work is performed; each rank keeps a fixed capacity.
    """
    cp_token_ranges(meta.chunk_size, size, 0)
    capacity = meta.chunk_size // size
    owners = [[] for _ in range(size)]
    item_owners = [set() for _ in meta.items]
    offset = 0
    short = []
    for index, item in enumerate(meta.items):
        length = item.token_length
        if item.segment_count > 1:
            for rank in range(size):
                for piece in (2*rank, 2*size-1-2*rank):
                    a, b = piece*length//(2*size), (piece+1)*length//(2*size)
                    if a < b:
                        owners[rank].extend(range(offset+a, offset+b))
                        item_owners[index].add(rank)
        else:
            short.append((length, index, offset))
        offset += length
    remainder = []
    for length, index, offset in sorted(short, key=lambda p: (-p[0], p[1])):
        eligible = [r for r in range(size) if len(owners[r])+length <= capacity]
        if eligible:
            rank = min(eligible, key=lambda r: (len(owners[r]), r))
            owners[rank].extend(range(offset, offset+length))
            item_owners[index].add(rank)
        else:
            remainder.append((length, index, offset))
    for length, index, offset in remainder:
        cursor = offset
        for rank in sorted(range(size), key=lambda r: (len(owners[r]), r)):
            count = min(length, capacity-len(owners[rank]))
            if count:
                owners[rank].extend(range(cursor, cursor+count))
                item_owners[index].add(rank)
                cursor += count
                length -= count
        if length:
            raise ValueError('CP packing exceeds fixed local capacity')
    return CpOwnership(tuple(tuple(sorted(row)) for row in owners),
                       tuple(i for i, ranks in enumerate(item_owners) if len(ranks) > 1))


def owned_query_runs(meta, indices):
    """Map packed local valid-prefix tokens back to sequence causal prefixes."""
    runs = []
    physical_offset = key_offset = cursor = 0
    for index, item in enumerate(meta.items):
        while cursor < len(indices) and indices[cursor] < physical_offset+item.token_length:
            begin = cursor
            first = indices[cursor]
            cursor += 1
            while (cursor < len(indices) and indices[cursor] == indices[cursor-1]+1
                   and indices[cursor] < physical_offset+item.token_length):
                cursor += 1
            runs.append(CpQueryRun(index, begin, cursor, key_offset,
                key_offset+item.token_offset+indices[cursor-1]+1-physical_offset,
                item.token_offset, item.token_offset+first-physical_offset))
        physical_offset += item.token_length
        key_offset += item.token_offset+item.token_length
    return tuple(runs)


def cp_query_runs(meta, size, rank):
    """Nonempty local Q runs and prefixes in assembled global packed K/V."""
    runs = []
    local_offset = 0
    for begin, end in cp_token_ranges(meta.chunk_size, size, rank):
        physical_offset = key_offset = 0
        for index, item in enumerate(meta.items):
            lo = max(begin, physical_offset)
            hi = min(end, physical_offset + item.token_length)
            if lo < hi:
                runs.append(CpQueryRun(
                    index, local_offset + lo - begin, local_offset + hi - begin,
                    key_offset, key_offset + item.token_offset + hi - physical_offset,
                    item.token_offset, item.token_offset + lo - physical_offset,
                ))
            physical_offset += item.token_length
            key_offset += item.token_offset + item.token_length
        local_offset += end - begin
    return tuple(runs)
