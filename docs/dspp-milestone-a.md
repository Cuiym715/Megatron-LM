# DSPP Milestone A: Variable-Length Batch and Packed KV Semantics

Status: implemented and tested on one NVIDIA L40

Date: 2026-09-02

## 1. Scope

Milestone A establishes the data and autograd contract required by DSPP before
pipeline scheduling is changed. It supports:

- a different number of fixed-size segments for every logical sequence;
- standalone full segments;
- residual packing with at most one long-sequence continuation followed by
  independent short sequences;
- fixed `[chunk_size, 1, ...]` tensors at the model/pipeline boundary;
- packed FlashAttention over valid tokens only;
- per-layer, per-sequence KV cache routing;
- reverse-segment backward with explicit historical K/V gradient accumulation;
- deterministic plan validation and materialization of tokens, labels,
  positions, and loss masks.

This milestone does not change the pipeline schedule or data iterator. The next
milestone will make the schedule create a shared `DsppSequenceKVState`, call
`set_microbatch(meta)` before each physical microbatch forward, and call
`finish_iteration()` after all reverse-order backward tasks complete.

## 2. Why the original SlimPipe path was insufficient

The original `get_flash_variable_sliced_batch` keeps every batch row as one
continuous context. It can pad different rows to the same segment count, but it
cannot represent the following DSPP residual pack safely:

```text
[sequence A continuation] [short sequence B] [short sequence C] [padding]
```

Only A should read historical KV. B and C must start at position zero and must
not attend to A or to each other. In addition, the backward of A's continuation
must return gradients to K/V produced by A's earlier physical microbatches.

The old `KVCache` is attached to a `MicroBatch` and indexed by layer only. The
new state is iteration-scoped and indexed by `(layer_id, sequence_id)`, which
matches the simulator's flattened physical-microbatch representation.

## 3. BatchPlan design

`DsppBatchPlan` is independent of the pipeline schedule. A `DsppSegment`
contains:

```text
sequence_id
segment_id
token_offset
token_length
sequence_length
segment_count
estimated_flops
```

The builder follows the simulator's residual-packing policy:

1. Every full `chunk_size` segment is a standalone physical microbatch.
2. Each long-sequence tail seeds a separate residual bin.
3. Short sequences are considered from long to short.
4. A feasible bin is selected by lowest accumulated FLOPs, then best remaining
   token fit, then stable bin index.
5. The number of residual bins starts at
   `max(ceil(residual_tokens/chunk_size), tail_count)` and increases only when
   packing fails.

The default cost is the attention-dominated prefix increment
`prefix_end^2 - prefix_start^2`. A stage-calibrated cost function can be passed
later without changing the plan representation.

The materializer accepts token sequences of length `training_length + 1` and
creates fixed `[1, chunk_size]` tensors. Query token `i` predicts token `i+1`.
Valid items occupy a contiguous prefix, and padding has a zero loss mask.

## 4. Packed attention layout

For a residual pack containing A's continuation and two short sequences:

```text
Q = [A_current, B_current, C_current]
K = [A_history, A_current, B_current, C_current]
```

Consequently, query and key cumulative lengths differ:

```text
cu_seqlens_q = [0, len(A_current), ..., total_current]
cu_seqlens_k = [0, len(A_history)+len(A_current), ..., total_key]
```

FlashAttention's bottom-right causal alignment makes A's current queries see
their history while keeping B and C as separate varlen sequences. Padding is
removed before attention and restored as zeros afterward, so the external
tensor shape remains fixed.

Position ids are materialized relative to the logical sequence: continuation
positions start at `token_offset`, while independent short sequences start at
zero.

## 5. Backward and state lifetime

`_DsppPackedFlashAttention` is a custom autograd function because different
physical microbatches are backwarded as separate pipeline tasks. Earlier K/V
tensors are intentionally retained outside the autograd graph.

Forward performs the following steps for every layer:

1. Look up history by `(layer_id, sequence_id)`.
2. Validate the expected segment id and history length in debug mode.
3. Assemble packed K/V and execute varlen FlashAttention.
4. Retain current K/V only when the segment has a successor.

Backward must execute segments in reverse order:

1. FlashAttention produces gradients for assembled current and historical K/V.
2. Historical gradients are accumulated by their original `segment_id`.
3. When the previous segment is backwarded, its locally computed K/V gradients
   are added to the accumulated gradients from all successors.
4. The retained segment and pending gradients are removed.
5. Backward of segment zero removes the sequence/layer entry completely.

`finish_iteration()` verifies that every cache entry was drained and then
releases device-side metadata. An early or out-of-order backward fails instead
of silently returning an incomplete gradient.

## 6. Megatron integration point

`ParallelAttention` detects `DsppSequenceKVState` on the local-transformer
FlashAttention path and calls `dspp_packed_flash_attention`. Existing SlimPipe
`KVCache` behavior is unchanged.

Required caller sequence:

```python
state = DsppSequenceKVState(validate_runtime=False)
for batch in plan.microbatches:
    state.set_microbatch(batch)
    output = model(..., kv_cache=state)

for output in reversed(outputs):
    output.backward(...)
state.finish_iteration()
```

The next pipeline milestone must share one state per vchunk for the whole
iteration. It must not create one cache per physical microbatch as the old
SlimPipe schedule does.

## 7. Performance-sensitive choices

- Full `BatchPlan.validate()` is run during construction and in unit tests.
- Runtime metadata validation is disabled by default and can be enabled with
  `DsppSequenceKVState(validate_runtime=True)` while debugging.
- There is no generic runtime DAG traversal or ready-set scan in Milestone A.
- `cu_seqlens_q/k` CUDA tensors are cached once per physical microbatch and
  reused across transformer layers.
- Pipeline-facing tensors stay fixed-size, avoiding a dynamic-shape handshake.
- Sequence-id dictionary lookups remain because they are required to route KV
  and gradients correctly. They can later be replaced by dense integer-indexed
  arrays if profiling shows measurable Python overhead.

## 8. Tests

Environment:

```text
container: nvidia_pytorch
GPU: NVIDIA L40, 46068 MiB
Python: 3.12.3
PyTorch: 2.10.0a0+a36e1d39eb.nv26.01.42222806
FlashAttention: 2.7.4.post1
```

Primary command:

```bash
docker exec nvidia_pytorch bash -lc \
  'cd /workspace/src/Megatron-LM-kwai && \
   /workspace/src/venvs/megatron/bin/python -m pytest -q tests/unit_tests/dspp'
```

Coverage includes:

- deterministic slicing and residual packing;
- all-short, exact-chunk, one-token-tail, heterogeneous multi-long-sequence,
  and padded packs;
- token/label/position/loss-mask preservation;
- plan rejection for broken segment chains;
- packed FlashAttention output versus unsliced causal SDPA;
- all Q/K/V gradients versus unsliced causal SDPA;
- cross-sequence attention isolation;
- input and every parameter gradient of a trainable attention block;
- language-model loss and every embedding/attention/output parameter gradient;
- grouped-query attention with different Q and K/V head counts;
- exact-multiple and non-exact long sequences in the same logical batch;
- missing-history rejection;
- 100 forward/reverse-backward iterations with drained state and bounded
  allocated-memory delta.

Result:

```text
19 passed, 0 failed
```

Existing smoke regression command:

```bash
docker exec nvidia_pytorch bash -lc \
  'cd /workspace/src/Megatron-LM-kwai && CUDA_VISIBLE_DEVICES=0 \
   /workspace/src/venvs/megatron/bin/python -m pytest -q \
   tests/unit_tests/test_basic.py tests/unit_tests/test_utils.py'
```

Result:

```text
7 passed, 0 failed
```

All new Python files also pass `py_compile` and `black --check`.

## 9. Current limitations and next milestone contract

- Only micro-batch size 1 is supported by the packed attention wrapper.
- Dropout is zero; deterministic dropout replay across sliced tasks is not
  implemented.
- TP, CP, DP, Transformer Engine, activation recomputation, attention
  offloading, and CUDA Graph are outside the MVP scope.
- The FlashAttention fast path currently uses private wrapped operators from
  FlashAttention 2.7.4. Its compatibility must be retested before changing the
  dependency version.
- The training data iterator and existing `MicroBatch` class do not yet emit
  `DsppBatchPlan`; that wiring belongs to the three-GPU pipeline milestone.
- Backward is required to respect reverse segment order. The DSPP stage-local
  schedule must preserve this invariant structurally; release mode will not add
  a general dependency checker to the hot path.

Milestone B can begin only with these invariants unchanged: stable
`sequence_id/segment_id`, one continuation per residual pack, fixed external
shape, shared iteration-level KV state, and reverse segment backward.
