# DSPP Milestone B1: Single-GPU Training Integration

Status: implemented and tested on one NVIDIA L40

Date: 2026-09-02

## 1. Scope

Milestone B1 connects the variable-length batch and packed-attention contract
from Milestone A to the real Megatron training path. It adds:

- an explicit-length dataloader contract for document-level samples;
- conversion from a padded logical batch to a `DsppBatchPlan` and fixed-shape
  physical microbatches directly on the training device;
- a dedicated one-stage DSPP executor selected by `--dspp`;
- one sequence-aware KV state shared by all physical microbatches in a batch
  plan;
- forward execution in plan order and backward execution in exact reverse
  order;
- token-weighted loss and gradients across Megatron gradient-accumulation
  microbatches;
- real `pretrain_llama.py` execution through model initialization, data
  broadcast, optimizer step, logging, and teardown.

B1 intentionally supports one GPU only. Pipeline P2P, two directed
communicators, virtual pipeline chunks, and stage-local V-ZB belong to B2/B3.

## 2. Runtime data flow

```text
DocumentGPTDataset
  -> variable_seq_collate: padded text + exact raw lengths
  -> tensor-parallel broadcast of text and lengths
  -> build_dspp_training_batch
       -> DsppBatchPlan
       -> tokens / labels / positions / loss mask on CUDA
  -> forward_backward_dspp_no_pipelining
       -> physical forward in plan order
       -> token-weighted loss
       -> physical backward in reverse plan order
       -> empty-state assertion
  -> ordinary Megatron gradient reduction and optimizer step
```

The collator carries explicit raw lengths instead of searching for a pad-token
value. This is required because the configured pad id can be a legitimate
token inside a document. Raw length includes the final label token; the plan's
sequence length is therefore `raw_length - 1`.

`DsppTrainingBatch` owns a validated plan and its materialized physical
microbatches. Every model-facing tensor remains `[1, chunk_size]`. Materialized
tensors stay on the input device, avoiding a CUDA-to-CPU-to-CUDA token copy.
Only the small vector of lengths is synchronized to CPU for deterministic plan
construction.

## 3. Executor and dependency semantics

The original SlimPipe `MicroBatch` creates one old-style `KVCache` per logical
microbatch and calls `detach().dump()` around every sequence slice. That
lifetime cannot represent a continuation whose earlier segment is a separate
physical microbatch. B1 therefore uses a separate executor instead of adding
DSPP branches throughout the legacy class.

For each batch plan, the executor performs:

```text
F(physical 0), F(physical 1), ..., F(physical N-1)
B(physical N-1), B(physical N-2), ..., B(physical 0)
```

All forwards use the same `DsppSequenceKVState`. Consequently:

```text
F(sequence, segment i-1) -> F(sequence, segment i)
B(sequence, segment i)   -> B(sequence, segment i-1)
```

is guaranteed by executor structure. Release mode does not run a generic DAG
or scan a ready set. The state still keeps low-cost shape and ownership checks
that prevent silent KV misrouting. Full plan validation and layout formatting
run only for the first `--variable-seq-debug-num-batches` process-wide batches;
the default value is zero.

Forward-only evaluation has no backward pass that could drain retained KV, so
the executor explicitly releases plan-owned state after computing the loss.

## 4. Loss normalization

A fixed-length Megatron run normally averages equally sized microbatches. In a
variable-length run, equal microbatch weighting would change the objective
when accumulation groups contain different valid-token counts.

B1 first reads the logical accumulation set and computes:

```text
weight(group) = valid_tokens(group) / valid_tokens(optimizer_iteration)
loss = sum(weight(group) * mean_valid_token_loss(group))
```

The same weight is applied to output gradients and logging values. Tests cover
two accumulation groups with different token counts and compare the result to
one unsliced all-token mean.

## 5. FLOPs are not runtime time

The B1 executor never reads `DsppSegment.estimated_flops`. There is no virtual
clock, duration simulation, sleep, FLOPs-driven ready queue, or global
cost-based event selection in the real training path.

The estimate remains only in CPU `DsppBatchPlan` construction as the
simulator-compatible residual-bin balancing heuristic. Later milestones may
replace it with profiled ordering costs without changing correctness.

## 6. CLI and safety boundary

`--dspp` selects the B1 executor. Argument validation fails early unless the
configuration satisfies the implemented boundary:

- world size, TP, PP, and CP are all one;
- no virtual pipeline chunk;
- FlashAttention enabled and `micro_seq_length > 0`;
- attention and hidden dropout are zero;
- no activation recomputation, MoE, fast RoPE, attention balancing, activation
  offload, or vocab-in-pipeline mode;
- `--dspp` and the old `--variable-seq-slicing` path are mutually exclusive.

These checks avoid silently running unsupported combinations. B2 will relax
the PP/world-size restriction after directed P2P is integrated and tested.

## 7. Main code changes

- `megatron/core/datasets/dspp_training.py`
  - `DsppTrainingBatch`;
  - explicit-length adapter and device-side materialization.
- `megatron/core/datasets/dspp_batch_plan.py`
  - optional materialization device.
- `megatron/data/data_samplers.py`
  - DSPP collate emits `text` and `lengths`.
- `megatron/data/gpt_dataset.py`
  - DSPP selects `DocumentGPTDataset`.
- `pretrain_llama.py`
  - broadcasts explicit lengths and returns `DsppTrainingBatch`.
- `megatron/core/pipeline_parallel/schedules.py`
  - DSPP schedule selection and the one-stage executor.
- `megatron/training.py`
  - threads `get_batch_func` through train/evaluation and selects DSPP.
- `megatron/arguments.py`
  - `--dspp` and B1 compatibility guards.
- `tests/unit_tests/dspp/test_training_adapter.py`
  - data-boundary tests.
- `tests/unit_tests/dspp/test_training_runtime.py`
  - CUDA executor, loss, full-gradient, accumulation, evaluation, and selector
    tests.

Existing SlimPipe and non-DSPP schedule selection remains unchanged when
`--dspp` is absent.

## 8. Automated tests

Environment:

```text
container: nvidia_pytorch
GPU: NVIDIA L40, 46068 MiB
Python: 3.12.3
PyTorch: 2.10.0a0+a36e1d39eb.nv26.01.42222806
FlashAttention: 2.7.4.post1
```

B1-specific coverage includes:

- exact length metadata when a real document contains the pad-token value;
- filtering a row with no LM target and rejecting inconsistent lengths;
- collate output with padded text and raw lengths;
- fixed CUDA physical batch construction;
- schedule loss and every parameter gradient versus unsliced causal SDPA;
- residual packing with long continuations and short sequences;
- token-weighted gradient accumulation across unequal valid-token counts;
- forward-only execution and KV release;
- rejection of PP greater than one in B1.

Command:

```bash
docker exec nvidia_pytorch bash -lc \
  'cd /workspace/src/Megatron-LM-kwai && \
   /workspace/src/venvs/megatron/bin/python -m pytest -q tests/unit_tests/dspp'
```

Result after Milestones A and B1:

```text
27 passed, 0 failed
```

Formatting and import checks:

```text
py_compile: passed
black --check on all DSPP modules/tests: passed
```

Existing smoke regression:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q \
  tests/unit_tests/test_basic.py tests/unit_tests/test_utils.py
```

Result:

```text
7 passed, 0 failed
```

Warnings are existing PyTorch/FlashAttention/SWIG deprecation warnings; no
test warning was promoted to a failure.

## 9. Real Megatron Llama acceptance run

A synthetic Megatron indexed dataset was created in `/tmp` with repeating raw
document lengths `67, 51, 25, 9`. With `chunk_size=32`, this produces training
lengths `66, 50, 24, 8`, including both multi-segment continuations and short
sequence residual packing.

Configuration:

```text
entry point: pretrain_llama.py
GPU / parallelism: L40, world=TP=PP=CP=1
model: 2 layers, hidden 64, FFN 128, 4 heads, BF16
logical microbatch: 4 documents
sequence cap / chunk: 96 / 32
optimizer: Adam
iterations: 100
dropout / recompute / offload: disabled
```

Observed debug layout includes a real continuation-plus-short pack:

```text
sequence_lengths=(50, 8, 8, 50)
layout=[
  [(0, 0, 0, 32)],
  [(3, 0, 0, 32)],
  [(0, 1, 32, 18), (1, 0, 0, 8)],
  [(3, 1, 32, 18), (2, 0, 0, 8)]
]
```

Result:

```text
100/100 optimizer iterations completed
skipped iterations: 0
NaN iterations: 0
iteration 20  lm loss: 1.179934E+01, grad norm: 1.590
iteration 40  lm loss: 1.155273E+01, grad norm: 1.407
iteration 60  lm loss: 1.138135E+01, grad norm: 1.576
iteration 80  lm loss: 1.127054E+01, grad norm: 2.141
iteration 100 lm loss: 1.122970E+01, grad norm: 1.428
allocated memory after iteration 20 and 40: 187.873 MiB
max allocated / reserved after iteration 20 and 40: 403.012 / 436.0 MiB
```

Peak reserved memory grew while new physical-layout shapes were first seen,
but live allocated memory remained constant. Every training iteration also
passed the executor's KV empty-state assertion. The process emitted the
repository's existing `destroy_process_group()` shutdown warning after normal
training completion.

## 10. Remaining limitations and B2 contract

- B1 is a correctness baseline, not a performance result. It executes all
  physical forwards followed by reverse backward on one GPU.
- A batch-plan state is independent from other gradient-accumulation groups;
  gradients and logs are nevertheless normalized over all valid tokens in the
  optimizer iteration.
- The first version still uses FlashAttention 2.7.4 private wrapped operators.
- TP, CP, DP, PP, VP, TE, dropout, recomputation, offload, MoE, and CUDA Graph
  are unsupported.
- The data adapter is wired to `pretrain_llama.py`; other model entry points
  must explicitly adopt the same batch contract.

B2 may start from this tested boundary. It must preserve fixed payload shapes,
explicit metadata, plan-order forward, reverse-order segment backward,
token-weighted loss, and empty-state behavior while adding PP=3 and two
directed P2P communication lanes.
