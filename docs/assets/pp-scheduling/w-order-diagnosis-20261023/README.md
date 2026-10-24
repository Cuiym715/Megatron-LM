# PP3 W-order numerical diagnosis

Follow-up: the user requested [fixed per-chunk W order](../fixed-w-order-20261023/README.md).
The current validator rejects the original permuted external plan; restoring
its W order while preserving insertion positions passes the original three-step
numerical check. The diagnosis below describes the earlier policy and remains
valid. Reproducing that old live permutation requires the preserved pre-change
source patch `/home/cuiym/tmp/dspp-w-diag-20261023/p1-with-w-diagnosis.patch`;
saved-capture/kernel/Adam replays remain usable without executing an invalid plan.

The LR=1e-3 three-step failure is reproduced and its first divergence is located
in FP32 accumulation order. In this fixture the W task mapping, deferred operand
lifetime, individual W contributions and B-direct gradients agree exactly.
Frozen-operand original-kernel replays and frozen-gradient original-Adam replays
reproduce the training differences bit for bit. This explains the numerical
assertion failure; it is neither a convergence experiment nor a fix that makes
the original acceptance criterion pass.

## Scope and controls

Branch `cuiym-dspp`, HEAD `9550175c1d14ab309e96f090142e105b1a8501fd`, existing
uncommitted P1 changes preserved. This follow-up adds three diagnostic tools and
documentation; production execution, decomposition and kernel sources are not
changed. PP3/VPP2, MHA, BF16, Adam beta1=.9/beta2=.999/eps=1e-8, LR=.001,
clip=1, dropout=0, seed=1234, same synthetic data and frozen profiles as the old
failure. All numerical runs use the existing deterministic FA backward option.
These results do not cover the separate default-FA legacy-repeat failure.

The candidate is the existing **handwritten external plan**, exchanging ready
W of stage 0/chunk 1 while preserving merged F/B order. It is not a newly
generated A plan or evidence that A itself generates this permutation.

Four three-step runs: legacy/permuted with normal Adam updates, and
legacy-frozen/permuted-frozen with the inner Adam step suppressed. Two additional
one-step runs capture gradients before reduction; `lr-decay-iters=3` preserves
the original first-step schedule. Separate replay controls use the actual fused
GEMM and Apex FusedAdam, not substitute arithmetic implementations. The temporary
container uses the local 26.01 image and existing dependencies read-only; no
downloads or changes to existing containers/installations.

## Located divergence

1. The first step executes 108/96/96 W closures on ranks 0/1/2. Every closure
   executes once, uses the corresponding task/parameter, and retains identical
   input/dy values from capture to execution. Between the two orders, every
   individual W contribution is bitwise identical. All B-direct gradients are
   identical before reduction, including input embedding/norm paths.
2. Replaying the **frozen original operands through the original fused GEMM**
   in each execution order exactly reproduces every deferred parameter's live
   pre-reduction gradient. On rank 0, the existing tied-embedding operation
   `first_grad.add_(last_grad); last_grad.copy_(first_grad)` also reproduces the
   two final embedding gradients exactly. This closes the B-direct/shared-weight
   path that an unadjusted W-only replay would omit. The maximum first-step
   final-gradient difference is 7.450580596923828e-9; ranks 1/2 remain identical.
   Independent FP32 contribution sums differ with order; the corresponding
   FP64 sums use the same contributions and provide a rounding reference.
3. First-step global grad norm is exactly the same, 15.013427827672093. After
   clipping, maximum gradient difference is 9.313225746154785e-10. Original
   Apex Adam replay from the same initial master weights and optimizer state
   is bitwise repeatable with identical gradients. Feeding the two captured
   clipped gradients reproduces **all** first-step master weights, m/v and BF16
   copies of both actual training runs exactly. On rank 0 the maximum FP32
   master difference is 1.6111880540847778e-7 and 38 BF16 coordinates differ;
   ranks 1/2 have no first-step update difference.
4. With weight updates suppressed, all three ranks' F outputs, dx, logits and
   loss remain bitwise identical for all three steps. Rank 0 still has only
   7.45e-9 maximum gradient differences per step; ranks 1/2 remain identical.
   With normal updates restored, step 2 introduces output/logit differences
   and step 3 reproduces the old failure: rank 1/2 output max error .01171875,
   rank 0 logits .01953125. The original .004/.01/.02 criteria were not relaxed.

The first-step diagnostic reference gradients and updated model parameters also
match the original paired-validation legacy reference snapshots (without the
new W/Adam diagnostic hooks) exactly.
The supplementary one-step runs match the full three-step runs' first gradients.

## A concrete BF16 boundary crossing

Input embedding `[6236, 505]` (flat index 3193337) starts with the same FP32
master value .01434326171875. This is one of the 38 differing BF16 coordinates,
not the coordinate of maximum FP32 master error.

| Value | Original W order | Permuted W order |
| --- | --- | --- |
| Raw gradient | -1.539186200716358e-8 | -1.5391876218018297e-8 |
| Clipped gradient | -1.0252062532245532e-9 | -1.0252072524252753e-9 |
| FP32 master after Adam | .014434814453125 | .014434815384447575 |
| BF16 model after copy | .014404296875 | .01446533203125 |

The FP32 masters differ by 9.313225746154785e-10. The reference lands exactly
on the midpoint of the two neighboring BF16 values; the candidate lands one
FP32 ULP above it. BF16 rounding therefore gives different model weights, with
6.103515625e-5 separation. This is a same-coordinate observation backed by the
optimizer replay, not an inference from unrelated global maxima.

## Reproduction and evidence

Raw data: `/home/cuiym/tmp/dspp-w-diag-20261023/` (container `/workspace/tmp/...`).
`commands.json`, `run.py`, `run_reduce.py`, and per-run logs/`.pt` captures retain
the exact commands and controls. The two dependency-import attempts before
training are preserved as `attempt1-*`/`attempt2-*`; neither is a training failure.
Compact checked evidence is copied here with SHA256 provenance in
`evidence-manifest.json`:

- `analysis.json`: 18 rank/iteration comparisons, W identity and contribution
  checks, independent FP32/FP64 sums and same-coordinate optimizer trace.
  `*_replay_vs_post_reduce_unadjusted` intentionally omits B-direct/tied-gradient
  addition and must be read alongside the next exact replay audit.
- `reduce-replay.json`: exact original-kernel, B-direct, tied-embedding and
  original-reference checks on all three ranks.
- `optimizer-replay.json`: identical-gradient repeat control and exact original
  Adam reproduction, including m/v/master and BF16 copy.
- `cpu-tests.log`: DSPP CPU regression, 103 passed / 243 GPU tests skipped.

Run the exact training commands with `tools/diagnose_dspp_w_order.py`. Analysis:

```bash
python tools/analyze_dspp_w_order.py --root "$DIAG_ROOT" --include-frozen
python tools/analyze_dspp_w_order.py --root "$DIAG_ROOT" --verify-reduce \
  --baseline-snapshots "$ORIGINAL_LEGACY_SNAPSHOTS"
python tools/replay_dspp_w_optimizer.py --root "$DIAG_ROOT"
```

The first two commands use CPU captures; the Adam replay needs the matching GPU
environment. Hooks synchronize/copy tensors and allocate independent targets;
these runs cannot certify timing, memory peaks or absence of all possible
stream races. The exact original-reference/replay checks support the cause of
this specific failure, rather than a general proof for every model/PP layout.

## Remaining work

The high-LR permutation still fails the original three-step numerical gate.
No threshold, learning rate, production accumulation policy or optimizer was
changed to report a pass. Longer multi-seed convergence comparison, the separate
default-FA reproducibility issue and the other P1 memory/protocol acceptance
items remain open. PP4/8 GPU validation was not performed. No commit/push.
