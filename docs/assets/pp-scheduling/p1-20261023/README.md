# P1 implementation evidence — 2026-10-23

Baseline: branch `cuiym-dspp`, HEAD `9550175c1d14ab309e96f090142e105b1a8501fd`.
`status.txt` is empty: the worktree was clean before implementation. `sha256.txt`
records the original four execution files and the two required design documents.
No commit, push, reset, or edits to historical experiments/related-work files.
No applicable AGENTS.md exists in this checkout or its ancestors; the one in
`../Megatron-LM/` belongs to another repository.

P1 is **implemented with CPU and bounded PP3 diagnostics; full acceptance remains open**.

Follow-up: [fixed per-chunk W order](../fixed-w-order-20261023/README.md)
narrows A to preserve each model chunk's template W FIFO order. The old manual
same-chunk W/W permutation is now rejected; restoring its W order passes the
original LR=.001 three-step check. Historical failures below are retained.

- `--dspp-schedule legacy`: original template/FIFO W path for paired validation.
- `--dspp-schedule template` (default): identical template through Plan IR and
  keyed W. `pipelining_with_dspp_stage_local_v(dspp_plan=...)` accepts an already
  built plan; it does not rerun A when one is supplied.
- `--dspp-schedule a`: synchronous, fixed-F/B, W-only construction, followed by
  final consumer-order FIFO replay and strict improvement acceptance. Requires
  fixed chunk, fused attention B, and an explicit execution profile. Missing,
  zero, extrapolated or low-confidence compute costs fall back to this input's
  template and expose the reason. Existing P2P profile/analytical fallback is
  used for communication; this is not new cost calibration.

Plan schema v1 binds iteration, CPU layout/length metadata hash, layout/cost/
capacity/protocol versions; full task keys, padding, B→W correspondence, per-rank
orders, and derived messages including sequence, directed lane, shape/dtype/bytes
are serialized and checked on roundtrip. The P1 hash does **not** bind token
contents or sampler IDs: the current data interface does not expose stable sample
IDs; `sample_ids` is reserved for P6. D/H are rejected by the P1 validator. The
legacy node phase/start/completion fields remain diagnostic template metadata;
execution follows final list order, never those nominal times.

The existing compute loop, virtual-rank setting, loss normalization, immediate-H
legacy behavior, directional NCCL groups, Gloo credit/ready/ack, receive lookahead,
send depth and send-tensor lifetimes are reused. Plan agreement/rejection is
exchanged across the PP group before protocol threads/tasks launch. End-of-round
checks include W, local V state, queued sends, posted receives and task tensors.
No execution-time rescheduling, D/H selection, new kernels, planner workers or
input prefetch were added.

## CPU evidence

`research-crosscheck.json` reads the eight original 027 inputs and frozen
`fixed-FB` r0 results. It reconstructs the archived unfused schedule and exact
unsplit W/KV/pending owner contract, then checks frozen replay and new A. All
eight have identical final task orders, makespan and per-rank owner peaks. This
includes PP4/PP8 tail/packing/multiple-logical-batch inputs;
64K remains archived extrapolated-cost evidence. Production code never imports
research modules. Reproduce (inside the established container):

```bash
python tools/validate_dspp_plan_research.py \
  --archive /workspace/docs/v4/design/history-dkv/assets/research/027-slack-generated-20261022 \
  --output /workspace/tmp/NEW-p1-research.json
```

The unit tests cover PP3/4/8 Plan roundtrip, padding, stable task identities,
F/B order, W completeness, invalid/missing/duplicate W, wrong task/chunk, early W,
stale binding, D/H rejection, wire tampering, owner/token limits, and receiver
FIFO generation. A finite-credit CPU model covers lookahead/depth `(0,1)`,
`(1,1)`, `(2,2)`, `(4,1)`. It models payload rendezvous atomically; it cannot
certify NCCL/stream liveness. Parameter tests preserve VPP2/layer/dropout checks.
A tiny dense stage uses the **actual** `_DsppVTaskBatch` interface and real
`WeightGradStore` to compare outputs, dx, parameter gradients and three optimizer
updates across legacy/template/A in CPU float64. It does not test FA kernels.

CPU suite: **103 passed, 243 skipped** (all 346 collected), in both torch 2.5
(`cpu-tests.log`) and the actual torch 2.10 training environment with CUDA hidden
(`cpu-tests-torch210.log`).
The CPU run deliberately has no accessible CUDA; skipped cases are not passes. Final syntax/compile and
`git diff --check` checks passed. `archive-integrity.json` also verifies all
22 entries in the original 027 manifest with zero mismatches. The final 8-case
report is identical across the initial and final runs with different hash seeds.

CPU environment: `nvidia_pytorch_v`, Python 3.10, torch
`2.5.0a0+b465a5843b.nv24.09`, no usable CUDA. The container's setuptools no longer
exports `pkg_resources.packaging`, so the CPU invocation uses a process-local
compatibility assignment; installed dependencies and production sources were
not modified for this environment issue:

```bash
docker exec -w /workspace/src/Megatron-LM-kwai nvidia_pytorch_v \
  /workspace/src/venvs/megatron/bin/python -c \
  'import packaging, pkg_resources, pytest; pkg_resources.packaging=packaging; raise SystemExit(pytest.main(["tests/unit_tests/dspp", "-q", "--tb=short"]))'
```

## Memory and GPU boundary

The CPU planner accepts explicit shared owners (multiple births/users and last
user release), capacities and F→B token limits. Research byte coefficients are
used **only by the validation adapter**, not claimed as real training owners.
For deployment before P2, A additionally requires the following conservative
W retention envelope: at every local merged F/B frontier, all W that the
template had consumed must also have been consumed by A. The same input's
validated template is retained if this fails. All eight 027 A outputs pass.
This is an explicit P1 storage/protocol restriction; it may reject other plans
that the eventual measured byte contract would accept. It is not a total GPU
memory bound or proof of ≤5% real-memory growth. Activation/P2P/allocator overlap
still needs real measurements.

The initially restricted host check failed; an escalated read-only check finds
three L40s. Existing training containers still have unusable NVML. A disposable
container uses the locally installed 26.01 image and read-only dependencies from
the existing 26.01 container: Python 3.12, torch 2.10.0a0, FA 2.7.4.post1,
three accessible GPUs. The earlier 24.09 trial cannot run the baseline's wrapped
FA API (FA 2.4.2). No existing container/installed dependency/driver was changed,
and no network/download was used. Failed trial logs remain outside Git under
`/home/cuiym/tmp/dspp-p1-20261023/`. PP4/8 GPU acceptance remains unperformed.

## PP3 paired diagnostics

Raw commands, logs, profiles, provenance, plan reports and snapshots are outside
Git at `/home/cuiym/tmp/dspp-p1-20261023/pp3-run7/` (container `/workspace/tmp/...`).
`gpu-pp3-summary.json` is the compact checked result. Model: PP3/VPP2, 6 layers,
hidden 512 / FFN 1024, Hq=4 / head dimension 128, BF16, chunk 64, capacity 256,
microbatch 4 / global 8, seed 1234, three optimizer steps, lengths
193/129/33/17, fused attention B, TP/CP/DP=1. Default receive lookahead/send
depth are 2/2, payloads 64 KiB. Profiles were collected on legacy first, frozen
before comparison, and fitted with the **existing** `from_records` interface.
These narrow execution fixtures are not P2 cost/owner calibration acceptance.

**Default FA numerical limitation:** in `pp3-run6`, template and a separate
legacy self-repeat both fail the pre-existing atol=.004/rtol=.01/relative-L2=.02
criteria at step 2. Step-1 forward outputs are identical; backward reduction
noise changes the optimizer trajectory. See `gpu-default-fa-boundary.json` and
its hashed raw failure logs. Criteria were not relaxed. The 3-step default-path
numerical gate is **not accepted**.

For isolation, numerical runs in `pp3-run7` uniformly select FA's existing
`deterministic=True` fused backward reduction via the diagnostic entry only.
Production attention/kernel sources are unchanged. Legacy→template and legacy→A
then have **zero difference** for every compared tensor: logits, loss, per-task
stage output/dx, all gradients, initial weights and three optimizer updates
(54 rank/iteration comparison reports). This is deterministic-FA diagnostic
acceptance, not default-FA or throughput acceptance.

| Heads | A accepted steps / 3 | A fallback | Default-FA clean allocated peaks by rank (MiB) | Reserved peaks (MiB) |
| --- | --- | --- | --- | --- |
| MHA, Hkv=4 | 3 | none | 749.662 / 130.086 / 130.112 | 796 / 144 / 144 |
| GQA, Hkv=2 | 3 | none | 737.100 / 118.061 / 117.729 | 778 / 132 / 130 |
| MQA, Hkv=1 | 2 | step 1: unreliable-compute-cost | 734.754 / 110.855 / 110.444 | 776 / 128 / 128 |

All eight accepted A step plans actually change W positions. Merged F/B orders,
full task sets and protocol drains are checked. MQA's fallback retains its exact
input template and is not counted as optimized A. Separate uninstrumented-tensor
memory passes use **default FA**, with per-step synchronized allocated/reserved
start/peak/end on every rank. All three methods have identical peaks in the
above runs; end allocation is stable after optimizer initialization. This short
observation is neither a complete owner contract nor a long-duration leak test.

Additional default-FA MHA protocol runs use lookahead=0/send depth=1. All
methods complete three steps; A accepts all three plans. Allocated/reserved
peaks match across methods. Whole-card NVML sampled every 50 ms has equal peaks
2816/1856/1716 MiB across methods (`gpu-small-protocol.json`); sampled peaks
cannot certify an exhaustive bound. Raw runs: `pp3-protocol-small/`.

External-plan stress uses a prebuilt serialized plan to exchange two ready W
of the same chunk while preserving F/B, dependencies and the P1 retention
envelope. Each of three steps permutes stage 0/chunk 1; runtime optimization is
bypassed by the public `dspp_plan=` input. These handwritten stress plans are
explicitly **not** algorithm A outputs/performance acceptance.
`gpu-external-w.json` preserves both results: LR=1e-3 passes steps 1/2 but fails
step 3 (first-step gradient error only 7.45e-9, then accumulated BF16/optimizer
trajectory drift). A separate LR=1e-4 legacy/fixture pair passes all three steps
at the unchanged criteria; max gradient/output/dx/logit/parameter-update errors
are .0004883 / 9.54e-7 / 2.29e-5 / .0039063 / .0002442. Both use deterministic
FA. The high-LR failure remains a numerical acceptance limit, not a success.
Raw fixture/failed run: `pp3-external-w-permutation/`; smaller-LR paired run:
`pp3-external-w-lr1e4/`. CPU tests additionally check exact closure identities,
not only aggregate gradients (summation can hide a wrong task-to-W association).

`examples/validate_dspp_plan_training.sh` and
`tools/validate_dspp_plan_training.py` reproduce paired numerical and independent
memory passes using fresh output/snapshot roots and externally frozen fused-B
profiles `hkv4.json`, `hkv2.json`, `hkv1.json`. Numerical isolation is opt-in with
`DSPP_PLAN_CHECK_DETERMINISTIC_FA=1`; memory always uses default FA in the shell
script. The shell driver rejects fallback-only A. `DSPP_PLAN_CHECK_PLAN_ROOT`
in the Python diagnostic accepts per-iteration serialized external plans.
Original failed trials and exact Python driver commands are preserved outside
Git; the shell driver itself was syntax-checked, while equivalent Python-driven
training invocations produced the measurements above.

Remaining P1 acceptance: default-FA multi-step numerical reproducibility and
high-LR non-FIFO W numerical stability;
complete real owner/whole-card memory contract and larger/longer protocol runs;
PP4/8 GPU acceptance when hardware is available. P2–P6 remain unimplemented.

All disposable P1 GPU containers were removed after validation. The original
training containers remain unchanged; new raw outputs are owned by the workspace
user. No commit/push was performed.

## Numerical root-cause follow-up

The observed three-step assertion failures are not evidence of degraded
convergence. FP32 accumulation order followed by optimizer/BF16 amplification
is a hypothesis, not an established root cause. The existing summaries do not
capture per-W contributions or complete Adam/master-parameter state. The
implementation plan §12.3 now specifies frozen-input/per-W/accumulation/optimizer
isolation, including same-coordinate checks and the original LR=1e-3 failure.
The isolation experiments have now run: exact original-kernel and original-Adam
replays locate the first divergence in FP32 accumulation order and reproduce
the subsequent BF16 boundary crossing. Frozen-weight three-step controls retain
identical outputs/dx/logits/loss. The high-LR numerical gate still fails and
longer convergence remains untested. See the [diagnosis report](../w-order-diagnosis-20261023/README.md)
and implementation plan §12.4. Earlier patch artifacts remain frozen snapshots;
the new diagnostic tools and results extend those snapshots.
