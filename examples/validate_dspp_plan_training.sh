#!/bin/bash
# P1 paired PP3 engineering checks. Run inside the established training container.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DSPP_PLAN_CHECK_ROOT:?set a new external output directory}"
: "${DSPP_PLAN_CHECK_SNAPSHOT_ROOT:?set a new external snapshot directory}"
: "${DSPP_PLAN_PROFILE_ROOT:?frozen fused-B profiles hkv4.json,hkv2.json,hkv1.json}"
PYTHON_BIN=${PYTHON_BIN:-/workspace/src/venvs/megatron/bin/python}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-/workspace/src/tokenizers/Mistral-7B-v0.1/tokenizer.model}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
numerical_deterministic_fa=${DSPP_PLAN_CHECK_DETERMINISTIC_FA:-0}
test ! -e "$DSPP_PLAN_CHECK_ROOT"
test ! -e "$DSPP_PLAN_CHECK_SNAPSHOT_ROOT"
# Fail before creating any run output if GPUs or profile inputs are unavailable.
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 3, "PP3 validation requires three accessible GPUs"'
for groups in 4 2 1; do test -f "$DSPP_PLAN_PROFILE_ROOT/hkv${groups}.json"; done
mkdir -p "$DSPP_PLAN_CHECK_ROOT" "$DSPP_PLAN_CHECK_SNAPSHOT_ROOT"
"$PYTHON_BIN" tools/build_dspp_synthetic_dataset.py \
  --output-prefix "$DSPP_PLAN_CHECK_SNAPSHOT_ROOT/data" --training-lengths 193,129,33,17 --repeats 32
for groups in 4 2 1; do
  extra=()
  if [[ "$groups" != 4 ]]; then extra+=(--group-query-attention --num-query-groups "$groups"); fi
  for method in legacy template a; do
    for check in numerical memory; do
      export DSPP_PLAN_CHECK_OUTPUT="$DSPP_PLAN_CHECK_ROOT/hkv${groups}/${method}/${check}"
      export DSPP_PLAN_CHECK_SNAPSHOTS="$DSPP_PLAN_CHECK_SNAPSHOT_ROOT/hkv${groups}"
      export DSPP_PLAN_CHECK_MODE=compare
      if [[ "$method" == legacy ]]; then export DSPP_PLAN_CHECK_MODE=reference; fi
      if [[ "$check" == memory ]]; then export DSPP_PLAN_CHECK_MODE=memory; fi
      # Optional numerical isolation only. Clean memory always uses production FA.
      export DSPP_PLAN_CHECK_DETERMINISTIC_FA=0
      if [[ "$check" == numerical ]]; then
        export DSPP_PLAN_CHECK_DETERMINISTIC_FA="$numerical_deterministic_fa"
      fi
      mkdir -p "$DSPP_PLAN_CHECK_OUTPUT"
      timeout 300 "$PYTHON_BIN" -m torch.distributed.run --nproc-per-node=3 \
        --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT:-29521}" \
        tools/validate_dspp_plan_training.py \
        --tensor-model-parallel-size 1 --pipeline-model-parallel-size 3 \
        --num-layers-per-virtual-pipeline-stage 1 --num-layers 6 \
        --hidden-size 512 --ffn-hidden-size 1024 --num-attention-heads 4 \
        --seq-length 256 --max-position-embeddings 256 --micro-seq-length 64 \
        --micro-batch-size 4 --global-batch-size 8 --train-iters 3 --seed 1234 \
        --lr .001 --min-lr .0001 --lr-decay-style cosine --weight-decay .1 --clip-grad 1 \
        --bf16 --use-flash-attn --no-masked-softmax-fusion --no-position-embedding --use-rotary-position-embeddings \
        --swiglu --rms-norm --disable-bias-linear --no-query-key-layer-scaling \
        --hidden-dropout 0 --attention-dropout 0 --make-main-grad-addresss-divisible-by 1 \
        --data-path "$DSPP_PLAN_CHECK_SNAPSHOT_ROOT/data" --data-impl mmap --split 100,0,0 \
        --tokenizer-type SentencePieceTokenizer --tokenizer-model "$TOKENIZER_MODEL" \
        --dataloader-type cyclic --dspp --dspp-v-layer-layout balanced --dspp-schedule "$method" \
        --dspp-execution-profile "$DSPP_PLAN_PROFILE_ROOT/hkv${groups}.json" \
        --dspp-microbatch-order warmup-short-steady-long --dspp-packing-strategy baseline \
        --dspp-recv-lookahead "${DSPP_PLAN_RECV_LOOKAHEAD:-2}" \
        --dspp-send-inflight-depth "${DSPP_PLAN_SEND_DEPTH:-2}" --variable-seq-pad-token-id 0 \
        --variable-seq-debug-num-batches 3 --eval-iters 0 --log-interval 1 \
        "${extra[@]}" > "$DSPP_PLAN_CHECK_OUTPUT/train.log" 2>&1
    done
  done
done
# A fallback-only run is useful evidence but is not an optimized-A acceptance.
"$PYTHON_BIN" - "$DSPP_PLAN_CHECK_ROOT" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
for groups in (4, 2, 1):
    reports = [json.loads(p.read_text()) for p in (root / f'hkv{groups}/a/numerical').glob('iteration*-rank*.json')]
    assert len(reports) == 9, (groups, 'incomplete PP3 records')
    accepted = [r for r in reports if r['plan']['algorithm'] == 'A' and not r['plan']['fallback_reason']]
    assert accepted, (groups, 'A only used fallback; inspect plan reasons, not an optimized-A pass')
PY
