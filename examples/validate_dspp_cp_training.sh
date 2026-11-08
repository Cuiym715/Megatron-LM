#!/bin/bash
# Stage 2: identical PP2/VPP2 model and data, CP1 reference -> CP2 candidate.
# Numerical snapshots synchronize; these logs are not performance evidence.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DSPP_CP_CHECK_ROOT:?set a new independent results directory}"
: "${DATA_PATH:?use an existing DSPP variable-length dataset}"
PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/megatron/bin/python}
tokenizer=(--tokenizer-type NullTokenizer --vocab-size 32001)
if [[ -n ${TOKENIZER_MODEL:-} ]]; then
  tokenizer=(--tokenizer-type SentencePieceTokenizer --tokenizer-model "$TOKENIZER_MODEL")
fi
test ! -e "$DSPP_CP_CHECK_ROOT"
mkdir -p "$DSPP_CP_CHECK_ROOT"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
export DSPP_PLAN_CHECK_SNAPSHOTS="$DSPP_CP_CHECK_ROOT/snapshots"
candidate_cp=${DSPP_CP_SIZE:-2}
pp=${DSPP_PP_SIZE:-2}
layers=${DSPP_NUM_LAYERS:-4}
variants=${DSPP_CP_CHECK_VARIANTS:-"cp1 cp${candidate_cp}-fused cp${candidate_cp}-immediate"}
for variant in $variants; do
  extra=()
  if [[ $variant == cp1 ]]; then
    cp=1
    export DSPP_PLAN_CHECK_MODE=reference DSPP_PLAN_CHECK_CP_REFERENCE=0
  else
    cp=$candidate_cp
    export DSPP_PLAN_CHECK_MODE=compare DSPP_PLAN_CHECK_CP_REFERENCE=1
  fi
  if [[ $variant == *-immediate ]]; then
    extra+=(--dspp-split-attention-backward --dspp-history-dkv-after-dx)
  fi
  export DSPP_PLAN_CHECK_OUTPUT="$DSPP_CP_CHECK_ROOT/$variant"
  mkdir "$DSPP_PLAN_CHECK_OUTPUT"
  cmd=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=$((pp * cp))
    tools/validate_dspp_plan_training.py
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size "$pp" --context-parallel-size "$cp"
    --num-layers-per-virtual-pipeline-stage "$((layers/(pp*2)))" --num-layers "$layers"
    --hidden-size 512 --ffn-hidden-size 1024 --num-attention-heads 4
    --seq-length "${SEQ_LENGTH:-256}" --max-position-embeddings "${SEQ_LENGTH:-256}"
    --micro-seq-length "${MICRO_SEQ_LENGTH:-64}"
    --micro-batch-size "${MICRO_BATCH_SIZE:-4}" --global-batch-size "${GLOBAL_BATCH_SIZE:-8}" --train-iters "${TRAIN_ITERS:-3}" --seed 1234
    --lr 0.001 --min-lr 0.0001 --lr-decay-style cosine --weight-decay 0.1 --clip-grad 1.0
    --bf16 --use-flash-attn --no-masked-softmax-fusion
    --no-context-parallel-comm-overlap-gemm --kaimm-context-parallel-impl "${DSPP_CP_IMPL:-key-value}"
    --no-position-embedding --use-rotary-position-embeddings --alibi-bias-max 0
    --swiglu --rms-norm --disable-bias-linear --no-query-key-layer-scaling
    --hidden-dropout 0 --attention-dropout 0 --make-main-grad-addresss-divisible-by 1
    --data-path "$DATA_PATH" --data-impl mmap --split 100,0,0
    "${tokenizer[@]}"
    --dataloader-type cyclic --dspp --dspp-schedule template --dspp-v-layer-layout balanced
    --dspp-microbatch-order warmup-short-steady-long --dspp-packing-strategy "${DSPP_PACKING_STRATEGY:-baseline}"
    --dspp-recv-lookahead 0 --dspp-send-inflight-depth 1 --variable-seq-pad-token-id 0
    --dspp-cp-layout "${DSPP_CP_LAYOUT:-hybrid}"
    --eval-iters 0 --log-interval 1 --timing-log-level 0 "${extra[@]}")
  printf '%q ' "${cmd[@]}" > "$DSPP_PLAN_CHECK_OUTPUT/command.sh"
  printf '\n' >> "$DSPP_PLAN_CHECK_OUTPUT/command.sh"
  "$PYTHON_BIN" tools/wait_dspp_gpus.py --output "$DSPP_PLAN_CHECK_OUTPUT/gpu.jsonl"
  timeout --kill-after=15 300 "${cmd[@]}" > "$DSPP_PLAN_CHECK_OUTPUT/train.log" 2>&1
done
