#!/bin/bash
# Paired real PP=3/VPP=2 correctness runs. Snapshots are intentionally external
# to Git; this script never replaces an existing output or reference directory.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DSPP_SPLIT_CHECK_ROOT:?set a new output directory}"
: "${DSPP_SPLIT_CHECK_SNAPSHOT_ROOT:?set a new external snapshot directory}"
PYTHON_BIN=${PYTHON_BIN:-/workspace/src/venvs/megatron/bin/python}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-/workspace/src/tokenizers/Mistral-7B-v0.1/tokenizer.model}
test ! -e "$DSPP_SPLIT_CHECK_ROOT"
test ! -e "$DSPP_SPLIT_CHECK_SNAPSHOT_ROOT"
mkdir -p "$DSPP_SPLIT_CHECK_ROOT" "$DSPP_SPLIT_CHECK_SNAPSHOT_ROOT"
"$PYTHON_BIN" tools/build_dspp_synthetic_dataset.py \
 --output-prefix "$DSPP_SPLIT_CHECK_SNAPSHOT_ROOT/data" --training-lengths 193,129,33,17 --repeats 32
for groups in 4 2 1; do
  for mode in fused split; do
    extra=()
    if [[ "${DSPP_SPLIT_CHECK_TASK_TIMING_ONLY:-0}" == 1 ]]; then
      extra+=(--dspp-task-timing-only --timing-log-level 0)
    fi
    if [[ "$mode" == split ]]; then extra+=(--dspp-split-attention-backward); fi
    if [[ "$mode" == split && "${DSPP_SPLIT_CHECK_AFTER_DX:-0}" == 1 ]]; then
      extra+=(--dspp-history-dkv-after-dx)
    fi
    if [[ "$groups" != 4 ]]; then extra+=(--group-query-attention --num-query-groups "$groups"); fi
    export DSPP_SPLIT_CHECK_OUTPUT="$DSPP_SPLIT_CHECK_ROOT/hkv${groups}-${mode}"
    export DSPP_SPLIT_CHECK_SNAPSHOTS="$DSPP_SPLIT_CHECK_SNAPSHOT_ROOT/hkv${groups}"
    mkdir -p "$DSPP_SPLIT_CHECK_OUTPUT"
    timeout 240 "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=3 \
      tools/validate_dspp_split_training.py \
      --tensor-model-parallel-size 1 --pipeline-model-parallel-size 3 \
      --num-layers-per-virtual-pipeline-stage 1 --num-layers 6 \
      --hidden-size 512 --ffn-hidden-size 1024 --num-attention-heads 4 \
      --seq-length 256 --max-position-embeddings 256 --micro-seq-length 64 \
      --micro-batch-size 4 --global-batch-size 8 --train-iters 3 --seed 1234 \
      --lr 0.001 --min-lr 0.0001 --lr-decay-style cosine --weight-decay 0.1 --clip-grad 1.0 \
      --bf16 --use-flash-attn --no-position-embedding --use-rotary-position-embeddings \
      --swiglu --rms-norm --disable-bias-linear --no-query-key-layer-scaling \
      --hidden-dropout 0 --attention-dropout 0 --make-main-grad-addresss-divisible-by 1 \
      --data-path "$DSPP_SPLIT_CHECK_SNAPSHOT_ROOT/data" --data-impl mmap --split 100,0,0 \
      --tokenizer-type SentencePieceTokenizer --tokenizer-model "$TOKENIZER_MODEL" \
      --dataloader-type cyclic --dspp --dspp-v-layer-layout balanced \
      --dspp-microbatch-order warmup-short-steady-long --dspp-packing-strategy baseline \
      --dspp-recv-lookahead 2 --dspp-send-inflight-depth 2 --variable-seq-pad-token-id 0 \
      --variable-seq-debug-num-batches 3 --eval-iters 0 --log-interval 1 \
      "${extra[@]}" > "$DSPP_SPLIT_CHECK_OUTPUT/train.log" 2>&1
  done
done
