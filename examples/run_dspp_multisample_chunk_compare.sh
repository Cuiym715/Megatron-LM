#!/bin/bash
set -u

cd "$(dirname "$0")/.."

RESULT_ROOT=${RESULT_ROOT:-/workspace/src/results/dspp_multisample_chunk_compare_20260912}
TRAIN_ITERS=${TRAIN_ITERS:-4}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-180}
SEEDS=${SEEDS:-"11 19 29"}
CHUNKS=${CHUNKS:-"2048 3072 4096"}

mkdir -p "$RESULT_ROOT"

for seed in $SEEDS; do
  data_path="/tmp/dspp_multisample_seed${seed}_max16k_drop_text_document"
  for chunk in $CHUNKS; do
    result_dir="$RESULT_ROOT/seed${seed}_chunk${chunk}"
    mkdir -p "$result_dir"
    echo "[multisample] seed=$seed chunk=$chunk"
    timeout "$TIMEOUT_SECONDS" env \
      TRAIN_ITERS="$TRAIN_ITERS" \
      LOG_INTERVAL=1 \
      NUM_LAYERS=24 \
      LAYERS_PER_VPP=4 \
      HIDDEN_SIZE=2048 \
      FFN_HIDDEN_SIZE=5504 \
      NUM_ATTENTION_HEADS=16 \
      SEQ_LENGTH=16384 \
      MAX_POSITION_EMBEDDINGS=16384 \
      MICRO_SEQ_LENGTH="$chunk" \
      MICRO_BATCH_SIZE=4 \
      GLOBAL_BATCH_SIZE=16 \
      DATALOADER_TYPE=single \
      DSPP_AUTO_CHUNK_SIZE=0 \
      DATA_PATH="$data_path" \
      DSPP_METRICS_PATH="$result_dir/metrics.json" \
      bash examples/run_dspp_b23_l40.sh >"$result_dir/train.log" 2>&1
    status=$?
    printf '%s\n' "$status" >"$result_dir/exit_status.txt"
    if [[ $status -ne 0 ]]; then
      echo "[multisample] failed seed=$seed chunk=$chunk status=$status"
    fi
  done
done
