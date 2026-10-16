#!/bin/bash
# Clean B/H boundaries only. Three independent repetitions rotate mode order;
# lr=0 keeps the weights fixed across modes/iterations. No audit/validation hook.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DSPP_CLEAN_MEASURE_ROOT:?set a new output directory}"
: "${DSPP_CLEAN_DATA_ROOT:?set a new external data directory}"
PYTHON_BIN=${PYTHON_BIN:-/workspace/src/venvs/megatron/bin/python}
test ! -e "$DSPP_CLEAN_MEASURE_ROOT"
test ! -e "$DSPP_CLEAN_DATA_ROOT"
mkdir -p "$DSPP_CLEAN_MEASURE_ROOT" "$DSPP_CLEAN_DATA_ROOT"
"$PYTHON_BIN" tools/build_dspp_synthetic_dataset.py --output-prefix "$DSPP_CLEAN_DATA_ROOT/data" \
  --training-lengths 6144,4097,1537,257 --repeats 32
for repeat in 1 2 3; do
  case "$repeat" in
    1) modes=(fused immediate after_dx);;
    2) modes=(after_dx fused immediate);;
    3) modes=(immediate after_dx fused);;
  esac
  for mode in "${modes[@]}"; do
    out="$DSPP_CLEAN_MEASURE_ROOT/repeat$repeat/$mode"
    mkdir -p "$out"
    split=0; after_dx=0
    if [[ "$mode" != fused ]]; then split=1; fi
    if [[ "$mode" == after_dx ]]; then after_dx=1; fi
    TRAIN_ITERS=3 HIDDEN_SIZE=512 FFN_HIDDEN_SIZE=1024 NUM_ATTENTION_HEADS=4 NUM_QUERY_GROUPS=2 \
    NUM_LAYERS=6 LAYERS_PER_VPP=1 SEQ_LENGTH=8192 MICRO_SEQ_LENGTH=2048 \
    MICRO_BATCH_SIZE=4 GLOBAL_BATCH_SIZE=8 DSPP_DEBUG_BATCHES=0 LEARNING_RATE=0 MIN_LEARNING_RATE=0 \
    DATA_PATH="$DSPP_CLEAN_DATA_ROOT/data" DSPP_TASK_TIMING_ONLY=1 \
    DSPP_SPLIT_ATTENTION_BACKWARD="$split" DSPP_HISTORY_DKV_AFTER_DX="$after_dx" \
    DSPP_TIMELINE_DIR="$out/timeline" DSPP_TIMELINE_ITERATION=3 \
    PYTHON_BIN="$PYTHON_BIN" TRAIN_ENTRYPOINT=pretrain_llama.py \
      timeout 300 bash examples/run_dspp_b23_l40.sh > "$out/train.log" 2>&1
  done
  "$PYTHON_BIN" tools/summarize_dspp_history_tasks.py --root "$DSPP_CLEAN_MEASURE_ROOT/repeat$repeat"
done
