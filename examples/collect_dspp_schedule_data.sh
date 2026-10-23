#!/bin/bash
# Run inside the prepared container, on the current checkout. Exactly 63 steps.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DSPP_COLLECTION_ROOT:?set a NEW raw-output directory}"
: "${DSPP_COLLECTION_DATA_PATH:?set the new dataset prefix}"
test ! -e "$DSPP_COLLECTION_ROOT"
mkdir -p "$DSPP_COLLECTION_ROOT"
export TORCH_CUDA_ARCH_LIST=8.9
export HIDDEN_SIZE=4096 FFN_HIDDEN_SIZE=8192 NUM_ATTENTION_HEADS=32 NUM_QUERY_GROUPS=16
export NUM_LAYERS=6 LAYERS_PER_VPP=1 SEQ_LENGTH=16384 MICRO_SEQ_LENGTH=2048
export MICRO_BATCH_SIZE=5 GLOBAL_BATCH_SIZE=10 DSPP_DEBUG_BATCHES=0
export LEARNING_RATE=0 MIN_LEARNING_RATE=0 LOG_INTERVAL=1
export DATA_PATH="$DSPP_COLLECTION_DATA_PATH" DSPP_TASK_TIMING_SCOPE=all
export TRAIN_ENTRYPOINT=tools/collect_dspp_schedule_data.py
export DSPP_ORDER=warmup-short-steady-long DSPP_PACKING_STRATEGY=baseline DSPP_AUTO_CHUNK_SIZE=0
export DSPP_RECV_LOOKAHEAD=2 DSPP_SEND_INFLIGHT_DEPTH=2
for mode in fused after_dx; do
  split=0; if [[ "$mode" == after_dx ]]; then split=1; fi
  out="$DSPP_COLLECTION_ROOT/$mode"
  mkdir -p "$out"
  echo "Starting $mode: 30 iterations"
  TRAIN_ITERS=30 DSPP_COLLECTION_OUTPUT="$out" DSPP_COLLECTION_REPLAY='' \
  DSPP_TASK_TIMING_ONLY=1 DSPP_TIMELINE_DIR="$out/timeline" DSPP_TIMELINE_ITERATIONS='26 27 28 29 30' \
  DSPP_SPLIT_ATTENTION_BACKWARD="$split" DSPP_HISTORY_DKV_AFTER_DX="$split" \
  DSPP_TORCH_PROFILER_DIR='' \
    timeout 900 bash examples/run_dspp_b23_l40.sh > "$out/train.log" 2>&1
  echo "Finished $mode"
done
out="$DSPP_COLLECTION_ROOT/diagnostic"
mkdir -p "$out"
echo 'Starting diagnostic: replay input26, 3 iterations'
TRAIN_ITERS=3 DSPP_COLLECTION_OUTPUT="$out" DSPP_COLLECTION_REPLAY="$DSPP_COLLECTION_ROOT/after_dx" \
DSPP_TASK_TIMING_ONLY=0 DSPP_TIMELINE_DIR="$out/timeline" DSPP_TIMELINE_ITERATIONS='3' \
DSPP_SPLIT_ATTENTION_BACKWARD=1 DSPP_HISTORY_DKV_AFTER_DX=1 \
DSPP_TORCH_PROFILER_DIR='' DSPP_TORCH_PROFILER_ITERATION=0 \
  timeout 900 bash examples/run_dspp_b23_l40.sh > "$out/train.log" 2>&1
echo 'Completed 63 iterations; no automatic performance repeats.'
