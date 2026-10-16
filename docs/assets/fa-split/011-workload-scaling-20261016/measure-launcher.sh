#!/bin/bash
# One paired run by default; reverse order explicitly only if results warrant it.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DSPP_COMBINED_MEASURE_ROOT:?set a new output directory}"
: "${DSPP_COMBINED_DATA_PATH:?set the existing input dataset prefix}"
PYTHON_BIN=${PYTHON_BIN:-/workspace/src/venvs/megatron/bin/python}
hidden=${DSPP_COMBINED_HIDDEN_SIZE:-512}
ffn=${DSPP_COMBINED_FFN_HIDDEN_SIZE:-1024}
heads=${DSPP_COMBINED_Q_HEADS:-4}
query_groups=${DSPP_COMBINED_QUERY_GROUPS:-2}
seq=${DSPP_COMBINED_SEQ_LENGTH:-8192}
chunk=${DSPP_COMBINED_CHUNK_LENGTH:-2048}
for value in "$hidden" "$ffn" "$heads" "$query_groups" "$seq" "$chunk"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]]
done
(( hidden == heads * 128 && heads % query_groups == 0 && seq >= chunk ))
# Equal Q/KV head counts are MHA: omit the GQA switch entirely.
if [[ "$query_groups" == "$heads" ]]; then query_groups=; fi
test ! -e "$DSPP_COMBINED_MEASURE_ROOT"
read -ra modes <<< "${DSPP_COMBINED_MODE_ORDER:-fused after_dx}"
[[ "${modes[*]}" == 'fused after_dx' || "${modes[*]}" == 'after_dx fused' ]]
mkdir -p "$DSPP_COMBINED_MEASURE_ROOT"
for mode in "${modes[@]}"; do
  out="$DSPP_COMBINED_MEASURE_ROOT/$mode"
  mkdir -p "$out"
  split=0
  if [[ "$mode" == after_dx ]]; then split=1; fi
  TRAIN_ITERS=3 HIDDEN_SIZE="$hidden" FFN_HIDDEN_SIZE="$ffn" NUM_ATTENTION_HEADS="$heads" NUM_QUERY_GROUPS="$query_groups" \
  NUM_LAYERS=6 LAYERS_PER_VPP=1 SEQ_LENGTH="$seq" MICRO_SEQ_LENGTH="$chunk" \
  MICRO_BATCH_SIZE=4 GLOBAL_BATCH_SIZE=8 DSPP_DEBUG_BATCHES=0 LEARNING_RATE=0 MIN_LEARNING_RATE=0 \
  DATA_PATH="$DSPP_COMBINED_DATA_PATH" DSPP_TASK_TIMING_ONLY=1 \
  DSPP_SPLIT_ATTENTION_BACKWARD="$split" DSPP_HISTORY_DKV_AFTER_DX="$split" \
  DSPP_TIMELINE_DIR="$out/timeline" DSPP_TIMELINE_ITERATION=3 \
  PYTHON_BIN="$PYTHON_BIN" TRAIN_ENTRYPOINT=pretrain_llama.py \
    timeout 300 bash examples/run_dspp_b23_l40.sh > "$out/train.log" 2>&1
done
"$PYTHON_BIN" tools/summarize_dspp_history_tasks.py --root "$DSPP_COMBINED_MEASURE_ROOT" --modes fused after_dx
