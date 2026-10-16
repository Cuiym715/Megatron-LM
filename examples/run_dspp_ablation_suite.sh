#!/bin/bash
set -u

cd "$(dirname "$0")/.."

RESULT_ROOT=${RESULT_ROOT:-/workspace/src/results/dspp_ablation_suite_20260913}
TRAIN_ITERS=${TRAIN_ITERS:-4}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-240}
SHARED_P2P_TIMEOUT_SECONDS=${SHARED_P2P_TIMEOUT_SECONDS:-60}
SEEDS=${SEEDS:-"11 19 29"}
SUITES=${SUITES:-"packing_ordering communication chunk"}

CHUNK_MENU=${CHUNK_MENU:-/workspace/src/results/dspp_chunk_dense_20260913/chunk_menu_dense.json}
EXECUTION_PROFILE=${EXECUTION_PROFILE:-/workspace/src/results/dspp_chunk_dense_20260913/compute_profile_dense_calibrated.json}

mkdir -p "$RESULT_ROOT"
manifest="$RESULT_ROOT/run_manifest.tsv"
printf 'suite\tconfig\tseed\tstatus\tresult_dir\n' > "$manifest"

run_one() {
  suite=$1
  config=$2
  seed=$3
  shift 3

  result_dir="$RESULT_ROOT/$suite/${config}_seed${seed}"
  data_path="/tmp/dspp_multisample_seed${seed}_max16k_drop_text_document"
  run_timeout=$TIMEOUT_SECONDS
  if [[ "$config" == shared_directional ]]; then
    run_timeout=$SHARED_P2P_TIMEOUT_SECONDS
  fi
  mkdir -p "$result_dir"
  echo "[ablation] suite=$suite config=$config seed=$seed"

  timeout "$run_timeout" env \
    TRAIN_ITERS="$TRAIN_ITERS" \
    LOG_INTERVAL=1 \
    NUM_LAYERS=24 \
    LAYERS_PER_VPP=4 \
    HIDDEN_SIZE=2048 \
    FFN_HIDDEN_SIZE=5504 \
    NUM_ATTENTION_HEADS=16 \
    SEQ_LENGTH=16384 \
    MAX_POSITION_EMBEDDINGS=16384 \
    MICRO_SEQ_LENGTH=3584 \
    MICRO_BATCH_SIZE=4 \
    GLOBAL_BATCH_SIZE=16 \
    DATALOADER_TYPE=single \
    DSPP_PACKING_STRATEGY=baseline \
    DSPP_RECV_LOOKAHEAD=2 \
    DSPP_SEND_INFLIGHT_DEPTH=2 \
    DATA_PATH="$data_path" \
    DSPP_METRICS_PATH="$result_dir/metrics.json" \
    "$@" \
    bash examples/run_dspp_b23_l40.sh >"$result_dir/train.log" 2>&1
  status=$?
  printf '%s\n' "$status" > "$result_dir/exit_status.txt"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$suite" "$config" "$seed" "$status" "$result_dir" >> "$manifest"
  if [[ $status -ne 0 ]]; then
    echo "[ablation] failed suite=$suite config=$config seed=$seed status=$status"
  fi
}

contains_suite() {
  case " $SUITES " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

if contains_suite packing_ordering; then
  for seed in $SEEDS; do
    run_one packing_ordering pack_input "$seed" \
      DSPP_ORDER=input DSPP_DISABLE_RESIDUAL_PACKING=0 DSPP_AUTO_CHUNK_SIZE=0
    run_one packing_ordering pack_ordered "$seed" \
      DSPP_ORDER=warmup-short-steady-long DSPP_DISABLE_RESIDUAL_PACKING=0 DSPP_AUTO_CHUNK_SIZE=0
    run_one packing_ordering no_pack_input "$seed" \
      DSPP_ORDER=input DSPP_DISABLE_RESIDUAL_PACKING=1 DSPP_AUTO_CHUNK_SIZE=0
    run_one packing_ordering no_pack_ordered "$seed" \
      DSPP_ORDER=warmup-short-steady-long DSPP_DISABLE_RESIDUAL_PACKING=1 DSPP_AUTO_CHUNK_SIZE=0
  done
fi

if contains_suite communication; then
  for seed in $SEEDS; do
    run_one communication split_directional "$seed" \
      DSPP_ORDER=warmup-short-steady-long DSPP_DISABLE_RESIDUAL_PACKING=0 \
      DSPP_SHARE_DIRECTIONAL_P2P=0 DSPP_AUTO_CHUNK_SIZE=0
    run_one communication shared_directional "$seed" \
      DSPP_ORDER=warmup-short-steady-long DSPP_DISABLE_RESIDUAL_PACKING=0 \
      DSPP_SHARE_DIRECTIONAL_P2P=1 DSPP_AUTO_CHUNK_SIZE=0
  done
fi

if contains_suite chunk; then
  for seed in $SEEDS; do
    run_one chunk fixed_2048 "$seed" \
      MICRO_SEQ_LENGTH=2048 DSPP_AUTO_CHUNK_SIZE=0
    run_one chunk fixed_4096 "$seed" \
      MICRO_SEQ_LENGTH=4096 DSPP_AUTO_CHUNK_SIZE=0
    run_one chunk max_feasible "$seed" \
      DSPP_AUTO_CHUNK_SIZE=1 DSPP_CHUNK_SELECTION=memory \
      DSPP_MIN_CHUNK_SIZE=256 DSPP_MAX_CHUNK_SIZE=0 \
      DSPP_MEMORY_SAFETY_FACTOR=1.05
    run_one chunk profiled_auto "$seed" \
      DSPP_AUTO_CHUNK_SIZE=1 DSPP_CHUNK_SELECTION=per-iteration-menu \
      DSPP_CHUNK_MENU="$CHUNK_MENU" DSPP_EXECUTION_PROFILE="$EXECUTION_PROFILE" \
      DSPP_MIN_CHUNK_SIZE=256 DSPP_MAX_CHUNK_SIZE=0 \
      DSPP_MEMORY_SAFETY_FACTOR=1.05
  done
fi

"${PYTHON_BIN:-/workspace/src/venvs/megatron/bin/python}" \
  tools/summarize_dspp_ablation.py \
  --result-root "$RESULT_ROOT" \
  --output "$RESULT_ROOT/summary.json"
