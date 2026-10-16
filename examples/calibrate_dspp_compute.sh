#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Offline, compute-only calibration.  Override these values for the target
# model and target accelerator; no PP process group, P2P traffic, dataloader,
# training iteration, or prior end-to-end result is used.
PYTHON_BIN=${PYTHON_BIN:-/workspace/src/venvs/megatron/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/src/results/dspp_offline_compute}
ALIGNMENT=${ALIGNMENT:-128}
HIDDEN_SIZE=${HIDDEN_SIZE:-2048}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-5504}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-16}
NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-$NUM_ATTENTION_HEADS}
LAYERS_PER_STAGE=${LAYERS_PER_STAGE:-4}
PROFILE_DTYPE=${PROFILE_DTYPE:-bf16}
PROFILE_WARMUP=${PROFILE_WARMUP:-3}
PROFILE_REPEATS=${PROFILE_REPEATS:-10}
REFINEMENT_ROUNDS=${REFINEMENT_ROUNDS:-2}
REFINEMENT_POINTS=${REFINEMENT_POINTS:-6}
EFFICIENCY_THRESHOLD=${EFFICIENCY_THRESHOLD:-0.90}
PROBE_TIMEOUT_SECONDS=${PROBE_TIMEOUT_SECONDS:-90}
PROBE_OPERATIONS=${PROBE_OPERATIONS:-B}
PROFILE_OPERATIONS=${PROFILE_OPERATIONS:-F,B,W}
HISTORY_RATIOS=${HISTORY_RATIOS:-0,1,2,4}
TAIL_RATIOS=${TAIL_RATIOS:-0.25,0.5,0.75}
ITEM_COUNTS=${ITEM_COUNTS:-2,4}
PROFILE_MAX_CAPACITY=${PROFILE_MAX_CAPACITY:-}
NO_CUDA_GRAPH=${NO_CUDA_GRAPH:-0}

mkdir -p "$OUTPUT_ROOT"
ADAPTER_KWARGS=$(printf \
  '{"hidden_size":%s,"ffn_hidden_size":%s,"num_heads":%s,"num_query_groups":%s,"layers_per_stage":%s,"dtype":"%s"}' \
  "$HIDDEN_SIZE" "$FFN_HIDDEN_SIZE" "$NUM_ATTENTION_HEADS" \
  "$NUM_QUERY_GROUPS" "$LAYERS_PER_STAGE" "$PROFILE_DTYPE")

CAPACITY_ARGS=(--auto-capacity-range)
if [[ -n "${MIN_CAPACITY:-}" || -n "${MAX_CAPACITY:-}" ]]; then
  if [[ -z "${MIN_CAPACITY:-}" || -z "${MAX_CAPACITY:-}" ]]; then
    echo "MIN_CAPACITY and MAX_CAPACITY must be set together" >&2
    exit 2
  fi
  CAPACITY_ARGS=(--min-capacity "$MIN_CAPACITY" --max-capacity "$MAX_CAPACITY")
fi
HISTORY_ARGS=()
if [[ -n "${MAX_HISTORY_TOKENS:-}" ]]; then
  HISTORY_ARGS=(--max-history-tokens "$MAX_HISTORY_TOKENS")
fi
PROFILE_MAX_ARGS=()
if [[ -n "$PROFILE_MAX_CAPACITY" ]]; then
  PROFILE_MAX_ARGS=(--profile-max-capacity "$PROFILE_MAX_CAPACITY")
fi
GRAPH_ARGS=()
if [[ "$NO_CUDA_GRAPH" == "1" ]]; then
  GRAPH_ARGS=(--no-cuda-graph)
fi

"$PYTHON_BIN" tools/profile_dspp_compute_shapes.py \
  --adapter megatron.core.datasets.dspp_local_compute_adapter:local_llama_adapter \
  --adapter-kwargs-json "$ADAPTER_KWARGS" \
  "${CAPACITY_ARGS[@]}" \
  --alignment "$ALIGNMENT" \
  "${HISTORY_ARGS[@]}" \
  --warmup "$PROFILE_WARMUP" \
  --repeats "$PROFILE_REPEATS" \
  --refinement-rounds "$REFINEMENT_ROUNDS" \
  --refinement-points-per-round "$REFINEMENT_POINTS" \
  --efficiency-threshold "$EFFICIENCY_THRESHOLD" \
  --probe-timeout-seconds "$PROBE_TIMEOUT_SECONDS" \
  --probe-operations "$PROBE_OPERATIONS" \
  --operations "$PROFILE_OPERATIONS" \
  --history-ratios "$HISTORY_RATIOS" \
  --tail-ratios "$TAIL_RATIOS" \
  --item-counts "$ITEM_COUNTS" \
  "${PROFILE_MAX_ARGS[@]}" \
  "${GRAPH_ARGS[@]}" \
  --output "$OUTPUT_ROOT/compute_profile.json"

"$PYTHON_BIN" tools/generate_dspp_compute_menu.py \
  --profile "$OUTPUT_ROOT/compute_profile.json" \
  --efficiency-threshold "$EFFICIENCY_THRESHOLD" \
  --output "$OUTPUT_ROOT/chunk_menu.json"

echo "[dspp][offline-compute] profile=$OUTPUT_ROOT/compute_profile.json"
echo "[dspp][offline-compute] menu=$OUTPUT_ROOT/chunk_menu.json"
