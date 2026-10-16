#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_ITERS=${TRAIN_ITERS:-10}
LOG_INTERVAL=${LOG_INTERVAL:-1}
NUM_LAYERS=${NUM_LAYERS:-6}
LAYERS_PER_VPP=${LAYERS_PER_VPP:-1}
DSPP_V_LAYER_LAYOUT=${DSPP_V_LAYER_LAYOUT:-balanced}
HIDDEN_SIZE=${HIDDEN_SIZE:-64}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-128}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-4}
SEQ_LENGTH=${SEQ_LENGTH:-96}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-$SEQ_LENGTH}
MICRO_SEQ_LENGTH=${MICRO_SEQ_LENGTH:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
DATALOADER_TYPE=${DATALOADER_TYPE:-cyclic}
DSPP_DEBUG_BATCHES=${DSPP_DEBUG_BATCHES:-0}
DSPP_ORDER=${DSPP_ORDER:-warmup-short-steady-long}
DSPP_DISABLE_RESIDUAL_PACKING=${DSPP_DISABLE_RESIDUAL_PACKING:-0}
DSPP_PACKING_STRATEGY=${DSPP_PACKING_STRATEGY:-baseline}
DSPP_ORDER_PROFILE=${DSPP_ORDER_PROFILE:-}
DSPP_EXECUTION_PROFILE=${DSPP_EXECUTION_PROFILE:-}
DSPP_GLOBAL_FIXED_TRACE=${DSPP_GLOBAL_FIXED_TRACE:-}
DSPP_GLOBAL_FIXED_CHUNKS=${DSPP_GLOBAL_FIXED_CHUNKS:-}
DSPP_GLOBAL_FIXED_MAX_ITERS=${DSPP_GLOBAL_FIXED_MAX_ITERS:-32}
DSPP_GLOBAL_FIXED_OUTPUT=${DSPP_GLOBAL_FIXED_OUTPUT:-/tmp/dspp_global_fixed_selection.json}
DSPP_GLOBAL_FIXED_ALLOW_LOW_COVERAGE=${DSPP_GLOBAL_FIXED_ALLOW_LOW_COVERAGE:-0}
DSPP_GLOBAL_FIXED_ACTIVATION_BUDGET_BYTES=${DSPP_GLOBAL_FIXED_ACTIVATION_BUDGET_BYTES:-0}
DSPP_AUTO_CHUNK_SIZE=${DSPP_AUTO_CHUNK_SIZE:-0}
DSPP_CHUNK_SELECTION=${DSPP_CHUNK_SELECTION:-memory}
DSPP_CHUNK_MENU=${DSPP_CHUNK_MENU:-}
DSPP_SATURATION_CHUNK_SIZE=${DSPP_SATURATION_CHUNK_SIZE:-0}
DSPP_SATURATION_SLACK=${DSPP_SATURATION_SLACK:-0.20}
DSPP_MIN_CHUNK_SIZE=${DSPP_MIN_CHUNK_SIZE:-256}
DSPP_MAX_CHUNK_SIZE=${DSPP_MAX_CHUNK_SIZE:-0}
DSPP_CHUNK_ALIGNMENT=${DSPP_CHUNK_ALIGNMENT:-256}
DSPP_EFFICIENCY_TOP_K=${DSPP_EFFICIENCY_TOP_K:-4}
DSPP_EFFICIENCY_SLACK=${DSPP_EFFICIENCY_SLACK:-0.05}
DSPP_MEMORY_UTILIZATION=${DSPP_MEMORY_UTILIZATION:-0.90}
DSPP_MEMORY_RESERVE_MIB=${DSPP_MEMORY_RESERVE_MIB:-1024}
DSPP_MEMORY_SAFETY_FACTOR=${DSPP_MEMORY_SAFETY_FACTOR:-1.05}
DSPP_ACTIVATION_BYTES_PER_TOKEN=${DSPP_ACTIVATION_BYTES_PER_TOKEN:-0}
DSPP_MAX_P2P_BYTES=${DSPP_MAX_P2P_BYTES:-0}
DSPP_RECV_LOOKAHEAD=${DSPP_RECV_LOOKAHEAD:-2}
DSPP_SEND_INFLIGHT_DEPTH=${DSPP_SEND_INFLIGHT_DEPTH:-2}
DSPP_SHARE_DIRECTIONAL_P2P=${DSPP_SHARE_DIRECTIONAL_P2P:-0}
DSPP_TIMELINE_DIR=${DSPP_TIMELINE_DIR:-}
DSPP_TIMELINE_ITERATION=${DSPP_TIMELINE_ITERATION:-0}
DSPP_TORCH_PROFILER_DIR=${DSPP_TORCH_PROFILER_DIR:-}
DSPP_TORCH_PROFILER_ITERATION=${DSPP_TORCH_PROFILER_ITERATION:-0}
DSPP_METRICS_PATH=${DSPP_METRICS_PATH:-}
DATA_PATH=${DATA_PATH:-/tmp/dspp_b1_varlen_20260902_text_document}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-/workspace/src/tokenizers/Mistral-7B-v0.1/tokenizer.model}
PYTHON_BIN=${PYTHON_BIN:-/workspace/src/venvs/megatron/bin/python}

# Optional startup-time global fixed selection. This is deliberately separate
# from per-iteration auto chunking: it evaluates a representative length trace
# once, then launches the entire run with one fixed model shape.
if [[ -n "$DSPP_GLOBAL_FIXED_TRACE" ]]; then
  if [[ -z "$DSPP_EXECUTION_PROFILE" ]]; then
    echo "DSPP_GLOBAL_FIXED_TRACE requires DSPP_EXECUTION_PROFILE" >&2
    exit 1
  fi
  if [[ -z "$DSPP_GLOBAL_FIXED_CHUNKS" ]]; then
    echo "DSPP_GLOBAL_FIXED_TRACE requires DSPP_GLOBAL_FIXED_CHUNKS" >&2
    exit 1
  fi
  GLOBAL_SELECTOR_ARGS=(
    --length-trace "$DSPP_GLOBAL_FIXED_TRACE"
    --chunks "$DSPP_GLOBAL_FIXED_CHUNKS"
    --pipeline-size 3
    --ordering-mode "$DSPP_ORDER"
    --packing-strategy "$DSPP_PACKING_STRATEGY"
    --sequence-group-size "$MICRO_BATCH_SIZE"
    --groups-per-iteration $((GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE))
    --raw-lengths-include-label
    --max-iterations "$DSPP_GLOBAL_FIXED_MAX_ITERS"
    --execution-profile "$DSPP_EXECUTION_PROFILE"
    --hidden-size "$HIDDEN_SIZE"
    --dtype-bytes 2
    --communication-cost-mode ignore
    --output "$DSPP_GLOBAL_FIXED_OUTPUT"
  )
  if [[ "$DSPP_GLOBAL_FIXED_ALLOW_LOW_COVERAGE" == 1 ]]; then
    GLOBAL_SELECTOR_ARGS+=(--allow-low-profile-coverage)
  fi
  if [[ "$DSPP_ACTIVATION_BYTES_PER_TOKEN" != 0 && "$DSPP_GLOBAL_FIXED_ACTIVATION_BUDGET_BYTES" != 0 ]]; then
    GLOBAL_SELECTOR_ARGS+=(
      --activation-bytes-per-live-token "$DSPP_ACTIVATION_BYTES_PER_TOKEN"
      --activation-budget-bytes "$DSPP_GLOBAL_FIXED_ACTIVATION_BUDGET_BYTES"
      --memory-safety-factor "$DSPP_MEMORY_SAFETY_FACTOR"
    )
  fi
  "$PYTHON_BIN" tools/evaluate_dspp_chunk_menu.py "${GLOBAL_SELECTOR_ARGS[@]}"
  MICRO_SEQ_LENGTH=$(
    "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_chunk_size"])' \
      "$DSPP_GLOBAL_FIXED_OUTPUT"
  )
  DSPP_AUTO_CHUNK_SIZE=0
  echo "[dspp][global-fixed] launching with micro_seq_length=$MICRO_SEQ_LENGTH"
fi

EXTRA_ARGS=()
if [[ -n "$DSPP_ORDER_PROFILE" ]]; then
  EXTRA_ARGS+=(--dspp-order-profile "$DSPP_ORDER_PROFILE")
fi
if [[ -n "$DSPP_EXECUTION_PROFILE" ]]; then
  EXTRA_ARGS+=(--dspp-execution-profile "$DSPP_EXECUTION_PROFILE")
fi
if [[ "$DSPP_DISABLE_RESIDUAL_PACKING" == 1 ]]; then
  EXTRA_ARGS+=(--dspp-disable-residual-packing)
fi
if [[ "$DSPP_SHARE_DIRECTIONAL_P2P" == 1 ]]; then
  EXTRA_ARGS+=(--dspp-share-directional-p2p)
fi
if [[ "$DSPP_AUTO_CHUNK_SIZE" == 1 ]]; then
  if [[ "$DSPP_CHUNK_SELECTION" == "per-iteration-menu" && -z "$DSPP_CHUNK_MENU" ]]; then
    echo "DSPP_CHUNK_MENU is required for per-iteration-menu selection" >&2
    exit 2
  fi
  EXTRA_ARGS+=(
    --dspp-auto-chunk-size
    --dspp-chunk-selection "$DSPP_CHUNK_SELECTION"
    --dspp-saturation-chunk-size "$DSPP_SATURATION_CHUNK_SIZE"
    --dspp-saturation-slack "$DSPP_SATURATION_SLACK"
    --dspp-min-chunk-size "$DSPP_MIN_CHUNK_SIZE"
    --dspp-max-chunk-size "$DSPP_MAX_CHUNK_SIZE"
    --dspp-chunk-alignment "$DSPP_CHUNK_ALIGNMENT"
    --dspp-efficiency-top-k "$DSPP_EFFICIENCY_TOP_K"
    --dspp-efficiency-slack "$DSPP_EFFICIENCY_SLACK"
    --dspp-memory-utilization "$DSPP_MEMORY_UTILIZATION"
    --dspp-memory-reserve-mib "$DSPP_MEMORY_RESERVE_MIB"
    --dspp-memory-safety-factor "$DSPP_MEMORY_SAFETY_FACTOR"
    --dspp-activation-bytes-per-token "$DSPP_ACTIVATION_BYTES_PER_TOKEN"
    --dspp-max-p2p-bytes "$DSPP_MAX_P2P_BYTES"
  )
  if [[ -n "$DSPP_CHUNK_MENU" ]]; then
    EXTRA_ARGS+=(--dspp-chunk-menu "$DSPP_CHUNK_MENU")
  fi
fi
if [[ -n "$DSPP_TIMELINE_DIR" ]]; then
  EXTRA_ARGS+=(--dspp-timeline-dir "$DSPP_TIMELINE_DIR")
fi
if [[ -n "$DSPP_TORCH_PROFILER_DIR" ]]; then
  EXTRA_ARGS+=(--dspp-torch-profiler-dir "$DSPP_TORCH_PROFILER_DIR")
fi
if [[ -n "$DSPP_METRICS_PATH" ]]; then
  EXTRA_ARGS+=(--dspp-metrics-path "$DSPP_METRICS_PATH")
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2} "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc-per-node=3 pretrain_llama.py \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 3 \
  --num-layers-per-virtual-pipeline-stage "$LAYERS_PER_VPP" \
  --num-layers "$NUM_LAYERS" \
  --hidden-size "$HIDDEN_SIZE" \
  --ffn-hidden-size "$FFN_HIDDEN_SIZE" \
  --num-attention-heads "$NUM_ATTENTION_HEADS" \
  --seq-length "$SEQ_LENGTH" \
  --max-position-embeddings "$MAX_POSITION_EMBEDDINGS" \
  --micro-seq-length "$MICRO_SEQ_LENGTH" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --train-iters "$TRAIN_ITERS" \
  --lr 0.001 \
  --min-lr 0.0001 \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --bf16 \
  --use-flash-attn \
  --no-position-embedding \
  --use-rotary-position-embeddings \
  --swiglu \
  --rms-norm \
  --disable-bias-linear \
  --no-query-key-layer-scaling \
  --hidden-dropout 0 \
  --attention-dropout 0 \
  --make-main-grad-addresss-divisible-by 1 \
  --data-path "$DATA_PATH" \
  --data-impl mmap \
  --split 100,0,0 \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --dataloader-type "$DATALOADER_TYPE" \
  --dspp \
  --dspp-v-layer-layout "$DSPP_V_LAYER_LAYOUT" \
  --dspp-microbatch-order "$DSPP_ORDER" \
  --dspp-packing-strategy "$DSPP_PACKING_STRATEGY" \
  --dspp-recv-lookahead "$DSPP_RECV_LOOKAHEAD" \
  --dspp-send-inflight-depth "$DSPP_SEND_INFLIGHT_DEPTH" \
  --dspp-timeline-iteration "$DSPP_TIMELINE_ITERATION" \
  --dspp-torch-profiler-iteration "$DSPP_TORCH_PROFILER_ITERATION" \
  --variable-seq-pad-token-id 0 \
  --variable-seq-debug-num-batches "$DSPP_DEBUG_BATCHES" \
  --eval-iters 0 \
  --log-interval "$LOG_INTERVAL" \
  "${EXTRA_ARGS[@]}"
