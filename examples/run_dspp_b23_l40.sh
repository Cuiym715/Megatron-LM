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
DSPP_DEBUG_BATCHES=${DSPP_DEBUG_BATCHES:-0}
DSPP_ORDER=${DSPP_ORDER:-warmup-short-steady-long}
DSPP_ORDER_PROFILE=${DSPP_ORDER_PROFILE:-}
DSPP_TIMELINE_DIR=${DSPP_TIMELINE_DIR:-}
DSPP_TIMELINE_ITERATION=${DSPP_TIMELINE_ITERATION:-0}
DSPP_TORCH_PROFILER_DIR=${DSPP_TORCH_PROFILER_DIR:-}
DSPP_TORCH_PROFILER_ITERATION=${DSPP_TORCH_PROFILER_ITERATION:-0}
DSPP_METRICS_PATH=${DSPP_METRICS_PATH:-}
DATA_PATH=${DATA_PATH:-/tmp/dspp_b1_varlen_20260902_text_document}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-/workspace/src/tokenizers/Mistral-7B-v0.1/tokenizer.model}
PYTHON_BIN=${PYTHON_BIN:-/workspace/src/venvs/megatron/bin/python}

EXTRA_ARGS=()
if [[ -n "$DSPP_ORDER_PROFILE" ]]; then
  EXTRA_ARGS+=(--dspp-order-profile "$DSPP_ORDER_PROFILE")
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
  --micro-batch-size 4 \
  --global-batch-size 8 \
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
  --dataloader-type cyclic \
  --dspp \
  --dspp-v-layer-layout "$DSPP_V_LAYER_LAYOUT" \
  --dspp-microbatch-order "$DSPP_ORDER" \
  --dspp-timeline-iteration "$DSPP_TIMELINE_ITERATION" \
  --dspp-torch-profiler-iteration "$DSPP_TORCH_PROFILER_ITERATION" \
  --variable-seq-pad-token-id 0 \
  --variable-seq-debug-num-batches "$DSPP_DEBUG_BATCHES" \
  --eval-iters 0 \
  --log-interval "$LOG_INTERVAL" \
  "${EXTRA_ARGS[@]}"
