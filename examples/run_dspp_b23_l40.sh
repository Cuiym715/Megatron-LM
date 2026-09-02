#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_ITERS=${TRAIN_ITERS:-10}
LOG_INTERVAL=${LOG_INTERVAL:-1}
DSPP_DEBUG_BATCHES=${DSPP_DEBUG_BATCHES:-0}
DATA_PATH=${DATA_PATH:-/tmp/dspp_b1_varlen_20260902_text_document}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-/workspace/src/tokenizers/Mistral-7B-v0.1/tokenizer.model}
PYTHON_BIN=${PYTHON_BIN:-/workspace/src/venvs/megatron/bin/python}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2} "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc-per-node=3 pretrain_llama.py \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 3 \
  --num-layers-per-virtual-pipeline-stage 1 \
  --num-layers 5 \
  --hidden-size 64 \
  --ffn-hidden-size 128 \
  --num-attention-heads 4 \
  --seq-length 96 \
  --max-position-embeddings 96 \
  --micro-seq-length 32 \
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
  --variable-seq-pad-token-id 0 \
  --variable-seq-debug-num-batches "$DSPP_DEBUG_BATCHES" \
  --eval-iters 0 \
  --log-interval "$LOG_INTERVAL"
