#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-/workspace/src/venvs/megatron/bin/python}
LENGTHS_FILE=${LENGTHS_FILE:-/workspace/src/InfiniPipe-code/datasets/dspp_compare_max16k.txt}
DATA_PATH=${DATA_PATH:-/tmp/dspp_infinipipe_figure2_seed7_n1024_max16k_text_document}
RESULT_DIR=${RESULT_DIR:-/workspace/src/results/dspp_infinipipe_compare_max16k_gbs16_p2popt}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-16384}

if [[ ! -s "$LENGTHS_FILE" ]]; then
  echo "Shared sequence-length file is missing or empty: $LENGTHS_FILE" >&2
  exit 1
fi

"$PYTHON_BIN" tools/build_dspp_synthetic_dataset.py \
  --output-prefix "$DATA_PATH" \
  --raw-lengths-file "$LENGTHS_FILE" \
  --repeats 1 \
  --vocab-size 32128 \
  --token-pattern infinipipe

mkdir -p "$RESULT_DIR"

TRAIN_ITERS=${TRAIN_ITERS:-12} \
NUM_LAYERS=24 \
LAYERS_PER_VPP=4 \
HIDDEN_SIZE=2048 \
FFN_HIDDEN_SIZE=5504 \
NUM_ATTENTION_HEADS=16 \
SEQ_LENGTH="$MAX_SEQUENCE_LENGTH" \
MAX_POSITION_EMBEDDINGS="$MAX_SEQUENCE_LENGTH" \
MICRO_SEQ_LENGTH=${MICRO_SEQ_LENGTH:-256} \
MICRO_BATCH_SIZE=4 \
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16} \
DATALOADER_TYPE=single \
DSPP_AUTO_CHUNK_SIZE=${DSPP_AUTO_CHUNK_SIZE:-1} \
DSPP_MIN_CHUNK_SIZE=${DSPP_MIN_CHUNK_SIZE:-256} \
DSPP_MAX_CHUNK_SIZE=${DSPP_MAX_CHUNK_SIZE:-$MAX_SEQUENCE_LENGTH} \
DSPP_CHUNK_ALIGNMENT=${DSPP_CHUNK_ALIGNMENT:-256} \
DSPP_MEMORY_UTILIZATION=${DSPP_MEMORY_UTILIZATION:-0.90} \
DSPP_MEMORY_RESERVE_MIB=${DSPP_MEMORY_RESERVE_MIB:-1024} \
DSPP_MEMORY_SAFETY_FACTOR=${DSPP_MEMORY_SAFETY_FACTOR:-1.20} \
DSPP_RECV_LOOKAHEAD=${DSPP_RECV_LOOKAHEAD:-2} \
DSPP_SEND_INFLIGHT_DEPTH=${DSPP_SEND_INFLIGHT_DEPTH:-2} \
DSPP_MAX_P2P_BYTES=${DSPP_MAX_P2P_BYTES:-0} \
DATA_PATH="$DATA_PATH" \
DSPP_METRICS_PATH="$RESULT_DIR/metrics.json" \
bash examples/run_dspp_b23_l40.sh
