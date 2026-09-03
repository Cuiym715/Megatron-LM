#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

export NUM_LAYERS=${NUM_LAYERS:-24}
export LAYERS_PER_VPP=${LAYERS_PER_VPP:-4}
export DSPP_V_LAYER_LAYOUT=${DSPP_V_LAYER_LAYOUT:-balanced}
export HIDDEN_SIZE=${HIDDEN_SIZE:-2048}
export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-5504}
export NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-16}
export SEQ_LENGTH=${SEQ_LENGTH:-768}
export MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-768}
export MICRO_SEQ_LENGTH=${MICRO_SEQ_LENGTH:-256}
DEFAULT_DATA_PATH=/tmp/dspp_saturated_c256_text_document
export DATA_PATH=${DATA_PATH:-$DEFAULT_DATA_PATH}

if [[ "$DATA_PATH" == "$DEFAULT_DATA_PATH" && ! -f "${DATA_PATH}.idx" ]]; then
  PYTHON_BIN=${PYTHON_BIN:-/workspace/src/venvs/megatron/bin/python}
  "$PYTHON_BIN" tools/build_dspp_synthetic_dataset.py \
    --output-prefix "$DATA_PATH" \
    --training-lengths 768,512,192,128 \
    --repeats 8
fi

exec bash examples/run_dspp_b23_l40.sh
