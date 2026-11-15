#!/usr/bin/env bash
# Native performance entry: pass model/data/profile arguments as usual.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON_BIN:-python3}" -m torch.distributed.run \
  --standalone --nproc-per-node="${NPROC_PER_NODE:-8}" \
  pretrain_llama.py \
  --dspp --dspp-schedule "${DSPP_SCHEDULE:-b}" \
  --dspp-prefetch --dspp-planner-backend cpp \
  --dspp-split-attention-backward --dspp-history-dkv-after-dx \
  "$@"
