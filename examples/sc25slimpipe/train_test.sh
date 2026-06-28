#!/bin/bash
set -euo pipefail

# SlimPipe smoke test for a small single-node server.
#
# Default layout:
#   - Use 2 GPUs out of a 3-GPU machine, because TP=1, PP=2, CP=1 requires
#     world_size to be divisible by TP*PP*CP=2.
#   - Enable sequence slicing with MICRO_SEQ_LENGTH < SEQ_LENGTH.
#   - Keep CP disabled first (CP_SIZE=1) so this checks SlimPipe scheduling.
#
# Usage:
#   cd /path/to/megatron-kwai
#   bash examples/sc25slimpipe/train_test.sh
#
# Optional overrides:
#   CUDA_VISIBLE_DEVICES=0,1 PP_SIZE=2 CP_SIZE=1 TRAIN_ITERS=20 \
#     bash examples/sc25slimpipe/train_test.sh
#   VARLEN=1 bash examples/sc25slimpipe/train_test.sh

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MEGATRON_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
RECORD_TIMER_EVENTS=1
TIMER_RECORD_START_ITER=10
TIMER_RECORD_END_ITER=15
SEQ_LENGTH=8192000
MICRO_SEQ_LENGTH=4096
GLOBAL_BATCH_SIZE=4
VARLEN_SCHEDULE=1f1b
PP_SIZE=3
NUM_LAYERS=6

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export PYTHONPATH="$MEGATRON_ROOT:${PYTHONPATH:-}"

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-6002}
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}

GPUS_PER_NODE=${GPUS_PER_NODE:-3}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-2}
CP_SIZE=${CP_SIZE:-1}
VARLEN=${VARLEN:-1}
VARLEN_SCHEDULE=${VARLEN_SCHEDULE:-1f1b}
if [[ "$VARLEN" == "1" ]]; then
    PP_ATTN_BALANCE=${PP_ATTN_BALANCE:-0}
    # DEBUG for variable length training.
    VARLEN_DEBUG=${VARLEN_DEBUG:-2}
else
    PP_ATTN_BALANCE=${PP_ATTN_BALANCE:-100}
    # DEBUG for variable length training.
    VARLEN_DEBUG=${VARLEN_DEBUG:-0}
fi

MODEL_PARALLEL_SIZE=$((TP_SIZE * PP_SIZE * CP_SIZE))
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
VISIBLE_GPU_COUNT=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
if (( VISIBLE_GPU_COUNT != GPUS_PER_NODE )); then
    echo "Invalid GPU visibility: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES exposes $VISIBLE_GPU_COUNT GPUs, but GPUS_PER_NODE=$GPUS_PER_NODE."
    echo "For PP_SIZE=3 on this machine, use CUDA_VISIBLE_DEVICES=0,1,2 GPUS_PER_NODE=3."
    exit 1
fi
if (( WORLD_SIZE % MODEL_PARALLEL_SIZE != 0 )); then
    echo "Invalid parallel layout: WORLD_SIZE=$WORLD_SIZE is not divisible by TP*PP*CP=$MODEL_PARALLEL_SIZE."
    echo "For three GPUs with PP_SIZE=2, use only two GPUs, e.g. CUDA_VISIBLE_DEVICES=0,1 GPUS_PER_NODE=2."
    echo "To use all three GPUs, choose a divisible layout such as PP_SIZE=3 CP_SIZE=1."
    exit 1
fi

DATA_PATH=${DATA_PATH:-/workspace/src/data/megatron/slimpajama_arxiv_50k_text_document}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-/workspace/src/tokenizers/Qwen2.5-7B}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-/workspace/src/data/cache_kwai_slimpipe_test}

NUM_LAYERS=${NUM_LAYERS:-4}
HIDDEN_SIZE=${HIDDEN_SIZE:-512}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-2048}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-8}

SEQ_LENGTH=${SEQ_LENGTH:-8192}
MICRO_SEQ_LENGTH=${MICRO_SEQ_LENGTH:-4096}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-$SEQ_LENGTH}
if [[ "$VARLEN" == "1" ]]; then
    if [[ "$VARLEN_SCHEDULE" == "vzb" ]]; then
        VIRTUAL_PP_LAYERS=${VIRTUAL_PP_LAYERS:-1}
    else
        VIRTUAL_PP_LAYERS=${VIRTUAL_PP_LAYERS:-}
    fi
else
    VIRTUAL_PP_LAYERS=${VIRTUAL_PP_LAYERS:-1}
fi

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
TRAIN_ITERS=${TRAIN_ITERS:-20}

LOG_DIR=${LOG_DIR:-$SCRIPT_DIR/logs/train_test}
RECORD_TIMER_EVENTS=${RECORD_TIMER_EVENTS:-0}
if [[ "$RECORD_TIMER_EVENTS" == "1" ]]; then
    TIMING_LOG_LEVEL=${TIMING_LOG_LEVEL:-2}
else
    TIMING_LOG_LEVEL=${TIMING_LOG_LEVEL:-0}
fi
TIMER_RECORD_START_ITER=${TIMER_RECORD_START_ITER:-0}
TIMER_RECORD_END_ITER=${TIMER_RECORD_END_ITER:--1}
RUN_TAG=${RUN_TAG:-pp${PP_SIZE}_varlen${VARLEN}${VARLEN_SCHEDULE}_seq${SEQ_LENGTH}_mseq${MICRO_SEQ_LENGTH}_gbs${GLOBAL_BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)}
TIMER_RECORD_DIR=${TIMER_RECORD_DIR:-/workspace/log/${RUN_TAG}/timers}
mkdir -p "$LOG_DIR" "$DATA_CACHE_PATH"
if [[ "$RECORD_TIMER_EVENTS" == "1" ]]; then
    mkdir -p "$TIMER_RECORD_DIR"
fi

if [[ "$VARLEN" != "1" ]] && (( SEQ_LENGTH % MICRO_SEQ_LENGTH != 0 )); then
    echo "SEQ_LENGTH=$SEQ_LENGTH must be divisible by MICRO_SEQ_LENGTH=$MICRO_SEQ_LENGTH."
    exit 1
fi

if [[ "$VARLEN" == "1" ]]; then
    NUM_SLICES=$(((SEQ_LENGTH + MICRO_SEQ_LENGTH - 1) / MICRO_SEQ_LENGTH))
else
    NUM_SLICES=$((SEQ_LENGTH / MICRO_SEQ_LENGTH))
fi
if (( NUM_SLICES < PP_SIZE )); then
    echo "SlimPipe interleaved slicing expects NUM_SLICES >= PP_SIZE; got NUM_SLICES=$NUM_SLICES PP_SIZE=$PP_SIZE."
    echo "Lower MICRO_SEQ_LENGTH or increase SEQ_LENGTH."
    exit 1
fi

if [[ -n "$VIRTUAL_PP_LAYERS" ]]; then
    if (( NUM_LAYERS % (PP_SIZE * VIRTUAL_PP_LAYERS) != 0 )); then
        echo "NUM_LAYERS=$NUM_LAYERS must be divisible by PP_SIZE*VIRTUAL_PP_LAYERS=$((PP_SIZE * VIRTUAL_PP_LAYERS))."
        exit 1
    fi
else
    if (( NUM_LAYERS % PP_SIZE != 0 )); then
        echo "NUM_LAYERS=$NUM_LAYERS must be divisible by PP_SIZE=$PP_SIZE."
        exit 1
    fi
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTENTION_HEADS"
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --no-position-embedding
    --use-rotary-position-embeddings
    --rope-theta 10000.0
    --swiglu
    --rms-norm
    --disable-bias-linear
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --no-query-key-layer-scaling
    --init-method-std 0.02
    --bf16
    --accumulate-allreduce-grads-in-fp32
    --seed 9527
)

TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-iters "$TRAIN_ITERS"
    --use-distributed-optimizer
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-5
    --lr 1e-4
    --min-lr 1e-5
    --lr-decay-style cosine
    --lr-decay-iters "$TRAIN_ITERS"
    --lr-warmup-iters 1
    --weight-decay 0.1
    --clip-grad 1.0
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
    --context-parallel-size "$CP_SIZE"
    --micro-seq-length "$MICRO_SEQ_LENGTH"
    --kaimm-kv-cache-impl chunked
    --kaimm-context-parallel-impl query-out
    --kaimm-pipeline-attn-balance "$PP_ATTN_BALANCE"
    --overlap-p2p-communication
)

if [[ -n "$VIRTUAL_PP_LAYERS" ]]; then
    PARALLEL_ARGS+=(--num-layers-per-virtual-pipeline-stage "$VIRTUAL_PP_LAYERS")
fi

if [[ "$VARLEN" == "1" ]]; then
    PARALLEL_ARGS+=(
        --variable-seq-slicing
        --variable-seq-schedule "$VARLEN_SCHEDULE"
        --variable-seq-pad-token-id "${PAD_TOKEN_ID:-0}"
        --variable-seq-pad-to-pipeline-size
        # DEBUG for variable length training.
        --variable-seq-debug-num-batches "$VARLEN_DEBUG"
    )
    if [[ "$VARLEN_SCHEDULE" == "vzb" ]]; then
        PARALLEL_ARGS+=(--gradient-accumulation-fusion)
    fi
fi

if (( TP_SIZE > 1 )); then
    PARALLEL_ARGS+=(--sequence-parallel)
fi

PERFORMANCE_ARGS=(
    --use-flash-attn
    --no-masked-softmax-fusion
    --no-bias-gelu-fusion
    --no-bias-dropout-fusion
    --no-context-parallel-comm-overlap-gemm
    --kaimm-warmup-iters 0
    --kaimm-cuda-synchronize-level 1
)

DATA_ARGS=(
    --data-path "$DATA_PATH"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --data-cache-path "$DATA_CACHE_PATH"
    --split 99,1,0
    --num-workers 1
)

LOGGING_ARGS=(
    --log-interval 1
    --timing-log-level "$TIMING_LOG_LEVEL"
    --eval-iters 0
    --eval-interval 1000
    --save-interval 100000
    --distributed-timeout-minutes 60
    --master-addr "${MASTER_ADDR}:${MASTER_PORT}"
)

if [[ "$RECORD_TIMER_EVENTS" == "1" ]]; then
    LOGGING_ARGS+=(
        --timer-record-dir "$TIMER_RECORD_DIR"
        --timer-record-start-iter "$TIMER_RECORD_START_ITER"
        --timer-record-end-iter "$TIMER_RECORD_END_ITER"
    )
fi

echo "MEGATRON_ROOT=$MEGATRON_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "TP_SIZE=$TP_SIZE PP_SIZE=$PP_SIZE CP_SIZE=$CP_SIZE"
echo "VARLEN=$VARLEN VIRTUAL_PP_LAYERS=${VIRTUAL_PP_LAYERS:-none}"
echo "VARLEN_SCHEDULE=$VARLEN_SCHEDULE"
# DEBUG for variable length training.
echo "VARLEN_DEBUG=$VARLEN_DEBUG"
if [[ "$VARLEN" == "1" ]]; then
    # DEBUG for variable length training.
    echo "VARIABLE_SEQ_ARGS=--variable-seq-slicing --variable-seq-schedule $VARLEN_SCHEDULE --variable-seq-pad-token-id ${PAD_TOKEN_ID:-0} --variable-seq-pad-to-pipeline-size --variable-seq-debug-num-batches $VARLEN_DEBUG"
fi
echo "PP_ATTN_BALANCE=$PP_ATTN_BALANCE"
echo "SEQ_LENGTH=$SEQ_LENGTH MICRO_SEQ_LENGTH=$MICRO_SEQ_LENGTH NUM_SLICES=$NUM_SLICES"
echo "DATA_PATH=$DATA_PATH"
echo "TOKENIZER_MODEL=$TOKENIZER_MODEL"
echo "LOG_DIR=$LOG_DIR"
echo "TIMING_LOG_LEVEL=$TIMING_LOG_LEVEL"
if [[ "$RECORD_TIMER_EVENTS" == "1" ]]; then
    echo "TIMER_RECORD_DIR=$TIMER_RECORD_DIR"
    echo "TIMER_RECORD_START_ITER=$TIMER_RECORD_START_ITER TIMER_RECORD_END_ITER=$TIMER_RECORD_END_ITER"
fi

sleep 10
cd "$MEGATRON_ROOT"

python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    pretrain_llama.py \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${PERFORMANCE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/train_test_$(date +%Y%m%d_%H%M%S).log"
