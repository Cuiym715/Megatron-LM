#!/bin/bash
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# Usage:
#   bash examples/train_test.sh smoke
#   bash examples/train_test.sh cp
#   bash examples/train_test.sh dcp
MODE=${1:-"smoke"}   # smoke | cp | dcp

DATA_PATH=${DATA_PATH:-"/workspace/src/data/megatron/slimpajama_arxiv_50k_text_document"}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-"/workspace/src/tokenizers/Qwen2.5-7B"}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-"/workspace/src/data/cache_small_${MODE}"}

mkdir -p "$DATA_CACHE_PATH"

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}

if [[ "$MODE" == "smoke" ]]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    GPUS_PER_NODE=1
    CP_SIZE=1
elif [[ "$MODE" == "cp" ]]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
    GPUS_PER_NODE=2
    CP_SIZE=2
elif [[ "$MODE" == "dcp" ]]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
    GPUS_PER_NODE=2
    CP_SIZE=2
else
    echo "Unknown MODE=$MODE. Use smoke, cp, or dcp."
    exit 1
fi

PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Small model for debugging / experiment baseline.
NUM_LAYERS=${NUM_LAYERS:-4}
HIDDEN_SIZE=${HIDDEN_SIZE:-512}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-2048}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-8}

SEQ_LENGTH=${SEQ_LENGTH:-2048}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-2048}

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
TRAIN_ITERS=${TRAIN_ITERS:-20}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTENTION_HEADS"
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --position-embedding-type rope
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --init-method-std 0.02
    --disable-bias-linear
    --untie-embeddings-and-output-weights
)

TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-iters "$TRAIN_ITERS"
    --lr 1e-4
    --min-lr 1e-5
    --lr-decay-style cosine
    --lr-decay-iters "$TRAIN_ITERS"
    --lr-warmup-iters 1
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
    --context-parallel-size "$CP_SIZE"
)

# Do not enable sequence parallel when TP_SIZE=1.
if [[ "$TP_SIZE" -gt 1 ]]; then
    MODEL_PARALLEL_ARGS+=(--sequence-parallel)
fi

DYNAMIC_CP_ARGS=()

if [[ "$MODE" == "dcp" ]]; then
    # In Dynamic CP, dp_size * context_parallel_size is the maximum dynamic CP group size.
    # With 2 GPUs, TP=1, PP=1, this is 2.
    TOTAL_DP_CP_RANKS=$((GPUS_PER_NODE * NUM_NODES / TP_SIZE / PP_SIZE))
    MAX_SEQLEN_PER_DP_CP_RANK=$(((SEQ_LENGTH + TOTAL_DP_CP_RANKS - 1) / TOTAL_DP_CP_RANKS))

    DYNAMIC_CP_ARGS+=(
        --dynamic-context-parallel
        --min-dynamic-context-parallel-size 1
        --max-seqlen-per-dp-cp-rank "$MAX_SEQLEN_PER_DP_CP_RANK"
        --sequence-packing-scheduler default_dynamic_cp
        --moe-token-dispatcher-type alltoall
    )
fi

# Data args:
#   smoke / cp:
#       Use normal GPTDataset from .bin/.idx.
#   dcp:
#       Use SFT mock verification dataset, because Dynamic CP sequence packing
#       requires cu_seqlens / max_seqlen in the batch.
if [[ "$MODE" == "dcp" ]]; then
    SFT_MIN_SEQ_LEN=${SFT_MIN_SEQ_LEN:-128}
    SFT_MAX_SEQ_LEN=${SFT_MAX_SEQ_LEN:-$SEQ_LENGTH}
    SFT_MEAN_SEQ_LEN=${SFT_MEAN_SEQ_LEN:-768}
    SFT_LOGNORMAL_SIGMA=${SFT_LOGNORMAL_SIGMA:-1.1}

    SFT_MOCK_CONFIG=${SFT_MOCK_CONFIG:-"{\"mode\":\"verification\",\"data_path\":\"${DATA_PATH}\",\"min_seq_len\":${SFT_MIN_SEQ_LEN},\"max_seq_len\":${SFT_MAX_SEQ_LEN},\"mean_seq_len\":${SFT_MEAN_SEQ_LEN},\"lognormal_sigma\":${SFT_LOGNORMAL_SIGMA}}"}

    DATA_ARGS=(
        --mock-data
        --sft
        --sft-mock-dataset-config-json "$SFT_MOCK_CONFIG"
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "$TOKENIZER_MODEL"
        --no-create-attention-mask-in-dataloader
        --num-workers 1
    )
else
    DATA_ARGS=(
        --data-path "$DATA_PATH"
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "$TOKENIZER_MODEL"
        --data-cache-path "$DATA_CACHE_PATH"
        --split 99,1,0
        --no-create-attention-mask-in-dataloader
        --num-workers 1
    )
fi

LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 1
    --eval-interval 1000
    --save-interval 100000
    --log-throughput
    --distributed-timeout-minutes 60
)

if [[ ! -f "$PRETRAIN_SCRIPT_PATH" ]]; then
    echo "Error: pretrain_gpt.py not found. Run this script from Megatron-LM root."
    exit 1
fi

echo "MODE=$MODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "CP_SIZE=$CP_SIZE"
echo "TP_SIZE=$TP_SIZE"
echo "PP_SIZE=$PP_SIZE"
echo "SEQ_LENGTH=$SEQ_LENGTH"
echo "DATA_PATH=$DATA_PATH"
echo "TOKENIZER_MODEL=$TOKENIZER_MODEL"

if [[ "$MODE" == "dcp" ]]; then
    echo "TOTAL_DP_CP_RANKS=$TOTAL_DP_CP_RANKS"
    echo "MAX_SEQLEN_PER_DP_CP_RANK=$MAX_SEQLEN_PER_DP_CP_RANK"
    echo "SFT_MOCK_CONFIG=$SFT_MOCK_CONFIG"
fi

python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    "$PRETRAIN_SCRIPT_PATH" \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${DYNAMIC_CP_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${LOGGING_ARGS[@]}"