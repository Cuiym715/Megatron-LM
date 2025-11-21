#!/bin/bash
#
# Strong scaling experiment for LLM pretraining
# This script tests different pipeline parallelism configurations

# ----- Base configuration -----
BASE_DIR="$HOME/Megatron-LM/examples/sc25slimpipe"
LOGS_DIR="${BASE_DIR}/logs/strong_scaling"
MEGATRON_PATH="$HOME/Megatron-LM"

# Training parameters
NUM_GPUS=32          # Maximal available GPUs
GPUS_PER_NODE=8
TRAIN_ITERS=10
GLOBAL_BATCH_SIZE=8
BASE_SEQ_LENGTH=1000 # In units of 1024 tokens
TP=8                 # Fixed tensor parallelism degree

# ----- Model definitions -----
# Model name to experiment name mapping
declare -A MODEL_EXPERIMENTS=(
    ["LLaMA-13B"]="slimpipe/llama"
    ["LLaMA-70B-GQA"]="slimpipe/llama"
    ["Mixtral-8x7B"]="slimpipe/mixtral"
    ["Mixtral-8x22B"]="slimpipe/mixtral"
)

declare -A MODEL_FILES=(
    ["LLaMA-13B"]="13b"
    ["LLaMA-70B-GQA"]="70bgqa"
    ["Mixtral-8x7B"]="8x7b"
    ["Mixtral-8x22B"]="8x22b"
)

# Pipeline parallelism degrees to test for each model
declare -A PIPELINE_CONFIGS=(
    ["LLaMA-13B"]="4 5 8 10 20"
    ["LLaMA-70B-GQA"]="5 8 10 16 20 40"
    ["Mixtral-8x7B"]="4 8 16"
    ["Mixtral-8x22B"]="4 8 14 28"
)

# ----- Run experiments -----
for MODEL_NAME in "${!MODEL_EXPERIMENTS[@]}"; do
    EXPERIMENT_NAME="${MODEL_EXPERIMENTS[$MODEL_NAME]}"
    MODEL_FILE="${MODEL_FILES[$MODEL_NAME]}"

    echo "======================================"
    echo "Running experiments for $MODEL_NAME"
    echo "Experiment: $EXPERIMENT_NAME"
    echo "Model file: $MODEL_FILE"
    echo "======================================"

    # Get pipeline parallel sizes for this model
    PP_VALUES=(${PIPELINE_CONFIGS[$MODEL_NAME]})

    for PP in "${PP_VALUES[@]}"; do
        # Calculate derived parameters
        TOTAL_PROCESSES=$((PP * GPUS_PER_NODE))
        NUM_SLICES=$((2 * PP))
        LAYERS_PER_STAGE=1

        if [ "$TOTAL_PROCESSES" -gt "$NUM_GPUS" ]; then
            break
        fi

        # Calculate sequence length
        SEQ_LENGTH=$((BASE_SEQ_LENGTH * 1024))
        if [[ "$MODEL_NAME" == "Mixtral-8x22B" ]]; then
            SEQ_LENGTH=$((SEQ_LENGTH + 128))
        fi

        if [[ "$MODEL_NAME" == Mixtral* ]]; then
            TP=1
            CP=8
            EP=8
        else
            TP=8
            CP=1
            EP=1
        fi

        echo "-----------------------------------------"
        echo "Configuration:"
        echo "  - Pipeline Parallel (PP): $PP"
        echo "  - Tensor Parallel (TP): $TP"
        echo "  - Context Parallel (TP): $CP"
        echo "  - Expert Parallel (EP): $EP"
        echo "  - Total Processes: $TOTAL_PROCESSES"
        echo "  - Sequence Length: $SEQ_LENGTH tokens"

        # Set environment variables and run training script
        echo "Starting training run..."

        cmd="NP=$TOTAL_PROCESSES \
            TP=$TP \
            PP=$PP \
            EP=$EP \
            PP_l=$LAYERS_PER_STAGE \
            TRAIN_ITERS=$TRAIN_ITERS \
            GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE \
            SEQUENCE_LENGTH=$SEQ_LENGTH \
            NUM_SLICES=$NUM_SLICES \
            MEGATRON_PATH=$MEGATRON_PATH \
            LOG_DIR=$LOGS_DIR/$MODEL_NAME/pp${PP} \
            EXP=$EXPERIMENT_NAME \
            $BASE_DIR/pretrain_llama.sh"

        echo "$cmd"
        eval "$cmd"

        echo "Training complete. Waiting 30 seconds before next run..."
        sleep 30
    done
done
