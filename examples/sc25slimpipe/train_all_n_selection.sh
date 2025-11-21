#!/bin/bash
#
# Script to run Llama pretraining with different sequence slicing configurations
# This tests various Pipeline Parallel (PP) sizes and slice counts (k)

# ===========================
# Configuration
# ===========================

# Directory paths
BASE_DIR="$HOME/Megatron-LM/examples/sc25slimpipe"
LOG_DIR="$BASE_DIR/logs/n_selection"
MEGATRON_PATH="$HOME/Megatron-LM"

# Training parameters
TRAIN_ITERS=5
GLOBAL_BATCH_SIZE=2
SEQ_LENGTH_BASES=(256 128)
RECORD_MEMORY_HISTORY=0

# Pipeline parallel configurations to test
TP=8
PP=4
LAYERS_PER_STAGE=2
TP_OVERLAP=1

# Models
MODELS=("13b")

# ===========================
# Helper functions
# ===========================

# Calculate ceiling division
ceil_div() {
    echo $((($1 + $2 - 1) / $2))
}

round_up_to_multiple() {
    local NUM=$1
    local MULTIPLE_OF=$2

    # Calculate the rounded value
    local remainder=$((NUM % MULTIPLE_OF))

    if [ $remainder -eq 0 ]; then
        echo $NUM
    else
        local rounded=$((NUM + MULTIPLE_OF - remainder))
        echo $rounded
    fi
}

# Run a specific configuration
run_config() {
    local model=$1    # Pipeline parallel size
    local seq_length_base=$2 # Sequence length base
    local k=$3        # n / p ratio
    local exp_name=$4 # Experiment name
    local pp_l=${5:-} # Optional PP layers parameter

    # Calculate derived parameters
    local total_processors=$((TP * PP))
    local num_slices=$((k * PP))
    local standard_seq_length=$((seq_length_base * 1024))
    local slice_length=$(ceil_div $standard_seq_length $num_slices)
    local rounded_slice_length=$(round_up_to_multiple $slice_length 128)
    local seq_length=$((rounded_slice_length * num_slices))
    local padding=$((seq_length - standard_seq_length))
    local micro_seq_length=$((seq_length / num_slices))

    # Print configuration details
    echo "======================================================"
    echo "Configuration: model=$model, k=$k, exp=$exp_name${pp_l:+, PP_l=$pp_l}"
    echo "  Processors:        $total_processors (TP=$TP, PP=$PP)"
    echo "  Slices:            $num_slices"
    echo "  Sequence Length:   $seq_length"
    echo "  Padding:           $padding"
    echo "  Micro Seq Length:  $micro_seq_length"
    echo "======================================================"

    # Build and execute command
    local cmd="NP=$total_processors \
        TP=$TP \
        PP=$PP \
        ${pp_l:+PP_l=$pp_l} \
        TRAIN_ITERS=$TRAIN_ITERS \
        GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE \
        SEQUENCE_LENGTH=$seq_length \
        MEGATRON_PATH=$MEGATRON_PATH \
        RECORD_MEMORY_HISTORY=$RECORD_MEMORY_HISTORY \
        TP_OVERLAP=$TP_OVERLAP \
        NUM_SLICES=$num_slices \
        LOG_DIR="${LOG_DIR}/${seq_length_base}K/${pp_l:+pp_l=$pp_l/}${model}/${k}p" \
        MODEL=$model \
        EXP=$exp_name \
        $BASE_DIR/pretrain_llama.sh"

    echo "Executing: $cmd"
    eval "$cmd"

    echo "Cooling period: waiting 10 seconds before next run..."
    sleep 10
}

# ===========================
# Main execution
# ===========================

# Switch to specific branch
cd $MEGATRON_PATH || exit
git switch wei/attn_balance
cd - || exit

echo "================ STANDARD CONFIGURATIONS ================"
for seq_length_base in "${SEQ_LENGTH_BASES[@]}"; do
    for model in "${MODELS[@]}"; do
        for k in $(seq 9 16); do
            run_config "$model" "$seq_length_base" "$k" "slimpipe/llama" "$LAYERS_PER_STAGE"
            # run_config "$model" "$seq_length_base" "$k" "slimpipe/llama"
        done
    done
done
