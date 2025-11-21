#!/bin/bash

# Paths
BASE_DIR="$HOME/Megatron-LM/examples/sc25slimpipe"
LOG_DIR="$BASE_DIR/logs/seq_length"
MEGATRON_PATH="$HOME/Megatron-LM"

# Configuration parameters
TRAIN_ITERS=20
GLOBAL_BATCH_SIZES=(4)
SEQUENCE_LENGTHS_K=(32 64 128 256) # in K (multiply by 1024 to get actual length)
RECORD_MEMORY_HISTORY=0
RECOMPUTE=full

# Parallelism settings
TENSOR_PARALLEL_SIZE=8
CONTEXT_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=4
VIRTUAL_PIPELINE_LAYERS=1                                      # Layers per virtual pipeline stage
TOTAL_NODES=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE)) # Total nodes

# Switch to specific branch
cd $MEGATRON_PATH || exit
git switch wei/attn_balance
cd - || exit

# Function to run experiments
run_experiment() {
    local experiment_name=$1
    local virtual_pipeline_layers=$2 # Optional parameter for slim-i experiment

    echo "===== Running $experiment_name experiments ====="

    for batch_size in "${GLOBAL_BATCH_SIZES[@]}"; do
        for seq_length_k in "${SEQUENCE_LENGTHS_K[@]}"; do
            # Calculate parameters
            local full_seq_length=$((seq_length_k * 1024))

            echo "  Batch size: $batch_size"
            echo "  Sequence length: $full_seq_length ($seq_length_k K)"

            # Build command
            local cmd="NP=$TOTAL_NODES"
            cmd="$cmd TP=$TENSOR_PARALLEL_SIZE"
            cmd="$cmd CP=$CONTEXT_PARALLEL_SIZE"
            cmd="$cmd PP=$PIPELINE_PARALLEL_SIZE"
            cmd="$cmd RECOMPUTE=$RECOMPUTE"
            cmd="$cmd TRAIN_ITERS=$TRAIN_ITERS"
            cmd="$cmd GLOBAL_BATCH_SIZE=$batch_size"
            cmd="$cmd SEQUENCE_LENGTH=$full_seq_length"
            cmd="$cmd RECORD_MEMORY_HISTORY=$RECORD_MEMORY_HISTORY"
            cmd="$cmd MEGATRON_PATH=$MEGATRON_PATH"
            cmd="$cmd LOG_DIR=${LOG_DIR}/${experiment_name} EXP=$experiment_name"

            # Add pipeline layers parameter if provided
            if [[ -n $virtual_pipeline_layers ]]; then
                cmd="$cmd PP_l=$virtual_pipeline_layers"
            fi

            # Add script to execute
            cmd="$cmd $BASE_DIR/pretrain_llama.sh"

            # Run experiment
            echo "$cmd"
            eval "$cmd"

            echo "Cooling period: waiting 30 seconds before next run..."
            sleep 30
        done
    done

    echo "===== Completed $experiment_name experiments ====="
}

# Run experiments
run_experiment "1f1b"
run_experiment "1f1b-i" $VIRTUAL_PIPELINE_LAYERS
