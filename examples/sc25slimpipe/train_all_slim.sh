#!/bin/bash

# Paths
BASE_DIR="$HOME/Megatron-LM/examples/sc25slimpipe"
LOG_DIR="$BASE_DIR/logs/seq_length"
MEGATRON_PATH="$HOME/Megatron-LM"

# Configuration parameters
TRAIN_ITERS=10
GLOBAL_BATCH_SIZES=(4)
SEQUENCE_LENGTHS_K=(32 64 128 256 512) # in K (multiply by 1024 to get actual length)
SLICE_FACTORS=(1)
RECORD_MEMORY_HISTORY=0
PROFILE=0

# Parallelism settings
TENSOR_PARALLEL_SIZE=8
CONTEXT_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=4
VIRTUAL_PIPELINE_LAYERS=2                                      # Layers per virtual pipeline stage
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
            for slice_factor in "${SLICE_FACTORS[@]}"; do
                # Calculate parameters
                local full_seq_length=$((seq_length_k * 1024))
                local num_slices=$((PIPELINE_PARALLEL_SIZE * slice_factor))
                local micro_batch_seq_length=$((full_seq_length / num_slices))
                local PP_ATTN_BALANCE=100
                if [[ "$seq_length_k" == 32 ]]; then
                    PP_ATTN_BALANCE=0
                fi

                echo "  Batch size: $batch_size"
                echo "  Sequence length: $full_seq_length ($seq_length_k K)"
                echo "  Number of slices: $num_slices"
                echo "  Micro-seq length: $micro_batch_seq_length"

                # Build command
                local cmd="NP=$TOTAL_NODES"
                cmd="$cmd TP=$TENSOR_PARALLEL_SIZE"
                cmd="$cmd CP=$CONTEXT_PARALLEL_SIZE"
                cmd="$cmd PP=$PIPELINE_PARALLEL_SIZE"
                cmd="$cmd TRAIN_ITERS=$TRAIN_ITERS"
                cmd="$cmd GLOBAL_BATCH_SIZE=$batch_size"
                cmd="$cmd RECORD_MEMORY_HISTORY=$RECORD_MEMORY_HISTORY"
                cmd="$cmd PROFILE=$PROFILE"
                cmd="$cmd SEQUENCE_LENGTH=$full_seq_length"
                cmd="$cmd NUM_SLICES=$num_slices"
                cmd="$cmd PP_ATTN_BALANCE=$PP_ATTN_BALANCE"
                cmd="$cmd MEGATRON_PATH=$MEGATRON_PATH"
                cmd="$cmd LOG_DIR=${LOG_DIR}/${experiment_name}/n=${slice_factor}p EXP=$experiment_name"

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
    done

    echo "===== Completed $experiment_name experiments ====="
}

# Run experiments
# run_experiment "slim"
run_experiment "slim-i" $VIRTUAL_PIPELINE_LAYERS
