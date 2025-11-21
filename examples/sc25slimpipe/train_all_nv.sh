#!/bin/bash

# Define paths
BASE_DIR="$HOME/Megatron-LM/examples/sc25slimpipe"
LOGS_DIR=$BASE_DIR/logs/seq_length
MEGATRON_DIR=$HOME/github/Megatron-LM

# Define test parameters
TRAIN_ITERS=10
GLOBAL_BATCH_SIZES=(4)
SEQ_LENGTH_BASES=(32 64 128 256) # Base values in K (will be multiplied by 1024)
TP_SIZE=8                            # Tensor Parallel size
PP_SIZE=4                            # Pipeline Parallel size
LAYERS_PER_STAGE=2
TOTAL_GPUS=$((TP_SIZE * PP_SIZE))
RECORD_MEMORY_HISTORY=

# Switch to correct branch
cd $MEGATRON_DIR || exit
git switch slimpipe_r0.8.0
cd - || exit

# Function to run experiments
run_experiment() {
    local exp_name=$1
    local layers_per_stage=$2 # Optional parameter

    echo "===== Running $exp_name experiments ====="

    for batch_size in "${GLOBAL_BATCH_SIZES[@]}"; do
        for seq_base in "${SEQ_LENGTH_BASES[@]}"; do
            # Calculate actual sequence length
            full_seq_length=$((seq_base * 1024))

            echo "Running: batch_size=$batch_size, sequence_length=$full_seq_length"

            # Build command
            cmd="NP=$TOTAL_GPUS TP=$TP_SIZE PP=$PP_SIZE"

            # Add virtual pipeline parameter if provided
            if [[ -n $layers_per_stage ]]; then
                cmd="PP_l=$layers_per_stage $cmd"
            fi

            # Add remaining parameters
            cmd="$cmd TRAIN_ITERS=$TRAIN_ITERS"
            cmd="$cmd RECORD_MEMORY_HISTORY=$RECORD_MEMORY_HISTORY"
            cmd="$cmd GLOBAL_BATCH_SIZE=$batch_size SEQUENCE_LENGTH=$full_seq_length"
            cmd="$cmd MEGATRON_PATH=$MEGATRON_DIR LOG_DIR=${LOGS_DIR}/${exp_name} EXP=$exp_name"
            cmd="$cmd $BASE_DIR/pretrain_llama_nv.sh"

            # Execute command
            echo $cmd
            eval $cmd

            echo "Cooling period: waiting 30 seconds before next run..."
            sleep 30
        done
    done

    echo "===== Completed $exp_name experiments ====="
}

# Run experiments
run_experiment "nv-1f1b"
run_experiment "nv-1f1b-i" $LAYERS_PER_STAGE
