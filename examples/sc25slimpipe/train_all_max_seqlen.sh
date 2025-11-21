#!/bin/bash

# Define constants
TRAIN_ITERS=5
BASE_DIR="$HOME/Megatron-LM/examples/sc25slimpipe"
RECORD_MEMORY_HISTORY=0

# Model configuration
MODEL="LLaMA-7B"
MODEL_FILE="7b"
NP=64

# Parallel configuration
t=8                   # Tensor Parallel
c=1                   # Context Parallel
p=8                   # Pipeline Parallel
d=$((NP / t / c / p)) # Data Parallel

n=$((8 * p)) # Number of slices
RECOMPUTE=no
OFFLOAD_ALPHA=0
ALLOC_CONF=0

LOGS_DIR="${BASE_DIR}/logs/max_seqlen"

# Function to run an experiment
run_experiment() {
    local seq_length_bases=($1)
    local batch_size="$2"
    local virtual_layers="$3"
    local megatron_path="$4"
    local entry_point="$5"
    local experiment_name="$6"
    local log_dir_suffix="${7:-}"

    for SEQ_LENGTH_BASE in "${seq_length_bases[@]}"; do
        SEQ_LENGTH=$((SEQ_LENGTH_BASE * 1024))
        MICRO_SEQ_LENGTH=$((SEQ_LENGTH / n))

        # Print configuration summary
        echo "---------- Processing entry for ${MODEL} with ${NP} GPUs ----------"
        echo "  Model: ${MODEL}, GPUs: ${NP}, Batch: ${batch_size}"
        echo "  TP: ${t}, CP: ${c}, PP: ${p}, DP: ${d}, Slices: ${n}, Virtual Layers: ${virtual_layers}"
        echo "  Sequence Length: ${SEQ_LENGTH}, Mirco Sequence Length: ${MICRO_SEQ_LENGTH}"
        echo "  Recompute: ${RECOMPUTE}, Offload: ${OFFLOAD_ALPHA}"

        # Switch to appropriate branch
        cd "$megatron_path" || exit
        if [[ "$megatron_path" == "/nlp_group/zhangwei/Megatron-LM" ]]; then
            git switch wei/attn_balance
        else
            git switch slimpipe_r0.8.0
        fi
        cd - || exit

        # Build command
        cmd="NP=$NP"
        cmd="$cmd RECORD_MEMORY_HISTORY=$RECORD_MEMORY_HISTORY"
        cmd="$cmd TRAIN_ITERS=$TRAIN_ITERS"
        cmd="$cmd GLOBAL_BATCH_SIZE=$batch_size"
        cmd="$cmd TP=$t CP=$c PP=$p"

        if [[ -n "$virtual_layers" ]]; then
            cmd="$cmd PP_l=$virtual_layers"
        fi

        cmd="$cmd RECOMPUTE=$RECOMPUTE"
        cmd="$cmd OFFLOAD_ALPHA=$OFFLOAD_ALPHA"
        cmd="$cmd ALLOC_CONF=$ALLOC_CONF"
        cmd="$cmd SEQUENCE_LENGTH=$SEQ_LENGTH"
        cmd="$cmd NUM_SLICES=$n MICRO_SEQ_LENGTH=$MICRO_SEQ_LENGTH"
        cmd="$cmd MEGATRON_PATH=$megatron_path"
        cmd="$cmd LOG_DIR=${LOGS_DIR}/${MODEL}/${experiment_name}${log_dir_suffix}"
        cmd="$cmd EXP=$experiment_name MODEL=$MODEL_FILE"
        cmd="$cmd ${BASE_DIR}/${entry_point}"

        # Display and execute command
        echo "Command: $cmd"
        echo "------------------------------------------------------------------"
        eval "$cmd"
    done
}

# Experiment 1: SlimPipe Llama
SLIMPIPE_SEQ_LENGTHS=(600)
run_experiment "${SLIMPIPE_SEQ_LENGTHS[*]}" 2 1 \
    "$HOME/Megatron-LM" \
    "pretrain_llama.sh" \
    "slimpipe/llama"

# Experiment 2: Megatron Llama without virtual layers
MEGATRON_SEQ_LENGTHS=(124)
run_experiment "${MEGATRON_SEQ_LENGTHS[*]}" 16 "" \
    "$HOME/github/Megatron-LM" \
    "pretrain_llama_nv.sh" \
    "megatron/llama"

# Experiment 3: Megatron Llama with virtual layers
MEGATRON_SEQ_LENGTHS=(80 88 92)
run_experiment "${MEGATRON_SEQ_LENGTHS[*]}" 16 1 \
    "$HOME/github/Megatron-LM" \
    "pretrain_llama_nv.sh" \
    "megatron/llama" \
    "/l=1"
