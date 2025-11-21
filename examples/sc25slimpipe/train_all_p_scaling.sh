#!/bin/bash

# Define constants
TRAIN_ITERS=2
SEQ_LENGTH_BASES=(96 64 32)
BASE_DIR="$HOME/Megatron-LM/examples/sc25slimpipe"
RECORD_MEMORY_HISTORY=0
SYNC_LEVEL=1
PP_ATTN_BALANCE=0

MEGATRON_PATH="$HOME/Megatron-LM"
ENTRY_POINT="pretrain_llama.sh"
LOGS_DIR="${BASE_DIR}/logs/p_scaling/slimpipe"

EXPERIMENT_NAME="slimpipe/llama"
MODEL="LLaMA-13B"
MODEL_FILE="13b"

PP_SIZES=(2 4 8)

t=8
c=1
e=1
d=1

L=40
B=2

ckpt=

for SEQ_LENGTH_BASE in "${SEQ_LENGTH_BASES[@]}"; do
    SEQ_LENGTH=$((SEQ_LENGTH_BASE * 1024))
    for p in "${PP_SIZES[@]}"; do
        NP=$((t * c * p * e * d))
        # v=1
        # l=$((L / p / v))
        l=1
        v=$((L / p / l))
        n=$((4 * p))
        MICRO_SEQ_LENGTH=$((SEQ_LENGTH / n))
        RECOMPUTE=$([[ "$ckpt" ]] && echo "full" || echo "no")

        # Print configuration summary
        echo "---------- Processing entry for ${MODEL} with ${NP} GPUs ----------"
        echo "  Model: ${MODEL}, GPUs: ${NP}, Batch: ${B}, TP: ${t}, CP: ${c}, PP: ${p}"
        echo "  EP: ${e}, DP: ${d}, Slices: ${n}, Trunks: ${v}, Virtual Layers: ${l}"
        echo "  Checkpoint: ${RECOMPUTE}, Sequence Length: ${SEQ_LENGTH}"
        echo "  Mirco Sequence Length: ${MICRO_SEQ_LENGTH}"

        cd $MEGATRON_PATH || exit
        git switch wei/attn_balance
        cd - || exit

        # Build command
        cmd="NP=$NP"
        cmd="$cmd RECORD_MEMORY_HISTORY=$RECORD_MEMORY_HISTORY"
        cmd="$cmd SYNC_LEVEL=$SYNC_LEVEL"
        cmd="$cmd PP_ATTN_BALANCE=$PP_ATTN_BALANCE"
        cmd="$cmd TRAIN_ITERS=$TRAIN_ITERS"
        cmd="$cmd GLOBAL_BATCH_SIZE=$B"
        cmd="$cmd TP_OVERLAP=0"
        cmd="$cmd TP=$t CP=$c PP=$p EP=$e"
        [[ "$v" -gt 1 ]] && cmd="$cmd PP_l=$l"
        cmd="$cmd RECOMPUTE=$RECOMPUTE"
        cmd="$cmd SEQUENCE_LENGTH=$SEQ_LENGTH"
        cmd="$cmd NUM_SLICES=$n MICRO_SEQ_LENGTH=$MICRO_SEQ_LENGTH"
        cmd="$cmd MEGATRON_PATH=$MEGATRON_PATH"
        cmd="$cmd LOG_DIR=${LOGS_DIR}/${MODEL}/tp${t}/pp${p}"
        cmd="$cmd EXP=$EXPERIMENT_NAME MODEL=$MODEL_FILE"
        cmd="$cmd ${BASE_DIR}/${ENTRY_POINT}"

        # Display command
        echo "Command: $cmd"
        echo "------------------------------------------------------------------"
        eval "$cmd"

    done
done
