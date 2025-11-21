#!/bin/bash

set -euo pipefail

TS=`date +%Y_%m_%d_%H_%M_%S`
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Get Megatron-LM diff
MEGATRON_PATH=${MEGATRON_PATH:-../../}
MEGATRON_COMMIT_SHA=$(cd "$MEGATRON_PATH" && git rev-parse HEAD)
MEGATRON_DIFF=$(cd "$MEGATRON_PATH" && git status -s)

# Util functions
function check_integer() {
    if [ ! -v "$1" ]; then echo "Variable $1 is not set."; exit 1; fi
    if [[ ! "${!1}" =~ ^[+-]?[0-9]+$ ]]; then echo "Variable $1 is not an integer."; exit 1; fi
}

function check_01() {
    if [ ! -v "$1" ]; then echo "Variable $1 is not set."; exit 1; fi
    if [[ ! "${!1}" =~ ^[01]$ ]]; then echo "Variable $1 is neither 0 nor 1."; exit 1; fi
}

function check_float() {
    if [ ! -v "$1" ]; then echo "Variable $1 is not set."; exit 1; fi
    if [[ ! "${!1}" =~ ^[+-]?([0-9]+([.][0-9]+)?)|(.[0-9]+)$ ]]; then echo "Variable $1 is not a float number."; exit 1; fi
}

function check_str() {
    if [ ! -v "$1" ]; then echo "Variable $1 is not set."; exit 1; fi
    if [[ -z "${!1}" ]]; then echo "Variable $1 is not a string."; exit 1; fi
}

# Cluster settings
HOSTFILE=${HOSTFILE:-}
if [ -f /etc/mpi/hostfile ]; then
    if [ ! -f /etc/mpi/hostfile_seq -a -z "$HOSTFILE" ]; then
        echo "Please use kai_launch to generate /etc/mpi/hostfile_seq"
        exit 1
    fi
    HOSTFILE=${HOSTFILE:-/etc/mpi/hostfile_seq}
fi
if [ -n "$HOSTFILE" ]; then
    # 多机任务
    if [ -z "${MY_NODE_IP:-}" ]; then echo "Variable MY_NODE_IP does not exist."; exit 1; fi
    if ! ifconfig | grep " $MY_NODE_IP " >/dev/null; then echo "MY_NODE_IP \"$MY_NODE_IP\" is not contained in \`ifconfig\`."; exit 1; fi
    MASTER_ADDR=$MY_NODE_IP
    if [ ! -f "$HOSTFILE" ]; then echo "Hostfile \"$HOSTFILE\" does not exist."; exit 1; fi
    NP=${NP:-$(cat "$HOSTFILE" | grep -v \# | cut -d'=' -f2 | awk '{sum += $0} END {print sum}')}
else
    # 单机任务
    MASTER_ADDR=127.0.0.1
    NP=${NP:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
fi
GPU_NAMES=`nvidia-smi --query-gpu=name --format=csv,noheader`
if echo "$GPU_NAMES" | grep "NVIDIA A800" >/dev/null; then CLUSTER=a800;
elif echo "$GPU_NAMES" | grep "NVIDIA A100" >/dev/null; then CLUSTER=a800;
elif echo "$GPU_NAMES" | grep "NVIDIA H800" >/dev/null; then CLUSTER=h800;
elif echo "$GPU_NAMES" | grep "NVIDIA H100" >/dev/null; then CLUSTER=h800;
else echo "Unknown GPU name."; exit 1; fi
source "$SCRIPT_DIR/cluster/$CLUSTER"  # (HOSTFILE, NP) => (CLUSTER_MPI_ARGS,)
check_str CLUSTER_MPI_ARGS

EXP=${EXP:-sample_llama_13b}
source "$SCRIPT_DIR/exp/$EXP"

# Check model args
check_integer SEQ_LENGTH
if [ "$SEQ_LENGTH" -le 4096 ]; then
    ROPE_THETA=10000.0
else
    ROPE_THETA=1000000.0
fi
check_integer HIDDEN_SIZE
check_integer FFN_HIDDEN_SIZE
check_integer NUM_LAYERS
check_integer NUM_ATTENTION_HEADS
check_01 GQA
if [ $GQA == 1 ]; then check_integer NUM_QUERY_GROUPS; fi
NUM_EXPERTS=${NUM_EXPERTS:-}
if [ -n "$NUM_EXPERTS" ]; then check_integer NUM_EXPERTS; check_integer MOE_LAYER_INTERVAL; check_integer MOE_ROUTER_TOPK; fi
check_str POSITION_EMBEDDING
if [ "$POSITION_EMBEDDING" != "rope" -a "$POSITION_EMBEDDING" != "alibi" ]; then echo "Unkown POSITION_EMBEDDING option \"$POSITION_EMBEDDING\""; exit 1; fi
check_str OPTIMIZER_ARGS

# Check dataset
check_str DATA_ARGS

# Check run settings
check_integer GLOBAL_BATCH_SIZE
check_integer TRAIN_ITERS
check_str LOG_DIR
LOG_INTERVAL=${LOG_INTERVAL:-1}
check_integer LOG_INTERVAL
if [ -z "${SAVE:-}" ]; then check_integer SAVE_INTERVAL; fi
FINETUNE=${FINETUNE:-0}
check_01 FINETUNE

# Check performance settings
check_integer MICRO_BATCH_SIZE
POST_LM_PP=${POST_LM_PP:-0}
check_01 POST_LM_PP
if [ $POST_LM_PP == 1 ]; then check_integer MICRO_SEQ_LENGTH; fi
PP_ATTN_BALANCE=${PP_ATTN_BALANCE:-0}
MICRO_SEQ_LENGTH=${MICRO_SEQ_LENGTH:-0}
check_integer MICRO_SEQ_LENGTH
# set default PP_l to L // p in slim 1F1B
if [ $MICRO_SEQ_LENGTH -gt 0 -a -z "$PP_l" ]; then PP_l=$(($NUM_LAYERS / $PP)); fi
KV_CACHE_IMPL=${KV_CACHE_IMPL:-extended}
check_str KV_CACHE_IMPL
check_integer TP
check_integer CP
check_integer EP
check_integer PP
if [ -n "$PP_l" ]; then check_integer PP_l; fi
check_str RECOMPUTE
if [ $RECOMPUTE != no -a $RECOMPUTE != partial -a $RECOMPUTE != partial+fc1 -a $RECOMPUTE != full ]; then echo "Unknown RECOMPUTE option."; exit 1; fi
check_float OFFLOAD_ALPHA
check_01 SP
check_01 TP_OVERLAP
check_01 DP_OVERLAP
CP_IMPL=${CP_IMPL:-key-value}
check_str CP_IMPL
PP_ATTN_BALANCE=${PP_ATTN_BALANCE:-0}
check_integer PP_ATTN_BALANCE
check_integer SYNC_LEVEL
check_01 ALLOC_CONF
check_01 CLONE_SCATTER_OUTPUT_IN_EMBEDDING
check_integer POST_LM_PROCESSING_SLICE_SIZE
check_integer SFT_PADDING
check_integer CLUSTER_DP_CTAS_MULTIPLIER

PROFILE=${PROFILE:-0}
check_01 PROFILE

# Verify variables
OFFLOAD_ALPHA_GT_0=`python3 -c "print([0, 1][$OFFLOAD_ALPHA > 0])"`
if [ $RECOMPUTE == full -a "$SYNC_LEVEL" -gt 2 ]; then echo "Using RECOMPUTE=full requires SYNC_LEVEL<=2"; exit 1; fi
if [ $OFFLOAD_ALPHA_GT_0 == 1 -a -z "$PP_l" ]; then echo "Using OFFLOAD_ALPHA>0 requires PP_l>=1"; exit 1; fi
if [ $OFFLOAD_ALPHA_GT_0 == 1 -a $RECOMPUTE == no ]; then echo "Using OFFLOAD_ALPHA>0 requires RECOMPUTE>=no"; exit 1; fi
if [ $DP_OVERLAP == 1 -a -z "$PP_l" ]; then echo "Using DP_OVERLAP=1 requires PP_l>=1"; exit 1; fi

MODEL_ARGS="
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --no-position-embedding \
    `if [ "$POSITION_EMBEDDING" == rope ]; then echo --use-rotary-position-embeddings --rope-theta $ROPE_THETA; fi` \
    `if [ "$POSITION_EMBEDDING" == alibi ]; then echo --use-alibi; fi` \
    --swiglu \
    --rms-norm \
    --disable-bias-linear \
    --hidden-dropout 0. \
    --attention-dropout 0. \
    --no-query-key-layer-scaling \
    `if [ $GQA == 1 ]; then echo --group-query-attention --num-query-groups $NUM_QUERY_GROUPS; fi` \
    `if [ -n "$NUM_EXPERTS" ]; then echo --num-experts $NUM_EXPERTS --moe-layer-interval $MOE_LAYER_INTERVAL --moe-router-topk $MOE_ROUTER_TOPK; fi` \
    $OPTIMIZER_ARGS \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --bf16 \
    --accumulate-allreduce-grads-in-fp32 \
    --seed 9527 \
"
#    --untie-embeddings-and-output-weights \

OUTPUT_ARGS="
    --log-interval $LOG_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 0 \
    ${TENSORBOARD_DIR:+--tensorboard-dir "$TENSORBOARD_DIR"} \
    ${LOAD:+--load "$LOAD"} \
    ${SAVE:+--save "$SAVE"} \
    ${SAVE:+--save-interval $SAVE_INTERVAL} \
    `if [ $FINETUNE == 1 ]; then echo --finetune; fi` \
"

PERFORMANCE_ENV="
    -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -x UB_AG_SM_MARGIN=0 \
    -x UB_RS_SM_MARGIN=1 \
    -x UB_MAX_STREAMS=2 \
    -x TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
"

PERFORMANCE_ARGS="
    --micro-batch-size $MICRO_BATCH_SIZE \
    --use-distributed-optimizer \
    --tensor-model-parallel-size $TP \
    `if [ $SP == 1 ]; then echo --sequence-parallel; fi` \
    `if [ $CP != 1 ]; then echo --context-parallel-size $CP; fi` \
    `if [ $EP != 1 ]; then echo --expert-model-parallel-size $EP; fi` \
    --pipeline-model-parallel-size $PP \
    ${PP_l:+--num-layers-per-virtual-pipeline-stage $PP_l} \
    --kaimm-context-parallel-impl $CP_IMPL \
    --micro-seq-length $MICRO_SEQ_LENGTH \
    --kaimm-kv-cache-impl $KV_CACHE_IMPL \
    --kaimm-pipeline-attn-balance $PP_ATTN_BALANCE \
    `if [ $RECOMPUTE == partial ]; then echo --kaimm-recompute-mlp-activation-func; fi` \
    `if [ $RECOMPUTE == partial+fc1 ]; then echo --kaimm-recompute-mlp-activation-func --kaimm-recompute-mlp-fc1 --kaimm-recompute-token-dispatcher; fi` \
    `if [ $RECOMPUTE == full ]; then echo --recompute-granularity full --recompute-method uniform --recompute-num-layers 1; fi` \
    `if [ $OFFLOAD_ALPHA_GT_0 == 1 ]; then echo --kaimm-offload-activation-ratio $OFFLOAD_ALPHA; fi` \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --use-flash-attn \
    --use-fast-rms-norm \
    --use-memory-efficient-norm \
    `if [ "$POSITION_EMBEDDING" == rope ]; then echo --use-fast-rope; fi` \
    `if [[ $MICRO_SEQ_LENGTH != 0 || -n "$PP_l" ]]; then echo --overlap-p2p-communication; fi` \
    `if [ $DP_OVERLAP == 1 ]; then echo --kaimm-overlap-optimizer-communication; fi` \
    `if [ $DP_OVERLAP == 1 ]; then echo --kaimm-overlap-reduce-ratio 0.$(printf "%06d" $(((999999 + $PP) / $PP))); fi` \
    `if [ $DP_OVERLAP == 1 ]; then echo --kaimm-overlap-gather-ratio 0.$(printf "%06d" $(((999999 + $PP) / $PP))); fi` \
    `if [ $DP_OVERLAP == 1 ]; then echo --kaimm-overlap-optimizer-slow-ctas $((8 / $TP * $CLUSTER_DP_CTAS_MULTIPLIER)); fi` \
    `if [ $TP_OVERLAP == 1 ]; then echo --overlap-sp-ag --overlap-sp-rs; fi` \
    `if [[ $GQA == 1 || $CP_IMPL != "key-value" ]]; then echo --no-context-parallel-comm-overlap-gemm; fi` \
    --no-context-parallel-comm-overlap-gemm \
    `if [ $CP != 1 ]; then echo --kaimm-overlap-cp-slow-ctas 8; fi` \
    `if [ $CLONE_SCATTER_OUTPUT_IN_EMBEDDING == 1 ]; then echo --clone-scatter-output-in-embedding; fi` \
    `if [ $POST_LM_PROCESSING_SLICE_SIZE != 0 ]; then echo --kaimm-post-lm-processing-slice-size $POST_LM_PROCESSING_SLICE_SIZE; fi` \
    `if [ $SYNC_LEVEL != 0 ]; then echo --kaimm-cuda-synchronize-level $SYNC_LEVEL; fi` \
    --kaimm-async-dataloader \
    --num-workers 2 \
    --prefetch-factor 64 \
    --kaimm-warmup-iters 0 \
    --moe-token-dispatcher-type alltoall \
    --moe-router-force-load-balancing \
    --moe-grouped-gemm \
    `if [ $POST_LM_PP == 1 ]; then echo --kaimm-vocab-in-pipeline-parallel; fi` \
"

if [ "${ONLY_K023_OPTIONS:-0}" != 1 ]; then
    PERFORMANCE_ARGS_NEW="
        `if [ $DP_OVERLAP == 1 ]; then echo --kaimm-overlap-optimizer-no-barrier; fi` \
        --kaimm-gc-interval 32 \
    "
else
    PERFORMANCE_ARGS_NEW=
fi

# PLM_RSH_ARGS
if [ -v TARGET_IP_PORT_FILE ]; then check_str TARGET_IP_PORT_FILE; PLM_RSH_ARGS="-F $TARGET_IP_PORT_FILE";
elif [ ! -v TARGET_IP_PORT_FILE -a -n "$HOSTFILE" ]; then PORT=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2); check_integer PORT; PLM_RSH_ARGS="-p $PORT";
else PLM_RSH_ARGS=;
fi

# NCCL_ENV_WRAPPER
if [ -n "$HOSTFILE" ] && command -v with_nccl_local_env >/dev/null; then NCCL_ENV_WRAPPER=with_nccl_local_env;
else NCCL_ENV_WRAPPER=; fi

# PROFILE_WRAPPER
if [[ "$PROFILE" == 1 ]]; then
    PROFILE_WRAPPER="$SCRIPT_DIR/nsys_profile_last_rank.sh";
    # PROFILE_WRAPPER="nsys profile \
    #                 --force-overwrite true \
    #                 -t cuda,nvtx \
    #                 -s none --cpuctxsw none \
    #                 --capture-range cudaProfilerApi --capture-range-end stop"
    # PROFILE_ARGS="--profile \
    #             --profile-step-start 4 \
    #             --profile-step-end 5 \
    #             --profile-ranks 0"
else
    PROFILE_WRAPPER=;
    PROFILE_ARGS=;
fi

# ENTRYPOINT
ENTRYPOINT=pretrain_llama.py;

mkdir -p "$LOG_DIR"
SEQ_LENGTH_BASE=$(($SEQ_LENGTH / 1024))
LOG_FILE_NAME="${TS}-${GLOBAL_BATCH_SIZE}-${SEQ_LENGTH_BASE}k"
exec &> >(tee "$LOG_DIR/${LOG_FILE_NAME}.txt")

# Echo Megatron-LM version
echo MEGATRON_COMMIT_SHA $MEGATRON_COMMIT_SHA
if [ -n "$MEGATRON_DIFF" ]; then
    echo "$MEGATRON_DIFF"
    echo "[WARNING] There are uncommitted changes in $MEGATRON_PATH"
fi

set -x
# nsys profile -t cuda,nvtx -s none --cpuctxsw none --python-sampling true --python-sampling-frequency 1000 \
mpirun --allow-run-as-root \
        ${HOSTFILE:+--hostfile "$HOSTFILE"} \
        --np $NP \
        --bind-to none --map-by slot \
        --mca plm_rsh_args "$PLM_RSH_ARGS" \
        $CLUSTER_MPI_ARGS \
        -x PATH \
        ${LD_LIBRARY_PATH:+-x LD_LIBRARY_PATH} \
        ${LD_PRELOAD:+-x LD_PRELOAD} \
        -x PYTHONPATH="$MEGATRON_PATH" \
        -x NCCL_DEBUG=WARN \
        -x CUDA_LAUNCH_BLOCKING=0 \
        -x NCCL_NVLS_ENABLE=0 \
        $PERFORMANCE_ENV \
    $NCCL_ENV_WRAPPER \
    $PROFILE_WRAPPER \
    python3 -u "$MEGATRON_PATH/$ENTRYPOINT" \
        --train-iters $TRAIN_ITERS \
        $MODEL_ARGS \
        ${EXTRA_MODEL_ARGS:-} \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $PERFORMANCE_ARGS \
        $PERFORMANCE_ARGS_NEW \
        ${STABILITY_ARGS:-} \
        ${PROFILE_ARGS:-} \
        --master-addr ${MASTER_ADDR}:6002

# failover forever
exit 1
