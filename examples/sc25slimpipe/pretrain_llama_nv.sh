#!/bin/bash

set -euo pipefail

TS=`date +%Y_%m_%d_%H_%M_%S`
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Get Megatron-LM diff
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}
MEGATRON_COMMIT_SHA=$(cd "$MEGATRON_PATH" && git rev-parse HEAD)
MEGATRON_DIFF=$(cd "$MEGATRON_PATH" && git status -s)

# Util functions
function check_integer() {
    if [ ! -v "$1" ]; then echo "Variable $1 is not set."; exit 1; fi
    if [[ ! "${!1}" =~ ^[0-9]+$ ]]; then echo "Variable $1 is not an integer."; exit 1; fi
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
MICRO_SEQ_LENGTH=${MICRO_SEQ_LENGTH:-0}
check_integer MICRO_SEQ_LENGTH
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
check_integer SYNC_LEVEL
check_01 ALLOC_CONF
# check_01 CLONE_SCATTER_OUTPUT_IN_EMBEDDING
# check_integer POST_LM_PROCESSING_SLICE_SIZE
# check_integer SFT_PADDING
# check_integer CLUSTER_DP_CTAS_MULTIPLIER

PROFILE=${PROFILE:-0}
check_01 PROFILE

# Verify variables
OFFLOAD_ALPHA_GT_0=`python3 -c "print([0, 1][$OFFLOAD_ALPHA > 0])"`
if [ $RECOMPUTE == full -a "$SYNC_LEVEL" -gt 2 ]; then echo "Using RECOMPUTE=full requires SYNC_LEVEL<=2"; exit 1; fi
# if [ $OFFLOAD_ALPHA_GT_0 == 1 -a -z "$PP_l" ]; then echo "Using OFFLOAD_ALPHA>0 requires PP_l>=1"; exit 1; fi
if [ $DP_OVERLAP == 1 -a -z "$PP_l" ]; then echo "Using DP_OVERLAP=1 requires PP_l>=1"; exit 1; fi

GPT_ARGS="
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    `if [ $GQA == 1 ]; then echo --group-query-attention --num-query-groups $NUM_QUERY_GROUPS; fi` \
    `if [ -n "$NUM_EXPERTS" ]; then echo --num-experts $NUM_EXPERTS --moe-router-topk $MOE_ROUTER_TOPK; fi` \
    --micro-batch-size 1 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr 1.5e-4 \
    --lr-decay-iters 500000 \
    --lr-decay-style cosine \
    --min-lr 1.5e-5 \
    --weight-decay 0.1 \
    --lr-warmup-iters 2000 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --bf16 \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --swiglu \
    --normalization RMSNorm \
    --disable-bias-linear \
    --hidden-dropout 0. \
    --attention-dropout 0. \
    --no-bias-gelu-fusion \
    --no-masked-softmax-fusion \
    --no-bias-dropout-fusion \
    $OPTIMIZER_ARGS \
"
    # --untie-embeddings-and-output-weights \

# DATA_PATH=/nlp_group/yexucheng/release/ATC2024/dataset/part1_text_document

# DATA_ARGS="
#     --data-path $DATA_PATH \
#     --tokenizer-type SentencePieceTokenizer \
#     --vocab-size 32004 \
#     --split 949,50,1
# "

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
    `if [ $ALLOC_CONF == 1 ]; then echo -x PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21; fi` \
"

# PLM_RSH_ARGS
if [ -v TARGET_IP_PORT_FILE ]; then check_str TARGET_IP_PORT_FILE; PLM_RSH_ARGS="-F $TARGET_IP_PORT_FILE";
elif [ ! -v TARGET_IP_PORT_FILE -a -n "$HOSTFILE" ]; then PORT=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2); check_integer PORT; PLM_RSH_ARGS="-p $PORT";
else PLM_RSH_ARGS=;
fi

# NCCL_ENV_WRAPPER
if [ -n "$HOSTFILE" ] && command -v with_nccl_local_env >/dev/null; then NCCL_ENV_WRAPPER=with_nccl_local_env;
else NCCL_ENV_WRAPPER=; fi

# PROFILE_WRAPPER
if [ $PROFILE == 1 ]; then PROFILE_WRAPPER="$SCRIPT_DIR/nsys_profile_last_rank.sh";
else PROFILE_WRAPPER=; fi

# ENTRYPOINT
ENTRYPOINT=pretrain_gpt.py;

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

HOSTNAME_WRAPPER=/usr/local/bin/wrap_mpi.sh

set -x
# nsys profile -t cuda,nvtx -s none --cpuctxsw none --python-sampling true --python-sampling-frequency 1000 \
mpirun --allow-run-as-root \
        ${HOSTFILE:+--hostfile "$HOSTFILE"} \
        -np $NP \
        --bind-to none --map-by slot \
        --mca plm_rsh_args "$PLM_RSH_ARGS" \
        $CLUSTER_MPI_ARGS \
        -x PATH \
        ${LD_LIBRARY_PATH:+-x LD_LIBRARY_PATH} \
        ${LD_PRELOAD:+-x LD_PRELOAD} \
        -x PYTHONPATH="$MEGATRON_PATH" \
        -x NCCL_DEBUG=WARN \
        -x CUDA_LAUNCH_BLOCKING=0 \
        $PERFORMANCE_ENV \
        -x MASTER_ADDR=$MASTER_ADDR -x MASTER_PORT=6002 \
    $NCCL_ENV_WRAPPER \
    $PROFILE_WRAPPER \
    python3 -u "$MEGATRON_PATH/$ENTRYPOINT" \
    --train-iters $TRAIN_ITERS \
    --use-distributed-optimizer \
    --accumulate-allreduce-grads-in-fp32 \
    --initial-loss-scale 1 \
    --tensor-model-parallel-size $TP \
    `if [ $SP == 1 ]; then echo --sequence-parallel; fi` \
    `if [[ $PP_l ]]; then echo --num-layers-per-virtual-pipeline-stage $PP_l; fi` \
    `if [ $CP != 1 ]; then echo --context-parallel-size $CP; fi` \
    `if [ $EP != 1 ]; then echo --expert-model-parallel-size $EP; fi` \
    --pipeline-model-parallel-size $PP \
    --use-flash-attn \
    --no-create-attention-mask-in-dataloader \
    `if [ $RECOMPUTE == full ]; then echo --recompute-granularity full --recompute-method uniform --recompute-num-layers 1; fi` \
    --use-mcore-models \
    --moe-token-dispatcher-type alltoall \
    --moe-router-force-load-balancing \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --manual-gc \
    --manual-gc-interval 9999 \
    --num-workers 2 \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --master-addr ${MASTER_ADDR}:6002

# failover forever
exit 1
