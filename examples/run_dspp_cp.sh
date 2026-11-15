#!/bin/bash
# Small fixed-capacity CP2 run. Correctness/profile/throughput use separate roots.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DSPP_CP_BENCH_OUTPUT:?set a new independent output directory}"
: "${DATA_PATH:?set the existing or independently constructed dataset}"
PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/megatron/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
export DSPP_CP_BENCH_WARMUP=${DSPP_CP_BENCH_WARMUP:-2}
export DSPP_CP_BENCH_TIMED=${DSPP_CP_BENCH_TIMED:-6}
export DSPP_BENCH_OUTPUT="$DSPP_CP_BENCH_OUTPUT"
export DSPP_BENCH_WARMUP="$DSPP_CP_BENCH_WARMUP"
export DSPP_BENCH_TIMED="$DSPP_CP_BENCH_TIMED"
mode=${DSPP_CP_MODE:-fused}
pp=${DSPP_PP_SIZE:-2}
cp=${DSPP_CP_SIZE:-2}
extra=()
case "$mode" in
 fused) ;;
 immediate) extra+=(--dspp-split-attention-backward --dspp-history-dkv-after-dx) ;;
 template|online)
  : "${DSPP_EXECUTION_PROFILE:?}" "${DSPP_P2P_PROFILE:?}" "${DSPP_MEMORY_PROFILE:?}"
  schedule=b
  [[ $mode == template ]] && schedule=template
  extra+=(--dspp-schedule "$schedule" --dspp-split-attention-backward --dspp-history-dkv-after-dx
    --dspp-execution-profile "$DSPP_EXECUTION_PROFILE" --dspp-p2p-profile "$DSPP_P2P_PROFILE"
    --dspp-memory-profile "$DSPP_MEMORY_PROFILE" --dspp-prefetch
    --dspp-b-budget-fraction 1.0)
  ;;
 *) exit 2 ;;
esac
extra+=("$@")
cmd=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node="$((pp*cp))"
 "${DSPP_CP_ENTRYPOINT:-tools/benchmark_dspp.py}"
 --tensor-model-parallel-size 1 --pipeline-model-parallel-size "$pp" --context-parallel-size "$cp"
 --num-layers-per-virtual-pipeline-stage "$((${DSPP_NUM_LAYERS:-4}/(pp*2)))" --num-layers "${DSPP_NUM_LAYERS:-4}"
 --hidden-size "${DSPP_HIDDEN_SIZE:-512}" --ffn-hidden-size "${DSPP_FFN_SIZE:-1024}" --num-attention-heads "${DSPP_HEADS:-4}"
 --seq-length "${DSPP_SEQ_LENGTH:-8192}" --max-position-embeddings "${DSPP_SEQ_LENGTH:-8192}" --micro-seq-length "${DSPP_CHUNK_SIZE:-2048}"
 --micro-batch-size 4 --global-batch-size 8
 --train-iters "$((DSPP_CP_BENCH_WARMUP+DSPP_CP_BENCH_TIMED))" --seed 1234
 --lr 0.001 --min-lr 0.0001 --lr-decay-style cosine --weight-decay 0.1 --clip-grad 1.0
 --bf16 --use-flash-attn --no-masked-softmax-fusion
 --no-context-parallel-comm-overlap-gemm --kaimm-context-parallel-impl "${DSPP_CP_IMPL:-key-value}"
 --no-position-embedding --use-rotary-position-embeddings --alibi-bias-max 0
 --swiglu --rms-norm --disable-bias-linear --no-query-key-layer-scaling
 --hidden-dropout 0 --attention-dropout 0 --make-main-grad-addresss-divisible-by 1
 --data-path "$DATA_PATH" --data-impl mmap --split 100,0,0
 --tokenizer-type NullTokenizer --vocab-size "${DSPP_VOCAB_SIZE:-32001}" --dataloader-type single
 --dspp --dspp-cp-layout "${DSPP_CP_LAYOUT:-hybrid}" --dspp-v-layer-layout balanced
 --dspp-microbatch-order warmup-short-steady-long --dspp-packing-strategy baseline
 --dspp-recv-lookahead "${DSPP_RECV_LOOKAHEAD:-0}"
 --dspp-send-inflight-depth "${DSPP_SEND_INFLIGHT_DEPTH:-1}" --variable-seq-pad-token-id 0
 --eval-iters 0 --log-interval 1000 --timing-log-level 0 "${extra[@]}")
test ! -e "$DSPP_CP_BENCH_OUTPUT"
mkdir -p "$(dirname "$DSPP_CP_BENCH_OUTPUT")"
"$PYTHON_BIN" tools/wait_dspp_gpus.py --output "${DSPP_CP_BENCH_OUTPUT}.gpu.jsonl"
printf '%q ' "${cmd[@]}" > "${DSPP_CP_BENCH_OUTPUT}.command.sh"
printf '\n' >> "${DSPP_CP_BENCH_OUTPUT}.command.sh"
timeout --kill-after=15 "${DSPP_CP_TIMEOUT:-600}" "${cmd[@]}" > "${DSPP_CP_BENCH_OUTPUT}.log" 2>&1
