#!/usr/bin/env bash

# DeepSeek-V4-Pro H200 single-node vllm **offline** benchmark — CANN-style
# (cann-recipes-infer infer.sh shape): in-process engine, one warmup batch +
# one timed lockstep batch of InfiniteBench prompts. ISL is 8192 input
# tokens; OSL is 256 *decode steps* (main-model forward passes), matching
# the 950DT reference's max_new_tokens loop bound — MTP bonus tokens are
# excluded from the headline metrics by construction.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    DP_ATTENTION \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RESULT_FILENAME

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

NUM_SPEC_TOKENS="$(dsv4_mtp_spec_tokens_for_spec_decoding)"
DPA_FLAG=()
[[ "${DP_ATTENTION}" == "true" ]] && DPA_FLAG=(--dp-attn)
start_gpu_monitor --output "$PWD/gpu_metrics.csv"

if [[ "${SLURM_NNODES:-1}" -gt 1 ]]; then
    export MASTER_PORT=${MASTER_PORT:-29501}
    echo "Multi-node: MASTER_ADDR=${MASTER_ADDR:-unset} MASTER_PORT=$MASTER_PORT node_rank=$SLURM_PROCID"
fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"

set -x
PYTHONNOUSERSITE=1 python3 utils/bench_offline/run_offline.py \
    --engine vllm \
    --model "$MODEL_PATH" \
    --tp "$TP" \
    --ep "$EP_SIZE" \
    --num-chips "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --mtp "$NUM_SPEC_TOKENS" \
    --temperature 1.0 \
    --infinitebench-input-len "$ISL" \
    --decode-steps "$OSL" \
    --routing-sim-strategy "${DSV4_OFFLINE_ROUTING_SIM:-uniform_random}" \
    --nnodes "${SLURM_NNODES:-1}" \
    --node-rank "${SLURM_PROCID:-0}" \
    --batch-size "$CONC" \
    --result-dir "$PWD/" \
    --result-filename "$RESULT_FILENAME" \
    --metadata "benchmark_input_len=$ISL" "benchmark_output_len=$OSL" \
    "${DPA_FLAG[@]}"
set +x

stop_gpu_monitor
