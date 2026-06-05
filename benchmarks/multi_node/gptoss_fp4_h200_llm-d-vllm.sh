#!/usr/bin/env bash
#
# Wrapper for the gpt-oss-120b H200 llm-d-vllm 1P+1D smoke benchmark.
# Sibling of dsr1_fp8_h200_llm-d-vllm.sh; the wrapper itself is
# model-agnostic (just sets topology env and calls the shared submit.sh).
# Kept as its own file so the runner's
#   SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_h200_llm-d-vllm.sh"
# rule resolves gptoss/fp4 here.

set -euo pipefail

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    CONC_LIST \
    ISL \
    OSL \
    IMAGE \
    MODEL_PATH \
    PREFILL_NODES \
    DECODE_NODES \
    RANDOM_RANGE_RATIO

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

set -x

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/llm-d" || exit 1

export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE

JOB_ID=$(bash ./submit.sh \
    "$PREFILL_NODES" \
    "$DECODE_NODES" \
    "$ISL" "$OSL" "${CONC_LIST// /x}" inf \
    "$RANDOM_RANGE_RATIO")

if [[ -z "$JOB_ID" ]]; then
    echo "Failed to submit job" >&2
    exit 1
fi

echo "$JOB_ID"
