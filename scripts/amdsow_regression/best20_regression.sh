#!/usr/bin/env bash
# Full 20-row best-config regression (Section 10 of the AMDSOW InferenceX user guide).
cd /mnt/shared_mango/amdsow-deliveries/regress/amdsow-inferencex-pr || exit 1
export SLURM_CONF=/run/slurm/conf/slurm.conf

export MODEL="deepseek-ai/DeepSeek-R1-0528"
export MODEL_PREFIX="dsr1"
export PRECISION="fp8"
export FRAMEWORK="vllm-disagg"
export IMAGE="docker.io/chaeminlimmb/vllm-mori-pd:milestone4-aiterwheel"
export ROUTER_TYPE=moriio
export RANDOM_RANGE_RATIO=0.8
export IS_MULTINODE=true
export KEEP_LOGS=1
export RUNNER_TYPE="mi300x-disagg"
export GITHUB_WORKSPACE="$PWD"
export RUN_EVAL=true
export EVAL_ONLY=false
export EVAL_SERVER_MAX_MODEL_LEN=20480
export EVAL_SERVER_BLOCK_SIZE=1
unset EVAL_CONC

export AMDSOW_SLURM_EXCLUDE_NODES="${AMDSOW_SLURM_EXCLUDE_NODES:-a05u43,a04u43}"
export RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
echo "RUN_TS=$RUN_TS" | tee /tmp/best20_run_ts.txt

submit_best() {
  local row="$1"
  export RUNNER_NAME="mi300x-best20-${row}-${RUN_TS}"
  export RESULT_FILENAME="validation_${row}"
  export BENCHMARK_LOGS_DIR="$PWD/benchmark_logs_best20_${row}_${RUN_TS}"
  setsid bash runners/launch_mi300x-amds.sh </dev/null >"/tmp/run_best20_${row}_${RUN_TS}.log" 2>&1 &
  echo "$row pid=$! launcher_log=/tmp/run_best20_${row}_${RUN_TS}.log logs=$BENCHMARK_LOGS_DIR"
  sleep 2
}

unset PREFILL_BLOCK_SIZE DECODE_BLOCK_SIZE
export EXP_NAME="dsr1_1k1k" ISL=1024 OSL=1024 SPEC_DECODING="mtp"
export PREFILL_NUM_WORKERS=1 PREFILL_TP=8 PREFILL_EP=1 PREFILL_DP_ATTN=false PREFILL_NODES=1 PREFILL_DP8EP=false
export DECODE_NUM_WORKERS=1 DECODE_TP=8 DECODE_EP=1 DECODE_DP_ATTN=false DECODE_NODES=1 DECODE_DP8EP=false DECODE_MTP_SIZE=3
export CONC_LIST="1 6 9 30 60 117"; submit_best "1k1k_c1_c117"

export DECODE_MTP_SIZE=1
export CONC_LIST="231"; submit_best "1k1k_c231"

export PREFILL_TP=8 PREFILL_EP=8 PREFILL_DP_ATTN=true PREFILL_NODES=1 PREFILL_DP8EP=true
export DECODE_TP=8 DECODE_EP=8 DECODE_DP_ATTN=true DECODE_NODES=1 DECODE_DP8EP=true DECODE_MTP_SIZE=1
export CONC_LIST="462 615 1229"; submit_best "1k1k_c462_c1229"

unset PREFILL_BLOCK_SIZE DECODE_BLOCK_SIZE
export EXP_NAME="dsr1_8k1k" ISL=8192 OSL=1024 SPEC_DECODING="mtp"
export PREFILL_NUM_WORKERS=1 PREFILL_TP=8 PREFILL_EP=1 PREFILL_DP_ATTN=false PREFILL_NODES=1 PREFILL_DP8EP=false
export DECODE_NUM_WORKERS=1 DECODE_TP=8 DECODE_EP=1 DECODE_DP_ATTN=false DECODE_NODES=1 DECODE_DP8EP=false DECODE_MTP_SIZE=4
export CONC_LIST="1"; submit_best "8k1k_c1"

export DECODE_NUM_WORKERS=2 DECODE_NODES=2 DECODE_MTP_SIZE=4
export CONC_LIST="2"; submit_best "8k1k_c2"

export DECODE_NUM_WORKERS=3 DECODE_NODES=3 DECODE_MTP_SIZE=3
export CONC_LIST="6 9 16 24 30"; submit_best "8k1k_c6_c30"

export DECODE_NUM_WORKERS=1 DECODE_NODES=1 DECODE_MTP_SIZE=3
export CONC_LIST="77"; submit_best "8k1k_c77"

export PREFILL_NUM_WORKERS=2 PREFILL_NODES=2
export DECODE_NUM_WORKERS=2 DECODE_NODES=2 DECODE_MTP_SIZE=3
export CONC_LIST="154"; submit_best "8k1k_c154"

export SPEC_DECODING="none"
export PREFILL_NUM_WORKERS=2 PREFILL_TP=8 PREFILL_EP=1 PREFILL_DP_ATTN=false PREFILL_NODES=2 PREFILL_DP8EP=false PREFILL_BLOCK_SIZE=1
export DECODE_NUM_WORKERS=1 DECODE_TP=8 DECODE_EP=8 DECODE_DP_ATTN=false DECODE_NODES=1 DECODE_DP8EP=true DECODE_BLOCK_SIZE=1 DECODE_MTP_SIZE=0
export CONC_LIST="256"; submit_best "8k1k_c256"

echo ALL_SUBMITTED
