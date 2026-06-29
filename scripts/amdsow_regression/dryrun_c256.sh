set -e
cd /mnt/shared_mango/amdsow-deliveries/regress/amdsow-inferencex-pr
export SLURM_CONF=/run/slurm/conf/slurm.conf
export EXP_NAME="dsr1_8k1k" ISL=8192 OSL=1024 CONC_LIST="256" SPEC_DECODING="none"
export PREFILL_NUM_WORKERS=2 PREFILL_TP=8 PREFILL_EP=1 PREFILL_DP_ATTN=false PREFILL_NODES=2 PREFILL_DP8EP=false PREFILL_BLOCK_SIZE=1
export DECODE_NUM_WORKERS=1 DECODE_TP=8 DECODE_EP=8 DECODE_DP_ATTN=false DECODE_NODES=1 DECODE_DP8EP=true DECODE_BLOCK_SIZE=1 DECODE_MTP_SIZE=0
export MODEL="deepseek-ai/DeepSeek-R1-0528" MODEL_PREFIX="dsr1" PRECISION=fp8 FRAMEWORK="vllm-disagg"
export IMAGE="docker.io/chaeminlimmb/vllm-mori-pd:milestone4-aiterwheel" RANDOM_RANGE_RATIO=0.8
export IS_MULTINODE=true KEEP_LOGS=1
export RUN_EVAL=true EVAL_ONLY=false EVAL_SERVER_MAX_MODEL_LEN=20480 EVAL_SERVER_BLOCK_SIZE=1
unset EVAL_CONC
export RUNNER_NAME="amdsow-dryrun-c256" RUNNER_TYPE="mi300x-disagg"
export GITHUB_WORKSPACE="$PWD" BENCHMARK_LOGS_DIR="$PWD/benchmark_logs"
export AMDSOW_SLURM_EXCLUDE_NODES="${AMDSOW_SLURM_EXCLUDE_NODES:-a05u43,a04u43}"
export RESULT_FILENAME="validation_8k1k_c256"
echo "=== SLURM_CONF in env: $SLURM_CONF ==="
SUBMIT_DRY_RUN=1 bash runners/launch_mi300x-amds.sh
echo DONE_DRYRUN
