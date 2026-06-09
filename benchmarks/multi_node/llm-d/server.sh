#!/usr/bin/env bash
#
# Per-node entrypoint for the llm-d-vllm wide-EP P/D disagg benchmark.
# NODE_RANK is set by srun (= $SLURM_PROCID) in job.slurm.
#
# Roles:
#   Rank 0                         -> prefill leader (DP rank 0)
#   Ranks 1 .. PREFILL_NODES-1     -> prefill workers
#   Rank PREFILL_NODES             -> decode leader (DP rank 0) + pd-sidecar
#                                     + EPP + Envoy + benchmark client
#                                     (the coordinator, like AMD's decode-0)
#   Ranks PREFILL_NODES+1 ..       -> decode workers
#
# Each "instance" (prefill or decode) is a single vLLM engine spanning
# PREFILL_NODES (or DECODE_NODES) nodes via --data-parallel-hybrid-lb. The
# leader pod accepts external traffic; workers handle their local DP ranks.

set -euo pipefail

source /workspace/benchmarks/benchmark_lib.sh

NODE_RANK="${NODE_RANK:-${SLURM_PROCID:-0}}"
PREFILL_NODES="${PREFILL_NODES:-1}"
DECODE_NODES="${DECODE_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
VLLM_PORT=8200
SIDECAR_PORT=8000
ENVOY_PORT=8080
EPP_GRPC_PORT=9002
EPP_HEALTH_PORT=9003
EPP_METRICS_PORT=9090

# Filesystem path to the weights inside the container. job.slurm mounts
# the host model directory at /models and sets MODEL_DIR=/models, so the
# weights live directly under MODEL_DIR. MODEL_NAME is the OpenAI-API
# served name passed via --served-model-name; it is not part of the
# filesystem path.
MODEL="${MODEL_DIR}"
# Resolve HOST_IP and DEFAULT_IFACE without relying on iproute2 (the
# `ip` binary is not present in the multi-arch arm64 vLLM base; the
# amd64 base ships it). python3 is guaranteed inside the vLLM image and
# its socket library exposes the kernel's source-IP / iface selection.
_HOST_INFO=$(python3 -c '
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("1.1.1.1", 80))
    ip = s.getsockname()[0]
finally:
    s.close()
iface = ""
try:
    with open("/proc/net/route") as f:
        f.readline()  # header
        for line in f:
            parts = line.split()
            if parts[1] == "00000000":  # default route dest
                iface = parts[0]; break
except OSError:
    pass
print(ip, iface)
' 2>/dev/null) || true
HOST_IP=$(echo "$_HOST_INFO" | awk '{print $1}')
DEFAULT_IFACE=$(echo "$_HOST_INFO" | awk '{print $2}')
DEFAULT_IFACE="${DEFAULT_IFACE:-eth0}"

VLLM_LOG="/benchmark_logs/vllm_rank${NODE_RANK}.log"
SIDECAR_LOG="/benchmark_logs/sidecar_rank${NODE_RANK}.log"
EPP_LOG="/benchmark_logs/epp.log"
ENVOY_LOG="/benchmark_logs/envoy.log"

echo "=== rank=$NODE_RANK host=$HOST_IP model=$MODEL ==="

# ----------------------------------------------------------------
# Role assignment
# ----------------------------------------------------------------
if [[ "$NODE_RANK" -lt "$PREFILL_NODES" ]]; then
    ROLE="prefill"
    DP_SIZE="$PREFILL_DP_SIZE"
    DP_ADDR="$PREFILL_DP_ADDR"
    LWS_WORKER_INDEX="$NODE_RANK"
    LWS_GROUP_SIZE="$PREFILL_NODES"
elif [[ "$NODE_RANK" -lt $((PREFILL_NODES + DECODE_NODES)) ]]; then
    ROLE="decode"
    DP_SIZE="$DECODE_DP_SIZE"
    DP_ADDR="$DECODE_DP_ADDR"
    LWS_WORKER_INDEX=$((NODE_RANK - PREFILL_NODES))
    LWS_GROUP_SIZE="$DECODE_NODES"
else
    echo "ERROR: NODE_RANK=$NODE_RANK out of range" >&2
    exit 1
fi

DP_SIZE_LOCAL="$GPUS_PER_NODE"
START_RANK=$((LWS_WORKER_INDEX * DP_SIZE_LOCAL))

# Defaults preserve the original H200 1P+1D shape: TP=1 + DP=role_total +
# expert-parallel on. Per-recipe overrides below.
TP_SIZE=1
ROLE_ENABLE_EP=true

echo "ROLE=$ROLE DP_SIZE=$DP_SIZE DP_ADDR=$DP_ADDR LWS_WORKER_INDEX=$LWS_WORKER_INDEX START_RANK=$START_RANK"

# ----------------------------------------------------------------
# Read role-specific extra-args and env from the recipe file.
#
# Recipe schema (per-role section, both prefill and decode):
#   tp:                       int  - --tensor-parallel-size override (default 1)
#   enable-expert-parallel:   bool - emit --enable-expert-parallel and the
#                                    DP/wide-EP knobs (default true)
#   extra-args:               str  - free-form vLLM CLI flags appended at the end
#   env:                      map  - role-only env vars exported before vllm serve
#
# A pure tensor-parallel decode (e.g. DSV4-Pro on GB200: TP=8, no DP, no EP)
# sets tp:8 and enable-expert-parallel:false. The original gpt-oss recipe
# omits both keys, so the H200 path is byte-identical.
# ----------------------------------------------------------------
ROLE_EXTRA_ARGS=""
if [[ -n "${CONFIG_FILE:-}" ]]; then
    RECIPE_PATH="/etc/llmd-recipes/${CONFIG_FILE}"
    if [[ -f "$RECIPE_PATH" ]]; then
        echo "Loading $ROLE recipe from $RECIPE_PATH"
        eval "$(python3 - <<PY
import yaml
recipe = yaml.safe_load(open('${RECIPE_PATH}'))
section = recipe.get('${ROLE}', {}) or {}
extra = (section.get('extra-args') or '').strip()
print(f'ROLE_EXTRA_ARGS={extra!r}')
tp = section.get('tp')
if tp is not None:
    print(f'TP_SIZE={int(tp)}')
ep = section.get('enable-expert-parallel')
if ep is not None:
    print(f'ROLE_ENABLE_EP={"true" if ep else "false"}')
for k, v in (section.get('env') or {}).items():
    print(f'export {k}={v!r}')
PY
)"
    else
        echo "WARNING: CONFIG_FILE=$CONFIG_FILE but $RECIPE_PATH not found; using defaults" >&2
    fi
fi
echo "Resolved $ROLE TP_SIZE=$TP_SIZE ROLE_ENABLE_EP=$ROLE_ENABLE_EP"

# ----------------------------------------------------------------
# Multi-node DP / NIXL P/D env: needed in any topology.
# ----------------------------------------------------------------
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$DEFAULT_IFACE}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$DEFAULT_IFACE}
export VLLM_SKIP_P2P_CHECK=1
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
export VLLM_USE_DEEP_GEMM=1
# DeepGEMM JIT-compiles CUDA kernels at warmup and links against
# libcuda.so.1. In ghcr.io/llm-d/llm-d-cuda the lib lives under
# /usr/local/cuda/compat/, which is in LD_LIBRARY_PATH (runtime) but
# NOT in LIBRARY_PATH (link time). Prepend it so ld can resolve
# -l:libcuda.so.1. The toolkit-injection fallback path is
# arch-specific (x86_64-linux-gnu on amd64, aarch64-linux-gnu on
# Grace/GB200), so resolve it from `uname -m` rather than hardcoding.
case "$(uname -m)" in
    aarch64|arm64) _NCT_LIB=/usr/lib/aarch64-linux-gnu ;;
    *)             _NCT_LIB=/usr/lib/x86_64-linux-gnu ;;
esac
export LIBRARY_PATH=/usr/local/cuda/compat:${_NCT_LIB}:${LIBRARY_PATH:-}
export VLLM_NIXL_SIDE_CHANNEL_HOST="$HOST_IP"
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

# Force NIXL/UCX onto IB verbs (Reliable-Connected) and turn on UCX +
# NCCL transport-selection logging so vllm_rank*.log records the chosen
# wire transport per endpoint. cuda_copy/cuda_ipc cover intra-node H2D
# and peer-GPU paths; rc covers cross-node KV via the IB HCAs that
# job.slurm exposes with --device /dev/infiniband + IPC_LOCK. Mirrors
# the dynamo minimax recipes (UCX_TLS=cuda_copy,rc).
export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,rc}
export UCX_LOG_LEVEL=${UCX_LOG_LEVEL:-info}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}

# ----------------------------------------------------------------
# Wide-EP NVSHMEM / ibgda env (from the llm-d wide-EP-lws guide
# manifests). Gated on LWS_GROUP_SIZE > 1 - the simple 1P+1D recipe
# explicitly avoids DeepEP, NVSHMEM ibgda, and full-mesh RDMA, so
# leaving these set on a single-node-per-role topology is misleading
# and could trigger ibgda code paths it does not need.
# ----------------------------------------------------------------
if [[ "$LWS_GROUP_SIZE" -gt 1 ]]; then
    export NVIDIA_GDRCOPY=enabled
    export NVSHMEM_REMOTE_TRANSPORT=ibgda
    export NVSHMEM_IB_ENABLE_IBGDA=true
    export NVSHMEM_SYMMETRIC_SIZE=16G
    export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-$DEFAULT_IFACE}
fi

# ----------------------------------------------------------------
# Start vLLM (every node, prefill or decode)
#
# Flags split into:
#   * COMMON_ARGS - always passed.
#   * MULTINODE_DP_ARGS - only when an instance spans more than one node
#     (LWS_GROUP_SIZE > 1, i.e. wide-EP topology). vLLM's
#     --data-parallel-hybrid-lb and the cross-process DP coordination
#     flags are wrong for the single-node-per-instance case where DP is
#     contained inside one engine process.
# ----------------------------------------------------------------
KV_TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'

COMMON_ARGS=(
    --port "$VLLM_PORT"
    --served-model-name "$MODEL_NAME"
    --trust-remote-code
    --disable-access-log-for-endpoints=/health,/metrics
    --tensor-parallel-size "$TP_SIZE"
    --kv_transfer_config "$KV_TRANSFER_CONFIG"
)
# --api-server-count is incompatible with --headless (vllm errors out
# because no API server runs in headless mode). The headless branch
# below is the only one that drops this; everywhere else keeps the
# original count=1 behavior.
if [[ "$ROLE_ENABLE_EP" == "true" ]] || [[ "$LWS_GROUP_SIZE" -le 1 ]] || [[ "$LWS_WORKER_INDEX" -eq 0 ]]; then
    COMMON_ARGS+=(--api-server-count 1)
fi
# --moe-backend is model-specific (DSR1-FP8 wants deep_gemm, gpt-oss-MXFP4
# rejects it - see vllm/.../oracle/mxfp4.py:163), so each recipe sets its
# own value via prefill/decode extra-args instead of inheriting one here.

# Expert-parallel + data-parallel knobs only apply when the recipe asks for
# DP-attention/EP. Pure tensor-parallel roles (e.g. DSV4-Pro decode TP=8)
# leave DP at vLLM's default 1 and skip --enable-expert-parallel; emitting
# --data-parallel-size with TP>1 conflicts with --tensor-parallel-size and
# vLLM rejects the combination.
if [[ "$ROLE_ENABLE_EP" == "true" ]]; then
    COMMON_ARGS+=(
        --enable-expert-parallel
        --data-parallel-size "$DP_SIZE"
    )
    if [[ "$LWS_GROUP_SIZE" -gt 1 ]]; then
        COMMON_ARGS+=(
            --data-parallel-hybrid-lb
            --data-parallel-size-local "$DP_SIZE_LOCAL"
            --data-parallel-address "$DP_ADDR"
            --data-parallel-rpc-port 5555
            --data-parallel-start-rank "$START_RANK"
        )
    fi
elif [[ "$LWS_GROUP_SIZE" -gt 1 ]]; then
    # Pure tensor-parallel that spans more than one node (e.g. DSV4-Pro
    # decode TP=8 on GB200's 4-GPU nodes). vLLM rejects TP > GPUs/node
    # without explicit cross-node coordination. We use vLLM's native
    # headless multi-node API - the same mechanism dynamo's vllm
    # launcher uses (NVIDIA/srt-slurm src/srtctl/backends/vllm.py):
    # leader rank-0 binds on --master-addr, followers join headless
    # with matching --nnodes/--node-rank. PyTorch distributed handles
    # the NCCL rendezvous.
    COMMON_ARGS+=(
        --master-addr "$DP_ADDR"
        --nnodes "$LWS_GROUP_SIZE"
        --node-rank "$LWS_WORKER_INDEX"
    )
    if [[ "$LWS_WORKER_INDEX" -gt 0 ]]; then
        COMMON_ARGS+=(--headless)
    fi
fi

echo "Starting vLLM ($ROLE) DP=$DP_SIZE local=$DP_SIZE_LOCAL start_rank=$START_RANK group_size=$LWS_GROUP_SIZE"
# shellcheck disable=SC2086
vllm serve "$MODEL" "${COMMON_ARGS[@]}" $ROLE_EXTRA_ARGS \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Every rank waits for its own engine to bind /health before falling
# through. For wide-EP (LWS_GROUP_SIZE > 1) this prevents the bench
# from starting before the worker-side DP shards have come up; for the
# single-node case it is a no-op extra check.
wait_for_server_ready --port "$VLLM_PORT" --server-log "$VLLM_LOG" --server-pid "$VLLM_PID"
echo "vLLM ready on rank $NODE_RANK ($ROLE worker_index=$LWS_WORKER_INDEX)"

# Only the leader of each instance accepts external requests on $VLLM_PORT.
if [[ "$LWS_WORKER_INDEX" -eq 0 ]]; then
    # ------------------------------------------------------------
    # Start pd-sidecar on each leader (prefill leader and decode leader).
    # The decode-side sidecar is what EPP routes to; the prefill-side
    # sidecar is the target the decode sidecar pulls KVs from.
    # ------------------------------------------------------------
    SIDECAR_CONNECTOR="nixlv2"
    SIDECAR_FLAGS=(--port="$SIDECAR_PORT" --vllm-port="$VLLM_PORT"
                   --kv-connector="$SIDECAR_CONNECTOR" --secure-proxy=false)
    if [[ "$ROLE" == "decode" ]]; then
        SIDECAR_FLAGS+=(--enable-prefiller-sampling)
    fi
    echo "Starting pd-sidecar ($ROLE leader): ${SIDECAR_FLAGS[*]}"
    pd-sidecar "${SIDECAR_FLAGS[@]}" > "$SIDECAR_LOG" 2>&1 &
    SIDECAR_PID=$!
    wait_for_server_ready --port "$SIDECAR_PORT" --server-log "$SIDECAR_LOG" --server-pid "$SIDECAR_PID"
    echo "pd-sidecar ready on $HOST_IP:$SIDECAR_PORT"
fi

# ----------------------------------------------------------------
# Coordinator: decode leader runs EPP + Envoy + benchmark client.
# ----------------------------------------------------------------
if [[ "$ROLE" == "decode" && "$LWS_WORKER_INDEX" -eq 0 ]]; then

    # Write endpoints.yaml. See benchmarks/multi_node/llm-d/README.md for
    # the discovery contract.
    # NOTE: endpoint 'namespace' must match EPP's --pool-namespace below
    # (file-discovery filters endpoints by namespace; the schema default
    # 'default' would otherwise drop every entry).
    python3 - <<PY
import os, yaml
NS = 'inferencex'
endpoints = [
    {'name': 'prefill-0',
     'namespace': NS,
     'address': os.environ['PREFILL_LEADER_IP'],
     'port': '$SIDECAR_PORT',
     'labels': {'llm-d.ai/role': 'prefill'}},
    {'name': 'decode-0',
     'namespace': NS,
     'address': os.environ['DECODE_LEADER_IP'],
     'port': '$SIDECAR_PORT',
     'labels': {'llm-d.ai/role': 'decode'}},
]
yaml.safe_dump({'endpoints': endpoints}, open('/tmp/endpoints.yaml', 'w'))
print('endpoints.yaml:')
print(open('/tmp/endpoints.yaml').read())
PY

    # EPP config: recipe override, else the default mounted by job.slurm
    # at /etc/epp/config.yaml (sourced from benchmarks/llm-d/epp-config.yaml).
    #
    # The recipe yaml is a single-file mix: EPP scheduler keys
    # (apiVersion, kind, plugins, schedulingProfiles, dataLayer) plus
    # per-role vLLM extra-args (prefill, decode) plus slurm hints. EPP's
    # strict YAML decoder rejects the latter ("unknown field \"prefill\""),
    # so when a recipe is in play we project it down to just the EPP keys
    # and hand EPP that.
    if [[ -n "$CONFIG_FILE" && -f "/etc/llmd-recipes/$CONFIG_FILE" ]]; then
        EPP_CONFIG="/tmp/epp-config-from-recipe.yaml"
        python3 - <<PY
import yaml
recipe = yaml.safe_load(open('/etc/llmd-recipes/${CONFIG_FILE}'))
# Keys EPP's strict decoder accepts. Anything else (prefill, decode,
# slurm, ...) is dropped before passing to --config-file.
keep = {'apiVersion', 'kind', 'plugins', 'schedulingProfiles', 'dataLayer'}
yaml.safe_dump({k: v for k, v in recipe.items() if k in keep},
               open('${EPP_CONFIG}', 'w'))
PY
    else
        EPP_CONFIG="/etc/epp/config.yaml"
    fi
    echo "EPP config: $EPP_CONFIG"

    # --secure-serving=false: EPP defaults to TLS gRPC; Envoy's `epp`
    # cluster in benchmarks/llm-d/envoy.yaml is plaintext HTTP/2, so
    # without this flag every ext_proc dial fails the TLS handshake,
    # the ext_proc filter trips, and Envoy returns 500 to the bench
    # client (the local-llmd-run smoke uses --secure-serving=false for
    # the same reason).
    epp \
        --pool-name=epp \
        --pool-namespace=inferencex \
        --config-file="$EPP_CONFIG" \
        --grpc-port="$EPP_GRPC_PORT" \
        --grpc-health-port="$EPP_HEALTH_PORT" \
        --metrics-port="$EPP_METRICS_PORT" \
        --secure-serving=false \
        --v=4 \
        > "$EPP_LOG" 2>&1 &
    EPP_PID=$!

    # Wait for EPP to bind its gRPC port before starting Envoy. Envoy's
    # ext_proc filter dials 127.0.0.1:$EPP_GRPC_PORT - if Envoy comes up
    # first the early bench requests hit ext_proc connection errors.
    # gRPC has no plain HTTP /health, so probe the TCP listener directly.
    echo "Waiting for EPP on 127.0.0.1:$EPP_GRPC_PORT"
    EPP_WAIT_DEADLINE=$(( $(date +%s) + 60 ))
    until (echo > "/dev/tcp/127.0.0.1/$EPP_GRPC_PORT") 2>/dev/null; do
        if ! kill -0 "$EPP_PID" 2>/dev/null; then
            echo "ERROR: EPP died before binding $EPP_GRPC_PORT" >&2
            exit 1
        fi
        if [[ "$(date +%s)" -ge "$EPP_WAIT_DEADLINE" ]]; then
            echo "ERROR: EPP did not bind $EPP_GRPC_PORT within 60s" >&2
            exit 1
        fi
        sleep 1
    done
    echo "EPP listening on $EPP_GRPC_PORT"

    envoy -c /etc/envoy/envoy.yaml > "$ENVOY_LOG" 2>&1 &
    ENVOY_PID=$!

    # Probe Envoy's admin /ready (port 9901) instead of /health on :8080.
    # /health on :8080 routes through ext_proc -> EPP -> ORIGINAL_DST, which
    # only resolves once a request has the right model/profile metadata for
    # EPP to set x-gateway-destination-endpoint. Health-style requests
    # without that metadata get 503 and the wait loop spins forever.
    echo "Waiting for envoy admin on 127.0.0.1:9901/ready"
    ENVOY_WAIT_DEADLINE=$(( $(date +%s) + 120 ))
    until [[ "$(curl --output /dev/null --silent --write-out '%{http_code}' \
                "http://127.0.0.1:9901/ready" 2>/dev/null)" == "200" ]]; do
        if ! kill -0 "$ENVOY_PID" 2>/dev/null; then
            echo "ERROR: envoy died before admin /ready returned 200" >&2
            tail -n 80 "$ENVOY_LOG" >&2 || true
            exit 1
        fi
        if [[ "$(date +%s)" -ge "$ENVOY_WAIT_DEADLINE" ]]; then
            echo "ERROR: envoy admin /ready did not return 200 within 120s" >&2
            tail -n 80 "$ENVOY_LOG" >&2 || true
            exit 1
        fi
        sleep 2
    done
    echo "Envoy admin ready; listener should be on $ENVOY_PORT"

    # Wait for the prefill leader's sidecar before starting the bench.
    # wait_for_server_ready can only probe localhost; the prefill leader
    # is on a different node, so poll directly with a deadline.
    echo "Waiting for prefill sidecar at $PREFILL_LEADER_IP:$SIDECAR_PORT/health"
    PREFILL_WAIT_DEADLINE=$(( $(date +%s) + 300 ))
    until curl --output /dev/null --silent --fail \
            "http://$PREFILL_LEADER_IP:$SIDECAR_PORT/health"; do
        if [[ "$(date +%s)" -ge "$PREFILL_WAIT_DEADLINE" ]]; then
            echo "ERROR: prefill sidecar did not become ready within 5 min" >&2
            exit 1
        fi
        sleep 5
    done
    echo "Prefill sidecar at $PREFILL_LEADER_IP:$SIDECAR_PORT is ready"

    # Sweep concurrency. BENCH_MAX_CONCURRENCY arrives from submit.sh as
    # an 'x'-delimited list (e.g. "2048x1024x512"); the runner / sweep
    # configs expect one bench run per level. Same shape as
    # benchmarks/multi_node/amd_utils/bench.sh.
    IFS='x' read -r -a CONCURRENCIES <<< "$BENCH_MAX_CONCURRENCY"
    for max_concurrency in "${CONCURRENCIES[@]}"; do
        num_prompts=$(( max_concurrency * BENCH_NUM_PROMPTS_MULTIPLIER ))
        [[ "$num_prompts" -lt 16 ]] && num_prompts=16
        # Bench against Envoy. EPP routes to decode (and decode sidecar
        # pulls from prefill via NIXL).
        # --bench-serving-dir resolves to the in-container repo bind-mount
        # (job.slurm bind-mounts $DI_REPO_DIR onto /workspace). Without it
        # run_benchmark_serving falls back to $(pwd), which on this image
        # is /home/vllm and does not contain utils/bench_serving/.
        #
        # --tokenizer points at /models (the in-container bind-mount of
        # MODEL_DIR). bench_serving.py loads a tokenizer locally for prompt
        # tokenization; without --tokenizer it derives one from --model,
        # but --model here is the *served-model-name* ("gpt-oss-120b",
        # "DeepSeek-R1-0528"), which is not a valid HF repo id - it would
        # try to fetch from huggingface.co and 401.
        # DSV4-Pro needs trust-remote-code so HF AutoTokenizer reads the
        # checkpoint's bundled configuration_deepseek_v4.py (the older
        # transformers wheel in vllm/vllm-openai:v0.20.0-ubuntu2404 does
        # not register `deepseek_v4` natively), plus --use-chat-template
        # and --dsv4 to match the prompt formatting the dynamo-vllm sa-bench
        # path uses for the same workload.
        bench_extra_args=()
        if [[ "${MODEL_NAME,,}" == *"deepseek-v4"* ]]; then
            # --tokenizer-mode deepseek_v4 routes _load_tokenizer in
            # benchmark_serving.py through vLLM's get_tokenizer wrapper
            # (which has DSV4-aware code), bypassing stock HF AutoTokenizer
            # whose transformers wheel on the v0.20.0 base does not register
            # the deepseek_v4 model type.
            bench_extra_args+=(
                --trust-remote-code
                --tokenizer-mode deepseek_v4
                --use-chat-template
                --dsv4
            )
        fi

        run_benchmark_serving \
            --bench-serving-dir /workspace \
            --tokenizer /models \
            --model "$MODEL_NAME" \
            --port "$ENVOY_PORT" \
            --backend openai \
            --input-len "$BENCH_INPUT_LEN" \
            --output-len "$BENCH_OUTPUT_LEN" \
            --random-range-ratio "$BENCH_RANDOM_RANGE_RATIO" \
            --num-prompts "$num_prompts" \
            --max-concurrency "$max_concurrency" \
            --result-filename "${RESULT_FILENAME}_c${max_concurrency}" \
            --result-dir "$BENCHMARK_LOGS_DIR/" \
            "${bench_extra_args[@]}"
    done

    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        run_eval --framework lm-eval --port "$ENVOY_PORT"
        append_lm_eval_summary
    fi

    # Signal job.slurm (running outside the container, where SLURM
    # client tools are available) to scancel the allocation. The image
    # does not bundle scancel, so calling it here would just trip
    # set -e. Workers end server.sh in `wait`; without this signal
    # they would hold the job until TIME_LIMIT.
    touch "$BENCHMARK_LOGS_DIR/.bench_done.$SLURM_JOB_ID"
else
    # Workers (prefill workers, decode workers, prefill leader): just keep vLLM alive.
    wait
fi
