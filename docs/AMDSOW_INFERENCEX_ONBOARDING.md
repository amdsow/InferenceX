# AMDSOW InferenceX Onboarding: MI300X Best Config

This document is for someone seeing this project for the first time. It explains enough background to understand what is being run, then gives the exact runbook for the current best AMDSOW MI300X config, and only then shows implementation details.

## How to use this document

| If you want to... | Read |
|---|---|
| Understand the system from zero | Sections 1-2 |
| Decide which row to run | Section 3 |
| Run the current best config through GitHub Actions | Section 4 |
| Run it manually on the cluster | Section 5 |
| Check what success/failure should look like | Sections 6-7 |
| Understand the implementation chain | Sections 8-9 |
| Change the benchmark config safely | Sections 10-11 |

---

# 1. Big picture in plain English

InferenceX is an automated benchmark system. A benchmark starts as a YAML row, becomes a GitHub Actions job, is picked up by a self-hosted runner, then uses Slurm to reserve GPU nodes and run Docker containers with vLLM servers and a benchmark client.

The current AMDSOW benchmark we care about is:

```text
Model:       DeepSeek-R1-0528
Hardware:    AMD MI300X
Framework:   vLLM PD-disaggregation
Precision:   FP8
Workload:    8k input / 1k output
Concurrency: 512
Topology:    2 prefill TP8 nodes + 1 decode DP8EP node
MTP:         3
```

Source of truth:

```text
.github/configs/amd-master.yaml
  key: dsr1-fp8-mi300x-vllm-disagg
```

Optional local wrappers such as `run_dsr1_mi300x_validation.sh` are not source of truth. They only pre-fill the same environment variables that GitHub Actions normally exports, and they may not exist in another checkout.

## 1.1 Tiny analogy

Think of the run like a restaurant:

```text
YAML config            = the order ticket
GitHub Actions         = the dispatcher
self-hosted runner     = the local manager who receives the ticket
Slurm                  = the reservation desk for kitchen stations
Docker                 = a prepared kitchen with fixed tools/software
prefill servers        = cooks doing the large prep work
decode servers         = cooks plating the output
bench.sh               = the customer sending orders and timing responses
```

## 1.2 The whole path, simplified

```text
YAML row
  -> GitHub Actions matrix
  -> self-hosted GitHub runner
  -> MI300X launcher script
  -> Slurm allocation
  -> Docker containers
  -> vLLM prefill/decode servers
  -> benchmark client
  -> JSON results
```

Keep that shape in mind. The rest of the document just explains each hop.

---

# 2. Five background concepts you need first

## 2.1 Slurm

Slurm is the cluster scheduler. It decides when GPU compute nodes are available and starts jobs on them.

In this repo:

| Slurm word | Meaning here |
|---|---|
| Job | One scheduled benchmark run. For the best row, it reserves 3 MI300X compute nodes and runs `job.slurm`. |
| Allocation | The nodes Slurm gives to the job. |
| Partition | The Slurm queue/pool. The MI300X launcher uses `compute`. |
| `sbatch` | Command used by `submit.sh` to submit a job. |
| `squeue` | Shows queued/running jobs. |
| `sinfo` | Shows node states, including idle nodes. |
| `NODELIST` | Optional comma-separated list of exact compute nodes to request. Leave it unset to let Slurm choose. |

Basic Slurm commands:

```bash
squeue -u amd                         # show jobs owned by user amd
squeue -j <jobid>                     # show one job
sinfo -p compute -N -o '%N %t'         # show compute-node state
sinfo -p compute -N -o '%N %t' | grep idle
scontrol show job <jobid>             # detailed job metadata and nodelist
scancel <jobid>                       # cancel a job
```

Typical job lifecycle:

```text
PENDING     Slurm accepted the request but is waiting for nodes.
RUNNING     Slurm allocated nodes and is executing job.slurm.
COMPLETING  The job is cleaning up.
COMPLETED   The script exited successfully.
FAILED      The script or one of its child commands failed.
CANCELLED   Someone or the launcher cancelled the allocation after results were collected.
```

Checked-in MI300X cluster facts used by this repo path:

```text
GitHub runner label:  mi300x-disagg
Actual runner names:  mi300x-amds_06, mi300x-amds_07, mi300x-amds_08
Launcher script:      runners/launch_mi300x-amds.sh
Slurm partition:      compute
GPUs per node:        8
Default model root:   /models/models
RDMA defaults:        Broadcom Thor device defaults from launch_mi300x-amds.sh
Log root:             BENCHMARK_LOGS_DIR
Best-row node count:  3 Slurm compute nodes
Best-row roles:       2 prefill TP8 nodes + 1 decode DP8EP node
```

Actual Slurm compute-node names are discovered at run time with `sinfo`/`squeue`. The repo does not hard-code which compute nodes you should use.

## 2.2 GitHub Actions and self-hosted runners

GitHub Actions runs workflow files under `.github/workflows/`.

A self-hosted runner is our own process that receives a GitHub Actions job. It is not the GPU compute node itself. It is the machine/process that starts the launcher, and the launcher then asks Slurm for GPU compute nodes.

For this config:

```yaml
runner: mi300x-disagg
```

That is a GitHub `runs-on` label/type. GitHub chooses one actual runner with that label, such as:

```text
mi300x-amds_06
```

The actual runner name chooses the launcher script:

```text
mi300x-amds_06 -> runners/launch_mi300x-amds.sh
```

## 2.3 InferenceX YAML to matrix workflow

InferenceX stores benchmark intent in YAML: model, Docker image, hardware runner label, sequence length, concurrency, and topology.

`utils/matrix_logic/generate_sweep_configs.py` turns the YAML into a GitHub Actions matrix. A matrix is just a list of concrete jobs.

For this guide, this YAML row:

```text
8k1k + concurrency 512
```

becomes a matrix job with values such as:

```text
ISL=8192
OSL=1024
CONC_LIST=512
PREFILL_NODES=2
DECODE_NODES=1
DECODE_DP8EP=true
DECODE_MTP_SIZE=3
```

## 2.4 Docker containers

Docker provides the fixed software environment. The host cluster provides GPUs, model files, network devices, and logs. Docker runs the vLLM server image.

Slurm starts `job.slurm` on compute nodes. `job.slurm` starts Docker containers and explicitly passes selected environment variables into them with `-e`.

Important: `--export=ALL` gets env vars into `job.slurm`, but not every env var automatically reaches Docker. If a new variable must be used inside Docker, `job.slurm` must pass it with `-e`.

## 2.5 PD prefill/decode

PD means prefill/decode disaggregation.

| Role | Plain meaning |
|---|---|
| Prefill | Handles the input prompt/context tokens. For `8k1k`, this is the heavy input side. |
| Decode | Generates output tokens. For high concurrency, decode can use a different profile. |

The best row uses:

```text
2 prefill TP8 nodes
1 decode DP8EP node
```

Acronym key:

| Acronym | Plain meaning |
|---|---|
| FP8 | 8-bit floating-point format; lower precision than 16-bit for speed/memory savings. |
| ISL / OSL | Input / Output Sequence Length. `8k1k` means 8192 input tokens and 1024 output tokens. |
| `c512` | Concurrency 512: 512 requests in flight. |
| TP8 | Tensor Parallelism across 8 GPUs. One model is split across the 8 GPUs of a node. |
| EP / DP8EP | Expert Parallelism for MoE models. `DP8EP` is this config's wide-EP decode profile. |
| MTP / MTP3 | Multi-Token Prediction / speculative decoding. `MTP3` predicts 3 speculative tokens per step. |
| `2P1D` | 2 prefill roles and 1 decode role. |

---

# 3. Which config row should I run?

The config key is the whole benchmark family:

```text
dsr1-fp8-mi300x-vllm-disagg
```

Inside it are many rows. Use this decision table:

| Goal | Row | Use this when |
|---|---|---|
| Default validation / current best row | `8k1k`, `conc=512`, mixed TP8-prefill + DP8EP-decode, MTP3 | You want the main AMDSOW row this guide is about. |
| Cheap smoke test | `8k1k`, `conc=6`, TP8 1P3D, MTP3 | You only want to check that the path launches. |
| Full config sweep | all rows in `dsr1-fp8-mi300x-vllm-disagg` | You want every checked-in concurrency/topology point. |

Current matrix:

| ISL/OSL | Concurrency | Topology | Slurm nodes | MTP | Notes |
|---|---:|---|---:|---:|---|
| 1k1k | 6, 9, 30, 60, 117 | TP8 1P1D | 2 | 3 | Low/mid concurrency TP8 path. |
| 1k1k | 231 | TP8 1P1D | 2 | 1 | TP8 with smaller MTP depth. |
| 1k1k | 462, 615, 1229 | DP8EP 1P1D | 2 | 1 | Wide-EP/DP8EP profile for high concurrency. |
| 8k1k | 6, 9, 16, 24 | TP8 1P3D | 4 | 3 | Prefill-heavy smoke/low-concurrency path. |
| 8k1k | 30 | TP8 1P2D | 3 | 3 | Current checked-in row is 1 prefill + 2 decode nodes. |
| 8k1k | 77 | TP8 2P1D | 3 | 2 | More prefill, one decode. |
| 8k1k | 154 | TP8 2P2D | 4 | 3 | Balanced 2 prefill / 2 decode. |
| 8k1k | 256 | TP8-prefill / DP8EP-decode 2P1D | 3 | 0 | Mixed profile, no MTP. |
| 8k1k | 512 | TP8-prefill / DP8EP-decode 2P1D | 3 | 3 | Default validation / current best row. |

Best-row YAML shape:

```yaml
- spec-decoding: "mtp"
  conc-list: [ 512 ]
  prefill:
    num-worker: 2
    tp: 8
    ep: 1
    dp-attn: false
    additional-settings:
    - "PREFILL_NODES=2"
  decode:
    num-worker: 1
    tp: 8
    ep: 8
    dp-attn: false
    additional-settings:
    - "DECODE_NODES=1"
    - "DECODE_DP8EP=true"
    - "DECODE_MTP_SIZE=3"
```

---

# 4. Runbook: GitHub Actions path

Use this path first. It is safer than manual cluster invocation.

## 4.1 Prerequisites

Before pasting commands:

- Run from the repo root.
- Use the branch/SHA you want to test as `<branch-or-sha-under-test>`.
- Make sure `gh` is installed and authenticated with access to the repo.
- Know that the dispatch command launches real benchmark work.
- Do not use manual `NODELIST` unless you are on the cluster login node and have checked idle nodes.

Check `gh` auth:

```bash
gh auth status
```

## 4.2 Validate the generated matrix locally

This does not launch GPUs. It only prints the matrix JSON.

```bash
.venv/bin/python utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files .github/configs/amd-master.yaml \
  --config-keys dsr1-fp8-mi300x-vllm-disagg \
  --seq-lens 8k1k \
  --conc 512 \
  --no-evals
```

Expected shape:

```text
runner: mi300x-disagg
framework: vllm-disagg
isl/osl: 8192/1024
conc: [512]
prefill: num-worker=2, tp=8, ep=1, PREFILL_NODES=2
decode:  num-worker=1, tp=8, ep=8, DECODE_NODES=1, DECODE_DP8EP=true, DECODE_MTP_SIZE=3
```

## 4.3 Dispatch the benchmark

```bash
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' \
  -f 'inputs[ref]=<branch-or-sha-under-test>' \
  -f 'inputs[test-name]=DSR1 MI300X vLLM disagg 8k1k c512' \
  -f 'inputs[generate-cli-command]=test-config --config-files .github/configs/amd-master.yaml --config-keys dsr1-fp8-mi300x-vllm-disagg --seq-lens 8k1k --conc 512 --no-evals' \
  -f 'inputs[duration-override]='
```

Important distinction:

```text
Workflow input: test-config --config-files ...
Local terminal: .venv/bin/python utils/matrix_logic/generate_sweep_configs.py test-config --config-files ...
```

## 4.4 Monitor the run

```bash
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX \
  --workflow e2e-tests.yml \
  --event workflow_dispatch \
  --limit 1 \
  --json databaseId \
  --jq '.[0].databaseId')

gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --exit-status
```

If it fails:

```bash
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --log-failed
```

## 4.5 Other workflow input fragments

These are fragments for `inputs[generate-cli-command]`, not standalone shell commands.

```bash
# Full config, throughput only
test-config --config-files .github/configs/amd-master.yaml \
  --config-keys dsr1-fp8-mi300x-vllm-disagg \
  --no-evals

# Cheap smoke row
test-config --config-files .github/configs/amd-master.yaml \
  --config-keys dsr1-fp8-mi300x-vllm-disagg \
  --seq-lens 8k1k \
  --conc 6 \
  --no-evals
```

To run either locally for matrix inspection, prepend:

```bash
.venv/bin/python utils/matrix_logic/generate_sweep_configs.py
```

---

# 5. Advanced runbook: manual cluster invocation

Use this only if you are on the cluster login node and need to reproduce the workflow by hand.

## 5.1 Safety checklist

Before running manually:

1. Re-open `.github/configs/amd-master.yaml` and confirm the row.
2. Check active jobs:

   ```bash
   squeue -u amd
   ```

3. Check idle compute nodes:

   ```bash
   sinfo -p compute -N -o '%N %t' | grep idle
   ```

4. Either omit `NODELIST` or choose exactly 3 currently idle nodes for c512.
5. Do not paste old example node names.

## 5.2 Manual c512 command

```bash
cd /mnt/shared_mango/amdsow-deliveries/regress/amdsow-inferencex-pr

export EXP_NAME="dsr1_8k1k" ISL=8192 OSL=1024 CONC_LIST="512" SPEC_DECODING="mtp" RESULT_FILENAME="validation_mixed_8k_c512"
export PREFILL_NUM_WORKERS=2 PREFILL_TP=8 PREFILL_EP=1 PREFILL_DP_ATTN=false PREFILL_NODES=2 PREFILL_DP8EP=false
export DECODE_NUM_WORKERS=1 DECODE_TP=8 DECODE_EP=8 DECODE_DP_ATTN=false DECODE_NODES=1 DECODE_DP8EP=true DECODE_MTP_SIZE=3
# Optional: pin exactly 3 currently idle nodes discovered with sinfo/squeue; omit NODELIST to let Slurm choose.
# export NODELIST="<idle-node-1>,<idle-node-2>,<idle-node-3>"
export GITHUB_WORKSPACE="$PWD" BENCHMARK_LOGS_DIR="$PWD/benchmark_logs"
export RUNNER_NAME="mi300x-amds_06" IS_MULTINODE=true KEEP_LOGS=1 PRECISION=fp8 FRAMEWORK="vllm-disagg"
export MODEL="deepseek-ai/DeepSeek-R1-0528" IMAGE="docker.io/chaeminlimmb/vllm-mori-pd:milestone4-aiterwheel" RANDOM_RANGE_RATIO=0.8

setsid bash runners/launch_mi300x-amds.sh </dev/null >/tmp/run_mixed_8k_c512.log 2>&1 &
tail -f /tmp/run_mixed_8k_c512.log
```

## 5.3 Manual smoke-test variant

Use this only to check that the path launches. It is not the c512 best row.

```text
CONC_LIST=6
PREFILL_NUM_WORKERS=1
PREFILL_NODES=1
DECODE_NUM_WORKERS=3
DECODE_NODES=3
DECODE_DP8EP=false
DECODE_MTP_SIZE=3
optional 4-node NODELIST if manually pinning nodes
```

---

# 6. What should happen during a successful run?

Expected high-level events:

1. A GitHub Actions job starts on a runner with label `mi300x-disagg`.
2. The actual runner name is something like `mi300x-amds_06`.
3. The workflow calls `runners/launch_mi300x-amds.sh`.
4. The launcher submits a Slurm job.
5. Slurm allocates 3 MI300X compute nodes for c512.
6. `job.slurm` starts Docker containers.
7. `server_vllm.sh` starts 2 prefill roles and 1 decode role.
8. `bench.sh` sends traffic at concurrency 512.
9. Result files named like `concurrency_512_..._ctx_..._gen_....json` are copied back for workflow processing.

---

# 7. Troubleshooting by symptom

## GitHub job does not start

- Check that the matrix has `runner: mi300x-disagg`.
- Check that GitHub has self-hosted runners with the `mi300x-disagg` label.
- Inside the job, check `runner.name`; it should look like `mi300x-amds_06`.
- The launcher path comes from `runner.name`, not from the config label.

## Slurm allocation fails

- Check current jobs: `squeue -u amd`.
- Check idle nodes: `sinfo -p compute -N -o '%N %t' | grep idle`.
- If using `NODELIST`, make sure it has exactly 3 nodes for c512.
- Check `SLURM_EXCLUDE_NODES` if Slurm rejects nodes.

## vLLM flags look wrong

- Check `benchmarks/multi_node/amd_utils/models_vllm.yaml`, entry `DeepSeek-R1-0528`.
- For the c512 row, prefill should be TP8 and decode should be DP8EP.
- If `DECODE_DP8EP=true`, decode uses the DP8EP profile.
- If `DECODE_MTP_SIZE=3`, MTP config is added to both prefill and decode.

## No result JSON appears

- The workflow expects `${RESULT_FILENAME}_*.json`.
- The launcher copies latest Slurm result files back to the workspace.
- `bench.sh` result files look like `concurrency_*_gpus_*_ctx_*_gen_*.json`.

---

# 8. How it works under the hood

You can skip this section unless you are debugging or changing scripts.

## 8.1 Full automatic path

```text
.github/configs/amd-master.yaml
  -> utils/matrix_logic/generate_sweep_configs.py
  -> .github/workflows/e2e-tests.yml
  -> .github/workflows/benchmark-multinode-tmpl.yml
  -> GitHub self-hosted runner
  -> runners/launch_mi300x-amds.sh
  -> benchmarks/multi_node/dsr1_fp8_mi300x_vllm-disagg.sh
  -> benchmarks/multi_node/amd_utils/submit.sh
  -> benchmarks/multi_node/amd_utils/job.slurm
  -> Docker container
  -> benchmarks/multi_node/amd_utils/server.sh
  -> benchmarks/multi_node/amd_utils/server_vllm.sh
  -> benchmarks/multi_node/amd_utils/bench.sh
```

## 8.2 Matrix to env

The workflow exports values like:

```text
EXP_NAME=dsr1_8k1k
IMAGE=docker.io/chaeminlimmb/vllm-mori-pd:milestone4-aiterwheel
MODEL=deepseek-ai/DeepSeek-R1-0528
FRAMEWORK=vllm-disagg
ISL=8192
OSL=1024
CONC_LIST=512
PREFILL_NUM_WORKERS=2
PREFILL_TP=8
DECODE_NUM_WORKERS=1
DECODE_TP=8
```

It also exports `additional-settings` from YAML:

```text
PREFILL_NODES=2
DECODE_NODES=1
DECODE_DP8EP=true
DECODE_MTP_SIZE=3
```

## 8.3 Launcher

`runners/launch_mi300x-amds.sh` injects cluster defaults:

```text
SLURM_ACCOUNT=$USER
SLURM_PARTITION=compute
MODEL_PATH=/models/models
GPUS_PER_NODE=8
Broadcom Thor RDMA defaults
```

It derives the recipe script:

```text
EXP_NAME=dsr1_8k1k
PRECISION=fp8
FRAMEWORK=vllm-disagg
script=benchmarks/multi_node/dsr1_fp8_mi300x_vllm-disagg.sh
```

## 8.4 Recipe driver

`benchmarks/multi_node/dsr1_fp8_mi300x_vllm-disagg.sh` validates the env vars and calls `submit.sh`.

It forwards:

```text
DECODE_MTP_SIZE        -> MTP config later in server_vllm.sh
PREFILL_DP8EP/DECODE_DP8EP -> choose DP8EP profile later
```

## 8.5 Slurm submitter

`submit.sh` computes:

```bash
NUM_NODES=$((PREFILL_NODES + DECODE_NODES))
xP=$PREFILL_NUM_WORKERS
yD=$DECODE_NUM_WORKERS
PREFILL_TP_SIZE=$((PREFILL_NODES * PREFILL_TP / PREFILL_NUM_WORKERS))
DECODE_TP_SIZE=$((DECODE_NODES * DECODE_TP / DECODE_NUM_WORKERS))
```

Then submits:

```bash
sbatch --export=ALL --exclusive -N "$NUM_NODES" -n "$NUM_NODES" ... job.slurm
```

Important caveat: `submit.sh` allocates `PREFILL_NODES + DECODE_NODES`, but `job.slurm` later selects server-role nodes from `xP + yD`. Current AMDSOW rows keep `*_NODES == *_NUM_WORKERS`, so this is safe. If someone tries a worker that spans multiple nodes, validate or update `job.slurm` first.

## 8.6 Slurm job and Docker

`job.slurm` runs inside the Slurm allocation.

It:

- chooses `models_vllm.yaml` for `ENGINE=vllm-disagg`,
- resolves model path,
- selects node/IP list,
- starts the external `vllm-router` on rank 0,
- starts Docker on selected nodes,
- passes only explicitly listed env vars into Docker with `-e`.

## 8.7 vLLM server roles

Inside Docker:

```text
ENGINE=vllm-disagg -> server_vllm.sh
```

For the c512 row:

```text
xP=2, yD=1
rank 0 -> prefill + proxy
rank 1 -> additional prefill
rank 2 -> decode
```

`server_vllm.sh` loads `DeepSeek-R1-0528` from `models_vllm.yaml`.

- Prefill uses the TP8 profile.
- Decode uses the DP8EP profile because `DECODE_DP8EP=true`.
- MTP3 is added to both prefill and decode because `DECODE_MTP_SIZE=3`.

---

# 9. File map reference

| File | Role |
|---|---|
| `.github/configs/amd-master.yaml` | Source of truth for benchmark rows. |
| `.github/configs/runners.yaml` | Maps runner labels/types to possible self-hosted runner names. |
| `utils/matrix_logic/generate_sweep_configs.py` | Converts YAML configs to matrix JSON. |
| `.github/workflows/e2e-tests.yml` | Main workflow. Runs the generator and dispatches matrix jobs. |
| `.github/workflows/benchmark-multinode-tmpl.yml` | Multi-node workflow. Converts matrix fields to env vars and calls the launcher. |
| `runners/launch_mi300x-amds.sh` | AMDSOW MI300X launcher. Adds cluster defaults and starts the recipe script. |
| `benchmarks/multi_node/dsr1_fp8_mi300x_vllm-disagg.sh` | DSR1 vLLM-disagg recipe driver. Converts env vars into `submit.sh` args. |
| `benchmarks/multi_node/amd_utils/submit.sh` | Builds and runs the Slurm `sbatch` command. |
| `benchmarks/multi_node/amd_utils/job.slurm` | Slurm job body. Starts Docker/router/server processes. |
| `benchmarks/multi_node/amd_utils/server.sh` | Docker-side engine dispatcher. |
| `benchmarks/multi_node/amd_utils/server_vllm.sh` | vLLM prefill/decode server launcher. |
| `benchmarks/multi_node/amd_utils/models_vllm.yaml` | vLLM model-specific flags/env profiles. |
| `benchmarks/multi_node/amd_utils/bench.sh` | Benchmark client and result writer. |
| `run_dsr1_mi300x_validation.sh` | Optional local wrapper if present; not source of truth. |

---

# 10. If you change something, edit the right file

| Change | File |
|---|---|
| Concurrency, ISL/OSL, prefill/decode worker count, node count, TP/EP, MTP size | `.github/configs/amd-master.yaml` |
| vLLM launch flags, AITER/MoRI/vLLM env, TP8 vs DP8EP profiles | `benchmarks/multi_node/amd_utils/models_vllm.yaml` |
| Cluster model path, NIC list, Slurm account/partition defaults | `runners/launch_mi300x-amds.sh` |
| Slurm submission mechanics, `NODELIST`, `sbatch` flags | `benchmarks/multi_node/amd_utils/submit.sh` |
| Docker env propagation, node/IP selection, router/container orchestration | `benchmarks/multi_node/amd_utils/job.slurm` |
| Engine dispatch inside Docker | `benchmarks/multi_node/amd_utils/server.sh` |
| Prefill/decode vLLM command construction | `benchmarks/multi_node/amd_utils/server_vllm.sh` |
| Benchmark request behavior and result file names | `benchmarks/multi_node/amd_utils/bench.sh` |
| Manual validation shortcut | Optional local `run_dsr1_mi300x_validation.sh`; always keep YAML as source of truth. |

---

# 11. Golden rules

1. Start from `.github/configs/amd-master.yaml`; wrappers are shortcuts only.
2. For the current best row, use `8k1k c512`, 2 prefill TP8 nodes, 1 decode DP8EP node, MTP3.
3. Do not mix GitHub runner names and Slurm node names.
4. Do not mix Slurm node count and worker count. Current rows keep them equal per role; changes need `job.slurm` validation.
5. `additional-settings` reach Slurm automatically, but they reach Docker only if `job.slurm` passes them with `-e`.
6. Local generator commands need the Python prefix; workflow inputs use only the `test-config ...` fragment.
