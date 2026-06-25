# AMDSOW InferenceX user guide

This guide is for the operator who runs the AMDSOW DeepSeek-R1-0528 PD-disaggregation benchmark in InferenceX. It is self-contained. It covers prerequisites, how to pick a row, how to validate that row without GPUs, how to dispatch one row through GitHub Actions, how to watch it, how to download and read the results, and a manual cluster path for when Actions is not an option.

The normal path is GitHub Actions. Use the manual cluster path (section 9) only as a fallback.

Quick reference:

| Field | Value |
|---|---|
| Repo path in this guide | `/mnt/shared_mango/amdsow-deliveries/regress/amdsow-inferencex-pr` |
| Repo slug (`origin`) | `amdsow/InferenceX` |
| Workflow | `.github/workflows/e2e-tests.yml` |
| Config file | `.github/configs/amd-master.yaml` |
| Config key | `dsr1-fp8-mi300x-vllm-disagg` |
| Model | `deepseek-ai/DeepSeek-R1-0528` |
| Hardware | AMD MI300X, 8 GPUs per node |
| Framework | `vllm-disagg` (prefill/decode disaggregation) |
| Runner label | `mi300x-disagg` |
| Best/default first row | `8k1k`, concurrency `256` |

## 1. How one run flows

You start from one YAML row. GitHub Actions expands the row into a matrix job. A self-hosted runner with the `mi300x-disagg` label accepts the job and submits a Slurm job in the `compute` partition. Slurm allocates MI300X nodes. Docker containers start vLLM prefill and decode workers. The benchmark client sends requests and writes JSON results, which the workflow uploads as artifacts.

Four terms that look alike but are not the same:

| Term | What it is | Example |
|---|---|---|
| Runner label | The tag the workflow targets with `runs-on`. It selects which class of self-hosted runner can take the job. | `mi300x-disagg` |
| Runner name | The specific self-hosted machine that took the job. The part before its first underscore selects the launcher script (`launch_<prefix>.sh`), and the full name becomes the Slurm job name. | `mi300x-amds_06` |
| Slurm compute node | A physical MI300X node that Slurm allocates for the job, in the `compute` partition. | `a04u01` |
| Benchmark worker | A vLLM server process. Each prefill or decode worker runs on its own node or nodes. | 2 prefill workers + 1 decode worker |

PD disaggregation splits serving in two. Prefill workers read the prompt and build the KV cache. Decode workers generate output tokens.

## 2. Prerequisites

Run every command from the repo root:

```bash
cd /mnt/shared_mango/amdsow-deliveries/regress/amdsow-inferencex-pr
```

Local tools on the machine you dispatch from:

- `git`
- `python3` with a `.venv` that has `pydantic`, `pyyaml`, and `tabulate`
- `gh`, authenticated against the repo that owns the workflow

Create the venv if it is missing:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install pydantic pyyaml tabulate
```

The local row generator needs `pydantic` and `pyyaml`. The `utils/summarize.py` reader in section 8 needs `tabulate`.

GitHub and cluster requirements:

| Requirement | Why |
|---|---|
| A self-hosted runner with label `mi300x-disagg`, online and idle | The workflow targets this label, and the runner must reach the MI300X Slurm cluster. |
| Actions secret `REPO_PAT` | Used for checkout and result upload. |
| Actions secret `INFERENCEX_OFFICIAL_RO_HF_TOKEN` | Exported as `HF_TOKEN` in the workflow and Slurm shell. `job.slurm` does not pass it into the vLLM Docker env for this recipe, so staged weights are still required. |
| Slurm partition `compute` | `runners/launch_mi300x-amds.sh` submits there. |
| Model weights staged on every allocated node | This recipe does not download weights during the run. |

The job looks for the weights on the host at:

```text
/models/models/DeepSeek-R1-0528
```

That path is `MODEL_DIR/MODEL_NAME`. Override `MODEL_DIR` with `AMDSOW_MODEL_DIR` if your site stages weights elsewhere. If the weights are missing, `job.slurm` stops with `FATAL ERROR: Model 'DeepSeek-R1-0528' not found on ALL allocated nodes` and the run produces no metrics.

Quick Slurm checks if you have cluster access:

```bash
squeue -u "$USER"
sinfo -p compute -N -o '%N %t'
```

Do not hard-code node names. Discover idle nodes at run time if you need to pin them.

## 3. Choose a row

For an Actions run you choose two things: a sequence length and a concurrency. The YAML maps that pair to a fixed topology (prefill and decode workers, node counts, MTP depth). You do not set TP, EP, or node counts yourself. The row already encodes them.

The config key `dsr1-fp8-mi300x-vllm-disagg` defines these concurrencies:

| Sequence length | ISL / OSL | Valid concurrencies |
|---|---|---|
| `1k1k` | 1024 / 1024 | 1, 6, 9, 30, 60, 117, 231, 462, 615, 1229 |
| `8k1k` | 8192 / 1024 | 1, 2, 6, 9, 16, 24, 30, 77, 154, 256 |

Pass a concurrency that exists in the list for the sequence length you pick. A value outside the list matches nothing.

Start with the default row: `8k1k` at concurrency `256`. It is the mixed PD shape used for validation: 2 prefill workers (TP8 each) plus 1 wide-EP decode worker (TP8, EP8), spread over 3 nodes, with speculative decoding off (`spec-decoding: none`, `DECODE_MTP_SIZE=0`). It is the simplest high-concurrency row to reason about and the one the delivery treats as the best-config baseline.

## 4. Validate the row locally

This step uses no GPUs. It expands the YAML and confirms you selected the row you intend to run. The generator prints a JSON array with one object per matrix row.

```bash
CONFIG_JSON=$(.venv/bin/python utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files .github/configs/amd-master.yaml \
  --config-keys dsr1-fp8-mi300x-vllm-disagg \
  --seq-lens 8k1k --conc 256 --no-evals)

echo "$CONFIG_JSON" | .venv/bin/python -c '
import sys, json
rows = json.load(sys.stdin)
if not rows:
    sys.exit("EMPTY: no row matched. Check --seq-lens and --conc against the matrix in section 3.")
for r in rows:
    print("isl/osl :", r.get("isl"), "/", r.get("osl"))
    print("conc    :", r.get("conc"))
    print("spec    :", r.get("spec-decoding"))
    print("runner  :", r.get("runner"))
    print("prefill :", r.get("prefill"))
    print("decode  :", r.get("decode"))
    print()
print(len(rows), "row(s)")
'
```

For `8k1k --conc 256` you should see one row:

- `isl/osl`: 8192 / 1024
- `conc`: `[256]`
- `spec`: `none`
- `runner`: `mi300x-disagg`
- `prefill`: `num-worker 2`, `tp 8`, `ep 1`, `additional-settings ["PREFILL_NODES=2"]`
- `decode`: `num-worker 1`, `tp 8`, `ep 8`, `additional-settings ["DECODE_NODES=1", "DECODE_DP8EP=true", "DECODE_MTP_SIZE=0"]`

To see the full objects, pipe the same command to a pretty printer:

```bash
echo "$CONFIG_JSON" | .venv/bin/python -m json.tool
```

If the check prints `EMPTY`, the concurrency you passed is not in the matrix for that sequence length. Fix `--conc` or `--seq-lens` before you dispatch. An empty selection still starts a workflow run, but every sweep job is skipped, so you get no benchmark.

To run the whole `8k1k` set, drop `--conc`. To run a different concurrency, change `--conc` to one of the values in section 3.

## 5. Dispatch one row through GitHub Actions

Set the repo, the branch that holds the workflow, the code under test, and a unique name:

```bash
export REPO="amdsow/InferenceX"
export WORKFLOW_REF="main"
export REF_UNDER_TEST="<branch-or-sha-under-test>"
export TEST_NAME="DSR1 MI300X vllm-disagg 8k1k c256 $(date -u +%Y%m%dT%H%M%SZ)"
```

`WORKFLOW_REF` is the branch where `e2e-tests.yml` lives. `REF_UNDER_TEST` is the code Actions checks out to generate the configs. Leave it empty to use the workflow branch's commit.

Dispatch the default row:

```bash
gh api -X POST \
  "/repos/${REPO}/actions/workflows/e2e-tests.yml/dispatches" \
  -f "ref=${WORKFLOW_REF}" \
  -f "inputs[ref]=${REF_UNDER_TEST}" \
  -f "inputs[test-name]=${TEST_NAME}" \
  -f 'inputs[generate-cli-command]=test-config --config-files .github/configs/amd-master.yaml --config-keys dsr1-fp8-mi300x-vllm-disagg --seq-lens 8k1k --conc 256 --no-evals' \
  -f 'inputs[duration-override]='
```

The `generate-cli-command` value is the same command you validated in section 4, minus the `.venv/bin/python` prefix. Keep `--seq-lens 8k1k --conc 256` unless you want a wider run. Removing both filters runs the entire checked-in matrix for this key. `--no-evals` runs throughput only; drop it to also run the gsm8k accuracy jobs the matrix selects.

Before dispatching, check for an active run so you do not collide with another user:

```bash
squeue -u "$USER"
```

The multi-node template cancels stale Slurm jobs whose name matches the runner it lands on. Do not dispatch over another user's live run.

## 6. Find and watch the run

The dispatch API does not return a run id. Find it by the unique `TEST_NAME`. The workflow names its run `e2e Test - <test-name>`.

```bash
sleep 8

RUN_ID=$(gh run list --repo "$REPO" \
  --workflow e2e-tests.yml \
  --event workflow_dispatch \
  --limit 30 \
  --json databaseId,displayTitle,createdAt \
  --jq "map(select(.displayTitle == \"e2e Test - ${TEST_NAME}\")) | sort_by(.createdAt) | last | .databaseId")

if [ -z "$RUN_ID" ] || [ "$RUN_ID" = "null" ]; then
  echo "No matching run yet. Wait a few seconds and run this block again."
else
  echo "RUN_ID=$RUN_ID"
fi
```

Watch it to completion:

```bash
gh run watch "$RUN_ID" --repo "$REPO" --exit-status
```

If it fails, read the failed step logs:

```bash
gh run view "$RUN_ID" --repo "$REPO" --log-failed
```

## 7. Download artifacts

After the run finishes:

```bash
gh run download "$RUN_ID" --repo "$REPO" --pattern 'bmk_*' --dir artifacts
gh run download "$RUN_ID" --repo "$REPO" --pattern 'multinode_server_logs_*' --dir artifacts || true
gh run download "$RUN_ID" --repo "$REPO" -n results_bmk --dir artifacts
gh run download "$RUN_ID" --repo "$REPO" -n run-stats --dir artifacts
```

| Artifact | Contents |
|---|---|
| `bmk_<RESULT_FILENAME>` | Per-config aggregated benchmark JSON (`agg_<RESULT_FILENAME>_*.json`). |
| `multinode_server_logs_<RESULT_FILENAME>` | `multinode_server_logs.tar.gz` when the launcher produced it. `launch_mi300x-amds.sh` does not create that tarball, so this artifact is empty or absent on the MI300X path. |
| `results_bmk` | `agg_bmk.json`, aggregated across the run. |
| `run-stats` | `run_stats.json`, with job and node status and the computed success rate. |

The workflow builds `RESULT_FILENAME` from the row's parameters and ends it with the runner name. The `bmk_*` pattern should match the result artifact for the selected runner; the `multinode_server_logs_*` pattern is harmless but may download nothing on MI300X.

## 8. Read results and judge success or failure

A healthy `8k1k c256` run moves through these stages:

1. `e2e-tests.yml` starts.
2. A runner with label `mi300x-disagg` accepts the job.
3. `runners/launch_mi300x-amds.sh` submits `benchmarks/multi_node/dsr1_fp8_mi300x_vllm-disagg.sh`.
4. Slurm allocates 3 MI300X nodes in `compute`.
5. vLLM starts 2 prefill workers and 1 decode worker.
6. The client drives concurrency 256 at ISL 8192 and OSL 1024.
7. The workflow uploads the per-config JSON. Raw per-node server logs survive only if `VLLM_LOG_ARCHIVE_DIR` was set for the run.

Read the metrics from the downloaded JSON. `utils/summarize.py` prints a table from a directory of result files:

```bash
.venv/bin/python utils/summarize.py artifacts
```

The table reports TTFT (mean, p75, p90, p95), TPOT (mean, p75), interactivity, end-to-end latency, and throughput per GPU, next to the prefill and decode TP, EP, worker, and GPU counts. You can also open `agg_bmk.json` directly.

There are no checked-in AMDSOW reference thresholds for these rows. Treat the first clean run as your baseline unless the delivery owner gives you target numbers.

Failure signals and what to check:

| Symptom | Check |
|---|---|
| Dispatch returns 403 | `gh auth status`, the repo slug, and the token scope. |
| Run starts but no runner picks it up | A runner with label `mi300x-disagg` is offline or busy. |
| Job fails before Slurm allocation | Secrets, `inputs[ref]`, and the runner's access to Slurm. |
| Slurm job stays pending | `squeue -u "$USER"` and `sinfo -p compute -N -o '%N %t'`. |
| `FATAL ERROR: Model ... not found` | Stage `DeepSeek-R1-0528` under the model root on every allocated node. |
| No benchmark artifact | `gh run view "$RUN_ID" --repo "$REPO" --log-failed`, then inspect Slurm output. Raw vLLM server logs require `VLLM_LOG_ARCHIVE_DIR` on MI300X. |
| Metrics look empty | Confirm the local validate step produced the intended row and that `--no-evals` was what you wanted. |

Per-concurrency result files inside the archives use this pattern:

```text
concurrency_<c>_req_rate_<r>_gpus_<total_gpus>_ctx_<prefill_gpus>_gen_<decode_gpus>.json
```

`<total_gpus>` is prefill plus decode GPUs, `<ctx>` is the prefill GPU count, and `<gen>` is the decode GPU count.

## 9. Manual cluster path (advanced fallback)

Use this only when you are on a cluster host that can submit Slurm jobs and cannot use GitHub Actions. It calls the same launcher the workflow calls.

```bash
cd /mnt/shared_mango/amdsow-deliveries/regress/amdsow-inferencex-pr
squeue -u "$USER"
sinfo -p compute -N -o '%N %t'
```

Launch the default `8k1k c256` row:

```bash
export EXP_NAME="dsr1_8k1k" ISL=8192 OSL=1024 CONC_LIST="256" SPEC_DECODING="none"
export PREFILL_NUM_WORKERS=2 PREFILL_TP=8 PREFILL_EP=1 PREFILL_DP_ATTN=false PREFILL_NODES=2 PREFILL_DP8EP=false
export DECODE_NUM_WORKERS=1 DECODE_TP=8 DECODE_EP=8 DECODE_DP_ATTN=false DECODE_NODES=1 DECODE_DP8EP=true DECODE_MTP_SIZE=0

export MODEL="deepseek-ai/DeepSeek-R1-0528" MODEL_PREFIX="dsr1" PRECISION=fp8 FRAMEWORK="vllm-disagg"
export IMAGE="docker.io/chaeminlimmb/vllm-mori-pd:milestone4-aiterwheel" RANDOM_RANGE_RATIO=0.8
export IS_MULTINODE=true KEEP_LOGS=1
export RUNNER_NAME="amdsow-dsr1-c256" RUNNER_TYPE="mi300x-disagg"
export GITHUB_WORKSPACE="$PWD" BENCHMARK_LOGS_DIR="$PWD/benchmark_logs"

# Optional. Preserve raw per-node vLLM logs on shared storage.
# export VLLM_LOG_ARCHIVE_DIR="$PWD/vllm_logs_8k1k_c256_$(date -u +%Y%m%dT%H%M%SZ)"
export RESULT_FILENAME="validation_8k1k_c256"

# Optional. Pin exactly 3 idle nodes you found with sinfo. Leave unset to let Slurm choose.
# export NODELIST="<node-a>,<node-b>,<node-c>"
# Optional. Skip known-bad nodes.
# export AMDSOW_SLURM_EXCLUDE_NODES="<node-x>,<node-y>"

setsid bash runners/launch_mi300x-amds.sh </dev/null >/tmp/run_8k1k_c256.log 2>&1 &
tail -f /tmp/run_8k1k_c256.log
```

Notes:

- `EXP_NAME` must start with `dsr1_`. The launcher takes the part before the first underscore to build the recipe name, so `dsr1_8k1k` resolves to `dsr1_fp8_mi300x_vllm-disagg.sh`.
- `RUNNER_NAME` becomes the Slurm job name (`submit.sh` passes it to `sbatch --job-name`). Use a unique value so you can find and cancel your job with `squeue` and `scancel --name`.
- If you pin `NODELIST`, the host count must equal the total nodes (`PREFILL_NODES + DECODE_NODES`), which is 3 for this row.
- Results land at the repo root as `<RESULT_FILENAME>_*.json`. `KEEP_LOGS=1` keeps `BENCHMARK_LOGS_DIR` instead of wiping it when the launcher exits.
