# AMDSOW InferenceX — best-config regression scripts

Helper scripts used to run the full 1k1k + 8k1k best-config matrix (all 20
datapoints) through the manual Slurm path described in Section 10 of the AMDSOW
InferenceX user guide. They wrap the repo's canonical launcher chain:

```
runners/launch_mi300x-amds.sh
  -> benchmarks/multi_node/dsr1_fp8_mi300x_vllm-disagg.sh
  -> benchmarks/multi_node/amd_utils/submit.sh   (sbatch)
  -> benchmarks/multi_node/amd_utils/job.slurm
```

| Script | Purpose |
|---|---|
| `best20_regression.sh` | Submits all 9 Slurm jobs covering the full 20-datapoint matrix (1k1k c1..c1229, 8k1k c1..c256). Prints `RUN_TS` and per-job launcher/log paths. |
| `best20_watch.sh` | Polls the Slurm queue until all jobs for a given `RUN_TS` finish, then runs `utils/summarize_slurm_results.py` to print the consolidated table. |
| `dryrun_c256.sh` | `SUBMIT_DRY_RUN=1` pre-flight for the default `8k1k c256` row; resolves the `sbatch` command and env without allocating GPUs. |
| `make_results_xlsx.py` | Generates a formatted XLSX of the summarized results (requires `openpyxl`). |

## Usage

```bash
cd /mnt/shared_mango/amdsow-deliveries/regress/amdsow-inferencex-pr

# Slurm here is configless; client tools need this to resolve the cluster:
export SLURM_CONF=/run/slurm/conf/slurm.conf

# (optional) pre-flight, no GPUs:
bash scripts/amdsow_regression/dryrun_c256.sh

# Submit all configs (run inside tmux so it survives disconnects):
bash scripts/amdsow_regression/best20_regression.sh

# Watch to completion + summarize (edit RUN_TS inside, or reuse the printed one):
bash scripts/amdsow_regression/best20_watch.sh
```

## Notes
- `best20_watch.sh` has `RUN_TS` hard-coded to the original run; update it to the
  value printed by `best20_regression.sh` for a new run.
- The 2 drained nodes are skipped via `AMDSOW_SLURM_EXCLUDE_NODES`.
- Outputs land on the shared `/mnt` NFS path: per-row `validation_*.json` at the
  repo root and Slurm `.out/.err` under `benchmark_logs_best20_<row>_<RUN_TS>/`.
