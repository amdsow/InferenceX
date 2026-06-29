export SLURM_CONF=/run/slurm/conf/slurm.conf
cd /mnt/shared_mango/amdsow-deliveries/regress/amdsow-inferencex-pr
RUN_TS=20260629T032703Z
while true; do
  n=$(squeue -h -o '%j' | grep -c "best20-.*-${RUN_TS}")
  ts=$(date -u +%H:%M:%S)
  echo "[$ts] jobs_remaining=$n"
  if [ "$n" -eq 0 ]; then
    echo "ALL_JOBS_DONE"
    break
  fi
  sleep 60
done
echo "=== SUMMARY ==="
python3 utils/summarize_slurm_results.py benchmark_logs_best20_*_"${RUN_TS}"/ 2>&1
echo "WATCH_COMPLETE"
