import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parent / "summarize_slurm_results.py"


def test_cli_prints_best_config_table_from_slurm_log(tmp_path: Path) -> None:
    log_dir = tmp_path / "benchmark_logs_best20_1k1k_c462_c1229_20260627T173426Z"
    log_dir.mkdir()
    log_path = log_dir / "slurm_job-284.out"
    log_path.write_text(
        "\n".join(
            [
                "export_file: /run_logs/slurm_job-284/vllm-disagg_isl_1024_osl_1024/concurrency_462_req_rate_inf_gpus_16_ctx_8_gen_8",
                "============ Serving Benchmark Result ============",
                "Output token throughput (tok/s):         8859.82",
                "Total Token throughput (tok/s):          17729.90",
                "Mean TPOT (ms):                          47.46",
                "Mean E2EL (ms):                          46301.77",
                "==================================================",
                "|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9553|±  |0.0057|",
                "|     |       |strict-match    |     5|exact_match|↑  |0.9583|±  |0.0055|",
            ]
        )
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(log_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Row          State" in result.stdout
    assert "1k1k_c462" in result.stdout
    assert "DONE" in result.stdout
    assert "1108.1" in result.stdout
    assert "1107.5" in result.stdout
    assert "95.83%" in result.stdout
    assert "95.53%" in result.stdout
