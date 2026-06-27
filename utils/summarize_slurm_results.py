#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
# --- How to run ---
# python utils/summarize_slurm_results.py benchmark_logs_best20_*20260627T173*/

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence


BEST20_ORDER: Final[tuple[str, ...]] = tuple(
    "1k1k_c1 1k1k_c6 1k1k_c9 1k1k_c30 1k1k_c60 1k1k_c117 1k1k_c231 "
    "1k1k_c462 1k1k_c615 1k1k_c1229 8k1k_c1 8k1k_c2 8k1k_c6 8k1k_c9 "
    "8k1k_c16 8k1k_c24 8k1k_c30 8k1k_c77 8k1k_c154 8k1k_c256".split()
)

EXPORT_RE: Final[re.Pattern[str]] = re.compile(
    r"export_file: .*vllm-disagg_isl_(?P<isl>\d+)_osl_(?P<osl>\d+)/"
    r"concurrency_(?P<conc>\d+)_req_rate_[^_]+_gpus_(?P<gpus>\d+)"
    r"_ctx_(?P<context_gpus>\d+)_gen_(?P<decode_gpus>\d+)"
)

NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"[-+]?[0-9]*\.?[0-9]+")


@dataclass(frozen=True, slots=True)
class CliError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class EvalScores:
    strict: float | None
    flex: float | None


@dataclass(frozen=True, slots=True)
class RunIdentity:
    isl: int
    osl: int
    concurrency: int
    total_gpus: int
    decode_gpus: int


@dataclass(frozen=True, slots=True)
class RunMetrics:
    total_token_throughput: float
    output_token_throughput: float
    mean_tpot_ms: float
    mean_e2el_ms: float


@dataclass(frozen=True, slots=True)
class RunRecord:
    identity: RunIdentity
    metrics: RunMetrics
    eval_scores: EvalScores
    log_path: Path

    @property
    def row_name(self) -> str:
        prefix = sequence_prefix(self.identity.isl, self.identity.osl)
        return f"{prefix}_c{self.identity.concurrency}"


@dataclass(frozen=True, slots=True)
class TableRow:
    name: str
    state: str
    total: float
    gen: float
    interactivity: float
    e2e: float
    cost: float
    strict: float | None
    flex: float | None


@dataclass(frozen=True, slots=True)
class CliArgs:
    paths: tuple[str, ...]
    cost_base: float


def sequence_prefix(isl: int, osl: int) -> str:
    if isl == 1024 and osl == 1024:
        return "1k1k"
    if isl == 8192 and osl == 1024:
        return "8k1k"
    return f"{isl}x{osl}"


def parse_args(argv: Sequence[str]) -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Summarize AMDSOW MI300X Slurm benchmark logs as a best-config table.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Slurm log directories or slurm_job-*.out files.",
    )
    parser.add_argument(
        "--cost-base",
        type=float,
        default=390.0,
        help="Numerator for Cost column. Default: 390.0.",
    )
    namespace = parser.parse_args(argv)
    return CliArgs(paths=tuple(namespace.paths), cost_base=namespace.cost_base)


def discover_logs(raw_paths: Sequence[str]) -> list[Path]:
    logs: set[Path] = set()
    missing: list[str] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if not path.exists():
            missing.append(raw_path)
            continue
        if path.is_dir():
            logs.update(path.rglob("slurm_job-*.out"))
        elif path.is_file():
            logs.add(path)
    if missing:
        raise CliError(f"Missing input path(s): {', '.join(missing)}")
    if not logs:
        raise CliError("No slurm_job-*.out files found in the provided path(s).")
    return sorted(logs)


def cell_before_pm(line: str) -> float | None:
    cells = [cell.strip() for cell in line.split("|")]
    for index, cell in enumerate(cells):
        if cell == "±" and index > 0:
            return float(cells[index - 1]) * 100.0
    return None


def parse_eval_scores(lines: Sequence[str]) -> EvalScores:
    strict: float | None = None
    flex: float | None = None
    for line in lines:
        if "|gsm8k|" in line and "flexible-extract" in line:
            flex = cell_before_pm(line)
        if "|     |" in line and "strict-match" in line:
            strict = cell_before_pm(line)
    return EvalScores(strict=strict, flex=flex)


def number_after_colon(line: str) -> float:
    value = line.split(":", 1)[1]
    match = NUMBER_RE.search(value)
    if match is None:
        raise CliError(f"Could not parse numeric metric from line: {line}")
    return float(match.group(0))


def identity_from_export(line: str) -> RunIdentity | None:
    match = EXPORT_RE.search(line)
    if match is None:
        return None
    return RunIdentity(
        isl=int(match.group("isl")),
        osl=int(match.group("osl")),
        concurrency=int(match.group("conc")),
        total_gpus=int(match.group("gpus")),
        decode_gpus=int(match.group("decode_gpus")),
    )


def parse_log(path: Path) -> list[RunRecord]:
    lines = path.read_text(errors="replace").splitlines()
    eval_scores = parse_eval_scores(lines)
    records: list[RunRecord] = []
    current_identity: RunIdentity | None = None
    total_token_throughput: float | None = None
    output_token_throughput: float | None = None
    mean_tpot_ms: float | None = None
    mean_e2el_ms: float | None = None
    for line in lines:
        identity = identity_from_export(line)
        if identity is not None:
            current_identity = identity
            total_token_throughput = None
            output_token_throughput = None
            mean_tpot_ms = None
            mean_e2el_ms = None
            continue
        if current_identity is None:
            continue
        if "Output token throughput (tok/s):" in line:
            output_token_throughput = number_after_colon(line)
        elif "Total Token throughput (tok/s):" in line:
            total_token_throughput = number_after_colon(line)
        elif "Mean TPOT (ms):" in line:
            mean_tpot_ms = number_after_colon(line)
        elif "Mean E2EL (ms):" in line:
            mean_e2el_ms = number_after_colon(line)
        elif "==================================================" in line:
            if (
                total_token_throughput is not None
                and output_token_throughput is not None
                and mean_tpot_ms is not None
                and mean_e2el_ms is not None
            ):
                records.append(
                    RunRecord(
                        identity=current_identity,
                        metrics=RunMetrics(
                            total_token_throughput=total_token_throughput,
                            output_token_throughput=output_token_throughput,
                            mean_tpot_ms=mean_tpot_ms,
                            mean_e2el_ms=mean_e2el_ms,
                        ),
                        eval_scores=eval_scores,
                        log_path=path,
                    )
                )
            current_identity = None
            total_token_throughput = None
            output_token_throughput = None
            mean_tpot_ms = None
            mean_e2el_ms = None
    return records


def make_table_row(record: RunRecord, cost_base: float) -> TableRow:
    identity = record.identity
    metrics = record.metrics
    total = metrics.total_token_throughput / identity.total_gpus
    gen = metrics.output_token_throughput / identity.decode_gpus
    return TableRow(
        name=record.row_name,
        state="DONE",
        total=total,
        gen=gen,
        interactivity=1000.0 / metrics.mean_tpot_ms,
        e2e=metrics.mean_e2el_ms / 1000.0,
        cost=cost_base / total,
        strict=record.eval_scores.strict,
        flex=record.eval_scores.flex,
    )


def select_latest(records: Sequence[RunRecord]) -> list[RunRecord]:
    selected: dict[str, RunRecord] = {}
    for record in records:
        previous = selected.get(record.row_name)
        if previous is None or record.log_path.stat().st_mtime >= previous.log_path.stat().st_mtime:
            selected[record.row_name] = record
    return list(selected.values())


def sort_rows(rows: Sequence[TableRow]) -> list[TableRow]:
    order = {name: index for index, name in enumerate(BEST20_ORDER)}
    return sorted(rows, key=lambda row: (order.get(row.name, len(order)), row.name))


def percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}%"


def render_table(rows: Sequence[TableRow]) -> str:
    output = [
        "Row          State        Total    Gen   Interactivity   E2E    Cost   GSM8K strict   GSM8K flex"
    ]
    for row in rows:
        output.append(
            f"{row.name:<12} {row.state:<10} {row.total:8.1f} {row.gen:7.1f}"
            f" {row.interactivity:12.1f} {row.e2e:7.1f}s {row.cost:7.2f}"
            f" {percent(row.strict):>11} {percent(row.flex):>12}"
        )
    return "\n".join(output)


def run(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    logs = discover_logs(args.paths)
    records = [record for path in logs for record in parse_log(path)]
    if not records:
        raise CliError("No completed benchmark result blocks found in the provided logs.")
    rows = [make_table_row(record, args.cost_base) for record in select_latest(records)]
    print(render_table(sort_rows(rows)))
    return 0


def main() -> int:  # noqa: BROAD_EXCEPT_OK
    try:
        return run(sys.argv[1:])
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
