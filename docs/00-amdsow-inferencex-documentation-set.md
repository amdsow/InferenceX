<div class="cover">
  <img class="cover-logo" src="mangoboost-logo.png" alt="MangoBoost logo">
  <div class="cover-org">MangoBoost</div>
  <div class="cover-title">AMDSOW InferenceX documentation set</div>
  <div class="cover-title2">Start here, choose the right guide, then run the benchmark</div>
  <div class="cover-sub">AMDSOW InferenceX DeepSeek-R1-0528 FP8 MI300X vLLM PD-disaggregation validation package</div>
  <div class="cover-conf">Confidential. Shared with AMD under NDA. &nbsp;|&nbsp; contact@mangoboost.io</div>
</div>
<div class="pagebreak"></div>

**Document control**

| Field | Value |
|---|---|
| Document | AMDSOW InferenceX documentation set |
| Version | v1.0.0 |
| Last updated | 2026-06-25 |
| Owner | MangoBoost AMDSOW delivery team |
| Contact | contact@mangoboost.io |
| Audience | AMD/customer operators, solution engineers, maintainers, and package reviewers |
| Status | Generated A4 PDF from maintained AMDSOW InferenceX Markdown documentation |
| Source format | Maintained Markdown rendered through HTML/CSS with Python-Markdown and WeasyPrint. |

<div class="pagebreak"></div>

[TOC]

# AMDSOW InferenceX onboarding

Start here if you are seeing this repository for the first time.

In plain terms: this documentation explains how to run the AMDSOW DeepSeek-R1-0528 FP8 MI300X vLLM prefill/decode benchmark through InferenceX. GitHub Actions starts the run, a self-hosted runner submits a Slurm job, Slurm allocates MI300X compute nodes, Docker starts vLLM workers, and the benchmark writes JSON results.

## Five-minute checklist

1. Read `AMDSOW_INFERENCEX_USER_GUIDE.md` sections 1-4 before dispatching anything.
2. Confirm your local shell has `.venv/bin/python` for matrix validation and `gh` for GitHub Actions dispatch.
3. Validate the default row locally: `8k1k`, concurrency `256`, config key `dsr1-fp8-mi300x-vllm-disagg`.
4. Confirm the shared `mi300x-disagg` runner and target Slurm cluster are idle using the site-approved status source. `squeue` is useful only from a Slurm-capable host and may not show every operator's jobs.
5. Use `AMDSOW_INFERENCEX_PROJECT_DESCRIPTION_DETAIL.md` before changing YAML, Slurm, Docker, or vLLM flags.

## Documents and PDFs

| Need | Markdown | PDF |
|---|---|---|
| Start here and choose the right document | `AMDSOW_INFERENCEX_ONBOARDING.md` | `00-amdsow-inferencex-documentation-set.pdf` |
| Run the benchmark, copy commands, monitor results, troubleshoot an operator run | `AMDSOW_INFERENCEX_USER_GUIDE.md` | `01-amdsow-inferencex-user-guide.pdf` |
| Understand the project wiring, config semantics, script flow, code excerpts, and safe edit rules | `AMDSOW_INFERENCEX_PROJECT_DESCRIPTION_DETAIL.md` | `02-amdsow-inferencex-project-detail.pdf` |

Use the user guide first if you are operating a benchmark. Use the project detail document only when you need internals or are changing configs/scripts.

## Source-of-truth rule

The source of truth for the AMDSOW InferenceX benchmark matrix is `.github/configs/amd-master.yaml`, key `dsr1-fp8-mi300x-vllm-disagg`. Local wrappers and validation scripts are convenience tools; do not treat them as the canonical config.
