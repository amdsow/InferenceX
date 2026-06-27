#!/usr/bin/env python3
"""Render the AMDSOW InferenceX documentation set to A4 PDFs."""
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
TODAY = dt.date.today().isoformat()
VERSION = "v1.1.0"


@dataclass(frozen=True)
class DocTarget:
    key: str
    source: str
    generated: str
    pdf: str
    title: str
    subtitle: str
    audience: str


TARGETS = [
    DocTarget(
        key="index",
        source="AMDSOW_INFERENCEX_ONBOARDING.md",
        generated="00-amdsow-inferencex-documentation-set.md",
        pdf="00-amdsow-inferencex-documentation-set.pdf",
        title="AMDSOW InferenceX documentation set",
        subtitle="Start here, choose the right guide, then run the benchmark",
        audience="AMD/customer operators, solution engineers, maintainers, and package reviewers",
    ),
    DocTarget(
        key="user",
        source="AMDSOW_INFERENCEX_USER_GUIDE.md",
        generated="01-amdsow-inferencex-user-guide.md",
        pdf="01-amdsow-inferencex-user-guide.pdf",
        title="AMDSOW InferenceX user guide",
        subtitle="Operator workflow and manual Slurm runbook",
        audience="AMD/customer operators and solution engineers",
    ),
    DocTarget(
        key="detail",
        source="AMDSOW_INFERENCEX_PROJECT_DESCRIPTION_DETAIL.md",
        generated="02-amdsow-inferencex-project-detail.md",
        pdf="02-amdsow-inferencex-project-detail.pdf",
        title="AMDSOW InferenceX project detail",
        subtitle="Config, workflow, Slurm, Docker, and vLLM wiring",
        audience="Maintainers, delivery owners, and package reviewers",
    ),
]

CSS_TEXT = r"""
@page {
  size: A4;
  margin: 14mm 13mm 15mm 13mm;
  @bottom-center {
    content: "AMDSOW InferenceX - " counter(page) " / " counter(pages);
    color: #6b7280;
    font-size: 8pt;
  }
}
html { font-family: "DejaVu Sans", "Noto Sans", Arial, sans-serif; color: #172033; }
body { font-size: 9.4pt; line-height: 1.38; }
a { color: #1e5aa8; text-decoration: none; }
.cover { min-height: 245mm; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
.cover-logo { width: 112px; margin-bottom: 18mm; }
.cover-org { color: #f97316; font-weight: 700; font-size: 18pt; letter-spacing: 0.08em; text-transform: uppercase; }
.cover-title { font-weight: 800; font-size: 29pt; margin-top: 8mm; max-width: 165mm; }
.cover-title2 { font-weight: 700; font-size: 17pt; margin-top: 3mm; color: #374151; max-width: 158mm; }
.cover-sub { margin-top: 9mm; font-size: 12pt; color: #4b5563; max-width: 150mm; }
.cover-conf { margin-top: 22mm; font-size: 9pt; color: #6b7280; }
.pagebreak { break-after: page; }
h1 { break-before: page; font-size: 20pt; margin: 0 0 7mm 0; padding-bottom: 2.5mm; border-bottom: 2px solid #f97316; color: #111827; }
h1:first-of-type { break-before: auto; }
h2 { font-size: 15pt; margin: 7mm 0 3mm 0; color: #111827; }
h3 { font-size: 11.6pt; margin: 5mm 0 2mm 0; color: #1f2937; }
h4 { font-size: 10.4pt; margin: 4mm 0 1.5mm 0; color: #374151; }
p { margin: 0 0 2.8mm 0; }
ul, ol { margin: 0 0 3mm 5mm; padding-left: 4mm; }
li { margin: 0.6mm 0; }
table { width: 100%; border-collapse: collapse; table-layout: fixed; margin: 3mm 0 4mm 0; font-size: 7.8pt; }
th, td { border: 0.35pt solid #d1d5db; padding: 1.15mm 1.35mm; vertical-align: top; overflow-wrap: anywhere; word-break: normal; }
th { background: #f3f4f6; font-weight: 700; color: #111827; }
pre { background: #0f172a; color: #e5e7eb; padding: 2.5mm; border-radius: 2mm; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 7.5pt; line-height: 1.28; margin: 3mm 0; }
code { font-family: "DejaVu Sans Mono", "Noto Sans Mono", monospace; font-size: 0.88em; background: #f3f4f6; color: #111827; padding: 0.15em 0.32em; border-radius: 0.8mm; }
pre code { background: transparent; color: inherit; padding: 0; border-radius: 0; font-size: inherit; }
blockquote { border-left: 3px solid #f97316; margin: 3mm 0; padding: 1.5mm 3mm; background: #fff7ed; color: #374151; }
.toc ul { list-style: none; margin-left: 0; padding-left: 0; }
.toc li { margin: 0.8mm 0; }
img { max-width: 100%; }
hr { border: 0; border-top: 1px solid #e5e7eb; margin: 4mm 0; }
"""


def cover(target: DocTarget) -> str:
    logo = '<img class="cover-logo" src="mangoboost-logo.png" alt="MangoBoost logo">' if (DOCS / "mangoboost-logo.png").exists() else ""
    return f"""<div class="cover">
  {logo}
  <div class="cover-org">MangoBoost</div>
  <div class="cover-title">{target.title}</div>
  <div class="cover-title2">{target.subtitle}</div>
  <div class="cover-sub">AMDSOW InferenceX DeepSeek-R1-0528 FP8 MI300X vLLM PD-disaggregation validation package</div>
  <div class="cover-conf">Confidential. Shared with AMD under NDA. &nbsp;|&nbsp; contact@mangoboost.io</div>
</div>
<div class="pagebreak"></div>

**Document control**

| Field | Value |
|---|---|
| Document | {target.title} |
| Version | {VERSION} |
| Last updated | {TODAY} |
| Owner | MangoBoost AMDSOW delivery team |
| Contact | contact@mangoboost.io |
| Audience | {target.audience} |
| Status | Generated A4 PDF from maintained AMDSOW InferenceX Markdown documentation |
| Source format | Maintained Markdown rendered through HTML/CSS with Python-Markdown and WeasyPrint. |

<div class="pagebreak"></div>

[TOC]
"""


def generated_markdown(target: DocTarget) -> str:
    source_path = DOCS / target.source
    body = source_path.read_text(encoding="utf-8").strip() + "\n"
    return cover(target) + "\n" + body


def render(target: DocTarget) -> None:
    try:
        import markdown
        from weasyprint import CSS, HTML
    except ImportError as exc:
        raise SystemExit(
            "Missing PDF render dependencies. Install with: "
            "python3 -m venv /tmp/amdsow-inferencex-pdfvenv && "
            "/tmp/amdsow-inferencex-pdfvenv/bin/pip install markdown pymdown-extensions weasyprint"
        ) from exc
    generated_path = DOCS / target.generated
    pdf_path = DOCS / target.pdf
    generated_path.write_text(generated_markdown(target), encoding="utf-8")
    html_body = markdown.markdown(
        generated_path.read_text(encoding="utf-8"),
        extensions=["tables", "attr_list", "def_list", "sane_lists", "toc", "pymdownx.superfences"],
        extension_configs={"toc": {"toc_depth": "1-3", "title": "Table of contents"}},
    )
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>{target.title}</title></head>
<body>{html_body}</body>
</html>
"""
    HTML(string=html, base_url=str(DOCS)).write_pdf(str(pdf_path), stylesheets=[CSS(string=CSS_TEXT)])
    print(f"wrote {generated_path.relative_to(ROOT)}")
    print(f"wrote {pdf_path.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", choices=["all", *(target.key for target in TARGETS)], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = TARGETS if args.doc == "all" else [target for target in TARGETS if target.key == args.doc]
    for target in selected:
        if not (DOCS / target.source).exists():
            raise SystemExit(f"missing source Markdown: docs/{target.source}")
        render(target)


if __name__ == "__main__":
    main()
