#!/usr/bin/env python3
"""Generate an XLSX of the AMDSOW InferenceX best-config regression results,
laid out like the reference MangoBoost sheet.

Source: run RUN_TS=20260629T032703Z (full 20-row best-config Slurm regression),
summarized via utils/summarize_slurm_results.py. Topology columns are derived
from the per-datapoint result JSON filenames (ctx/gen GPU counts -> nodes).
"""
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

# Row: (config_id, conc, nodes_tp_ep, prefill_nodes, decode_nodes,
#        total, generation, interactivity, e2e, mtp)
ROWS_1K1K = [
    ("#1",    1,   "2 TP8", 1, 1,   17.2,   17.3, 143.1,  6.7, 3),
    ("#2",    6,   "2 TP8", 1, 1,   67.8,   67.6,  96.1,  9.9, 3),
    ("#3",    9,   "2 TP8", 1, 1,   91.3,   91.1,  87.1, 10.9, 3),
    ("#4",   30,   "2 TP8", 1, 1,  212.9,  212.8,  60.6, 15.6, 3),
    ("#5",   60,   "2 TP8", 1, 1,  356.4,  356.8,  50.8, 18.7, 3),
    ("#6",  117,   "2 TP8", 1, 1,  582.1,  580.5,  42.8, 22.2, 3),
    ("#7",  231,   "2 TP8", 1, 1,  809.7,  810.1,  29.9, 31.8, 1),
    ("#8",  462,   "2 EP8", 1, 1, 1104.8, 1104.2,  21.1, 46.4, 1),
    ("#9",  615,   "2 EP8", 1, 1, 1238.3, 1237.9,  17.7, 55.1, 1),
    ("#10", 1229,  "2 EP8", 1, 1, 1624.5, 1623.8,  11.7, 84.8, 1),
]

# 8K1K. Note: our regression ran 8k1k c6/c9 inside the 1P3D (4-node) group,
# so their topology differs from the reference (which used 1P1D for c6/c9).
ROWS_8K1K = [
    ("#11",  2,  "3 TP8",          1, 2,  111.3,  18.7, 172.7,  6.0, 4),
    ("#12",  6,  "4 TP8",          1, 3,  166.6,  24.8, 118.9,  8.7, 3),
    ("#13",  9,  "4 TP8",          1, 3,  246.6,  36.7, 112.5,  9.1, 3),
    ("#14", 16,  "4 TP8",          1, 3,  390.3,  57.6, 104.8,  9.9, 3),
    ("#15", 24,  "4 TP8",          1, 3,  530.0,  78.5,  96.3, 11.3, 3),
    ("#16", 30,  "4 TP8",          1, 3,  582.5,  86.5,  89.4, 12.6, 3),
    ("#17", 77,  "2 TP8",          1, 1, 1324.0, 292.9,  49.5, 29.0, 3),
    ("#18", 154, "4 TP8",          2, 2, 1320.9, 292.5,  50.1, 29.0, 3),
    ("#19", 256, "3 TP8-P|EP8-D",  2, 1, 1445.6, 481.5,  16.8, 58.7, 0),
]

# Extra datapoint we measured that is not numbered in the reference sheet.
ROWS_EXTRA = [
    ("8k1k c1", 1, "2 TP8", 1, 1, 86.8, 19.2, 168.9, 6.0, 4),
]

HEADERS = [
    "Config ID", "Concurrency", "# nodes (TP/EP)", "# prefill nodes",
    "# decode nodes", "Total (tok/s/gpu)", "Generation (tok/s/gpu)",
    "Interactivity (tok/s/user)", "E2E Latency (s)", "MTP",
]

ORANGE = PatternFill("solid", fgColor="F4B183")
ORANGE_HDR = PatternFill("solid", fgColor="FFC000")
GREY = PatternFill("solid", fgColor="D9D9D9")
WHITE = Font(bold=True)
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center")

wb = Workbook()
ws = wb.active
ws.title = "best-config results"

# Group header row (row 1)
ws.merge_cells("A1:B1"); ws["A1"] = "MangoBoost Current"
ws.merge_cells("C1:E1"); ws["C1"] = "Our Config"
ws.merge_cells("F1:I1"); ws["F1"] = "Ours"
for c in ("A1", "C1", "F1", "J1"):
    ws[c].fill = ORANGE_HDR; ws[c].font = WHITE; ws[c].alignment = CENTER; ws[c].border = BORDER

# Column header row (row 2)
for j, h in enumerate(HEADERS, start=1):
    cell = ws.cell(row=2, column=j, value=h)
    cell.fill = ORANGE; cell.font = WHITE; cell.alignment = CENTER; cell.border = BORDER

r = 3

def section(title):
    global r
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=10)
    cell = ws.cell(row=r, column=1, value=title)
    cell.fill = GREY; cell.font = Font(bold=True); cell.alignment = CENTER; cell.border = BORDER
    for j in range(1, 11):
        ws.cell(row=r, column=j).border = BORDER
    r += 1

def datarow(vals):
    global r
    for j, v in enumerate(vals, start=1):
        cell = ws.cell(row=r, column=j, value=v)
        cell.border = BORDER
        cell.alignment = LEFT if j == 3 else CENTER
        if j in (6, 7, 8, 9):
            cell.number_format = "0.0"
    r += 1

section("1K1K")
for row in ROWS_1K1K:
    datarow(row)
section("8K1K")
for row in ROWS_8K1K:
    datarow(row)
section("Additional datapoint (measured this run, not in reference sheet)")
for row in ROWS_EXTRA:
    datarow(row)

widths = [11, 12, 16, 13, 13, 14, 16, 16, 14, 7]
for i, w in enumerate(widths, start=1):
    ws.column_dimensions[chr(64 + i)].width = w
ws.row_dimensions[2].height = 46
ws.freeze_panes = "A3"

out = "AMDSOW_InferenceX_best_config_results_20260629.xlsx"
wb.save(out)
print("wrote", out)
