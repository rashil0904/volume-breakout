#!/usr/bin/env python3
"""
run_all_analysis.py — Full analysis pipeline
=============================================
Runs all four analysis scripts in sequence, overwriting existing result files.

Order:
  1. prepare_data.py   → results/diagnostic_table.csv  (LB=30, VM=7 baseline)
  2. run_backtest.py   → results/trades_*.csv + results/summary.csv  (8 combos)
  3. parameter_sweep.py→ results/full_sweep_results.xlsx + .parquet   (1,536 runs)
  4. monthly_report.py → results/monthly_performance_report.xlsx       (8 sheets)
"""

import time
import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import warnings
warnings.filterwarnings("ignore")


def section(title):
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n")


# ── 1. Rebuild diagnostic table (LB=30, VM=7 — feeds run_backtest.py) ─────────

section("STEP 1 / 4  —  prepare_data.py  (LB=30, VM=7)")
t0 = time.time()

from prepare_data import build_diagnostic_table

diag = build_diagnostic_table(
    vol_window=30,
    vol_mult=7.0,
    output_path=BASE / "results" / "diagnostic_table.csv",
    save_csv=True,
    verbose=True,
)
print(f"\n  Done in {time.time() - t0:.1f}s — {len(diag):,} rows")


# ── 2. Run all 8 entry/exit combos → trade CSVs ───────────────────────────────

section("STEP 2 / 4  —  run_backtest.py  (8 trade CSVs + summary)")
t0 = time.time()

from run_backtest import run_all_combos

print("  Running all combinations …")
summary, trades_dict = run_all_combos(diag, verbose=True)

print("\n" + "=" * 72)
print(f"  {'#':>2}  {'Entry / Exit Combo':<40}  {'Trades':>7}  {'Total%':>9}  {'Win%':>7}  {'Avg%':>8}")
print("      " + "─" * 68)
for i, row in summary.iterrows():
    print(f"  {i+1:>2}  {row['combo']:<40}  {int(row['total_trades']):>7,}  "
          f"{row['total_return_pct']:>+9.2f}  {row['win_rate_pct']:>6.2f}%  "
          f"{row['avg_ret_per_trade_pct']:>+8.4f}")
print(f"\n  Done in {time.time() - t0:.1f}s")


# ── 3. Full 1,536-run parameter sweep ─────────────────────────────────────────

section("STEP 3 / 4  —  parameter_sweep.py  (1,536 runs)")

import parameter_sweep
parameter_sweep.main()


# ── 4. Monthly performance report (8-sheet Excel) ─────────────────────────────

section("STEP 4 / 4  —  monthly_report.py  (8-sheet Excel)")

import monthly_report
monthly_report.main()


# ── Done ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("  ALL ANALYSIS COMPLETE — results/ updated:")
print(f"    diagnostic_table.csv")
print(f"    summary.csv")
print(f"    trades_*.csv  (8 files)")
print(f"    full_sweep_results.xlsx  +  .parquet")
print(f"    monthly_performance_report.xlsx")
print("=" * 72)
