# -*- coding: utf-8 -*-
"""
final_performance_report_4L.py
==============================
Identical to final_performance_report.py in EVERY respect except the capital base:
  pool 500000 -> 400000, per-trade allocation 100000 -> 80000 (proportional, /5).
Non-compounded sizing uses the SAME per-day rule scaled to ₹4L (₹80k cap, ₹4L/n
when >5 signals) so the file is directly comparable to the ₹5L version.
Output: final_performance_report_4L.xlsx (same 7 sheets, same columns).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import exit_time_sweep as ets
import final_performance_report as fpr

POOL_4L, ALLOC_4L, MAXPS_4L = 400_000, 100_000, 100_000       # 4L pool, 1L max/trade (unchanged)


def load_base_4L():
    """Base positions re-sized with a ₹4L daily pool (₹80k cap) — same run_standard
    _day_target rule as the ₹5L 1_Standard sheet, just scaled proportionally."""
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "passes_all_three",
                                "entry_price_315pm", "exit_3pm_open"], parse_dates=["date"])
    v = diag[diag["passes_all_three"]].dropna(
        subset=["entry_price_315pm", "exit_3pm_open"]).reset_index(drop=True)
    entry = v["entry_price_315pm"].values.astype(float)
    n_per_date = v.groupby("date")["entry_price_315pm"].transform("count").values
    thr = POOL_4L // MAXPS_4L                                  # 5 (same ratio as ₹5L)
    tgt = np.where(n_per_date <= thr, MAXPS_4L, POOL_4L / n_per_date)
    sh = np.floor(tgt / entry).astype(np.int64)
    df = pd.DataFrame({"date": pd.to_datetime(v["date"]), "symbol": v["symbol"].values,
                       "entry": entry, "shares": sh})
    df["cap"] = df["shares"] * df["entry"]
    return df[df["shares"] > 0].reset_index(drop=True)


# ── monkeypatch the ₹5L report to run on the ₹4L basis ──
ets.load_base_positions = load_base_4L
fpr.BASE_POOL = POOL_4L
fpr.BASE_ALLOC = ALLOC_4L
ets.CAPITAL_BASE = POOL_4L                                     # harmless safety
fpr.XLSX = fpr.OUTDIR / "final_performance_report_4L.xlsx"


if __name__ == "__main__":
    fpr.main()
