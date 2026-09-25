# -*- coding: utf-8 -*-
"""plot_volume_breakout_equity_curves.py — cumulative-returns line charts for the main NSE Volume-Breakout
BTST strategy (locked baseline: mcap 1500-5000 Cr, LB36/VM6, Category A/B/C UC-entry, split long exit +
double-down short). Uses the EXISTING daily_performance sheet from baseline_final_performance.xlsx (already
computed, not recomputed) -- net_A basis, fixed-base % return, confirmed as the project's standard reporting
methodology by reproducing the summary sheet's total_return_fixedbase_pct exactly via cumsum.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUTDIR = rb.RESULTS / "baseline_and_cross_final"
SRC = OUTDIR / "baseline_final_performance.xlsx"
COL = "net_A_total_return_fixedbase_pct"

WINDOW_START = pd.Timestamp("2024-10-01")
WINDOW_END = pd.Timestamp("2026-07-31")


def plot_curve(D, out_png, title):
    D = D.sort_values("date").reset_index(drop=True)
    D["cum_return_pct"] = D[COL].cumsum()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(D["date"], D["cum_return_pct"], color="#1f6feb", linewidth=1.4)
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Return (%, fixed-base, net_A)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"{title}")
    print(f"  days: {len(D)} | window {D['date'].min().date()} -> {D['date'].max().date()} | final cum return {round(D['cum_return_pct'].iloc[-1],2)}%")
    print(f"  saved -> {out_png}")


def main():
    D_all = pd.read_excel(SRC, sheet_name="daily_performance")
    D_all["date"] = pd.to_datetime(D_all["date"])

    # ---- Graph 1: full backtest range ----
    plot_curve(D_all.copy(), OUTDIR / "volume_breakout_returns_full_range.png",
               "NSE Volume-Breakout BTST — Locked Baseline (net_A, fixed-base %) — Full Backtest Range")

    # ---- Graph 2: Oct-2024 to Jul-2026 window, cumulative RESTARTS at 0 for this sub-window ----
    D_win = D_all[(D_all["date"] >= WINDOW_START) & (D_all["date"] <= WINDOW_END)].copy()
    plot_curve(D_win, OUTDIR / "volume_breakout_returns_oct24_jul26.png",
               "NSE Volume-Breakout BTST — Locked Baseline (net_A, fixed-base %) — Oct 2024 to Jul 2026 Window")


if __name__ == "__main__":
    main()
