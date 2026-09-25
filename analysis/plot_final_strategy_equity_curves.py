# -*- coding: utf-8 -*-
"""plot_final_strategy_equity_curves.py — cumulative-returns line charts for the two FINAL strategies,
plotted from their already-computed trade sequences (no recomputation). Each PNG saved into that
strategy's own output folder only.
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

CS_DIR = rb.RESULTS / "weekly_credit_spread"
CS_TRADES = CS_DIR / "weekly_credit_spread_FINAL_v3_capped_july2026_trades.csv"
CS_OUT = CS_DIR / "weekly_credit_spread_returns.png"

BT_DIR = rb.RESULTS / "btst_close_direction_FINAL"
BT_TRADES = BT_DIR / "btst_close_direction_FINAL_v3_trades.csv"
BT_OUT = BT_DIR / "btst_close_direction_returns.png"


def plot_equity(trades_csv, out_png, title, date_col="entry_date", pnl_col="pnl_points"):
    T = pd.read_csv(trades_csv)
    T[date_col] = pd.to_datetime(T[date_col])
    T = T.sort_values(date_col).reset_index(drop=True)
    T["cum_pnl"] = T[pnl_col].cumsum()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(T[date_col], T["cum_pnl"], color="#1f6feb", linewidth=1.6)
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Return (points)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"{title}: {len(T)} trades | final cum P&L {round(T['cum_pnl'].iloc[-1],1)} | last entry {T[date_col].max().date()}")
    print(f"  saved -> {out_png}")


def main():
    plot_equity(CS_TRADES, CS_OUT,
                "NIFTY Weekly Credit Spread — FINAL (EOD-close SL, capped 22-Jul-2026)")
    plot_equity(BT_TRADES, BT_OUT,
                "NIFTY Daily Close Direction BTST — FINAL v3 (DTE-1 removed, capped 30/31-Jul-2026)")


if __name__ == "__main__":
    main()
