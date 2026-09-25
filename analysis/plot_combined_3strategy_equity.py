# -*- coding: utf-8 -*-
"""plot_combined_3strategy_equity.py — single overlaid cumulative-return chart for all 3 finalized strategies,
Oct-2024 to Jul-2026 window. Uses EXISTING computed trade/daily data for each (no backtest re-run).

UNIT NORMALIZATION (flagged, not silent): Volume Breakout already reports fixed-base % return (real 5L
capital pool). BTST and Credit Spread have no capital base anywhere in this project -- always reported as
raw GROSS option-premium points. For THIS comparison chart only, each option-strategy trade's % return is
derived from columns already in its own trade sheet: BTST = pnl_points/entry_cost*100 (return on premium
paid); Credit Spread = pnl_points/net_credit*100 (return on credit received, standard credit-spread
convention). This is a new convention for cross-strategy comparison ONLY -- not part of either strategy's
official reported results elsewhere in this project.
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

VB_SRC = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
BTST_SRC = rb.RESULTS / "btst_close_direction_FINAL" / "btst_close_direction_FINAL_v3_trades.csv"
CS_SRC = rb.RESULTS / "weekly_credit_spread" / "weekly_credit_spread_FINAL_v3_capped_july2026_trades.csv"
OUT_PNG = rb.RESULTS / "combined_3strategy_equity_oct24_jul26.png"

WIN_START = pd.Timestamp("2024-10-01"); WIN_END = pd.Timestamp("2026-07-31")


def main():
    # ---- 1. Volume Breakout: existing fixed-base % (net_B basis, per user instruction), already-computed ----
    D = pd.read_excel(VB_SRC, sheet_name="daily_performance")
    D["date"] = pd.to_datetime(D["date"])
    vb = D[(D["date"] >= WIN_START) & (D["date"] <= WIN_END)].sort_values("date").copy()
    vb["cum_pct"] = vb["net_B_total_return_fixedbase_pct"].cumsum()

    # ---- 2. BTST FINAL v3: 1.5-pt-per-trade expense (GRAPH ONLY), then x65 lot size, /70,000 fixed-base ----
    LOT_SIZE = 65; FIXED_CAP = 70000; EXPENSE_PTS_PER_TRADE = 1.5
    B = pd.read_csv(BTST_SRC); B["entry_date"] = pd.to_datetime(B["entry_date"])
    B = B[(B["entry_date"] >= WIN_START) & (B["entry_date"] <= WIN_END)].sort_values("entry_date").copy()
    B["pnl_inr"] = (B["pnl_points"] - EXPENSE_PTS_PER_TRADE) * LOT_SIZE
    B["cum_pct"] = B["pnl_inr"].cumsum() / FIXED_CAP * 100

    # ---- 3. Weekly Credit Spread FINAL (capped 22-Jul-2026): same 1.5-pt expense + 70,000 fixed-base ----
    C = pd.read_csv(CS_SRC); C["entry_date"] = pd.to_datetime(C["entry_date"])
    C = C[(C["entry_date"] >= WIN_START) & (C["entry_date"] <= WIN_END)].sort_values("entry_date").copy()
    C["pnl_inr"] = (C["pnl_points"] - EXPENSE_PTS_PER_TRADE) * LOT_SIZE
    C["cum_pct"] = C["pnl_inr"].cumsum() / FIXED_CAP * 100

    print(f"Volume Breakout : {len(vb)} days   | final cum % = {round(vb['cum_pct'].iloc[-1],2)}")
    print(f"BTST FINAL v3   : {len(B)} trades | final cum % = {round(B['cum_pct'].iloc[-1],2)}")
    print(f"Credit Spread   : {len(C)} trades | final cum % = {round(C['cum_pct'].iloc[-1],2)}")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(vb["date"], vb["cum_pct"], color="#1f6feb", linewidth=1.6, label="Volume Breakout (fixed-base %, net_B, ₹5L pool)")
    ax.plot(B["entry_date"], B["cum_pct"], color="#e0742a", linewidth=1.4, label="Nifty BTST Close Direction (net of 1.5pt/trade, ₹70K/trade cap)")
    ax.plot(C["entry_date"], C["cum_pct"], color="#2ba84a", linewidth=1.4, label="Nifty Weekly Credit Spread (net of 1.5pt/trade, ₹70K/trade cap)")
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")

    ax.set_title("3-Strategy NET Cumulative Return Comparison — Oct 2024 to Jul 2026\n"
                  "(Volume Breakout: net_B  |  BTST & Credit Spread: net of 1.5 pts/trade, ₹70,000/trade cap — see note below)",
                  fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    fig.autofmt_xdate()
    note = ("NOTE: Volume Breakout uses its established real capital base (₹5L pool) on the net_B cost basis (existing methodology, unchanged).\n"
            "BTST/Credit Spread have no capital base anywhere else in this project (always reported in raw GROSS premium points, no per-trade\n"
            "expense assumption). For THIS comparison chart only, per user instruction: a flat 1.5-point-per-trade expense is deducted from each\n"
            "trade's pnl_points, then pnl_INR = net_pnl_points × 65 (lot size), cum_% = cumsum(pnl_INR) / ₹70,000 × 100. Not part of either\n"
            "strategy's official results or files -- graph-only convention.")
    fig.text(0.5, 0.05, note, ha="center", va="bottom", fontsize=7.5, style="italic", color="#555555")
    fig.tight_layout(rect=[0.02, 0.17, 0.98, 1])
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"\nSaved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
