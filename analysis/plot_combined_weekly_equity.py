# -*- coding: utf-8 -*-
"""plot_combined_weekly_equity.py — WEEKLY-aggregated versions of the daily combined 3-strategy chart.
Same unit-normalization convention as plot_combined_3strategy_equity.py (confirmed by user, graph-only,
no changes to any strategy's own files): Volume Breakout stays on its real net_B fixed-base % (5L pool);
BTST/Credit Spread get 1.5-pt/trade expense deducted, x65 lot size, /70,000 fixed-base cap. Weekly boundary
= W-SUN (week ending Sunday), matching the W-SUN convention already used in this project's cross-strategy
correlation analysis. Returns/pnl within each calendar week are SUMMED (matches each strategy's existing
additive fixed-base % / points convention -- none of the three compound trade-to-trade), then cumulative
curve is the running cumsum of weekly totals. Uses EXISTING computed results only, no backtest re-run.
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
OUT_3 = rb.RESULTS / "combined_3strategy_weekly_oct24_jul26.png"
OUT_2 = rb.RESULTS / "combined_2strategy_weekly_oct24_jul26.png"

WIN_START = pd.Timestamp("2024-10-01"); WIN_END = pd.Timestamp("2026-07-31")
LOT_SIZE = 65; FIXED_CAP = 70000; EXPENSE_PTS_PER_TRADE = 1.5


def to_weekly(dates, values):
    # .to_numpy() strips any pre-existing index off `values` so pd.Series assigns positionally
    # against the new date index, instead of label-aligning against values' original row index.
    s = pd.Series(pd.Series(values).to_numpy(), index=pd.to_datetime(dates).to_numpy())
    w = s.resample("W-SUN").sum()
    return w


def main():
    # ---- 1. Volume Breakout: existing daily net_B fixed-base %, resampled to weekly sums ----
    D = pd.read_excel(VB_SRC, sheet_name="daily_performance")
    D["date"] = pd.to_datetime(D["date"])
    vb_d = D[(D["date"] >= WIN_START) & (D["date"] <= WIN_END)].sort_values("date")
    vb_w = to_weekly(vb_d["date"], vb_d["net_B_total_return_fixedbase_pct"])
    vb_cum = vb_w.cumsum()

    # ---- 2. BTST: 1.5-pt expense, x65 lot, resampled to weekly sums, /70,000 fixed-base ----
    B = pd.read_csv(BTST_SRC); B["entry_date"] = pd.to_datetime(B["entry_date"])
    B = B[(B["entry_date"] >= WIN_START) & (B["entry_date"] <= WIN_END)].sort_values("entry_date")
    B_pnl_inr = (B["pnl_points"] - EXPENSE_PTS_PER_TRADE) * LOT_SIZE
    btst_w = to_weekly(B["entry_date"], B_pnl_inr) / FIXED_CAP * 100
    btst_cum = btst_w.cumsum()

    # ---- 3. Credit Spread: same 1.5-pt expense + 70,000 fixed-base, resampled to weekly sums ----
    C = pd.read_csv(CS_SRC); C["entry_date"] = pd.to_datetime(C["entry_date"])
    C = C[(C["entry_date"] >= WIN_START) & (C["entry_date"] <= WIN_END)].sort_values("entry_date")
    C_pnl_inr = (C["pnl_points"] - EXPENSE_PTS_PER_TRADE) * LOT_SIZE
    cs_w = to_weekly(C["entry_date"], C_pnl_inr) / FIXED_CAP * 100
    cs_cum = cs_w.cumsum()

    # ---- align all three onto a common weekly index spanning the window (missing weeks = flat/no trade) ----
    # NOTE: build the shared index via .union() of the actual resampled indices (not a fresh pd.date_range) --
    # a fresh date_range can carry a different datetime64 unit (ns vs us) than the resampled data, which makes
    # reindex() silently match nothing and fillna(0) zero out the whole series.
    full_idx = vb_cum.index.union(btst_cum.index).union(cs_cum.index).sort_values()
    full_idx = full_idx[(full_idx >= WIN_START) & (full_idx <= WIN_END + pd.Timedelta(days=6))]
    vb_cum = vb_cum.reindex(full_idx).ffill().fillna(0.0)
    btst_cum = btst_cum.reindex(full_idx).ffill().fillna(0.0)
    cs_cum = cs_cum.reindex(full_idx).ffill().fillna(0.0)

    print(f"Volume Breakout : {len(vb_w.dropna())} active weeks | final cum % = {round(vb_cum.iloc[-1],2)}")
    print(f"BTST FINAL v3   : {len(btst_w.dropna())} active weeks | final cum % = {round(btst_cum.iloc[-1],2)}")
    print(f"Credit Spread   : {len(cs_w.dropna())} active weeks | final cum % = {round(cs_cum.iloc[-1],2)}")

    NOTE = ("NOTE: Weekly = W-SUN (week ending Sunday), matching this project's existing weekly-alignment convention.\n"
            "Volume Breakout uses its real capital base (₹5L pool, net_B cost basis, unchanged methodology). BTST/Credit\n"
            "Spread have no capital base elsewhere in this project (raw GROSS premium points only). For THIS chart only,\n"
            "per user-confirmed convention: 1.5-point-per-trade expense deducted, pnl_INR = net_pts × 65 (lot size), summed\n"
            "per week, cum_% = cumsum(weekly pnl_INR) / ₹70,000 × 100. Graph-only -- not part of either strategy's official results.")

    def style_axes(ax, title):
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Week", fontsize=11)
        ax.set_ylabel("Cumulative Return (%)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))

    # ---- GRAPH 1: all 3 strategies, weekly ----
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(full_idx, vb_cum, color="#1f6feb", linewidth=1.6, label="Volume Breakout (fixed-base %, net_B, ₹5L pool)")
    ax.plot(full_idx, btst_cum, color="#e0742a", linewidth=1.4, label="Nifty BTST Close Direction (net of 1.5pt/trade, ₹70K/trade cap)")
    ax.plot(full_idx, cs_cum, color="#2ba84a", linewidth=1.4, label="Nifty Weekly Credit Spread (net of 1.5pt/trade, ₹70K/trade cap)")
    style_axes(ax, "3-Strategy WEEKLY Cumulative Return Comparison — Oct 2024 to Jul 2026\n"
                    "(Volume Breakout: net_B  |  BTST & Credit Spread: net of 1.5 pts/trade, ₹70,000/trade cap — see note below)")
    fig.autofmt_xdate()
    fig.text(0.5, 0.05, NOTE, ha="center", va="bottom", fontsize=7.5, style="italic", color="#555555")
    fig.tight_layout(rect=[0.02, 0.17, 0.98, 1])
    fig.savefig(OUT_3, dpi=150)
    plt.close(fig)
    print(f"\nSaved -> {OUT_3}")

    # ---- GRAPH 2: Volume Breakout + Credit Spread only, weekly ----
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(full_idx, vb_cum, color="#1f6feb", linewidth=1.6, label="Volume Breakout (fixed-base %, net_B, ₹5L pool)")
    ax.plot(full_idx, cs_cum, color="#2ba84a", linewidth=1.4, label="Nifty Weekly Credit Spread (net of 1.5pt/trade, ₹70K/trade cap)")
    style_axes(ax, "2-Strategy WEEKLY Cumulative Return Comparison — Oct 2024 to Jul 2026\n"
                    "(Volume Breakout: net_B  |  Credit Spread: net of 1.5 pts/trade, ₹70,000/trade cap — see note below)")
    fig.autofmt_xdate()
    fig.text(0.5, 0.05, NOTE, ha="center", va="bottom", fontsize=7.5, style="italic", color="#555555")
    fig.tight_layout(rect=[0.02, 0.17, 0.98, 1])
    fig.savefig(OUT_2, dpi=150)
    plt.close(fig)
    print(f"Saved -> {OUT_2}")


if __name__ == "__main__":
    main()
