# -*- coding: utf-8 -*-
"""
returns_by_mcap_band.py
=======================
Segments the LOCKED main strategy's trades (₹1,500–5,000 Cr, lookback 36, volume 6x,
+5% move, 3:15pm entry, 09:45/12:00 split exit + 14% target) into ₹500 Cr market-cap
sub-bands, by each trade's ENTRY-DAY market cap — the cap the strategy actually screened
on at 3:15pm. Reuses the canonical trade list (final_performance_report.build_trades,
the deployed 1_Standard base sheet); does NOT recompute the strategy.

Entry-day mcap comes from diagnostic_table.csv (market_cap_value, per symbol/date) — the
same source the entry filter uses.

Boundary convention: sub-bands are [lower, upper) so each trade lands in exactly one band;
the top band is [4500, 5000] inclusive to mirror the main filter's inclusive upper bound
(>=1500 & <=5000). Actual mcap range is 1500.3–4987.4, so no trade is dropped.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import final_performance_report as fpr

OUTDIR = rb.RESULTS / "mcap_subbands"
BASE_POOL = fpr.BASE_POOL
EDGES = [1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]   # 7 sub-bands
SMALL = 20                                                 # small-sample flag


def band_label(lo, hi):
    return f"{lo:,}-{hi:,}"


def assign_band(mc):
    """[lower, upper) for bands 1-6; top band [4500, 5000] inclusive."""
    for i in range(len(EDGES) - 1):
        lo, hi = EDGES[i], EDGES[i + 1]
        last = (i == len(EDGES) - 2)
        if (lo <= mc < hi) or (last and mc == hi):
            return band_label(lo, hi)
    return None


def metrics(df, label):
    n = len(df)
    gp = df["gross_pnl"].sum()
    win = df[df["gross_pnl"] > 0]
    los = df[df["gross_pnl"] <= 0]
    return {
        "mcap_band_cr": label,
        "n_trades": n,
        "win_rate_pct": round((df["gross_pnl"] > 0).mean() * 100, 2) if n else 0,
        "avg_return_per_trade_pct": round(df["gross_ret"].mean(), 4) if n else 0,
        "median_return_per_trade_pct": round(df["gross_ret"].median(), 4) if n else 0,
        "avg_return_winning_trades_pct": round(win["gross_ret"].mean(), 4) if len(win) else 0,
        "avg_return_losing_trades_pct": round(los["gross_ret"].mean(), 4) if len(los) else 0,
        "total_return_fixedbase_pct": round(gp / BASE_POOL * 100, 4),
        "total_pnl_inr": round(gp, 0),
        "avg_capital_deployed_per_trade": round(df["capital_deployed"].mean(), 0) if n else 0,
        "small_sample_flag": "n_trades<20" if n < SMALL else "",
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── canonical trade list (reused, not recomputed) ──
    T = fpr.build_trades()
    print(f"Main-strategy trades: {len(T):,}")

    # ── attach entry-day market cap from the diagnostic table ──
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "market_cap_value"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    mc_map = diag.set_index(["symbol", "date"])["market_cap_value"].to_dict()
    T = T.copy()
    T["entry_mcap_cr"] = [mc_map.get((s, d), np.nan)
                          for s, d in zip(T["symbol"], T["entry_date"])]

    missing = int(T["entry_mcap_cr"].isna().sum())
    if missing:
        print(f"  WARNING: {missing} trades missing entry-day mcap (dropped from banding)")
    T = T[T["entry_mcap_cr"].notna()].copy()
    print(f"  mcap range across trades: {T['entry_mcap_cr'].min():.1f}–{T['entry_mcap_cr'].max():.1f} Cr")

    T["mcap_band"] = T["entry_mcap_cr"].map(assign_band)
    unbanded = int(T["mcap_band"].isna().sum())
    if unbanded:
        print(f"  WARNING: {unbanded} trades outside [1500,5000] — not banded")
        T = T[T["mcap_band"].notna()].copy()

    # ── per-band metrics + ALL reconciliation row ──
    rows = []
    for i in range(len(EDGES) - 1):
        lbl = band_label(EDGES[i], EDGES[i + 1])
        rows.append(metrics(T[T["mcap_band"] == lbl], lbl))
    rows.append(metrics(T, "ALL (1,500-5,000)"))
    tbl = pd.DataFrame(rows)

    # reconciliation check (on UNROUNDED pnl — per-band display values are rounded to ₹1)
    band_rows = tbl[tbl["mcap_band_cr"] != "ALL (1,500-5,000)"]
    all_row = tbl[tbl["mcap_band_cr"] == "ALL (1,500-5,000)"].iloc[0]
    n_ok = band_rows["n_trades"].sum() == all_row["n_trades"]
    pnl_ok = abs(T[T["mcap_band"].notna()]["gross_pnl"].sum()
                 - T["gross_pnl"].sum()) < 1e-6

    tbl.to_csv(OUTDIR / "returns_by_mcap_band.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "returns_by_mcap_band.xlsx", engine="openpyxl") as w:
        tbl.to_excel(w, sheet_name="by_mcap_band", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 130)
    print("RETURNS BY ENTRY-DAY MARKET-CAP SUB-BAND  (main strategy ₹1,500–5,000 Cr; gross)")
    print("=" * 130)
    print(tbl.to_string(index=False))
    print(f"\nReconciliation: Σ band n_trades == ALL n_trades ? {n_ok}  |  "
          f"Σ band pnl == ALL pnl ? {pnl_ok}")
    flagged = band_rows[band_rows["small_sample_flag"] != ""]
    print(f"Small-sample bands (n_trades<{SMALL}): "
          f"{', '.join(flagged['mcap_band_cr']) if len(flagged) else 'none'}")

    # ── bar chart ──
    fig, ax = plt.subplots(figsize=(11, 6))
    x = band_rows["mcap_band_cr"].values
    y = band_rows["total_return_fixedbase_pct"].values
    small = band_rows["small_sample_flag"].values != ""
    colors = ["#b0b0b0" if s else "#1f4e79" for s in small]
    bars = ax.bar(range(len(x)), y, color=colors, edgecolor="white")
    for b, nt, s in zip(bars, band_rows["n_trades"].values, small):
        ax.annotate(f"n={nt}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8, color="#888" if s else "#333")
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=30, ha="right")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("entry-day market-cap sub-band (₹ Cr)")
    ax.set_ylabel("total return (fixed ₹5L base) %")
    ax.set_title("Where the edge lives — total return by entry-day market-cap band\n"
                 "(main ₹1,500–5,000 Cr strategy; bar label = n_trades)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "total_return_by_mcap_band.png", dpi=130)
    plt.close(fig)

    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
