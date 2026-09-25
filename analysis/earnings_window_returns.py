# -*- coding: utf-8 -*-
"""
earnings_window_returns.py
==========================
Compares RETURNS across three groups for the main strategy (mcap ₹1,500-5,000 Cr,
lookback 36, volume 6x, 3:15pm entry, +5% day, 09:45/12:00 split + 14% target,
₹5L pool / ₹1L per trade). Trades classified by ENTRY date; existing set reused.

  Group 1  earnings windows (any of the 4, pooled over the dataset)
  Group 2  non-earnings period (the complement — all trades NOT in any window)
  Group 3  all trades (baseline)

total_return_fixedbase_pct isn't comparable across groups of different length (Group 2
spans far more days), so per-trade metrics lead and per-trading-day normalized figures
(+ avg_trades_per_day) are added. Windows recur annually, endpoints inclusive.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import final_performance_report as fpr

OUTDIR = rb.RESULTS / "earnings_window_freq"
BASE_POOL = fpr.BASE_POOL

WINDOWS = [
    ("Q4-results  Jan 15 -> Feb 16", (1, 15), (2, 16)),
    ("Q1-results  Apr 20 -> May 31", (4, 20), (5, 31)),
    ("Q2-results  Jul 15 -> Aug 16", (7, 15), (8, 16)),
    ("Q3-results  Oct 15 -> Nov 16", (10, 15), (11, 16)),
]


def which_window(dt):
    md = (dt.month, dt.day)
    for name, s, e in WINDOWS:
        if s <= md <= e:
            return name
    return None


def group_metrics(name, T, n_days):
    n = len(T)
    if n == 0:
        return {"group": name, "n_trades": 0}
    gp = T["gross_pnl"].sum(); npl = T["net_pnl"].sum()
    gw, gl = T[T["gross_pnl"] > 0], T[T["gross_pnl"] <= 0]
    total_g = gp / BASE_POOL * 100
    total_n = npl / BASE_POOL * 100
    return {
        "group": name,
        "n_trades": n,
        "trading_days_in_group": n_days,
        "avg_trades_per_day": round(n / n_days, 4) if n_days else np.nan,
        # per-trade (trade-count-normalized — lead with these)
        "win_rate_pct": round((T["gross_pnl"] > 0).mean() * 100, 2),
        "avg_return_per_trade_pct": round(T["gross_ret"].mean(), 4),
        "median_return_per_trade_pct": round(T["gross_ret"].median(), 4),
        "avg_return_winning_trades_pct": round(gw["gross_ret"].mean(), 4) if len(gw) else np.nan,
        "avg_return_losing_trades_pct": round(gl["gross_ret"].mean(), 4) if len(gl) else np.nan,
        "net_avg_return_per_trade_pct": round(T["net_ret"].mean(), 4),
        "net_win_rate_pct": round((T["net_pnl"] > 0).mean() * 100, 2),
        # totals (NOT comparable across groups of different length)
        "total_return_fixedbase_pct": round(total_g, 4),
        "total_pnl_inr": round(gp, 0),
        "net_total_return_fixedbase_pct": round(total_n, 4),
        "net_total_pnl_inr": round(npl, 0),
        # length-normalized
        "total_return_per_trading_day_pct": round(total_g / n_days, 4) if n_days else np.nan,
        "avg_capital_deployed_per_trade": round(T["capital_deployed"].mean(), 0),
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    T = fpr.build_trades().copy()
    T["entry_dt"] = pd.to_datetime(T["entry_date"])
    T["window"] = [which_window(d) for d in T["entry_dt"]]
    T["in_earnings"] = T["window"].notna()

    # trading-day calendar (all market-open days)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["date"], parse_dates=["date"])
    cal = pd.DatetimeIndex(sorted(diag["date"].dt.normalize().unique()))
    cal_win = pd.Series([which_window(d) for d in cal], index=cal)
    n_days_earn = int(cal_win.notna().sum())
    n_days_non = int(cal_win.isna().sum())
    n_days_all = len(cal)
    print(f"Calendar {len(cal):,} days ({cal.min().date()}..{cal.max().date()}) | trades {len(T):,}")

    g1 = group_metrics("1_earnings_windows", T[T["in_earnings"]], n_days_earn)
    g2 = group_metrics("2_non_earnings", T[~T["in_earnings"]], n_days_non)
    g3 = group_metrics("3_all_trades", T, n_days_all)
    comp = pd.DataFrame([g1, g2, g3])

    # per-window breakdown (optional output 3)
    win_rows = []
    for name, s, e in WINDOWS:
        sub = T[T["window"] == name]
        days = int((cal_win.values == name).sum())
        win_rows.append(group_metrics(name, sub, days))
    per_window = pd.DataFrame(win_rows)

    with pd.ExcelWriter(OUTDIR / "earnings_window_returns.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="three_group_comparison", index=False)
        per_window.to_excel(w, sheet_name="per_window", index=False)

    pd.set_option("display.width", 240)
    lead = ["group", "n_trades", "win_rate_pct", "avg_return_per_trade_pct",
            "median_return_per_trade_pct", "avg_return_winning_trades_pct",
            "avg_return_losing_trades_pct", "net_avg_return_per_trade_pct"]
    length = ["group", "n_trades", "trading_days_in_group", "avg_trades_per_day",
              "total_return_fixedbase_pct", "total_pnl_inr", "total_return_per_trading_day_pct",
              "avg_capital_deployed_per_trade"]
    print("\n" + "=" * 120)
    print("THREE-GROUP COMPARISON — PER-TRADE METRICS (lead; trade-count-normalized)")
    print("=" * 120)
    print(comp[lead].to_string(index=False))
    print("\n" + "=" * 120)
    print("THREE-GROUP COMPARISON — TOTALS + LENGTH-NORMALIZED (totals NOT comparable across groups)")
    print("=" * 120)
    print(comp[length].to_string(index=False))
    print("\n" + "=" * 120)
    print("PER-WINDOW (all years pooled) — per-trade metrics")
    print("=" * 120)
    print(per_window[["group", "n_trades", "avg_trades_per_day", "win_rate_pct",
                      "avg_return_per_trade_pct", "median_return_per_trade_pct",
                      "total_return_per_trading_day_pct"]].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
