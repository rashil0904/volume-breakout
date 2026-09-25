# -*- coding: utf-8 -*-
"""
earnings_window_trade_frequency.py
==================================
Average number of trades PER TRADING DAY during four recurring quarterly-earnings
windows, for the main strategy (mcap ₹1,500-5,000 Cr, lookback 36, volume 6x, 3:15pm
entry, +5% day, 09:45/12:00 split + 14% target). Trades counted by ENTRY date; existing
trade set reused (not recomputed).

Denominator = ALL market-open days in each window across all years (from the trading-day
calendar), INCLUDING zero-trade days — a true per-trading-day average, not an average over
active days only. Windows are calendar month-day ranges recurring annually, endpoints
INCLUSIVE, no year-wrap.
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

# (name, (start_month, start_day), (end_month, end_day)) — inclusive, no year wrap
WINDOWS = [
    ("Q4-results  Jan 15 -> Feb 16", (1, 15), (2, 16)),
    ("Q1-results  Apr 20 -> May 31", (4, 20), (5, 31)),
    ("Q2-results  Jul 15 -> Aug 16", (7, 15), (8, 16)),
    ("Q3-results  Oct 15 -> Nov 16", (10, 15), (11, 16)),
]


def in_window(m, d, start_md, end_md):
    return start_md <= (m, d) <= end_md          # tuple compare; safe since no year-wrap


def which_window(dt):
    m, d = dt.month, dt.day
    for name, s, e in WINDOWS:
        if in_window(m, d, s, e):
            return name
    return None


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── trades (by entry date) ──
    T = fpr.build_trades()
    entry_dates = pd.to_datetime(T["entry_date"])
    trades_by_date = entry_dates.dt.normalize().value_counts()

    # ── trading-day calendar = all market-open days in the dataset ──
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["date"], parse_dates=["date"])
    cal = pd.DatetimeIndex(sorted(diag["date"].dt.normalize().unique()))
    data_min, data_max = cal.min(), cal.max()
    print(f"Trading-day calendar: {len(cal):,} days | {data_min.date()} -> {data_max.date()}")
    print(f"Total trades: {len(T):,}")

    cal_win = pd.Series([which_window(d) for d in cal], index=cal)
    trade_win = pd.Series([which_window(d) for d in entry_dates], index=entry_dates.index)

    # ── OUTPUT 2: per window, all years pooled ──
    per_rows = []
    for name, s, e in WINDOWS:
        days = cal[cal_win.values == name]
        n_days = len(days)
        n_trades = int(trade_win.eq(name).sum())
        yrs = sorted(set(days.year))
        # partial-coverage detection at dataset boundaries
        partial = []
        for y in range(data_min.year, data_max.year + 1):
            wstart = pd.Timestamp(year=y, month=s[0], day=s[1])
            wend = pd.Timestamp(year=y, month=e[0], day=e[1])
            if wend < data_min or wstart > data_max:
                continue                                  # window-year outside data
            if wstart < data_min or wend > data_max:
                partial.append(y)
        per_rows.append({
            "window_name": name, "total_trades": n_trades, "total_trading_days": n_days,
            "avg_trades_per_day": round(n_trades / n_days, 4) if n_days else np.nan,
            "n_years_contributing": len(yrs),
            "years": ",".join(str(y) for y in yrs),
            "partial_boundary_years": ",".join(str(y) for y in partial) if partial else "",
        })
    per_window = pd.DataFrame(per_rows)

    # ── OUTPUT 1: all four windows combined ──
    any_win = cal_win.notna()
    tot_days = int(any_win.sum())
    tot_trades = int(trade_win.notna().sum())
    combined = {
        "scope": "ALL FOUR WINDOWS COMBINED (all years)",
        "total_trades": tot_trades, "total_trading_days": tot_days,
        "avg_trades_per_day": round(tot_trades / tot_days, 4) if tot_days else np.nan,
    }

    # ── OUTPUT 3: full-period baseline + coverage context ──
    baseline = {
        "scope": "FULL-PERIOD BASELINE (all trading days)",
        "total_trades": len(T), "total_trading_days": len(cal),
        "avg_trades_per_day": round(len(T) / len(cal), 4),
    }
    lift = round(combined["avg_trades_per_day"] / baseline["avg_trades_per_day"], 3)

    overview = pd.DataFrame([combined, baseline])

    # ── save ──
    with pd.ExcelWriter(OUTDIR / "earnings_window_trade_frequency.xlsx", engine="openpyxl") as w:
        overview.to_excel(w, sheet_name="summary_combined_vs_baseline", index=False)
        per_window.to_excel(w, sheet_name="per_window", index=False)
        pd.DataFrame([{"metric": "earnings_window_avg_trades_per_day", "value": combined["avg_trades_per_day"]},
                      {"metric": "baseline_avg_trades_per_day", "value": baseline["avg_trades_per_day"]},
                      {"metric": "lift_earnings_vs_baseline_x", "value": lift}]).to_excel(
            w, sheet_name="lift", index=False)

    # ── prints ──
    pd.set_option("display.width", 200)
    print("\n" + "=" * 90)
    print("OUTPUT 1 — ALL FOUR WINDOWS COMBINED")
    print("=" * 90)
    print(f"  total_trades        : {combined['total_trades']:,}")
    print(f"  total_trading_days  : {combined['total_trading_days']:,}")
    print(f"  avg_trades_per_day  : {combined['avg_trades_per_day']}")

    print("\n" + "=" * 90)
    print("OUTPUT 2 — PER WINDOW (all years pooled)")
    print("=" * 90)
    print(per_window.to_string(index=False))

    print("\n" + "=" * 90)
    print("OUTPUT 3 — CONTEXT / BASELINE")
    print("=" * 90)
    print(f"  full-period baseline avg_trades_per_day : {baseline['avg_trades_per_day']}  "
          f"({len(T):,} trades / {len(cal):,} days)")
    print(f"  earnings-window avg_trades_per_day      : {combined['avg_trades_per_day']}")
    print(f"  LIFT (earnings vs baseline)             : {lift}x  "
          f"({'higher' if lift > 1 else 'lower'} frequency during earnings windows)")
    pj = per_window[per_window["partial_boundary_years"] != ""]
    if len(pj):
        print("  PARTIAL-COVERAGE windows (truncated at data boundary):")
        for _, r in pj.iterrows():
            print(f"     {r['window_name']}: partial year(s) {r['partial_boundary_years']}")
    else:
        print("  No windows truncated at the dataset boundary.")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
