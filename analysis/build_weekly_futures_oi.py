# -*- coding: utf-8 -*-
"""
build_weekly_futures_oi.py
==========================
Weekly aggregation of the daily Nifty futures COMBINED total OI
(data/nifty_futures_oi_daily.csv -> data/nifty_futures_oi_weekly.csv).

NOTE: Nifty FUTURES are monthly (near/next/far expiries) — there is NO weekly futures
contract (weekly expiries exist only for OPTIONS). So "weekly OI" here = the daily combined
futures OI resampled to weekly frequency (week ending on the last trading day, Fri-anchored).

Per week: week_ending (last trading day), n_trading_days, oi_week_last (week-ending OI, the
standard weekly print), oi_week_avg / high / low, oi_week_change_vs_prev_week (last-vs-last),
contains_expiry (a monthly expiry fell in the week), near_expiry_at_week_end.
OI units = NSE OPEN_INT/OpnIntrst as reported: number of UNITS (contracts × lot size), NOT
contracts; lot size changed over the period, so not contract-comparable across lot changes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DAILY = REPO / "data" / "nifty_futures_oi_daily.csv"
OUT = REPO / "data" / "nifty_futures_oi_weekly.csv"


def main():
    d = pd.read_csv(DAILY)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    oi = "combined_oi_all_expiries"

    di = d.set_index("date")
    grp = di.resample("W-FRI")
    week_ending = grp.apply(lambda g: g.index.max().date() if len(g) else pd.NaT)
    w = pd.DataFrame({
        "week_ending": week_ending,
        "n_trading_days": grp[oi].count(),
        "oi_week_last": grp[oi].last(),
        "oi_week_avg": grp[oi].mean().round(0),
        "oi_week_high": grp[oi].max(),
        "oi_week_low": grp[oi].min(),
        "contains_expiry": grp["is_expiry_day"].apply(lambda s: bool(s.any())),
        "near_expiry_at_week_end": grp["near_expiry"].last(),
    })
    w = w[w["n_trading_days"] > 0].reset_index(drop=True)
    w["oi_week_last"] = w["oi_week_last"].astype("Int64")
    w["oi_week_high"] = w["oi_week_high"].astype("Int64")
    w["oi_week_low"] = w["oi_week_low"].astype("Int64")
    w["oi_week_avg"] = w["oi_week_avg"].astype("Int64")
    w["oi_week_change_vs_prev_week"] = w["oi_week_last"].diff().astype("Int64")
    w = w[["week_ending", "n_trading_days", "oi_week_last", "oi_week_change_vs_prev_week",
           "oi_week_avg", "oi_week_high", "oi_week_low", "contains_expiry", "near_expiry_at_week_end"]]
    w.to_csv(OUT, index=False)

    pd.set_option("display.width", 200)
    print("=" * 70)
    print("WEEKLY Nifty futures combined OI  (week-ending, Fri-anchored)")
    print("=" * 70)
    print(f"  Saved to           : {OUT}")
    print(f"  Weeks              : {len(w)}")
    print(f"  Range              : {w['week_ending'].min()} → {w['week_ending'].max()}")
    print(f"  n_trading_days dist: {dict(w['n_trading_days'].value_counts().sort_index())}")
    print(f"  Weeks w/ an expiry : {int(w['contains_expiry'].sum())}")
    print(f"  oi_week_last range : {int(w['oi_week_last'].min()):,} → {int(w['oi_week_last'].max()):,}")
    print("\n  first 3 weeks:\n" + w.head(3).to_string(index=False))
    print("\n  last 3 weeks:\n" + w.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
