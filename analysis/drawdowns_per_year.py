# -*- coding: utf-8 -*-
"""
drawdowns_per_year.py
=====================
Top 10 DRAWDOWNS per year for the main strategy — peak-to-trough declines in the
cumulative daily-P&L (equity) curve, grouped by EXIT day, equity reset at each year's
start. Reuses canonical trades (fpr.build_trades); no recompute. Primary curve = gross
day P&L; net depth over the same span reported alongside.

A drawdown episode = from a running-equity peak to the lowest point before a new high
(or year end). depth = equity[peak] − equity[trough] (₹, positive). Ranked by gross depth.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import final_performance_report as fpr

OUTDIR = rb.RESULTS / "worst_pnl_days"


def find_drawdowns(dates, g_eq, n_eq, day_g):
    """Return list of drawdown episodes on the gross equity curve g_eq."""
    eps = []
    peak_v = -np.inf; peak_i = 0
    in_dd = False; tr_v = None; tr_i = None
    for i, eq in enumerate(g_eq):
        if eq >= peak_v:
            if in_dd:                                   # recovery -> close episode
                eps.append((peak_i, tr_i, i)); in_dd = False
            peak_v, peak_i = eq, i
        else:
            if not in_dd:
                in_dd, tr_v, tr_i = True, eq, i
            elif eq < tr_v:
                tr_v, tr_i = eq, i
    if in_dd:
        eps.append((peak_i, tr_i, None))                # unrecovered at year end
    out = []
    for p, t, r in eps:
        depth_g = g_eq[p] - g_eq[t]
        depth_n = n_eq[p] - n_eq[t]
        n_down = int((day_g[p + 1:t + 1] < 0).sum())    # losing days within the decline
        out.append({
            "peak_date": dates[p], "trough_date": dates[t],
            "drawdown_depth_inr_gross": round(depth_g, 0),
            "drawdown_depth_inr_net": round(depth_n, 0),
            "duration_peak_to_trough_days": t - p,
            "recovery_date": dates[r] if r is not None else "not recovered in year",
            "recovery_days_trough_to_new_high": (r - t) if r is not None else np.nan,
            "n_losing_days_in_decline": n_down,
        })
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = fpr.build_trades()
    T["exit_d"] = pd.to_datetime(T["exit_date"])
    daily = (T.groupby("exit_d").agg(day_g=("gross_pnl", "sum"), day_n=("net_pnl", "sum"))
             .reset_index().sort_values("exit_d"))
    daily["year"] = daily["exit_d"].dt.year

    blocks = []
    for y, g in daily.groupby("year"):
        g = g.sort_values("exit_d").reset_index(drop=True)
        dates = [d.date() for d in g["exit_d"]]
        g_eq = np.cumsum(g["day_g"].values)
        n_eq = np.cumsum(g["day_n"].values)
        eps = find_drawdowns(dates, g_eq, n_eq, g["day_g"].values)
        dd = pd.DataFrame(eps).sort_values("drawdown_depth_inr_gross", ascending=False).head(10)
        dd.insert(0, "rank_in_year", range(1, len(dd) + 1))
        dd.insert(0, "year", y)
        blocks.append(dd)
    ddtbl = pd.concat(blocks, ignore_index=True)
    ddtbl.to_csv(OUTDIR / "drawdowns_top10_per_year.csv", index=False)

    # append to the existing workbook (keep prior sheets)
    xlsx = OUTDIR / "worst_pnl_days.xlsx"
    mode = "a" if xlsx.exists() else "w"
    kw = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(xlsx, engine="openpyxl", mode=mode, **kw) as w:
        ddtbl.to_excel(w, sheet_name="drawdowns_top10_per_year", index=False)

    pd.set_option("display.width", 220)
    print("=" * 130)
    print("TOP 10 DRAWDOWNS PER YEAR (peak-to-trough of cumulative daily P&L; equity reset each year)")
    print("=" * 130)
    show = ["year", "rank_in_year", "peak_date", "trough_date", "drawdown_depth_inr_gross",
            "drawdown_depth_inr_net", "duration_peak_to_trough_days", "recovery_date",
            "n_losing_days_in_decline"]
    print(ddtbl[show].to_string(index=False))
    print("\n--- WORST DRAWDOWN EACH YEAR ---")
    worst = ddtbl[ddtbl.rank_in_year == 1]
    for _, r in worst.iterrows():
        print(f"  {r['year']}: ₹{r['drawdown_depth_inr_gross']:,.0f} gross  "
              f"({r['peak_date']} -> {r['trough_date']}, {r['duration_peak_to_trough_days']}d, "
              f"recovered {r['recovery_date']})")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
