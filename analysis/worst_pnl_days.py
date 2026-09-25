# -*- coding: utf-8 -*-
"""
worst_pnl_days.py
=================
Ranks the main strategy's worst single-day P&L days BY day_pnl_inr (₹), grouped by EXIT
day (P&L realization). Reuses canonical trades (fpr.build_trades); no recompute.

day_pnl_inr = Σ gross_trade_pnl of trades exiting that day (net version = Σ net_pnl).
Context joins the four earnings windows and the CNXMIDCAP100 morning move (prev-close ->
exit-day 12:00) to test for a market-wide driver. Selloff flag = index morning ret <= -1%.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import final_performance_report as fpr

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "worst_pnl_days"
INDEX_CSV = rb.REPO / "data" / "cnxmidcap100_15min_ohlc.csv" if hasattr(rb, "REPO") \
    else Path(__file__).resolve().parent.parent / "data" / "cnxmidcap100_15min_ohlc.csv"
SELLOFF_THR = -1.0
WINDOWS = [("Jan15-Feb16", (1, 15), (2, 16)), ("Apr20-May31", (4, 20), (5, 31)),
           ("Jul15-Aug16", (7, 15), (8, 16)), ("Oct15-Nov16", (10, 15), (11, 16))]


def which_window(d):
    md = (d.month, d.day)
    for name, s, e in WINDOWS:
        if s <= md <= e:
            return name
    return ""


def index_morning_returns():
    """CNXMIDCAP100 morning move: (12:00 open − prior-day last close)/prior close × 100, by date."""
    d = pd.read_csv(INDEX_CSV)
    ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(IST)
    d["date"] = ts.dt.date
    d["hm"] = ts.dt.hour * 60 + ts.dt.minute
    last_close = d.sort_values("hm").groupby("date")["close"].last()
    o1200 = d[d["hm"] == 720].groupby("date")["open"].last()
    dates = sorted(last_close.index)
    prev_close = {dates[i]: last_close.loc[dates[i - 1]] for i in range(1, len(dates))}
    out = {}
    for dt in dates:
        pc = prev_close.get(dt); o = o1200.get(dt, np.nan)
        if pc and pc == pc and o == o:
            out[dt] = (o - pc) / pc * 100
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = fpr.build_trades()
    T["exit_d"] = pd.to_datetime(T["exit_date"]).dt.date

    idx_ret = index_morning_returns()

    rows = []
    for d, g in T.groupby("exit_d"):
        gp = g["gross_pnl"].sum(); npl = g["net_pnl"].sum()
        worst = g.loc[g["gross_pnl"].idxmin()]
        ir = idx_ret.get(d, np.nan)
        rows.append({
            "date": d, "day_pnl_inr_gross": round(gp, 0), "day_pnl_inr_net": round(npl, 0),
            "n_trades_that_day": len(g), "n_losers": int((g["gross_pnl"] <= 0).sum()),
            "win_rate_that_day": round((g["gross_pnl"] > 0).mean() * 100, 2),
            "worst_trade_symbol": worst["symbol"], "worst_trade_return_pct": round(worst["gross_ret"], 2),
            "year": pd.Timestamp(d).year,
            "earnings_window": which_window(pd.Timestamp(d)),
            "cnxmidcap100_morning_ret_pct": round(ir, 3) if ir == ir else np.nan,
            "index_selloff_day": (ir <= SELLOFF_THR) if ir == ir else False,
        })
    daily = pd.DataFrame(rows).sort_values("day_pnl_inr_gross").reset_index(drop=True)

    cols = ["rank", "date", "day_pnl_inr_gross", "day_pnl_inr_net", "n_trades_that_day",
            "win_rate_that_day", "worst_trade_symbol", "worst_trade_return_pct",
            "earnings_window", "cnxmidcap100_morning_ret_pct", "index_selloff_day"]

    # ── TABLE 1: top 20 worst all-time ──
    t1 = daily.head(20).copy()
    t1.insert(0, "rank", range(1, len(t1) + 1))
    t1 = t1[cols]

    # ── TABLE 2: top 10 worst per year ──
    blocks = []
    for y, g in daily.groupby("year"):
        b = g.sort_values("day_pnl_inr_gross").head(10).copy()
        b["rank_in_year"] = range(1, len(b) + 1)
        blocks.append(b[["year", "rank_in_year"] + cols[1:]])
    t2 = pd.concat(blocks, ignore_index=True)

    with pd.ExcelWriter(OUTDIR / "worst_pnl_days.xlsx", engine="openpyxl") as w:
        t1.to_excel(w, sheet_name="top20_alltime", index=False)
        t2.to_excel(w, sheet_name="top10_per_year", index=False)

    # ── context ──
    worst = daily.iloc[0]
    n_ew = int((t1["earnings_window"] != "").sum())
    n_sell = int(t1["index_selloff_day"].sum())

    pd.set_option("display.width", 240)
    print("=" * 130)
    print("TABLE 1 — TOP 20 WORST DAYS ALL-TIME (by day_pnl_inr gross; grouped by EXIT day)")
    print("=" * 130)
    print(t1.to_string(index=False))
    print("\n" + "=" * 130)
    print("TABLE 2 — TOP 10 WORST DAYS PER YEAR")
    print("=" * 130)
    print(t2.to_string(index=False))
    print("\n" + "=" * 130)
    print("CONTEXT")
    print("=" * 130)
    print(f"  WORST DAY OVERALL : {worst['date']}  gross ₹{worst['day_pnl_inr_gross']:,.0f} | "
          f"net ₹{worst['day_pnl_inr_net']:,.0f} | {worst['n_trades_that_day']} trades, "
          f"{worst['n_losers']} losers | CNXMIDCAP100 morning {worst['cnxmidcap100_morning_ret_pct']}%")
    print(f"  day_pnl_inr GROSS : mean ₹{daily['day_pnl_inr_gross'].mean():,.0f} | "
          f"median ₹{daily['day_pnl_inr_gross'].median():,.0f}  (over {len(daily):,} trading days)")
    print(f"  day_pnl_inr NET   : mean ₹{daily['day_pnl_inr_net'].mean():,.0f} | "
          f"median ₹{daily['day_pnl_inr_net'].median():,.0f}")
    print(f"  of TOP 20 worst days: {n_ew}/20 fall in an earnings window | "
          f"{n_sell}/20 were CNXMIDCAP100 selloff days (morning ret <= {SELLOFF_THR}%)")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
