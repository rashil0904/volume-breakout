# -*- coding: utf-8 -*-
"""vb_extended_holding_monthly_pnl.py — monthly P&L attribution for the extended-holding overlay's
no-overlap-fixed trades (V1/V2/V3). Since these positions can span multiple calendar months, a trade's
full P&L is broken into per-month mark-to-market slices instead of being dumped entirely into its exit
month: for each month a trade is open, price_end_of_month - price_start_of_month (or avg_entry / exit_price
at the trade's own boundaries) * shares. A month's total = REALIZED (from trades that actually exited that
month, their final slice) + UNREALIZED (month-end mark-to-market on positions still open past month-end).
Reads the already-computed no-overlap trade lists; does not touch the baseline or any locked file.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SRC = rb.RESULTS / "baseline_and_cross_final" / "extended_holding_bb_overlay" / "extended_holding_no_overlap_fixed.xlsx"
MD = rb.BASE / "master_data"
OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "extended_holding_bb_overlay"


def daily_close_series(sym, _cache={}):
    if sym in _cache:
        return _cache[sym]
    fn = MD / f"{sym}.parquet"
    if not fn.exists():
        _cache[sym] = None; return None
    df = pd.read_parquet(fn, columns=["timestamp", "close"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    daily = df.groupby(ts.dt.normalize())["close"].last().sort_index()
    daily.index.name = "date"
    _cache[sym] = daily
    return daily


def month_end_marks(daily, entry_date, exit_date):
    """Return list of (date, price, is_final) marks between entry_date (exclusive) and exit_date
    (inclusive): one per calendar month-end the trade is still open through, plus the exit itself."""
    span = daily.loc[(daily.index > entry_date) & (daily.index <= exit_date)]
    if span.empty:
        return []
    months = span.index.to_series().dt.to_period("M")
    marks = []
    for period, idx in span.groupby(months).groups.items():
        last_date = idx.max()
        is_final = last_date == exit_date
        marks.append((last_date, float(span.loc[last_date]), is_final, period))
    return marks


def monthly_pnl_for_variant(trades):
    rows = []
    for _, r in trades.iterrows():
        sym = r["symbol"]; ed = pd.Timestamp(r["entry_date"]); xd = pd.Timestamp(r["exit_date"])
        shares = float(r["shares"]); avg = float(r["avg_entry"]); exit_price = float(r["exit_price"])
        daily = daily_close_series(sym)
        if daily is None:
            continue
        marks = month_end_marks(daily, ed, xd)
        if not marks:
            continue
        prev_price = avg
        for last_date, price, is_final, period in marks:
            px = exit_price if is_final else price   # force exact exit price on the final slice
            pnl = shares * (px - prev_price)
            rows.append({"symbol": sym, "month": str(period), "pnl": pnl,
                         "kind": "realized" if is_final else "unrealized",
                         "entry_date": ed, "exit_date": xd})
            prev_price = px
    R = pd.DataFrame(rows)
    if R.empty:
        return R, pd.DataFrame()
    monthly = R.pivot_table(index="month", columns="kind", values="pnl", aggfunc="sum", fill_value=0.0)
    for c in ["realized", "unrealized"]:
        if c not in monthly.columns:
            monthly[c] = 0.0
    monthly["total_pnl"] = monthly["realized"] + monthly["unrealized"]
    monthly = monthly.sort_index()
    monthly["cumulative_pnl"] = monthly["total_pnl"].cumsum()
    monthly = monthly.reset_index()[["month", "realized", "unrealized", "total_pnl", "cumulative_pnl"]]
    return R, monthly


def main():
    variants = ["V1", "V2", "V3"]
    all_monthly = {}
    with pd.ExcelWriter(OUTDIR / "extended_holding_monthly_pnl.xlsx", engine="openpyxl") as w:
        for v in variants:
            trades = pd.read_excel(SRC, sheet_name=f"{v}_no_overlap_trades")
            detail, monthly = monthly_pnl_for_variant(trades)
            all_monthly[v] = monthly
            print(f"\n=== {v} MONTHLY P&L (realized + unrealized mark-to-market) ===")
            pd.set_option("display.width", 200)
            print(monthly.round(1).to_string(index=False))
            check = monthly["total_pnl"].sum()
            actual = trades.assign(exit_price=trades["exit_price"]).apply(
                lambda r: r["shares"] * (r["exit_price"] - r["avg_entry"]), axis=1).sum()
            print(f"  reconciliation: sum(monthly total_pnl)={check:,.1f}  vs  sum(trade gross_pnl)={actual:,.1f}  "
                  f"(diff={check-actual:,.2f}, should be ~0)")
            monthly.round(2).to_excel(w, sheet_name=f"{v}_Monthly_PnL", index=False)
            detail.to_excel(w, sheet_name=f"{v}_Monthly_Detail", index=False)

    print(f"\nSaved -> {OUTDIR}/extended_holding_monthly_pnl.xlsx")


if __name__ == "__main__":
    main()
