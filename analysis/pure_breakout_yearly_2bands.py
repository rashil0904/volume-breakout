# -*- coding: utf-8 -*-
"""
pure_breakout_yearly_2bands.py
==============================
Yearly breakup for TWO market-cap bands (₹5,000-10,000 Cr and ₹10,000-15,000 Cr) of the
pure 52-week-high breakout strategy (exit 09:30/11:00). Reuses the exact trade set from
pure_breakout_mcap_bands.build() — same entries/sizing/exit; also caches the trade list
so subsequent band queries don't rescan parquets.

Band by ENTRY-DAY market cap; year by ENTRY date. total_return_fixedbase_pct on the fixed
₹5,00,000 base (additive up to each band's all-years total).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import pure_breakout_mcap_bands as pbm

OUTDIR = pbm.OUTDIR
CACHE = OUTDIR / "pure_breakout_trades.csv"
BASE_POOL = pbm.BASE_POOL
BANDS = ["5,000-10,000", "10,000-15,000"]
SMALL = 10


def load_trades():
    if CACHE.exists():
        print(f"Reusing cached trade list -> {CACHE.name}")
        return pd.read_csv(CACHE, parse_dates=["date"])
    print("Building pure-breakout trade set (one-time scan) …")
    bk, _, _ = pbm.build()
    bk.to_csv(CACHE, index=False)
    return bk


def year_rows(df, band):
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    rows = []
    for y, g in df.groupby("year"):
        rows.append({
            "band_cr": band, "year": int(y), "n_trades": len(g),
            "total_return_fixedbase_pct": round(g["gross_pnl"].sum() / BASE_POOL * 100, 4),
            "total_pnl_inr": round(g["gross_pnl"].sum(), 0),
            "net023_total_return_fixedbase_pct": round(g["net023_pnl"].sum() / BASE_POOL * 100, 4),
            "net023_total_pnl_inr": round(g["net023_pnl"].sum(), 0),
            "small_sample_flag": "n_trades<10" if len(g) < SMALL else "",
        })
    # ALL YEARS total row
    rows.append({
        "band_cr": band, "year": "ALL YEARS", "n_trades": len(df),
        "total_return_fixedbase_pct": round(df["gross_pnl"].sum() / BASE_POOL * 100, 4),
        "total_pnl_inr": round(df["gross_pnl"].sum(), 0),
        "net023_total_return_fixedbase_pct": round(df["net023_pnl"].sum() / BASE_POOL * 100, 4),
        "net023_total_pnl_inr": round(df["net023_pnl"].sum(), 0),
        "small_sample_flag": "",
    })
    return pd.DataFrame(rows)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    bk = load_trades()

    tables = {b: year_rows(bk[bk["band"] == b], b) for b in BANDS}
    combined = pd.concat(tables.values(), ignore_index=True)
    with pd.ExcelWriter(OUTDIR / "pure_breakout_yearly_2bands.xlsx", engine="openpyxl") as w:
        combined.to_excel(w, sheet_name="yearly_by_band", index=False)

    pd.set_option("display.width", 200)
    show = ["year", "n_trades", "total_return_fixedbase_pct", "total_pnl_inr",
            "net023_total_return_fixedbase_pct", "net023_total_pnl_inr", "small_sample_flag"]
    for b in BANDS:
        print("\n" + "=" * 110)
        print(f"BAND ₹{b} Cr — yearly breakup (pure 52w-high breakout, exit 09:30/11:00)")
        print("=" * 110)
        print(tables[b][show].to_string(index=False))
    print(f"\nSaved -> {OUTDIR / 'pure_breakout_yearly_2bands.xlsx'}")


if __name__ == "__main__":
    main()
