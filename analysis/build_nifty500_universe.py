# -*- coding: utf-8 -*-
"""build_nifty500_universe.py — ADDITIVE universe for the PEAD expansion. Pulls CURRENT Nifty 500
constituents (NSE), tags fno_eligible from the CURRENT F&O list, and flags data coverage (daily OHLCV
present? already in the reconciled F&O announcement set?). CAVEATS emitted: membership + F&O eligibility
are current-snapshot, not point-in-time. Does NOT touch any existing pipeline/config.
"""
import sys, io
from pathlib import Path
import requests, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import fetch_bse_results_announcements as fb

OUT = rb.BASE / "data" / "nifty500_universe.csv"
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
ANN = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements.csv"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


def nifty500():
    s = requests.Session(); s.headers.update({"User-Agent": UA, "Accept": "*/*"})
    s.get("https://www.nseindia.com/", timeout=15)
    r = s.get("https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv", timeout=25)
    df = pd.read_csv(io.StringIO(r.text)); df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Company Name": "company", "Industry": "industry", "Symbol": "symbol", "ISIN Code": "isin"})
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    return df[["symbol", "company", "industry", "isin"]]


def main():
    n500 = nifty500()
    print(f"Nifty 500 constituents (current snapshot): {len(n500)}", flush=True)
    fo = set(fb.fo_universe())
    print(f"current F&O universe: {len(fo)} stocks", flush=True)
    daily_syms = set(pd.read_parquet(DAILY, columns=["symbol"])["symbol"].unique())
    ann_syms = set(pd.read_csv(ANN, usecols=["symbol"])["symbol"].unique())

    n500["fno_eligible"] = n500["symbol"].isin(fo)
    n500["has_daily_data"] = n500["symbol"].isin(daily_syms)
    n500["has_bse_announcements"] = n500["symbol"].isin(ann_syms)   # already pulled (F&O reconciled set)
    n500["in_nifty500"] = True
    n500["membership_asof"] = "2026-08-16 (current snapshot; NOT point-in-time)"
    n500["fno_eligibility_asof"] = "2026-08-16 (current; F&O inclusion changes over time)"

    n_fo = int(n500["fno_eligible"].sum()); n_non = len(n500) - n_fo
    n_nodata = int((~n500["has_daily_data"]).sum())
    n_need_ann = int((~n500["has_bse_announcements"]).sum())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n500.to_csv(OUT, index=False)

    print("\n" + "=" * 78 + "\nNIFTY 500 UNIVERSE (additive; F&O pipeline untouched)\n" + "=" * 78)
    print(f"total Nifty500        : {len(n500)}")
    print(f"  fno_eligible=True   : {n_fo}   (both long+short in backtest)")
    print(f"  fno_eligible=False  : {n_non}   (LONG-only in backtest)")
    print(f"daily OHLCV present   : {len(n500)-n_nodata}/{len(n500)}   (missing {n_nodata} -> would need price pull)")
    print(f"BSE announcements have: {len(n500)-n_need_ann}/{len(n500)}   (need pull for {n_need_ann} new names)")
    if n_nodata:
        miss = n500[~n500["has_daily_data"]]["symbol"].tolist()
        print(f"MISSING daily data ({n_nodata}): {miss[:30]}{' ...' if n_nodata>30 else ''}")
    print("\nCAVEAT: constituents + F&O flags are a CURRENT snapshot. Point-in-time membership and")
    print("point-in-time F&O eligibility are NOT reconstructed (survivorship/inclusion drift possible).")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
