# -*- coding: utf-8 -*-
"""
build_combined_oi_price.py
==========================
Nifty futures daily series: COMBINED OI (Σ across ALL expiries) + NEAR-MONTH OHLC + close change.
Re-cut from the cached F&O bhavcopy per-contract data (results/fo_price_cache/) — no re-fetch.

Columns (per spec, ONLY these + a rollover flag requested by STEP 5 / flag c):
  date, futures_open/high/low/close (near-month), close_change_pct (near-month continuous),
  combined_oi (Σ all expiries), combined_oi_change_num, combined_oi_change_pct,
  contract_expiry + rollover_day  (metadata so rollover days are identifiable/flaggable).

close_change_pct here is the NEAR-MONTH CONTINUOUS change: (close - prev-day near-month close)/
prev × 100. On a rollover day the near contract switches, so that day's % spans two contracts
(reflects the contract switch, not a pure price move) — flagged via rollover_day.

NO spot, NO basis, NO price_oi_signal (excluded per instruction).
OI units = NSE OPEN_INT/OpnIntrst as reported: UNITS (contracts × lot size), NOT contracts.
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fetch_nifty_futures_oi as FO

REPO = Path(__file__).resolve().parent.parent
PCACHE = REPO / "results" / "fo_price_cache"
OUT_XLSX = REPO / "data" / "nifty_futures_combined_oi_price_daily.xlsx"


def main():
    print("STEP 1 — calendar from existing stock data …")
    cal = FO.trading_calendar()
    calset = set(cal)
    print(f"    {len(cal):,} trading days | {cal[0]} → {cal[-1]}")

    rows = []
    for d in cal:
        cf = PCACHE / f"{d.strftime('%Y%m%d')}.json"
        if not cf.exists():
            continue
        r = json.loads(cf.read_text())
        cons = r["contracts"]
        combined_oi = int(sum(c["oi"] for c in cons))                 # STEP 2: Σ OI all expiries
        near = min(cons, key=lambda c: c["expiry"])                   # STEP 3: near-month contract
        rows.append({"date": pd.to_datetime(r["date"]).date(), "contract_expiry": pd.to_datetime(near["expiry"]).date(),
                     "futures_open": near["o"], "futures_high": near["h"], "futures_low": near["l"],
                     "futures_close": near["c"], "combined_oi": combined_oi})

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["close_change_pct"] = (df["futures_close"].pct_change() * 100).round(4)      # near-month continuous
    df["combined_oi_change_num"] = df["combined_oi"].diff().astype("Int64")
    df["combined_oi_change_pct"] = (df["combined_oi"].pct_change() * 100).round(4)
    df["rollover_day"] = df["contract_expiry"].ne(df["contract_expiry"].shift(1))
    df.loc[0, "rollover_day"] = False                                              # first obs: no prior contract

    out = df[["date", "contract_expiry", "futures_open", "futures_high", "futures_low", "futures_close",
              "close_change_pct", "combined_oi", "combined_oi_change_num", "combined_oi_change_pct",
              "rollover_day"]]

    rollover_dates = out.loc[out["rollover_day"], "date"].tolist()
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="combined_oi_price_daily", index=False)
        pd.DataFrame({"rollover_date": rollover_dates}).to_excel(w, sheet_name="rollover_dates", index=False)
        pd.DataFrame({"note": [
            "combined_oi = Σ open interest across ALL Nifty futures expiries that day (near+next+far).",
            "futures_open/high/low/close = NEAR-MONTH (front) contract OHLC (representative futures price).",
            "close_change_pct = near-month CONTINUOUS change; on rollover_day it spans two contracts "
            "(reflects the contract switch, not a pure price move).",
            "OI units: NSE OPEN_INT/OpnIntrst as reported = UNITS (contracts × lot size), NOT contracts.",
        ]}).to_excel(w, sheet_name="notes", index=False)

    # ── sanity ──
    got = set(out["date"]); missing = sorted(calset - got)
    pd.set_option("display.width", 220)
    print("\n" + "=" * 74)
    print("Nifty futures COMBINED OI + near-month OHLC — saved")
    print("=" * 74)
    print(f"  Saved to        : {OUT_XLSX}")
    print(f"  Columns         : {list(out.columns)}")
    print(f"  Date range      : {out['date'].min()} → {out['date'].max()}")
    print(f"  Trading days    : {len(out)}  (stock calendar: {len(cal)})")
    print(f"  Missing vs stocks: {len(missing)}"
          + ("  -> " + ", ".join(str(x) for x in missing[:15]) if missing else ""))
    print(f"  combined_oi range: {out['combined_oi'].min():,} → {out['combined_oi'].max():,}")
    print(f"  Rollover days    : {len(rollover_dates)}")
    print("     " + ", ".join(str(x) for x in rollover_dates[:14]) + (" …" if len(rollover_dates) > 14 else ""))
    print("\n  first 3 rows:\n" + out.head(3).to_string(index=False))
    print("\n  around a rollover (rows flagged rollover_day):\n"
          + out[out["rollover_day"]].head(3).to_string(index=False))
    print("\n  last 3 rows:\n" + out.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
