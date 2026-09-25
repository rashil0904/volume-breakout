# -*- coding: utf-8 -*-
"""classify_new_listing_candidates.py — for each ETF-excluded 'new' candidate (in current NSE_EQ/BE Upstox
master but not in Companies List.csv), check its DAILY candle history to distinguish a GENUINE new listing
(no data before ~2025) from an old stock that simply wasn't in the original 1609-symbol list for some
other reason. Metadata-scale check only (daily candles, one wide-range call/symbol) -- no 1-min data
pulled. Read-only scoping pass.
"""
import sys, time, pickle
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_loading as dl
import run_backtest as rb

H = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}
CUTOFF_NEW = "2025-01-01"   # data starting on/after this -> treated as a genuine new listing
QUERY_FLOOR = "2022-01-01"  # v2 daily endpoint rejects far-back ranges (e.g. 2015) with "Invalid date range" --
                             # 2022-01-01 works AND matches our own dataset floor, which is the right question anyway


def get_earliest_daily(key):
    url = f"https://api.upstox.com/v2/historical-candle/{key.replace('|', '%7C')}/day/2026-09-20/{QUERY_FLOOR}"
    for a in range(3):
        try:
            r = requests.get(url, headers=H, timeout=30)
        except Exception:
            time.sleep(1); continue
        if r.status_code == 200:
            d = r.json().get("data", {}).get("candles", [])
            return min(c[0][:10] for c in d) if d else None
        if r.status_code in (429, 500, 502, 503):
            time.sleep(1.5); continue
        return f"http_{r.status_code}"
    return "failed"


def main():
    with open(rb.RESULTS / "_new_stock_candidates.pkl", "rb") as f:
        candidates = pickle.load(f)
    print(f"classifying {len(candidates)} ETF-excluded candidates...", flush=True)

    rows = []
    t0 = time.time()
    for i, inst in enumerate(candidates, 1):
        earliest = get_earliest_daily(inst["instrument_key"])
        rows.append({"symbol": inst["trading_symbol"], "name": inst.get("name"), "isin": inst.get("isin"),
                     "instrument_key": inst["instrument_key"], "earliest_daily_date": earliest})
        if i % 150 == 0:
            print(f"  ...{i}/{len(candidates)} ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(0.05)

    import pandas as pd
    R = pd.DataFrame(rows)
    R["is_genuine_new_listing"] = R["earliest_daily_date"].apply(
        lambda d: isinstance(d, str) and len(d) == 10 and d >= CUTOFF_NEW)
    R["no_data_at_all"] = R["earliest_daily_date"].isna()

    genuine = R[R["is_genuine_new_listing"]]
    old_excluded = R[(~R["is_genuine_new_listing"]) & (~R["no_data_at_all"]) & (~R["earliest_daily_date"].astype(str).str.startswith("http")) & (R["earliest_daily_date"] != "failed")]
    no_data = R[R["no_data_at_all"] | (R["earliest_daily_date"] == "failed") | R["earliest_daily_date"].astype(str).str.startswith("http")]

    print(f"\n=== RESULTS ({time.time()-t0:.0f}s total) ===")
    print(f"genuine new listings (data starts >= {CUTOFF_NEW}): {len(genuine)}")
    print(f"pre-existing stocks, just excluded originally (data starts earlier): {len(old_excluded)}")
    print(f"no data / fetch issues: {len(no_data)}")

    R.to_csv(rb.RESULTS / "new_listing_candidates_classified.csv", index=False)
    print(f"\nSaved -> {rb.RESULTS}/new_listing_candidates_classified.csv")
    print("\n--- genuine new listings, sample ---")
    print(genuine[["symbol", "name", "earliest_daily_date"]].sort_values("earliest_daily_date").to_string(index=False))


if __name__ == "__main__":
    main()
