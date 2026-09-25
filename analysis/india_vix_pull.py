# -*- coding: utf-8 -*-
"""india_vix_pull.py — pull India VIX 1-minute (and daily) history from Upstox v2 historical-candle.
Instrument key: NSE_INDEX|India VIX. 1-min available from 2022-01-03 (verified). 1minute endpoint caps at
~1 month/request -> pull in monthly chunks. Daily pulled in one call as a reference series."""
import sys, time
from pathlib import Path
import requests, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb, data_loading as dl

KEY = "NSE_INDEX|India VIX"
ENC = KEY.replace("|", "%7C").replace(" ", "%20")
H = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}
BASE = "https://api.upstox.com/v2/historical-candle"
COLS = ["timestamp", "open", "high", "low", "close", "volume", "oi"]
OUT = rb.BASE / "data"


def hist(interval, to, frm, tries=4):
    url = f"{BASE}/{ENC}/{interval}/{to}/{frm}"
    for t in range(tries):
        try:
            r = requests.get(url, headers=H, timeout=40)
            if r.status_code == 200:
                return r.json().get("data", {}).get("candles", []) or []
            if r.status_code == 429:
                time.sleep(3 * (t + 1)); continue
            return []
        except Exception:
            time.sleep(2)
    return []


def main():
    today = pd.Timestamp.today().normalize()
    months = pd.date_range("2022-01-01", today, freq="MS")
    allc = []
    for m in months:
        frm = m.strftime("%Y-%m-%d"); to = (m + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
        c = hist("1minute", to, frm)
        allc += c
        print(f"  {frm[:7]}: {len(c)} candles")
        time.sleep(0.3)
    df = pd.DataFrame(allc, columns=COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_csv(OUT / "india_vix_1min.csv", index=False)

    dc = hist("day", today.strftime("%Y-%m-%d"), "2022-01-01")
    dd = pd.DataFrame(dc, columns=COLS); dd["timestamp"] = pd.to_datetime(dd["timestamp"])
    dd = dd.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    dd.to_csv(OUT / "india_vix_daily.csv", index=False)

    print("\n" + "=" * 70 + "\nINDIA VIX PULL COMPLETE")
    print(f"1-min: {len(df):,} rows | {df.timestamp.min()} .. {df.timestamp.max()}")
    print(f"       trading days: {df.timestamp.dt.normalize().nunique()} | VIX close range {df.close.min()} .. {df.close.max()}")
    print(f"daily: {len(dd):,} rows | {dd.timestamp.min().date()} .. {dd.timestamp.max().date()}")
    print(f"saved -> {OUT/'india_vix_1min.csv'} , {OUT/'india_vix_daily.csv'}")


if __name__ == "__main__":
    main()
