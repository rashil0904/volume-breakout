# -*- coding: utf-8 -*-
"""fetch_nifty_daily_2020_2021.py — one-off pull of NIFTY DAILY OHLCV, 2020-01-01 through 2021-12-31,
via Upstox V3 historical-candle (days/1 interval, confirmed correct endpoint format via direct API test --
'day/1' and 'day' both 400, 'days/1' is correct). Standalone daily dataset, distinct granularity from the
1-min intraday files -- NOT merged with them.
"""
import time
from pathlib import Path
import requests
import pandas as pd
import data_loading as dl

REPO = Path(__file__).parent
OUT_CSV = REPO / "data" / "nifty_daily_2020_2021.csv"
KEY = "NSE_INDEX|Nifty 50"
SCHEMA = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]
HEADERS = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}
FROM, TO = "2020-01-01", "2021-12-31"


def enc(k): return k.replace("|", "%7C")


url = f"https://api.upstox.com/v3/historical-candle/{enc(KEY)}/days/1/{TO}/{FROM}"
r = requests.get(url, headers=HEADERS, timeout=40)
print(f"status: {r.status_code}")
candles = r.json().get("data", {}).get("candles", [])
print(f"candles returned: {len(candles)}")

df = pd.DataFrame(candles, columns=SCHEMA)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
for c in ("open", "high", "low", "close"):
    df[c] = df[c].astype(float)
df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0).astype("int64")

print(f"\nfetched: {len(df)} rows | {df['timestamp'].min()} -> {df['timestamp'].max()}")

# ---- quality checks ----
ts_dates = df["timestamp"].dt.date
bdays_expected = pd.bdate_range(FROM, TO).date
missing_bdays = sorted(set(bdays_expected) - set(ts_dates))
print(f"\nbusiness weekdays with ZERO candles (holiday OR missing): {len(missing_bdays)}")
if missing_bdays:
    print("   " + ", ".join(str(d) for d in missing_bdays))

dup = df["timestamp"].duplicated().sum()
nan_ct = df[["open", "high", "low", "close"]].isna().sum().sum()
bad_hl = (df["high"] < df["low"]).sum()
bad_ho = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
bad_lo = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
print(f"\nduplicate timestamps: {dup} | NaN OHLC: {nan_ct} | high<low: {bad_hl} | high<max(o,c): {bad_ho} | low>min(o,c): {bad_lo}")

df.to_csv(OUT_CSV, index=False)
print(f"\nsaved -> {OUT_CSV}")
