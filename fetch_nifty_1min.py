# -*- coding: utf-8 -*-
"""
fetch_nifty_1min.py — fetch Nifty-50 SPOT 1-minute OHLC from Upstox v3, SAME instrument
(NSE_INDEX|Nifty 50) as the 15-min series used for the hourly RSI (spot-vs-spot consistency,
flag g). Range = the 15-min dataset's span so hold-till-hit trades can always resolve.
Saves data/nifty_1min_ohlc.csv with the same schema as nifty_15min_ohlc.csv.
"""
import time
from pathlib import Path
from datetime import date
import requests
import pandas as pd
from dateutil.relativedelta import relativedelta
import data_loading as dl

REPO = Path(__file__).parent
OUT_CSV = REPO / "data" / "nifty_1min_ohlc.csv"
IST = "Asia/Kolkata"
KEY = "NSE_INDEX|Nifty 50"
SCHEMA = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]
HEADERS = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}

# span of the existing 15-min series (the signal series)
r15 = pd.read_csv(REPO / "data" / "nifty_15min_ohlc.csv")
ts15 = pd.to_datetime(r15["timestamp"], utc=True).dt.tz_convert(IST)
DMIN, DMAX = ts15.dt.date.min(), ts15.dt.date.max()
print(f"fetching 1-min Nifty {DMIN} -> {DMAX}")


def fetch_chunk(cfrom, cto):
    url = f"https://api.upstox.com/v3/historical-candle/{KEY}/minutes/1/{cto}/{cfrom}"
    for attempt in range(5):
        r = requests.get(url, headers=HEADERS, timeout=40)
        if r.status_code == 200:
            return r.json().get("data", {}).get("candles", [])
        if r.status_code == 429 or "UDAPI10005" in r.text:
            time.sleep(2 * (attempt + 1)); continue
        print(f"    [{cfrom}->{cto}] HTTP {r.status_code}: {r.text[:160]}")
        return []
    return []


def month_chunks(a, b):
    cur = a
    while cur <= b:
        end = min(cur + relativedelta(months=1) - relativedelta(days=1), b)
        yield cur.isoformat(), end.isoformat()
        cur = end + relativedelta(days=1)


chunks = list(month_chunks(DMIN, DMAX))
all_c = []
for i, (cf, ct) in enumerate(chunks, 1):
    all_c.extend(fetch_chunk(cf, ct))
    time.sleep(0.25)
    if i % 6 == 0 or i == len(chunks):
        print(f"    {i}/{len(chunks)} monthly chunks | {len(all_c):,} candles")

df = pd.DataFrame(all_c, columns=SCHEMA)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
for c in ("open", "high", "low", "close"):
    df[c] = df[c].astype(float)
df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0).astype("int64")
df.to_csv(OUT_CSV, index=False)

ts = df["timestamp"].dt.tz_convert(IST)
hm = ts.dt.hour * 60 + ts.dt.minute
per_day = df.groupby(ts.dt.date).size()
print("=" * 60)
print(f"saved {len(df):,} 1-min candles -> {OUT_CSV}")
print(f"range {ts.dt.date.min()} -> {ts.dt.date.max()} | {df.groupby(ts.dt.date).ngroups:,} days")
print(f"first candle {hm.min()//60:02d}:{hm.min()%60:02d}  last {hm.max()//60:02d}:{hm.max()%60:02d}")
print(f"candles/day median {int(per_day.median())}  min {per_day.min()}  max {per_day.max()}")
