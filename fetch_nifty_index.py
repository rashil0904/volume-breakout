# -*- coding: utf-8 -*-
"""
fetch_nifty_index.py
====================
Fetch Nifty 50 index 15-min OHLC candles covering the SAME date range as the
existing stock dataset (master_data/*.parquet), via the same Upstox API/auth
already used by the pipeline. Saves to data/nifty_15min_ohlc.csv with the exact
same column schema as the stock candle data.
"""

import io, gzip, json, time
from pathlib import Path
from datetime import date

import requests
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dateutil.relativedelta import relativedelta

import data_loading as dl   # reuse the project's Upstox ACCESS_TOKEN

REPO      = Path(__file__).parent
MASTER    = REPO / "master_data"
OUT_DIR   = REPO / "data"
OUT_CSV   = OUT_DIR / "nifty_15min_ohlc.csv"
IST       = "Asia/Kolkata"
MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
SCHEMA    = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]

# ── STEP 1: derive date range + trading-day list from the existing stock data ──
print("STEP 1 — scanning master_data/ for the trading-day calendar …")
files = sorted(MASTER.glob("*.parquet"))
stock_days = set()
for i, f in enumerate(files, 1):
    ts = pd.to_datetime(pq.read_table(f, columns=["timestamp"]).column("timestamp").to_pandas(),
                        utc=True).dt.tz_convert(IST)
    stock_days.update(ts.dt.date.unique())
    if i % 400 == 0:
        print(f"    …{i}/{len(files)} files scanned")
stock_days = sorted(stock_days)
dmin, dmax = stock_days[0], stock_days[-1]
print(f"    stock dataset: {len(files)} symbols | {len(stock_days):,} trading days "
      f"| {dmin} → {dmax}")

# ── STEP 2: resolve the Nifty 50 index instrument key, then fetch ──
print("\nSTEP 2 — resolving Nifty 50 index instrument key from Upstox master …")
resp = requests.get(MASTER_URL, timeout=60); resp.raise_for_status()
raw = json.load(gzip.GzipFile(fileobj=io.BytesIO(resp.content)))
nifty_key = None
for inst in raw:
    if inst.get("segment") == "NSE_INDEX" and inst.get("trading_symbol", "").strip().lower() == "nifty 50":
        nifty_key = inst["instrument_key"]; break
if nifty_key is None:
    nifty_key = "NSE_INDEX|Nifty 50"   # documented fallback
    print(f"    not found in master; using documented fallback: {nifty_key}")
else:
    print(f"    resolved: {nifty_key}")

HEADERS = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}

def fetch_chunk(cfrom, cto):
    url = f"https://api.upstox.com/v3/historical-candle/{nifty_key}/minutes/15/{cto}/{cfrom}"
    for attempt in range(4):
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json().get("data", {}).get("candles", [])
        if r.status_code == 429 or "UDAPI10005" in r.text:
            time.sleep(2 * (attempt + 1)); continue
        print(f"    [{cfrom}->{cto}] HTTP {r.status_code}: {r.text[:200]}")
        return []
    return []

def month_chunks(a, b):
    cur = a
    while cur <= b:
        end = min(cur + relativedelta(months=1) - relativedelta(days=1), b)
        yield cur.isoformat(), end.isoformat()
        cur = end + relativedelta(days=1)

print(f"\n    fetching 15-min candles {dmin} → {dmax} …")
all_candles = []
chunks = list(month_chunks(dmin, dmax))
for idx, (cf, ct) in enumerate(chunks, 1):
    all_candles.extend(fetch_chunk(cf, ct))
    time.sleep(0.2)
    if idx % 12 == 0 or idx == len(chunks):
        print(f"    …{idx}/{len(chunks)} monthly chunks done ({len(all_candles):,} candles)")

# ── STEP 3: format to the existing schema + save ──
print("\nSTEP 3 — formatting + saving …")
df = pd.DataFrame(all_candles, columns=SCHEMA)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
for c in ("open", "high", "low", "close"):
    df[c] = df[c].astype(float)
df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0).astype("int64")
OUT_DIR.mkdir(exist_ok=True)
df.to_csv(OUT_CSV, index=False)
# also drop a parquet next to the stock data for pipeline parity
df.to_parquet(MASTER / "NIFTY_50_INDEX.parquet", index=False, compression="snappy")

# ── STEP 4: sanity check ──
ts = df["timestamp"].dt.tz_convert(IST)
df["_date"] = ts.dt.date
df["_hm"] = ts.dt.hour * 60 + ts.dt.minute
nifty_days = sorted(df["_date"].unique())
per_day = df.groupby("_date").size()
missing = sorted(set(stock_days) - set(nifty_days))
extra   = sorted(set(nifty_days) - set(stock_days))

print("\n" + "=" * 64)
print("SANITY CHECK — Nifty 50 index 15-min fetch")
print("=" * 64)
print(f"  Saved to               : {OUT_CSV}")
print(f"  Schema                 : {list(df[SCHEMA].columns)}")
print(f"  Date range covered     : {nifty_days[0]} → {nifty_days[-1]}")
print(f"  Trading days (Nifty)   : {len(nifty_days):,}")
print(f"  Trading days (stocks)  : {len(stock_days):,}")
print(f"  Total 15-min candles   : {len(df):,}")
print(f"  Candles/day - counts   : {dict(per_day.value_counts().sort_index())}")
print(f"  First candle time      : {df['_hm'].min()//60:02d}:{df['_hm'].min()%60:02d}   "
      f"Last candle time: {df['_hm'].max()//60:02d}:{df['_hm'].max()%60:02d}")
print(f"  Volume all zero?       : {(df['volume'] == 0).all()}   "
      f"(index volume is typically not meaningful)")
print(f"  Missing days (in stocks, not Nifty): {len(missing)}")
if missing:
    print("     " + ", ".join(str(d) for d in missing[:20]) + (" …" if len(missing) > 20 else ""))
print(f"  Extra days (in Nifty, not stocks)  : {len(extra)}")
if extra:
    print("     " + ", ".join(str(d) for d in extra[:20]) + (" …" if len(extra) > 20 else ""))
print("=" * 64)
