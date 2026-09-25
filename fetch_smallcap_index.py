# -*- coding: utf-8 -*-
"""
fetch_smallcap_index.py
=======================
Fetch NIFTY SMALLCAP 100 and NIFTY SMALLCAP 250 15-min OHLC candles covering the SAME
date range as the existing stock dataset (master_data/*.parquet), via the same Upstox
v3 API/auth the pipeline already uses for CNXMIDCAP100. Saves each to
  data/cnxsmallcap100_15min_ohlc.csv
  data/cnxsmallcap250_15min_ohlc.csv
with the same column schema as the stock/index candle data.

Upstox NSE index master names them 'NIFTY SMLCAP 100' / 'NIFTY SMLCAP 250'
(instrument_key NSE_INDEX|NIFTY SMLCAP 100 / …250). Index volume is not meaningful.
"""

import io, gzip, json, time
from pathlib import Path

import requests
import pandas as pd
import pyarrow.parquet as pq
from dateutil.relativedelta import relativedelta

import data_loading as dl   # reuse the project's Upstox ACCESS_TOKEN

REPO       = Path(__file__).parent
MASTER     = REPO / "master_data"
OUT_DIR    = REPO / "data"
IST        = "Asia/Kolkata"
MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
SCHEMA     = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]
# (index name in Upstox master, output filename)
TARGETS = [
    ("NIFTY SMLCAP 100", OUT_DIR / "cnxsmallcap100_15min_ohlc.csv"),
    ("NIFTY SMLCAP 250", OUT_DIR / "cnxsmallcap250_15min_ohlc.csv"),
]

# ── STEP 1: derive date range + trading-day list from the existing stock data ──
print("STEP 1 — scanning master_data/ for the trading-day calendar …")
files = [f for f in sorted(MASTER.glob("*.parquet")) if "INDEX" not in f.stem.upper()]
stock_days = set()
for i, f in enumerate(files, 1):
    ts = pd.to_datetime(pq.read_table(f, columns=["timestamp"]).column("timestamp").to_pandas(),
                        utc=True).dt.tz_convert(IST)
    stock_days.update(ts.dt.date.unique())
    if i % 400 == 0:
        print(f"    …{i}/{len(files)} files scanned")
stock_days = sorted(stock_days)
dmin, dmax = stock_days[0], stock_days[-1]
print(f"    stock dataset: {len(files)} symbols | {len(stock_days):,} trading days | {dmin} → {dmax}")

# ── resolve index instrument keys from the Upstox NSE master (once) ──
print("\nResolving smallcap index instrument keys from the Upstox NSE master …")
resp = requests.get(MASTER_URL, timeout=90); resp.raise_for_status()
raw = json.load(gzip.GzipFile(fileobj=io.BytesIO(resp.content)))
keys = {}
for name, _ in TARGETS:
    k = None
    for inst in raw:
        if inst.get("segment") != "NSE_INDEX":
            continue
        if name.upper() in (str(inst.get("name", "")).upper(),
                            str(inst.get("trading_symbol", "")).upper()):
            k = inst["instrument_key"]; break
    if k is None:
        raise SystemExit(f"'{name}' NOT found in the Upstox NSE index master — different source needed.")
    keys[name] = k
    print(f"    {name:20s} -> {k}")

HEADERS = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}


def fetch_chunk(key, cfrom, cto):
    url = f"https://api.upstox.com/v3/historical-candle/{key}/minutes/15/{cto}/{cfrom}"
    for attempt in range(5):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException as e:
            print(f"    [{cfrom}->{cto}] connection error ({type(e).__name__}), retrying …")
            time.sleep(3 * (attempt + 1)); continue
        if r.status_code == 200:
            return r.json().get("data", {}).get("candles", [])
        if r.status_code == 429 or "UDAPI10005" in r.text:
            time.sleep(2 * (attempt + 1)); continue
        print(f"    [{cfrom}->{cto}] HTTP {r.status_code}: {r.text[:200]}")
        return []
    print(f"    [{cfrom}->{cto}] GAVE UP after retries")
    return []


def month_chunks(a, b):
    cur = a
    while cur <= b:
        end = min(cur + relativedelta(months=1) - relativedelta(days=1), b)
        yield cur.isoformat(), end.isoformat()
        cur = end + relativedelta(days=1)


def fetch_and_save(name, key, out_csv):
    print(f"\n{'='*66}\nFETCH — {name}  ({key})\n{'='*66}")
    all_candles = []
    chunks = list(month_chunks(dmin, dmax))
    for idx, (cf, ct) in enumerate(chunks, 1):
        all_candles.extend(fetch_chunk(key, cf, ct))
        time.sleep(0.2)
        if idx % 12 == 0 or idx == len(chunks):
            print(f"    …{idx}/{len(chunks)} monthly chunks done ({len(all_candles):,} candles)")

    df = pd.DataFrame(all_candles, columns=SCHEMA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0).astype("int64")
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)

    # ── sanity check ──
    ts = df["timestamp"].dt.tz_convert(IST)
    df["_date"] = ts.dt.date
    df["_hm"] = ts.dt.hour * 60 + ts.dt.minute
    idx_days = sorted(df["_date"].unique())
    per_day = df.groupby("_date").size()
    missing = sorted(set(stock_days) - set(idx_days))
    extra = sorted(set(idx_days) - set(stock_days))
    print(f"  Saved to              : {out_csv}")
    print(f"  Schema                : {list(df[SCHEMA].columns)}")
    print(f"  Date range covered    : {idx_days[0]} → {idx_days[-1]}")
    print(f"  Trading days (index)  : {len(idx_days):,}   (stocks: {len(stock_days):,})")
    print(f"  Total 15-min candles  : {len(df):,}")
    print(f"  Candles/day counts    : {dict(per_day.value_counts().sort_index())}")
    print(f"  First / last candle   : {df['_hm'].min()//60:02d}:{df['_hm'].min()%60:02d}"
          f" / {df['_hm'].max()//60:02d}:{df['_hm'].max()%60:02d}")
    print(f"  Volume all zero?      : {(df['volume'] == 0).all()}  (index volume not meaningful)")
    print(f"  Missing days (stocks, not index): {len(missing)}"
          + ("  -> " + ", ".join(str(d) for d in missing[:15]) + (" …" if len(missing) > 15 else "") if missing else ""))
    print(f"  Extra days (index, not stocks)  : {len(extra)}"
          + ("  -> " + ", ".join(str(d) for d in extra[:15]) + (" …" if len(extra) > 15 else "") if extra else ""))
    return len(df), len(idx_days), len(missing), len(extra)


summary = []
for name, out_csv in TARGETS:
    summary.append((name, *fetch_and_save(name, keys[name], out_csv)))

print("\n" + "=" * 66)
print("ALL DONE — smallcap index fetch summary")
print("=" * 66)
for name, ncand, ndays, nmiss, nextra in summary:
    print(f"  {name:20s} | {ncand:>7,} candles | {ndays:>4} days | missing {nmiss} | extra {nextra}")
print("=" * 66)
