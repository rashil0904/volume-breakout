# -*- coding: utf-8 -*-
"""
build_daily_from_1min.py — aggregate the 1-min master_data parquets into ONE cached daily OHLCV
table (data/daily_ohlcv_all.parquet) for the daily-timeframe strategies. Regular session only
([09:15, 15:30)). Per symbol/day: open=first, high=max, low=min, close=last, volume=sum.
Run once; the daily strategy reads the cache.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

MASTER = rb.BASE / "master_data"
OUT = rb.BASE / "data" / "daily_ohlcv_all.parquet"
IST = "Asia/Kolkata"

files = sorted(MASTER.glob("*.parquet"))
print(f"aggregating {len(files):,} symbols -> daily")
frames = []
t0 = time.time()
for i, f in enumerate(files, 1):
    sym = f.stem
    tb = pq.read_table(f, columns=["timestamp", "open", "high", "low", "close", "volume"]).to_pandas()
    ts = pd.to_datetime(tb["timestamp"], utc=True).dt.tz_convert(IST)
    hm = ts.dt.hour * 60 + ts.dt.minute
    keep = (hm >= 555) & (hm < 930)
    tb = tb[keep]; ts = ts[keep]
    if tb.empty:
        continue
    tb = tb.assign(date=ts.dt.date)
    g = tb.groupby("date")
    d = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum()})
    d.insert(0, "symbol", sym)
    d = d.reset_index()
    frames.append(d)
    if i % 200 == 0 or i == len(files):
        print(f"  {i}/{len(files)} | {sum(len(x) for x in frames):,} daily rows | {time.time()-t0:.0f}s")

alld = pd.concat(frames, ignore_index=True)
alld["date"] = pd.to_datetime(alld["date"])
alld = alld.sort_values(["symbol", "date"]).reset_index(drop=True)
alld.to_parquet(OUT, index=False, compression="snappy")
print("=" * 60)
print(f"saved {len(alld):,} daily rows, {alld['symbol'].nunique():,} symbols -> {OUT}")
print(f"date range {alld['date'].min().date()} -> {alld['date'].max().date()}")
