# -*- coding: utf-8 -*-
"""build_daily_vwapclose.py — daily OHLCV where CLOSE = VWAP of the last 30 one-min candles
(15:00-15:29), typical price (H+L+C)/3, volume-weighted — the NSE official-close convention used by
prepare_data.py. open/high/low = full regular session [09:15,15:30); volume = session sum.
Saves data/daily_ohlcv_vwapclose.parquet.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

MASTER = rb.BASE / "master_data"
OUT = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"
IST = "Asia/Kolkata"

files = sorted(MASTER.glob("*.parquet"))
print(f"building VWAP-close daily for {len(files):,} symbols")
frames = []; t0 = time.time()
for i, f in enumerate(files, 1):
    sym = f.stem
    tb = pq.read_table(f, columns=["timestamp", "open", "high", "low", "close", "volume"]).to_pandas()
    ts = pd.to_datetime(tb["timestamp"], utc=True).dt.tz_convert(IST)
    hm = ts.dt.hour * 60 + ts.dt.minute
    tb = tb.assign(date=ts.dt.date, hm=hm)
    reg = tb[(tb["hm"] >= 555) & (tb["hm"] < 930)]
    if reg.empty:
        continue
    g = reg.groupby("date")
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                     "low": g["low"].min(), "volume": g["volume"].sum()})
    # VWAP close over 15:00-15:29 (hm 900..929), tp=(H+L+C)/3
    last30 = reg[(reg["hm"] >= 900) & (reg["hm"] <= 929)].copy()
    last30["tpv"] = (last30["high"] + last30["low"] + last30["close"]) / 3 * last30["volume"]
    lg = last30.groupby("date")[["tpv", "volume"]].sum()
    vwap = (lg["tpv"] / lg["volume"]).where(lg["volume"] > 0, np.nan)
    d["close"] = vwap
    # fallback: days with zero last-30 volume -> use last regular close
    lastclose = g["close"].last()
    d["close"] = d["close"].fillna(lastclose)
    d.insert(0, "symbol", sym); d = d.reset_index()
    frames.append(d[["symbol", "date", "open", "high", "low", "close", "volume"]])
    if i % 300 == 0 or i == len(files):
        print(f"  {i}/{len(files)} | {sum(len(x) for x in frames):,} rows | {time.time()-t0:.0f}s", flush=True)

alld = pd.concat(frames, ignore_index=True)
alld["date"] = pd.to_datetime(alld["date"])
alld = alld.sort_values(["symbol", "date"]).reset_index(drop=True)
alld.to_parquet(OUT, index=False, compression="snappy")
print(f"saved {len(alld):,} rows, {alld['symbol'].nunique():,} symbols -> {OUT}")
print(f"date range {alld['date'].min().date()} -> {alld['date'].max().date()}")
