# -*- coding: utf-8 -*-
"""sensex_spot_pull.py — pull SENSEX (BSE_INDEX|SENSEX) spot 1-min OHLC via Upstox regular historical-candle
v2, Oct-2024..2026-07-31, monthly chunks. Saves data/sensex_1min_ohlc.csv matching NIFTY spot schema
(timestamp,open,high,low,close,volume,open_interest). Index has no volume/OI -> 0. Idempotent (rewrites file)."""
import sys, time
from pathlib import Path
import requests, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import opt_pull_nifty_full as op
import run_backtest as rb

KEY = "BSE_INDEX|SENSEX"; OUT = rb.BASE / "data" / "sensex_1min_ohlc.csv"
H = op.H; enc = op.enc


def get(url):
    for a in range(5):
        try: r = requests.get(url, headers=H, timeout=40)
        except Exception: time.sleep(1.0 * (a + 1)); continue
        if r.status_code == 200:
            d = r.json().get("data", {}); return d.get("candles", []) if isinstance(d, dict) else []
        if r.status_code in (429, 500, 502, 503): time.sleep(1.3 * (a + 1)); continue
        return None
    return None


def main():
    months = pd.date_range("2024-10-01", "2026-07-31", freq="MS")
    rows = []
    for m in months:
        lo = m.date(); hi = (m + pd.offsets.MonthEnd(1)).date()
        if str(hi) > "2026-07-31": hi = pd.Timestamp("2026-07-31").date()
        c = get(f"https://api.upstox.com/v2/historical-candle/{enc(KEY)}/1minute/{hi.isoformat()}/{lo.isoformat()}")
        n = len(c) if c else 0; rows += (c or [])
        print(f"  {lo}..{hi}: {n} candles", flush=True); time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df.to_csv(OUT, index=False)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    print(f"\nSAVED {len(df):,} rows -> {OUT} | {ts.min()} .. {ts.max()} | trading days {ts.dt.normalize().nunique()}")


if __name__ == "__main__":
    main()
