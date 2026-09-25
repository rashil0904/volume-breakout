# -*- coding: utf-8 -*-
"""densify_banknifty_futures.py — PHASE 2 densification for BankNifty futures (1 contract per expiry, no
strike/CE-PE dimension). Same convention as densify_nifty_options.py: fill each ACTIVE day (>=1 real candle)
to a full 375-row 09:15-15:29 regular-session grid; minutes Upstox omitted at source -> OHLC=last price
(ffill/bfill), OI carried forward, volume=0, is_synthetic=True. Real rows keep is_synthetic=False. Rows
outside the regular window (Diwali Muhurat evenings 2024-11-01 18:00-18:59 / shortened Muhurat afternoon
2025-10-21 13:45-14:44, and the Aug-2026 contract's real 15:30-15:39 extension) are kept AS REAL ROWS,
NOT gridded/synthesised. Whole no-trade days (e.g. the Nov-2024 contract's confirmed mid-2024 gap) are NOT
fabricated. In-place, resumable (skips files that already have is_synthetic).
"""
import sys, glob, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

ROOT = rb.BASE / "data" / "futures_intraday_full" / "BANKNIFTY"
WORKERS = 4
COLS = ["contract_month", "expiry_date", "symbol", "timestamp", "DTE", "open", "high", "low", "close", "volume", "OI", "is_synthetic"]


def densify_df(df):
    cm = df["contract_month"].iloc[0]; exp = df["expiry_date"].iloc[0]; sym = df["symbol"].iloc[0]
    exp_ts = pd.Timestamp(exp).normalize()
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    out = []
    for day, g in df.groupby(df["timestamp"].dt.normalize()):
        m = g["timestamp"].dt.hour * 60 + g["timestamp"].dt.minute
        daytime = g[(m >= 555) & (m <= 929)]                 # regular session 09:15-15:29
        extra = g[(m < 555) | (m > 929)]                     # Muhurat evening / extended session -> keep as real, untouched
        if len(daytime):
            grid = pd.date_range(day + pd.Timedelta(hours=9, minutes=15), day + pd.Timedelta(hours=15, minutes=29), freq="1min")
            gg = daytime.set_index("timestamp").reindex(grid)
            syn = gg["close"].isna(); gg["close"] = gg["close"].ffill().bfill()
            for col in ("open", "high", "low"):
                gg[col] = gg[col].where(~syn, gg["close"])
            gg["volume"] = gg["volume"].fillna(0); gg["OI"] = gg["OI"].ffill().bfill().fillna(0)
            gg["is_synthetic"] = syn.values
            gg = gg.reset_index().rename(columns={"index": "timestamp"})
            out.append(gg)
        if len(extra):
            ee = extra.copy(); ee["is_synthetic"] = False
            out.append(ee)
    for g in out:
        g["contract_month"] = cm; g["expiry_date"] = exp; g["symbol"] = sym
        g["DTE"] = (exp_ts - g["timestamp"].dt.normalize()).dt.days
    d = pd.concat(out, ignore_index=True).sort_values("timestamp")
    d["volume"] = d["volume"].fillna(0).astype("int64"); d["OI"] = d["OI"].fillna(0).astype("int64"); d["DTE"] = d["DTE"].astype("int64")
    return d[COLS]


def densify_file(fn):
    df = pd.read_parquet(fn)
    if "is_synthetic" in df.columns:
        return 0, 0
    n_raw = len(df); d = densify_df(df); d.to_parquet(fn, index=False)
    return n_raw, len(d)


def main():
    files = sorted(glob.glob(str(ROOT / "*" / "*.parquet")))
    print(f"densifying {len(files)} BankNifty futures contracts | {WORKERS} processes (resumable, in-place) ...", flush=True)
    t0 = time.time(); done = skip = raw_tot = dense_tot = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(densify_file, fn): fn for fn in files}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                nr, nd = fut.result()
                if nr == 0: skip += 1
                else: done += 1; raw_tot += nr; dense_tot += nd
            except Exception as e:
                print(f"  ERR {Path(futs[fut]).name}: {str(e)[:80]}", flush=True)
            print(f"  {i}/{len(files)} | densified {done} skip {skip} | rows {raw_tot:,}->{dense_tot:,} | {time.time()-t0:.0f}s", flush=True)
    print(f"\nCOMPLETE: densified {done} | already-done {skip} | rows {raw_tot:,} -> {dense_tot:,} (+{dense_tot-raw_tot:,}) | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
