# -*- coding: utf-8 -*-
"""densify_nifty_options.py — PHASE 2: densify each NIFTY option contract to a full regular-session grid
for every ACTIVE day (a day with >=1 real candle). Session window is CAS-aware: 09:15-15:29 (375-min grid)
for days before 2026-08-03, 09:15-15:39 (385-min grid) from 2026-08-03 onward (NSE Closing Auction Session
rollout extended F&O trading by 10 minutes — see nse-closing-auction-session-cas memory). Minutes Upstox
omitted (pre-first-trade, post-last-trade, and interior zero-activity gaps) are FILLED: OHLC = forward/
back-filled last price, OI carried forward, volume=0, is_synthetic=True. Real rows keep their values,
is_synthetic=False. Whole no-trade days are NOT fabricated. In-place rewrite (raw recoverable via
is_synthetic==False). Resumable: skips any file already densified (has is_synthetic column). Test mode:
pass a file path to densify one.
"""
import sys, time, glob, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

ROOT = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
WORKERS = max(1, (os.cpu_count() or 4))                      # CPU-bound -> use all cores
COLS = ["symbol", "strike", "option_type", "expiry_date", "timestamp", "DTE", "open", "high", "low", "close", "volume", "OI", "is_synthetic"]
CAS_START = pd.Timestamp("2026-08-03")                        # session end 15:29 -> 15:39 from this date


def densify_df(df):
    sym = df["symbol"].iloc[0]; strike = df["strike"].iloc[0]; ot = df["option_type"].iloc[0]; exp = df["expiry_date"].iloc[0]
    exp_ts = pd.Timestamp(exp).normalize()
    df = df.sort_values("timestamp").drop_duplicates("timestamp")   # raw re-pulls can have chunk-overlap dups
    out = []
    for day, g in df.groupby(df["timestamp"].dt.normalize()):
        session_end_m = 939 if day >= CAS_START else 929     # 15:39 post-CAS, else 15:29
        m = g["timestamp"].dt.hour * 60 + g["timestamp"].dt.minute
        daytime = g[(m >= 555) & (m <= session_end_m)]        # regular session 09:15 -> session_end (CAS-aware)
        evening = g[(m < 555) | (m > session_end_m)]          # special/Muhurat evening session -> keep as real rows
        if len(daytime):                                     # full-session grid, CAS-aware length
            grid = pd.date_range(day + pd.Timedelta(hours=9, minutes=15), day + pd.Timedelta(minutes=session_end_m), freq="1min")
            gg = daytime.set_index("timestamp").reindex(grid)
            syn = gg["close"].isna(); gg["close"] = gg["close"].ffill().bfill()
            for col in ("open", "high", "low"):
                gg[col] = gg[col].where(~syn, gg["close"])
            gg["volume"] = gg["volume"].fillna(0); gg["OI"] = gg["OI"].ffill().bfill().fillna(0)
            gg["is_synthetic"] = syn.values
            gg = gg.reset_index().rename(columns={"index": "timestamp"})
            out.append(gg)
        if len(evening):                                     # real Muhurat/special-session candles, NOT gridded/synthesised
            ee = evening.copy(); ee["is_synthetic"] = False
            out.append(ee)
    for g in out:
        g["symbol"] = sym; g["strike"] = strike; g["option_type"] = ot; g["expiry_date"] = exp
        g["DTE"] = (exp_ts - g["timestamp"].dt.normalize()).dt.days
    d = pd.concat(out, ignore_index=True).sort_values("timestamp")
    d["volume"] = d["volume"].fillna(0).astype("int64"); d["OI"] = d["OI"].fillna(0).astype("int64"); d["DTE"] = d["DTE"].astype("int64")
    return d[COLS]


def densify_file(fn):
    df = pd.read_parquet(fn)
    if "is_synthetic" in df.columns:
        return 0, 0                                          # already densified
    n_raw = len(df); d = densify_df(df); d.to_parquet(fn, index=False)
    return n_raw, len(d)


def main():
    if len(sys.argv) > 1 and sys.argv[1] != "--run":        # TEST one file
        fn = sys.argv[1]; before = pd.read_parquet(fn)
        d = densify_df(before) if "is_synthetic" not in before.columns else before
        pd.set_option("display.width", 200)
        print(f"TEST {Path(fn).name}: raw {len(before)} -> dense {len(d)} rows (+{len(d)-len(before)})")
        day1 = d[d["timestamp"].dt.normalize() == d["timestamp"].dt.normalize().iloc[0]]
        exp_rows = 385 if day1["timestamp"].dt.normalize().iloc[0] >= CAS_START else 375
        print(f"first active day: {len(day1)} rows (expect {exp_rows}), synthetic {int(day1.is_synthetic.sum())}")
        print(day1.head(3)[["timestamp", "open", "close", "volume", "OI", "is_synthetic"]].to_string(index=False))
        print("..."); print(day1[day1.is_synthetic].head(2)[["timestamp", "open", "close", "volume", "OI", "is_synthetic"]].to_string(index=False))
        return
    files = sorted(glob.glob(str(ROOT / "*" / "*.parquet")))
    print(f"densifying {len(files):,} contracts | {WORKERS} processes (resumable, in-place) ...", flush=True)
    t0 = time.time(); done = skip = raw_tot = dense_tot = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(densify_file, fn): fn for fn in files}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                nr, nd = fut.result()
                if nr == 0:
                    skip += 1
                else:
                    done += 1; raw_tot += nr; dense_tot += nd
            except Exception as e:
                print(f"  ERR {Path(futs[fut]).name}: {str(e)[:70]}", flush=True)
            if i % 1000 == 0 or i == len(files):
                print(f"  {i}/{len(files)} | densified {done:,} skip {skip:,} | rows {raw_tot:,}->{dense_tot:,} | {time.time()-t0:.0f}s", flush=True)
    print(f"\nCOMPLETE: densified {done:,} | already-done {skip:,} | rows {raw_tot:,} -> {dense_tot:,} (+{dense_tot-raw_tot:,}) | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
