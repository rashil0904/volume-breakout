# -*- coding: utf-8 -*-
"""fix_special_session_sensex.py — SENSEX counterpart of fix_special_session.py. The densifier gridded the
Diwali-2025 Muhurat day (2025-10-21, BSE open only 13:45-14:44) to a full 375-min window, fabricating ~315
synthetic closed-hour rows/contract. This drops synthetic rows OUTSIDE the actual session window on that date
(real candles are within it and untouched). 2024-11-01 Diwali evening (18:00-18:59) was handled correctly by
densify (0 synthetic) so it is NOT in SPECIAL. In-place, no API/re-pull. A contract with a 2025-10-21 row must
have expiry in [2025-10-21 .. ~+150d] (pull LOOKBACK), so only those folders are scanned.
"""
import sys, time, glob
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

ROOT = rb.BASE / "data" / "options_intraday_full" / "SENSEX"
SPECIAL = {pd.Timestamp("2025-10-21").date(): (13 * 60 + 45, 14 * 60 + 44)}   # Diwali-2025 Muhurat 13:45-14:44


def main():
    dirs = sorted([d for d in ROOT.iterdir() if d.is_dir() and "20251021" <= d.name <= "20260401"])
    files = [f for d in dirs for f in glob.glob(str(d / "*.parquet"))]
    print(f"scanning {len(files):,} files (expiries 2025-10-21..2026-04-01) for Muhurat over-synthesis ...", flush=True)
    t0 = time.time(); fixed = 0; removed = 0
    for i, fn in enumerate(files, 1):
        d = pd.read_parquet(fn)
        dts = d["timestamp"].dt.date; mn = d["timestamp"].dt.hour * 60 + d["timestamp"].dt.minute
        drop = pd.Series(False, index=d.index)
        for sd, (lo, hi) in SPECIAL.items():
            drop |= (dts == sd) & d["is_synthetic"].astype(bool) & ((mn < lo) | (mn > hi))
        if drop.any():
            removed += int(drop.sum()); d[~drop].to_parquet(fn, index=False); fixed += 1
        if i % 1000 == 0 or i == len(files):
            print(f"  {i}/{len(files)} | contracts trimmed {fixed} | closed-hour synthetic rows removed {removed:,} | {time.time()-t0:.0f}s", flush=True)
    print(f"\nCOMPLETE: trimmed {fixed} contracts | removed {removed:,} fabricated closed-hour rows | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
