# -*- coding: utf-8 -*-
"""fix_special_session.py — trim OVER-SYNTHESISED closed-hour rows on SHORTENED special sessions. The
densifier grids every active day to the full 09:15-15:29 (375-min) window; on a shortened session (e.g.
Diwali-2025 Muhurat, 2025-10-21, market open only 13:45-14:44) that fabricated ~315 synthetic rows for
hours the market was CLOSED. This drops synthetic rows outside the actual session window on those dates
(real candles are within it and untouched). In-place, no API, no re-pull. Reports rows removed.
"""
import sys, time, glob
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

ROOT = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
# special shortened sessions: date -> (session_start_minute, session_end_minute)  [minute = hh*60+mm]
SPECIAL = {pd.Timestamp("2025-10-21").date(): (13 * 60 + 45, 14 * 60 + 44)}   # Diwali-2025 Muhurat 13:45-14:44


def main():
    # only expiries that could be alive on a special date (>= special date, within ~150d)
    dirs = sorted([d for d in ROOT.iterdir() if d.is_dir() and "20251021" <= d.name <= "20260401"])
    files = [f for d in dirs for f in glob.glob(str(d / "*.parquet"))]
    print(f"scanning {len(files):,} files in expiries 2025-10-28..2026-03-31 for special-session over-synthesis ...", flush=True)
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
            print(f"  {i}/{len(files)} | contracts trimmed {fixed} | synthetic closed-hour rows removed {removed:,} | {time.time()-t0:.0f}s", flush=True)
    print(f"\nCOMPLETE: trimmed {fixed} contracts | removed {removed:,} fabricated closed-hour rows | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
