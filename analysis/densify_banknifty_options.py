# -*- coding: utf-8 -*-
"""densify_banknifty_options.py — PHASE 2 densification for BankNifty OPTIONS (monthly-only chain), reusing
the exact NIFTY logic (densify_nifty_options.densify_file: fill each active day to a 375-row/day 09:15-15:29
grid; API-omitted minutes -> OHLC=last price, OI ffill, volume=0, is_synthetic=True; real rows
is_synthetic=False; whole no-trade days NOT fabricated; Diwali/Muhurat evening candles kept as real). Same
NSE session hours as NIFTY, so the grid is identical. In-place, resumable (skips files already having
is_synthetic). Only ROOT differs from NIFTY.
"""
import sys, time, glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import densify_nifty_options as dn          # reuse densify_file / densify_df unchanged

ROOT = rb.BASE / "data" / "options_intraday_full" / "BANKNIFTY"
WORKERS = dn.WORKERS


def main():
    files = sorted(glob.glob(str(ROOT / "*" / "*.parquet")))
    print(f"BANKNIFTY OPTIONS densify: {len(files):,} contracts | {WORKERS} processes (in-place, resumable) ...", flush=True)
    t0 = time.time(); done = skip = raw_tot = dense_tot = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(dn.densify_file, fn): fn for fn in files}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                nr, nd = fut.result()
                if nr == 0: skip += 1
                else: done += 1; raw_tot += nr; dense_tot += nd
            except Exception as e:
                print(f"  ERR {Path(futs[fut]).name}: {str(e)[:70]}", flush=True)
            if i % 1000 == 0 or i == len(files):
                print(f"  {i}/{len(files)} | densified {done:,} skip {skip:,} | rows {raw_tot:,}->{dense_tot:,} | {time.time()-t0:.0f}s", flush=True)
    print(f"\nCOMPLETE: densified {done:,} | already-done {skip:,} | rows {raw_tot:,} -> {dense_tot:,} (+{dense_tot-raw_tot:,}) | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
