#!/usr/bin/env python3
"""
fetch_be_symbols.py — Targeted fetch for symbols previously unresolved
due to instrument_type == "BE" (trade-to-trade category).

Finds all symbols in unresolved_symbols.txt that:
  1. Have an existing checkpoint CSV (meaning we had data before)
  2. Are now resolvable via ISIN in the Upstox master (EQ or BE types)

Then incrementally fetches any missing historical candles + today's intraday
and updates master_data/<SYMBOL>.parquet.
"""

import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from data_loading import (
    download_nse_instrument_master,
    load_symbols_and_isins_from_csv,
    fetch_historical_data,
    fetch_intraday_today,
    _candles_to_df,
    update_master_parquet,
    UNIT, INTERVAL, FROM_DATE, TO_DATE,
    CHECKPOINT_DIR, MASTER_DIR,
)

# ── Resolve which symbols to fetch ───────────────────────────────────────────

unresolved_path = BASE / "unresolved_symbols.txt"
if not unresolved_path.exists():
    print("unresolved_symbols.txt not found — nothing to do.")
    sys.exit(0)

unresolved = unresolved_path.read_text().strip().splitlines()
has_checkpoint = [s for s in unresolved if (CHECKPOINT_DIR / f"{s}.csv").exists()]
print(f"Unresolved symbols:    {len(unresolved)}")
print(f"With checkpoint data:  {len(has_checkpoint)}")

# Load Companies List ISINs
csv_symbol_isin = load_symbols_and_isins_from_csv(BASE / "Companies List.csv")

# Download master (now includes BE type)
master_by_symbol, master_by_isin = download_nse_instrument_master()

# Resolve instrument keys for checkpoint symbols
to_fetch = {}
still_unresolved = []
for s in has_checkpoint:
    isin = str(csv_symbol_isin.get(s, ""))
    entry = master_by_symbol.get(s)
    if entry is None and isin and isin.startswith("IN"):
        entry = master_by_isin.get(isin)
    if entry:
        to_fetch[s] = entry["instrument_key"]
    else:
        still_unresolved.append(s)

print(f"\nResolved for fetch:    {len(to_fetch)}")
print(f"Still unresolved:      {len(still_unresolved)}")
if still_unresolved:
    print(f"  {still_unresolved}")
print()

# ── Fetch loop ────────────────────────────────────────────────────────────────

failed = []
intraday_added = 0
t0 = time.time()

for i, (symbol, instrument_key) in enumerate(to_fetch.items(), start=1):
    print(f"[{i}/{len(to_fetch)}] {symbol}  ({instrument_key})")
    try:
        df = fetch_historical_data(
            symbol, instrument_key, UNIT, INTERVAL, FROM_DATE, TO_DATE
        )
        if df.empty:
            failed.append(symbol)
            continue

        intraday_candles = fetch_intraday_today(instrument_key)
        if intraday_candles:
            intraday_df = _candles_to_df(intraday_candles)
            before = len(df)
            df = pd.concat([df, intraday_df], ignore_index=True)
            df = (df.drop_duplicates(subset="timestamp")
                    .sort_values("timestamp")
                    .reset_index(drop=True))
            added = len(df) - before
            if added > 0:
                print(f"    {symbol}: +{added} intraday candles for today")
                intraday_added += added

        update_master_parquet(symbol, df)

    except Exception as e:
        print(f"  ERROR for {symbol}: {e}")
        failed.append(symbol)

# ── Summary ───────────────────────────────────────────────────────────────────

elapsed = time.time() - t0
parquet_count = len(list(MASTER_DIR.glob("*.parquet")))
print("\n" + "=" * 60)
print(f"BE-symbol fetch complete  ({elapsed:.1f}s)")
print(f"  Fetched:           {len(to_fetch) - len(failed)}")
print(f"  Intraday added:    {intraday_added:,}")
print(f"  master_data/:      {parquet_count} parquet files")
if failed:
    print(f"  Failed:  {failed}")
if still_unresolved:
    print(f"  Still unresolved (no instrument key found): {still_unresolved}")
print("=" * 60)
