#!/usr/bin/env python3
"""
data_loading.py — Daily update pipeline
----------------------------------------
Step A: Reads Companies List.csv → resolves instrument keys via Upstox master.
Step B: Fetches new/missing 15-min candles (incremental — only new rows since
        last checkpoint date). Appends to checkpoints/<SYMBOL>.csv with
        deduplication on timestamp.
Step C: Updates master_data/<SYMBOL>.parquet (same dedup/sort logic).
        On first run (master_data/ empty), builds all parquets from existing
        checkpoint CSVs without any API calls.

Rate limits respected: 6 req/sec (well within 50/sec, 500/min, 2000/30min).
"""

import io
import gzip
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta

# ── CONFIG ────────────────────────────────────────────────────────────────────
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiJEUjEyOTMiLCJqdGkiOiI2YTUxM2I4OGUzNDQ3MjQwMjRmNmQ5ODUiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlzRXh0ZW5kZWQiOnRydWUsImlhdCI6MTc4MzcwODU1MiwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxODE1MjU2ODAwfQ.b2kgnyorOf1XQgIiXyvZmRnu74rmMXlQ7jDbM2G2nHo"
COMPANIES_CSV_PATH = "Companies List.csv"

UNIT           = "minutes"
INTERVAL       = "15"
FROM_DATE      = "2022-01-01"
TO_DATE        = str(date.today())
CHUNK_MONTHS   = 1
REQUESTS_PER_SECOND = 6
MAX_RETRIES    = 5
INITIAL_BACKOFF_SEC = 2
OVERLAP_DAYS   = 7   # days of overlap when fetching incrementally (safety buffer)

BASE           = Path(__file__).parent
CHECKPOINT_DIR = BASE / "checkpoints"
MASTER_DIR     = BASE / "master_data"
INSTRUMENT_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR.mkdir(exist_ok=True)
MASTER_DIR.mkdir(exist_ok=True)

_min_gap = 1.0 / REQUESTS_PER_SECOND
_last_request_time = [0.0]


def throttle():
    now = time.monotonic()
    elapsed = now - _last_request_time[0]
    if elapsed < _min_gap:
        time.sleep(_min_gap - elapsed)
    _last_request_time[0] = time.monotonic()


# ── Instrument resolution ─────────────────────────────────────────────────────

def load_symbols_and_isins_from_csv(path):
    raw = pd.read_csv(path, header=None, encoding="utf-8-sig")
    df = raw.iloc[:, :2].copy()
    df.columns = ["symbol", "isin"]
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["isin"]   = df["isin"].astype(str).str.strip()
    df = df[df["symbol"].notna() & (df["symbol"] != "") & (df["symbol"] != "nan")]
    df = df.drop_duplicates(subset="symbol")
    valid_isin_mask = df["isin"].str.match(r"^IN[A-Z0-9]{10}$")
    df.loc[~valid_isin_mask, "isin"] = ""
    return dict(zip(df["symbol"], df["isin"]))


def download_nse_instrument_master():
    print("Downloading Upstox NSE instrument master...")
    resp = requests.get(INSTRUMENT_MASTER_URL)
    resp.raise_for_status()
    with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as f:
        raw_instruments = json.load(f)
    by_symbol = {}
    by_isin   = {}
    for inst in raw_instruments:
        if inst.get("segment") == "NSE_EQ" and inst.get("instrument_type") in ("EQ", "BE"):
            entry = {
                "instrument_key": inst["instrument_key"],
                "isin": inst.get("isin", ""),
            }
            by_symbol[inst["trading_symbol"]] = entry
            isin = inst.get("isin", "")
            if isin:
                by_isin[isin] = entry
    print(f"  Loaded {len(by_symbol)} NSE equity instruments (EQ + BE).")
    return by_symbol, by_isin


def build_instruments(csv_path):
    csv_symbol_isin = load_symbols_and_isins_from_csv(csv_path)
    master_by_symbol, master_by_isin = download_nse_instrument_master()
    instruments = {}
    mismatches  = []
    unresolved  = []
    for symbol, csv_isin in csv_symbol_isin.items():
        master_entry = master_by_symbol.get(symbol)
        if master_entry is None and csv_isin and csv_isin.startswith("IN"):
            master_entry = master_by_isin.get(csv_isin)
        if master_entry is None:
            unresolved.append(symbol)
            continue
        instruments[symbol] = master_entry["instrument_key"]
        master_isin = master_entry["isin"]
        if csv_isin and master_isin and csv_isin != master_isin:
            mismatches.append({"symbol": symbol, "csv_isin": csv_isin, "master_isin": master_isin})
    return instruments, mismatches, unresolved


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def date_chunks(from_date_str, to_date_str, months=1):
    start = date.fromisoformat(from_date_str)
    end   = date.fromisoformat(to_date_str)
    cur   = start
    while cur <= end:
        chunk_end = min(cur + relativedelta(months=months) - relativedelta(days=1), end)
        yield (cur.isoformat(), chunk_end.isoformat())
        cur = chunk_end + relativedelta(days=1)


def fetch_one_chunk(instrument_key, unit, interval, chunk_from, chunk_to):
    url = (
        f"https://api.upstox.com/v3/historical-candle/"
        f"{instrument_key}/{unit}/{interval}/{chunk_to}/{chunk_from}"
    )
    headers = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    backoff = INITIAL_BACKOFF_SEC
    for attempt in range(1, MAX_RETRIES + 1):
        throttle()
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            return resp.json().get("data", {}).get("candles", [])
        is_rate_limited = (
            resp.status_code == 429
            or "Too Many Request" in resp.text
            or "UDAPI10005" in resp.text
        )
        if is_rate_limited and attempt < MAX_RETRIES:
            print(f"      Rate limited on [{chunk_from}->{chunk_to}], retrying in {backoff}s")
            time.sleep(backoff)
            backoff *= 2
            continue
        print(f"      [{chunk_from}->{chunk_to}] HTTP {resp.status_code}: {resp.text[:200]}")
        return []
    return []


def fetch_intraday_today(instrument_key):
    """Fetch today's live intraday candles via the intraday endpoint."""
    url = f"https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/{UNIT}/{INTERVAL}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    backoff = INITIAL_BACKOFF_SEC
    for attempt in range(1, MAX_RETRIES + 1):
        throttle()
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            return resp.json().get("data", {}).get("candles", [])
        is_rate_limited = (
            resp.status_code == 429
            or "Too Many Request" in resp.text
            or "UDAPI10005" in resp.text
        )
        if is_rate_limited and attempt < MAX_RETRIES:
            time.sleep(backoff)
            backoff *= 2
            continue
        print(f"      [intraday] HTTP {resp.status_code}: {resp.text[:200]}")
        return []
    return []


def _candles_to_df(candles):
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)


# ── Checkpoint (CSV) management ───────────────────────────────────────────────

def checkpoint_path(symbol):
    return CHECKPOINT_DIR / f"{symbol}.csv"


def load_checkpoint(symbol):
    path = checkpoint_path(symbol)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"])


def save_checkpoint(symbol, df):
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_csv(checkpoint_path(symbol), index=False)


def fetch_historical_data(symbol, instrument_key, unit, interval, from_date, to_date):
    """
    Fetch candles for a symbol, incrementally if a checkpoint already exists.
    Returns the complete (existing + new) DataFrame, or empty if nothing was fetched.
    """
    existing = load_checkpoint(symbol)

    if not existing.empty:
        last_ts    = pd.to_datetime(existing["timestamp"]).max()
        fetch_from = (last_ts - pd.Timedelta(days=OVERLAP_DAYS)).date().isoformat()
        if fetch_from >= to_date:
            print(f"    {symbol} already up to date (last candle: {last_ts.date()}), skipping.")
            return existing
        print(f"    {symbol}: fetching {fetch_from} → {to_date} (last candle was {last_ts.date()})")
    else:
        fetch_from = from_date
        print(f"    {symbol}: full fetch {fetch_from} → {to_date}")

    all_candles = []
    chunks = list(date_chunks(fetch_from, to_date, CHUNK_MONTHS))
    for idx, (chunk_from, chunk_to) in enumerate(chunks, start=1):
        candles = fetch_one_chunk(instrument_key, unit, interval, chunk_from, chunk_to)
        if candles:
            all_candles.extend(candles)
        if idx % 10 == 0 or idx == len(chunks):
            print(f"      ...{idx}/{len(chunks)} chunks done")

    if not all_candles:
        if not existing.empty:
            print(f"    No new candles returned for {symbol}.")
        return existing

    new_df   = _candles_to_df(all_candles)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    save_checkpoint(symbol, combined)
    added = len(combined) - len(existing)
    print(f"    {symbol}: +{added} new candles → {len(combined)} total")
    return combined


# ── master_data (parquet) management ─────────────────────────────────────────

def master_parquet_path(symbol):
    return MASTER_DIR / f"{symbol}.parquet"


def update_master_parquet(symbol, df):
    """Write or incrementally update master_data/<symbol>.parquet."""
    if df.empty:
        return
    path = master_parquet_path(symbol)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(path, index=False, compression="snappy")


def build_master_from_checkpoints():
    """
    One-time full build of master_data/ from all existing checkpoint CSVs.
    Called automatically on first run when master_data/ is empty.
    Makes no API calls.
    """
    csv_files = sorted(CHECKPOINT_DIR.glob("*.csv"))
    print(f"\nBuilding master_data/ from {len(csv_files)} checkpoint CSVs...")
    t0 = time.time()
    for i, csv_path in enumerate(csv_files, start=1):
        symbol = csv_path.stem
        parquet_path = master_parquet_path(symbol)
        if parquet_path.exists():
            continue  # already converted
        try:
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
            df.to_parquet(parquet_path, index=False, compression="snappy")
        except Exception as e:
            print(f"  SKIP {symbol}: {e}")
        if i % 100 == 0 or i == len(csv_files):
            elapsed = time.time() - t0
            print(f"  [{i:>4}/{len(csv_files)}]  {elapsed:.1f}s elapsed")
    print(f"master_data/ build complete ({time.time() - t0:.1f}s)\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Auto-build master_data/ from checkpoints on first run
    existing_parquets = list(MASTER_DIR.glob("*.parquet"))
    if not existing_parquets:
        build_master_from_checkpoints()

    instruments, mismatches, unresolved = build_instruments(COMPANIES_CSV_PATH)
    print(f"\nResolved {len(instruments)} companies via Upstox instrument master.")

    if mismatches:
        print(f"\n{len(mismatches)} ISIN mismatches (CSV vs master — using master as authoritative):")
        for m in mismatches:
            print(f"  {m['symbol']}: CSV={m['csv_isin']}, master={m['master_isin']}")
        pd.DataFrame(mismatches).to_csv(BASE / "isin_mismatches.csv", index=False)

    if unresolved:
        print(f"\n{len(unresolved)} symbols not found in Upstox master (delisted/renamed?):")
        print(", ".join(unresolved))
        with open(BASE / "unresolved_symbols.txt", "w") as f:
            f.write("\n".join(unresolved))

    print(f"\nFetching updates for {len(instruments)} symbols...\n")
    failed        = []
    intraday_added = 0

    for i, (symbol, instrument_key) in enumerate(instruments.items(), start=1):
        print(f"[{i}/{len(instruments)}] {symbol}")
        try:
            # Step 1: historical fetch (backfill up to yesterday)
            df = fetch_historical_data(
                symbol, instrument_key, UNIT, INTERVAL, FROM_DATE, TO_DATE
            )
            if df.empty:
                failed.append(symbol)
                continue

            # Step 2: intraday fetch (today's live candles)
            intraday_candles = fetch_intraday_today(instrument_key)
            if intraday_candles:
                intraday_df = _candles_to_df(intraday_candles)
                before = len(df)
                df = pd.concat([df, intraday_df], ignore_index=True)
                df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
                added = len(df) - before
                if added > 0:
                    print(f"    {symbol}: +{added} intraday candles for today")
                    intraday_added += added

            update_master_parquet(symbol, df)
        except Exception as e:
            print(f"  ERROR for {symbol}: {e}")
            failed.append(symbol)

    print("\n" + "=" * 60)
    print("Daily update complete.")
    parquet_count = len(list(MASTER_DIR.glob("*.parquet")))
    print(f"  master_data/: {parquet_count} parquet files")
    print(f"  Intraday candles added today: {intraday_added:,}")
    if failed:
        print(f"  Failed symbols: {', '.join(failed)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
