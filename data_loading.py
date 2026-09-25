#!/usr/bin/env python3
"""
data_loading.py — Daily 3:00 PM intraday fetch (parallel, ~4-6 min for 1,610 symbols)
=======================================================================================
Run at 3:00 PM each trading day BEFORE prepare_data.py.

Fetches TODAY's 15-min intraday candles for every symbol in Companies List.csv
via the Upstox intraday endpoint (no date params — always returns today).

Merges new candles into:
  checkpoints/<SYMBOL>.csv    — durable source of truth (append, dedup, sort)
  master_data/<SYMBOL>.parquet — fast-read cache (same merge logic)

Handles both EQ and BE instrument types with ISIN fallback.

Usage:
    python data_loading.py

Update ACCESS_TOKEN each morning after running upstox_auth.py.
"""

import io
import gzip
import json
import os
import time
import threading
import requests
import pandas as pd
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Read from env var (GitHub Actions secret) with hardcoded fallback for local runs.
ACCESS_TOKEN   = os.environ.get(
    "UPSTOX_ACCESS_TOKEN",
    "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiJEUjEyOTMiLCJqdGkiOiI2YTUxM2I4OGUzNDQ3MjQwMjRmNmQ5ODUiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlzRXh0ZW5kZWQiOnRydWUsImlhdCI6MTc4MzcwODU1MiwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxODE1MjU2ODAwfQ.b2kgnyorOf1XQgIiXyvZmRnu74rmMXlQ7jDbM2G2nHo"
)
COMPANIES_CSV  = "Companies List.csv"
MAX_WORKERS    = 8
MAX_RETRIES    = 3
INITIAL_BACKOFF = 1.0

BASE           = Path(__file__).parent
CHECKPOINT_DIR = BASE / "checkpoints"
MASTER_DIR     = BASE / "master_data"
INSTRUMENT_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
INTRADAY_URL   = "https://api.upstox.com/v3/historical-candle/intraday/{key}/minutes/15"

CHECKPOINT_DIR.mkdir(exist_ok=True)
MASTER_DIR.mkdir(exist_ok=True)

# ── Rate limiter (stress-tested at 12-worker concurrency) ────────────────────
RATE_LIMITS = [(1, 50), (60, 500), (1800, 2000)]

_request_timestamps = deque()
_rate_lock          = threading.Lock()


def throttle():
    while True:
        with _rate_lock:
            now        = time.monotonic()
            max_window = RATE_LIMITS[-1][0]
            while _request_timestamps and now - _request_timestamps[0] > max_window:
                _request_timestamps.popleft()
            wait_time = 0.0
            for window_sec, max_requests in RATE_LIMITS:
                count = sum(1 for t in _request_timestamps if now - t <= window_sec)
                if count >= max_requests:
                    oldest    = next(t for t in _request_timestamps if now - t <= window_sec)
                    wait_time = max(wait_time, window_sec - (now - oldest) + 0.05)
            if wait_time <= 0:
                _request_timestamps.append(now)
                return
        time.sleep(wait_time)


# ── Instrument resolution (EQ + BE with ISIN fallback) ───────────────────────

def load_symbols_and_isins_from_csv(path):
    raw = pd.read_csv(path, header=None, encoding="utf-8-sig")
    df  = raw.iloc[:, :2].copy()
    df.columns = ["symbol", "isin"]
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["isin"]   = df["isin"].astype(str).str.strip()
    df = df[df["symbol"].notna() & (df["symbol"] != "") & (df["symbol"] != "nan")]
    df = df.drop_duplicates(subset="symbol")
    valid_isin = df["isin"].str.match(r"^IN[A-Z0-9]{10}$")
    df.loc[~valid_isin, "isin"] = ""
    return dict(zip(df["symbol"], df["isin"]))


def download_nse_instrument_master():
    print("Downloading Upstox NSE instrument master …")
    resp = requests.get(INSTRUMENT_MASTER_URL, timeout=30)
    resp.raise_for_status()
    with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as f:
        raw = json.load(f)
    by_symbol, by_isin = {}, {}
    for inst in raw:
        if inst.get("segment") == "NSE_EQ" and inst.get("instrument_type") in ("EQ", "BE"):
            entry = {"instrument_key": inst["instrument_key"], "isin": inst.get("isin", "")}
            by_symbol[inst["trading_symbol"]] = entry
            if entry["isin"]:
                by_isin[entry["isin"]] = entry
    print(f"  Loaded {len(by_symbol)} NSE equity instruments (EQ + BE).")
    return by_symbol, by_isin


def build_instruments(csv_path):
    csv_map = load_symbols_and_isins_from_csv(csv_path)
    by_sym, by_isin = download_nse_instrument_master()
    instruments, unresolved = {}, []
    for symbol, csv_isin in csv_map.items():
        entry = by_sym.get(symbol)
        if entry is None and csv_isin and csv_isin.startswith("IN"):
            entry = by_isin.get(csv_isin)
        if entry is None:
            unresolved.append(symbol)
        else:
            instruments[symbol] = entry["instrument_key"]
    return instruments, unresolved


# ── Candle parsing ────────────────────────────────────────────────────────────

def _candles_to_df(candles):
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles,
                      columns=["timestamp", "open", "high", "low",
                               "close", "volume", "open_interest"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return (df.drop_duplicates(subset="timestamp")
              .sort_values("timestamp")
              .reset_index(drop=True))


# ── Per-symbol worker ─────────────────────────────────────────────────────────

_print_lock = threading.Lock()


def _merge_and_save(symbol, new_df):
    """Merge new candles into checkpoint CSV and parquet in one pass."""
    pq_path  = MASTER_DIR     / f"{symbol}.parquet"
    csv_path = CHECKPOINT_DIR / f"{symbol}.csv"

    # Load existing from parquet (fast) or CSV (fallback)
    if pq_path.exists():
        existing = pd.read_parquet(pq_path)
    elif csv_path.exists():
        existing = pd.read_csv(csv_path, parse_dates=["timestamp"])
    else:
        existing = pd.DataFrame()

    if existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)

    combined = (combined
                .drop_duplicates(subset="timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True))

    # Atomic CSV write (temp + rename prevents corruption on crash)
    tmp = csv_path.with_suffix(".tmp")
    combined.to_csv(tmp, index=False)
    tmp.rename(csv_path)

    combined.to_parquet(pq_path, index=False, compression="snappy")
    return len(new_df), len(combined)


def fetch_symbol(args):
    """Fetch today's intraday candles for one symbol and merge into both files."""
    symbol, instrument_key = args
    url     = INTRADAY_URL.format(key=instrument_key)
    headers = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}

    backoff  = INITIAL_BACKOFF
    candles  = []
    for attempt in range(1, MAX_RETRIES + 1):
        throttle()
        try:
            resp = requests.get(url, headers=headers, timeout=10)
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                return symbol, 0, f"connection error: {e}"
            time.sleep(backoff); backoff *= 2
            continue

        if resp.status_code == 200:
            candles = resp.json().get("data", {}).get("candles", [])
            break
        if resp.status_code == 429 or "UDAPI10005" in resp.text:
            time.sleep(backoff); backoff *= 2
            continue
        return symbol, 0, f"HTTP {resp.status_code}"

    if not candles:
        return symbol, 0, None   # no data today (pre-market, holiday, etc.)

    new_df = _candles_to_df(candles)
    if new_df.empty:
        return symbol, 0, None

    new_count, _ = _merge_and_save(symbol, new_df)
    return symbol, new_count, None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 64)
    print("data_loading.py — Daily 3pm intraday fetch")
    print("=" * 64)

    instruments, unresolved = build_instruments(COMPANIES_CSV)
    print(f"\nResolved: {len(instruments):,} symbols  |  Unresolved: {len(unresolved)}")
    if unresolved:
        print(f"  Unresolved: {', '.join(unresolved[:10])}{'…' if len(unresolved)>10 else ''}")

    print(f"\nFetching {len(instruments):,} symbols with {MAX_WORKERS} workers …")
    print(f"Expected: 12–18 min at 500 req/min rate limit\n")

    tasks       = list(instruments.items())
    no_data     = []
    errors      = []
    completed   = 0
    total_new   = 0
    n           = len(tasks)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(fetch_symbol, t): t[0] for t in tasks}
        for future in as_completed(future_map):
            symbol, new_count, err = future.result()
            completed += 1
            if err:
                errors.append((symbol, err))
            elif new_count == 0:
                no_data.append(symbol)
            else:
                total_new += new_count

            if completed % 100 == 0 or completed == n:
                elapsed = time.time() - t_start
                rate    = completed / elapsed * 60
                eta     = (n - completed) / (completed / elapsed) if completed > 0 else 0
                with _print_lock:
                    print(f"  [{completed:>4}/{n}]  {elapsed:>5.1f}s  "
                          f"{rate:>5.0f} req/min  ETA {eta:>3.0f}s  "
                          f"new_candles={total_new:,}  no_data={len(no_data)}")

    elapsed = time.time() - t_start

    # Save no-data list
    no_data_path = BASE / "no_data_today.txt"
    with open(no_data_path, "w") as f:
        f.write("\n".join(no_data))

    print()
    print("=" * 64)
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Symbols fetched with data : {n - len(no_data) - len(errors):,}")
    print(f"  New candles added today   : {total_new:,}")
    print(f"  No data today             : {len(no_data):,}  → no_data_today.txt")
    if errors:
        print(f"  Errors                    : {len(errors)}")
        for sym, msg in errors[:10]:
            print(f"    {sym}: {msg}")
    print("=" * 64)


if __name__ == "__main__":
    main()
