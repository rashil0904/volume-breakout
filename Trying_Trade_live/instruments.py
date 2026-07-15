"""
Resolves NSE trading symbols (e.g. "RELIANCE") to Upstox instrument_keys
(e.g. "NSE_EQ|INE002A01018"), which is what the order API needs.

Caches the NSE instrument list locally for a day so you're not re-downloading
it on every run.
"""

import gzip
import json
import os
import time
import requests

INSTRUMENTS_URL     = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
CACHE_FILE          = "nse_instruments_cache.json"
CACHE_MAX_AGE_SECONDS = 20 * 60 * 60  # ~20 hours; file refreshes daily at 6 AM


def _download_instruments():
    resp = requests.get(INSTRUMENTS_URL)
    resp.raise_for_status()
    data = json.loads(gzip.decompress(resp.content))
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f)
    return data


def _load_instruments():
    if os.path.exists(CACHE_FILE):
        age = time.time() - os.path.getmtime(CACHE_FILE)
        if age < CACHE_MAX_AGE_SECONDS:
            with open(CACHE_FILE) as f:
                return json.load(f)
    return _download_instruments()


def resolve_symbol(trading_symbol: str) -> str:
    """
    Given e.g. 'RELIANCE', returns the instrument_key e.g. 'NSE_EQ|INE002A01018'.
    Raises ValueError if not found.
    """
    instruments = _load_instruments()
    trading_symbol = trading_symbol.upper().strip()

    for row in instruments:
        if row.get("instrument_type") == "EQ" and row.get("trading_symbol") == trading_symbol:
            return row["instrument_key"]

    raise ValueError(f"Could not find instrument_key for symbol '{trading_symbol}'")


if __name__ == "__main__":
    import sys
    for sym in sys.argv[1:] or ["RELIANCE", "TCS", "INFY"]:
        print(sym, "->", resolve_symbol(sym))
