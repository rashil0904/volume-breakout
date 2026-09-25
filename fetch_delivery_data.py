# -*- coding: utf-8 -*-
"""
fetch_delivery_data.py
======================
Fetches NSE security-wise delivery data (sec_bhavdata_full) for the exact set of trading
dates in the main strategy's signal set, filters to the 691-symbol universe, and caches
one file per date (resumable). Columns kept: SYMBOL, SERIES, TTL_TRD_QNTY, DELIV_QTY,
DELIV_PER. SERIES flags EQ vs BE/T2T (delivery ~100% by rule in T2T).

NSE anti-bot: a Session primes cookies from nseindia.com; requests are spaced and cookies
re-primed periodically / on failure. Only signal dates are fetched (975), not the calendar.
"""
import sys, io, time
from pathlib import Path

import requests
import numpy as np
import pandas as pd

REPO = Path(__file__).parent
RESULTS = REPO / "results"
CACHE = RESULTS / "delivery_cache"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept": "text/csv,*/*", "Accept-Language": "en-US,en;q=0.9",
           "Referer": "https://www.nseindia.com/"}
URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"


def new_session():
    s = requests.Session(); s.headers.update(HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=15)
        s.get("https://www.nseindia.com/all-reports", timeout=15)
    except requests.RequestException:
        pass
    return s


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "passes_all_three"], parse_dates=["date"])
    sig = diag[diag["passes_all_three"]]
    universe = set(sig["symbol"].str.upper().str.strip())
    dates = sorted(sig["date"].dt.normalize().unique())
    print(f"Universe: {len(universe)} symbols | dates to fetch: {len(dates)}")

    s = new_session()
    done = failed = 0
    for i, d in enumerate(dates, 1):
        ts = pd.Timestamp(d)
        ddmmyyyy = ts.strftime("%d%m%Y")
        out = CACHE / f"{ddmmyyyy}.csv"
        if out.exists():
            done += 1
            continue
        url = URL.format(ddmmyyyy=ddmmyyyy)
        ok = False
        for attempt in range(4):
            try:
                r = s.get(url, timeout=25)
            except requests.RequestException:
                time.sleep(2 * (attempt + 1)); s = new_session(); continue
            txt = r.text.strip()
            is_csv = r.status_code == 200 and txt[:20].upper().lstrip().startswith("SYMBOL")
            if r.status_code == 200 and not is_csv:      # HTML block / error page, not CSV
                time.sleep(3 * (attempt + 1)); s = new_session(); continue
            if is_csv:
                try:
                    df = pd.read_csv(io.StringIO(r.text))
                except pd.errors.ParserError:
                    time.sleep(2 * (attempt + 1)); s = new_session(); continue
                df.columns = [c.strip() for c in df.columns]
                df["SYMBOL"] = df["SYMBOL"].astype(str).str.upper().str.strip()
                df = df[df["SYMBOL"].isin(universe)].copy()
                keep = ["SYMBOL", "SERIES", "TTL_TRD_QNTY", "DELIV_QTY", "DELIV_PER"]
                df = df[[c for c in keep if c in df.columns]]
                df.insert(0, "date", ts.date())
                df.to_csv(out, index=False)
                ok = True; done += 1
                break
            if r.status_code in (401, 403, 429):
                time.sleep(3 * (attempt + 1)); s = new_session(); continue
            if r.status_code == 404:                    # no file (holiday slipped in / not published)
                out.write_text("date,SYMBOL,SERIES,TTL_TRD_QNTY,DELIV_QTY,DELIV_PER\n")
                ok = True; done += 1
                break
            time.sleep(2 * (attempt + 1))
        if not ok:
            failed += 1
            print(f"  [{i}/{len(dates)}] {ddmmyyyy}: GAVE UP")
        time.sleep(0.4)
        if i % 50 == 0:
            print(f"  [{i}/{len(dates)}] done={done} failed={failed} … re-priming cookies")
            s = new_session()
    print(f"\nDONE: cached {done} dates, failed {failed}. -> {CACHE}")


if __name__ == "__main__":
    main()
