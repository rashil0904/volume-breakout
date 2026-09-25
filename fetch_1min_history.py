#!/usr/bin/env python3
"""
fetch_1min_history.py — FULL 1-minute historical refresh for all stocks
=======================================================================
Fetches 1-min candles from 2022-01-01 → today for every symbol in Companies List.csv,
via the Upstox v3 historical endpoint (capped at 1 month/request → monthly chunks).
Overwrites master_data/<SYMBOL>.parquet with the complete, clean 1-min series.

Reuses data_loading.py: throttle() rate limiter, build_instruments(), _candles_to_df(),
ACCESS_TOKEN, MASTER_DIR. 5 parallel workers; the shared rate limiter keeps total request
rate under Upstox limits (50/s, 500/min, 2000/30min).

RESUMABLE: each completed symbol is appended to results/refresh_1min_done.txt; re-running
skips them. Live progress in results/refresh_1min.log. Safe to stop/restart anytime.

Usage:  python fetch_1min_history.py
"""
import sys, time, threading
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data_loading as dl

# SPEEDUP: drop the over-conservative 2000/30min cap — sustain 500/min (~15× faster bulk
# backfill). 429s (if Upstox pushes back) are still handled with backoff in fetch_month().
dl.RATE_LIMITS = [(1, 50), (60, 500)]

REPO      = Path(__file__).parent
RESULTS   = REPO / "results"
DONE_FILE = RESULTS / "refresh_1min_done.txt"
LOG_FILE  = RESULTS / "refresh_1min.log"
START     = date(2022, 1, 1)
MAX_WORKERS = 5
HIST_URL  = "https://api.upstox.com/v3/historical-candle/{key}/minutes/1/{to}/{frm}"
HEADERS   = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}

_lock = threading.Lock()


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _lock:
        print(line, flush=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def month_ranges(start, end):
    cur, out = date(start.year, start.month, 1), []
    while cur <= end:
        nxt = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
        out.append((cur.isoformat(), min(nxt - timedelta(days=1), end).isoformat()))
        cur = nxt
    return out


def fetch_month(key, frm, to):
    """Return (candles_list, error). error='AUTH' on 401; None on ok/empty."""
    backoff = 2.0
    for attempt in range(5):
        dl.throttle()
        try:
            r = requests.get(HIST_URL.format(key=key, to=to, frm=frm), headers=HEADERS, timeout=30)
        except requests.RequestException:
            time.sleep(backoff); backoff *= 1.6; continue
        if r.status_code == 200:
            return r.json().get("data", {}).get("candles", []), None
        if r.status_code == 400:
            return [], None                                   # pre-listing / no data this month
        if r.status_code == 401:
            return None, "AUTH"                                # token expired
        if r.status_code == 429 or "UDAPI10005" in r.text:
            time.sleep(backoff); backoff *= 1.6; continue
        return None, f"HTTP{r.status_code}"
    return None, "retries"


def do_symbol(symbol, key, today):
    allc, hard = [], False
    for frm, to in month_ranges(START, today):
        c, err = fetch_month(key, frm, to)
        if err == "AUTH":
            return symbol, 0, "AUTH"
        if c is None:
            hard = True; continue                              # month failed — don't mark done
        allc.extend(c)
    if hard:
        return symbol, len(allc), "partial"
    df = dl._candles_to_df(allc)
    if len(df):
        df.to_parquet(dl.MASTER_DIR / f"{symbol}.parquet", index=False, compression="snappy")
    return symbol, len(df), "ok"


def main():
    RESULTS.mkdir(exist_ok=True)
    log("=" * 70)
    log("FULL 1-min refresh starting")
    instruments, unresolved = dl.build_instruments(dl.COMPANIES_CSV)
    log(f"symbols resolved: {len(instruments)} | unresolved: {len(unresolved)}")
    done = set(DONE_FILE.read_text().split()) if DONE_FILE.exists() else set()
    todo = [(s, k) for s, k in instruments.items() if s not in done]
    today = date.today()
    n_months = len(month_ranges(START, today))
    log(f"to fetch: {len(todo)} symbols × {n_months} months (~{len(todo)*n_months:,} requests) "
        f"| skipping {len(done)} already done | {MAX_WORKERS} workers")

    t0 = time.time(); n = 0; ok = 0; partial = 0; authfail = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(do_symbol, s, k, today): s for s, k in todo}
        for fut in as_completed(futs):
            sym, rows, status = fut.result(); n += 1
            if status == "ok":
                ok += 1
                with _lock, open(DONE_FILE, "a", encoding="utf-8") as f:
                    f.write(sym + "\n")
            elif status == "partial":
                partial += 1
            elif status == "AUTH":
                authfail += 1
            if n % 20 == 0 or status != "ok":
                el = time.time() - t0
                eta = el / n * (len(todo) - n) / 3600 if n else 0
                log(f"[{n}/{len(todo)}] {sym}: {rows:,} rows ({status}) | ok={ok} partial={partial} "
                    f"auth={authfail} | {el/60:.0f}m elapsed, ETA {eta:.1f}h")
    log(f"COMPLETE. ok={ok} partial={partial} auth_fail={authfail}. "
        f"Re-run to retry partial/failed (resumes via {DONE_FILE.name}).")
    if authfail:
        log("NOTE: 401 auth failures — refresh ACCESS_TOKEN in data_loading.py, then re-run.")


if __name__ == "__main__":
    main()
