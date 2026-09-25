#!/usr/bin/env python3
"""
pull_new_listed_stocks.py — full 1-min history pull (2022-01-01 -> today) for stocks currently on NSE
(Upstox NSE_EQ/BE master) but NOT in the existing Companies List.csv, with ETF-like names excluded.
Reuses fetch_1min_history.py's exact proven do_symbol()/fetch_month() pattern unmodified. Whichever
candidates are genuine new listings will naturally show a recent start date in their own resulting
parquet (Upstox simply has no earlier data for them) -- reported after the fact, not pre-filtered.
"""
import sys, time, threading, pickle
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data_loading as dl

dl.RATE_LIMITS = [(1, 50), (60, 500)]

REPO = Path(__file__).parent
RESULTS = REPO / "results"
DONE_FILE = RESULTS / "new_listed_stocks_done.txt"
LOG_FILE = RESULTS / "new_listed_stocks.log"
OUT_DIR = REPO / "master_data_new_listings"
OUT_DIR.mkdir(exist_ok=True)
START = date(2022, 1, 1)
MAX_WORKERS = 5
HIST_URL = "https://api.upstox.com/v3/historical-candle/{key}/minutes/1/{to}/{frm}"
HEADERS = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}
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
            return [], None
        if r.status_code == 401:
            return None, "AUTH"
        if r.status_code == 429 or "UDAPI10005" in r.text:
            time.sleep(backoff); backoff *= 1.6; continue
        return None, f"HTTP{r.status_code}"
    return None, "retries"


def do_symbol(symbol, key, today):
    allc, hard = [], False
    for frm, to in month_ranges(START, today):
        c, err = fetch_month(key, frm, to)
        if err == "AUTH":
            return symbol, 0, "AUTH", None
        if c is None:
            hard = True; continue
        allc.extend(c)
    if hard:
        return symbol, len(allc), "partial", None
    df = dl._candles_to_df(allc)
    if len(df):
        df.to_parquet(OUT_DIR / f"{symbol}.parquet", index=False, compression="snappy")
        first_date = pd.to_datetime(df["timestamp"], utc=True).min().tz_convert("Asia/Kolkata").date()
        return symbol, len(df), "ok", str(first_date)
    return symbol, 0, "ok_empty", None


def main():
    RESULTS.mkdir(exist_ok=True)
    log("=" * 70)
    log("NEW-LISTED-STOCKS full history pull starting")
    with open(RESULTS / "_new_stock_candidates.pkl", "rb") as f:
        candidates = pickle.load(f)
    instruments = {inst["trading_symbol"]: inst["instrument_key"] for inst in candidates}
    log(f"candidates (ETF-excluded, not in existing 1609): {len(instruments)}")

    done = set(DONE_FILE.read_text().split()) if DONE_FILE.exists() else set()
    todo = [(s, k) for s, k in instruments.items() if s not in done]
    today = date.today()
    n_months = len(month_ranges(START, today))
    log(f"to fetch: {len(todo)} symbols x {n_months} months (~{len(todo)*n_months:,} requests) | skipping {len(done)} done")

    t0 = time.time(); n = 0; ok = 0; partial = 0; authfail = 0; first_dates = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(do_symbol, s, k, today): s for s, k in todo}
        for fut in as_completed(futs):
            sym, rows, status, first_date = fut.result(); n += 1
            if status in ("ok", "ok_empty"):
                ok += 1
                with _lock, open(DONE_FILE, "a", encoding="utf-8") as f:
                    f.write(sym + "\n")
                first_dates.append({"symbol": sym, "rows": rows, "first_date": first_date})
            elif status == "partial":
                partial += 1
            elif status == "AUTH":
                authfail += 1
            if n % 50 == 0 or status not in ("ok", "ok_empty"):
                el = time.time() - t0
                eta = el / n * (len(todo) - n) / 3600 if n else 0
                log(f"[{n}/{len(todo)}] {sym}: {rows:,} rows ({status}, first={first_date}) | ok={ok} partial={partial} "
                    f"auth={authfail} | {el/60:.0f}m elapsed, ETA {eta:.1f}h")

    log(f"COMPLETE. ok={ok} partial={partial} auth_fail={authfail}. Re-run to retry partial (resumes via {DONE_FILE.name}).")
    pd.DataFrame(first_dates).to_csv(RESULTS / "new_listed_stocks_first_dates.csv", index=False)
    log(f"First-date detail saved -> {RESULTS}/new_listed_stocks_first_dates.csv")
    if authfail:
        log("NOTE: 401 auth failures — refresh ACCESS_TOKEN in data_loading.py, then re-run.")


if __name__ == "__main__":
    main()
