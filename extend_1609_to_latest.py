#!/usr/bin/env python3
"""
extend_1609_to_latest.py — INCREMENTAL extension of the existing 1609-stock master_data/ parquet files,
picking up right after each stock's own last date through today. Appends + dedupes rather than
re-downloading full history (unlike fetch_1min_history.py's full-refresh design).
"""
import sys, time, threading
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
LOG_FILE = RESULTS / "extend_1609.log"
FLOOR = date(2026, 8, 1)
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
        out.append((max(cur, start).isoformat(), min(nxt - timedelta(days=1), end).isoformat()))
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
    fn = dl.MASTER_DIR / f"{symbol}.parquet"
    if not fn.exists():
        return symbol, 0, "no_existing_file"
    old = pd.read_parquet(fn)
    old_ts = pd.to_datetime(old["timestamp"], utc=True)
    last_date = old_ts.max().tz_convert("Asia/Kolkata").date()
    start_from = max(FLOOR, last_date + timedelta(days=1))
    if start_from > today:
        return symbol, 0, "already_current"

    allc, hard = [], False
    for frm, to in month_ranges(start_from, today):
        c, err = fetch_month(key, frm, to)
        if err == "AUTH":
            return symbol, 0, "AUTH"
        if c is None:
            hard = True; continue
        allc.extend(c)
    if not allc:
        return symbol, 0, "no_new_data" if not hard else "partial_no_data"

    new_df = dl._candles_to_df(allc)
    if len(new_df) == 0:
        return symbol, 0, "no_new_rows"
    combined = pd.concat([old, new_df], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined = combined.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    combined.to_parquet(fn, index=False, compression="snappy")
    new_last = combined["timestamp"].max().tz_convert("Asia/Kolkata").date()
    return symbol, len(new_df), ("partial" if hard else "ok") + f"|last={new_last}|old_last={last_date}"


def main():
    RESULTS.mkdir(exist_ok=True)
    log("=" * 70)
    log(f"Extending 1609 stocks from {FLOOR} to today")
    instruments, unresolved = dl.build_instruments(dl.COMPANIES_CSV)
    log(f"symbols resolved: {len(instruments)} | unresolved: {len(unresolved)}")
    today = date.today()

    t0 = time.time(); n = 0; ok = 0; already_current = 0; issues = []
    results_detail = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(do_symbol, s, k, today): s for s, k in instruments.items()}
        for fut in as_completed(futs):
            sym, rows, status = fut.result(); n += 1
            results_detail.append({"symbol": sym, "new_rows": rows, "status": status})
            if status.startswith("ok"):
                ok += 1
            elif status == "already_current":
                already_current += 1
            else:
                issues.append((sym, status))
            if n % 200 == 0:
                el = time.time() - t0
                log(f"[{n}/{len(instruments)}] elapsed {el/60:.1f}m | ok={ok} already_current={already_current} issues={len(issues)}")

    log(f"COMPLETE. ok={ok} already_current={already_current} issues={len(issues)} total={n} in {(time.time()-t0)/60:.1f}m")
    if issues:
        log(f"ISSUES (flagged, not silently skipped): {issues[:50]}")
    pd.DataFrame(results_detail).to_csv(RESULTS / "extend_1609_detail.csv", index=False)
    log(f"Detail saved -> {RESULTS}/extend_1609_detail.csv")


if __name__ == "__main__":
    main()
