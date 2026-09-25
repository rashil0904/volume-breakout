# -*- coding: utf-8 -*-
"""fetch_nifty_1min_extend_aug2026.py — extend data/nifty_1min_ohlc.csv (currently ending 2026-07-15)
forward through 2026-08-31, same source/instrument/schema as fetch_nifty_1min.py (Upstox v3
historical-candle, NSE_INDEX|Nifty 50, 1-min). Verifies no gap/overlap at the boundary, flags any
missing trading days in the new window, then appends and overwrites the CSV in place.
"""
import time
from pathlib import Path
import requests
import pandas as pd
from dateutil.relativedelta import relativedelta
import data_loading as dl

REPO = Path(__file__).parent
OUT_CSV = REPO / "data" / "nifty_1min_ohlc.csv"
IST = "Asia/Kolkata"
KEY = "NSE_INDEX|Nifty 50"
SCHEMA = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]
HEADERS = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}

NEW_FROM = pd.Timestamp("2026-07-16").date()
NEW_TO = pd.Timestamp("2026-08-31").date()


def fetch_chunk(cfrom, cto):
    url = f"https://api.upstox.com/v3/historical-candle/{KEY}/minutes/1/{cto}/{cfrom}"
    for attempt in range(5):
        r = requests.get(url, headers=HEADERS, timeout=40)
        if r.status_code == 200:
            return r.json().get("data", {}).get("candles", [])
        if r.status_code == 429 or "UDAPI10005" in r.text:
            time.sleep(2 * (attempt + 1)); continue
        print(f"    [{cfrom}->{cto}] HTTP {r.status_code}: {r.text[:160]}")
        return []
    return []


def month_chunks(a, b):
    cur = a
    while cur <= b:
        end = min(cur + relativedelta(months=1) - relativedelta(days=1), b)
        yield cur.isoformat(), end.isoformat()
        cur = end + relativedelta(days=1)


# ---- load existing dataset ----
existing = pd.read_csv(OUT_CSV)
existing["timestamp"] = pd.to_datetime(existing["timestamp"])
ts_e = existing["timestamp"].dt.tz_convert(IST)
old_last_ts = existing["timestamp"].max()
old_last_date = ts_e.dt.date.max()
old_days = set(ts_e.dt.date.unique())
print(f"existing dataset: {len(existing):,} rows | {ts_e.dt.date.min()} -> {old_last_date} | last ts {old_last_ts}")

# ---- fetch new range ----
print(f"\nfetching 1-min Nifty {NEW_FROM} -> {NEW_TO} …")
chunks = list(month_chunks(NEW_FROM, NEW_TO))
all_c = []
for i, (cf, ct) in enumerate(chunks, 1):
    all_c.extend(fetch_chunk(cf, ct))
    time.sleep(0.25)
    print(f"    chunk {i}/{len(chunks)} ({cf}->{ct}): {len(all_c):,} candles so far")

new_df = pd.DataFrame(all_c, columns=SCHEMA)
if new_df.empty:
    raise SystemExit("No new candles returned — aborting, existing file untouched.")
new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
new_df = new_df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
for c in ("open", "high", "low", "close"):
    new_df[c] = new_df[c].astype(float)
new_df["volume"] = pd.to_numeric(new_df["volume"], errors="coerce").fillna(0).astype("int64")
new_df["open_interest"] = pd.to_numeric(new_df["open_interest"], errors="coerce").fillna(0).astype("int64")

ts_n = new_df["timestamp"].dt.tz_convert(IST)
new_first_date, new_last_date = ts_n.dt.date.min(), ts_n.dt.date.max()
new_days = set(ts_n.dt.date.unique())
print(f"\nfetched: {len(new_df):,} rows | {new_first_date} -> {new_last_date}")

# ---- boundary checks ----
print("\n" + "=" * 64 + "\nBOUNDARY CHECK\n" + "=" * 64)
overlap_days = old_days & new_days
gap = (pd.Timestamp(new_first_date) - pd.Timestamp(old_last_date)).days
print(f"old last trading day : {old_last_date}")
print(f"new first trading day: {new_first_date}")
print(f"calendar gap (days)  : {gap}  (>1 is fine if it spans a weekend/holiday)")
print(f"overlapping days     : {len(overlap_days)}  {sorted(overlap_days) if overlap_days else ''}")
overlap_ts = set(existing["timestamp"]) & set(new_df["timestamp"])
print(f"overlapping timestamps (exact): {len(overlap_ts)}")

# ---- missing trading day check within new window (vs NSE calendar via existing stock-day proxy not available here;
#      flag any weekday with < 350 candles as incomplete, and any multi-business-day silent gap) ----
per_day_new = new_df.groupby(ts_n.dt.date).size()
bdays_expected = pd.bdate_range(NEW_FROM, min(NEW_TO, new_last_date)).date
missing_bdays = sorted(set(bdays_expected) - new_days)
incomplete_days = per_day_new[per_day_new < 350]
print(f"\nweekday calendar days with ZERO candles in new window (possible holiday OR missing data): {len(missing_bdays)}")
if missing_bdays:
    print("   " + ", ".join(str(d) for d in missing_bdays))
print(f"days with <350 candles (possibly incomplete/half day): {len(incomplete_days)}")
if len(incomplete_days):
    print(incomplete_days.to_string())

# ---- merge ----
combined = pd.concat([existing[SCHEMA], new_df[SCHEMA]], ignore_index=True)
combined = combined.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
combined.to_csv(OUT_CSV, index=False)

ts_c = combined["timestamp"].dt.tz_convert(IST)
print("\n" + "=" * 64 + "\nFINAL COMBINED DATASET\n" + "=" * 64)
print(f"rows          : {len(combined):,}  (was {len(existing):,}, added {len(combined) - len(existing):,})")
print(f"date range    : {ts_c.dt.date.min()} -> {ts_c.dt.date.max()}")
print(f"first ts      : {combined['timestamp'].min()}")
print(f"last ts       : {combined['timestamp'].max()}")
print(f"saved -> {OUT_CSV}")
