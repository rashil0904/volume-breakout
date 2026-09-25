# -*- coding: utf-8 -*-
"""fetch_india_vix_daily_2020_2021.py — one-off pull of India VIX DAILY OHLC, 2020-01-01 through
2021-12-31, via Upstox V3 historical-candle (days/1 interval, confirmed correct format). Schema matches
the existing data/india_vix_daily.csv exactly for consistent future merging.
"""
from pathlib import Path
import requests
import pandas as pd
import data_loading as dl

REPO = Path(__file__).parent
OUT_CSV = REPO / "data" / "india_vix_daily_2020_2021.csv"
KEY = "NSE_INDEX|India VIX"
COLS = ["timestamp", "open", "high", "low", "close", "volume", "oi"]
HEADERS = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}
FROM, TO = "2020-01-01", "2021-12-31"


def enc(k): return k.replace("|", "%7C").replace(" ", "%20")


url = f"https://api.upstox.com/v3/historical-candle/{enc(KEY)}/days/1/{TO}/{FROM}"
r = requests.get(url, headers=HEADERS, timeout=40)
print(f"status: {r.status_code}")
candles = r.json().get("data", {}).get("candles", [])
print(f"candles returned: {len(candles)}")

df = pd.DataFrame(candles, columns=COLS)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
for c in ("open", "high", "low", "close"):
    df[c] = df[c].astype(float)
df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0).astype("int64")

print(f"\nfetched: {len(df)} rows | {df['timestamp'].min()} -> {df['timestamp'].max()}")

# ---- quality checks ----
ts_dates = df["timestamp"].dt.date
bdays_expected = pd.bdate_range(FROM, TO).date
missing_bdays = sorted(set(bdays_expected) - set(ts_dates))
print(f"\nbusiness weekdays with ZERO candles (holiday OR missing): {len(missing_bdays)}")
if missing_bdays:
    print("   " + ", ".join(str(d) for d in missing_bdays))

dup = df["timestamp"].duplicated().sum()
nan_ct = df[["open", "high", "low", "close"]].isna().sum().sum()
bad_hl = (df["high"] < df["low"]).sum()
bad_ho = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
bad_lo = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
print(f"\nduplicate timestamps: {dup} | NaN OHLC: {nan_ct} | high<low: {bad_hl} | high<max(o,c): {bad_ho} | low>min(o,c): {bad_lo} | non-positive: {nonpos}")

# ---- VIX-specific sanity check: known COVID crash spike, March 2020 ----
mar2020 = df[(df["timestamp"] >= "2020-03-01") & (df["timestamp"] <= "2020-03-31")]
print(f"\nMarch 2020 (COVID crash) VIX range: min={mar2020['close'].min():.2f}  max={mar2020['close'].max():.2f}  "
      f"(expect a dramatic spike, historically India VIX peaked around 80+ in late March 2020)")
peak_row = mar2020.loc[mar2020["close"].idxmax()]
print(f"peak close: {peak_row['close']:.2f} on {peak_row['timestamp'].date()}")
overall_range = f"min={df['close'].min():.2f} (on {df.loc[df['close'].idxmin(),'timestamp'].date()})  max={df['close'].max():.2f} (on {df.loc[df['close'].idxmax(),'timestamp'].date()})"
print(f"\noverall 2020-2021 VIX close range: {overall_range}")

df.to_csv(OUT_CSV, index=False)
print(f"\nsaved -> {OUT_CSV}")
