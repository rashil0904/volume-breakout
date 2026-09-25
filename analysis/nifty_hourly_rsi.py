# -*- coding: utf-8 -*-
"""
nifty_hourly_rsi.py — STEP 1 of the Nifty RSI-divergence strategy.
Builds NON-STANDARD hourly candles from Nifty-50 SPOT 15-min data and stamps Wilder RSI-14 on
the hourly closes. No divergence logic yet — just the verifiable hourly OHLC + RSI series.

DATA (flag a): Nifty-50 SPOT index, data/nifty_15min_ohlc.csv (Upstox, 15-min, volume=0 for index).
  -> RSI/divergence is on the SPOT index (not futures), which is the usual convention.

HOURLY BUCKETS (flag b: [start, end) by the 15-min candle's START time):
  candle 1  09:07-10  -> 15-min 09:15, 09:30, 09:45      (NOTE flag d: OPEN = 09:15, NOT 9:07 —
                                                          the 9:07 pre-open print is NOT in the data;
                                                          Upstox candles begin at 09:15 regular session.)
  candle 2  10-11     -> 10:00,10:15,10:30,10:45
  candle 3  11-12     -> 11:00,11:15,11:30,11:45
  candle 4  12-13     -> 12:00,12:15,12:30,12:45
  candle 5  13-14     -> 13:00,13:15,13:30,13:45
  candle 6  14-15     -> 14:00,14:15,14:30,14:45
  candle 7  15-15:30  -> 15:00, 15:15                    (flag e: 30-min stub treated as a full bar)
  Per candle: open=first 15-min open, high=max, low=min, close=last 15-min close, volume=sum(=0).
  Evening/muhurat (>15:30) and pre-open (<09:15) 15-min candles are excluded.

RSI-14 (flag c: CONTINUOUS across days — candle 7 of a day is followed by candle 1 of the next;
RSI is NOT reset each day). Wilder's smoothing: first avg gain/loss = SMA of the first 14 hourly
changes; then avg = (prior*13 + current)/14. RSI = 100 - 100/(1+RS). First RSI at the 15th candle.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = "Asia/Kolkata"
CSV = rb.BASE / "data" / "nifty_15min_ohlc.csv"
OUTDIR = rb.RESULTS / "nifty_hourly_rsi"
RSI_N = 14

BUCKETS = [(555, 600, 1, "09:07-10"), (600, 660, 2, "10-11"), (660, 720, 3, "11-12"),
           (720, 780, 4, "12-13"), (780, 840, 5, "13-14"), (840, 900, 6, "14-15"),
           (900, 930, 7, "15-15:30")]


def bucket_of(hm):
    for lo, hi, idx, lbl in BUCKETS:
        if lo <= hm < hi:
            return idx, lbl
    return None, None


def wilder_rsi(close, n=14):
    """Wilder RSI-14 on a continuous close series. Returns array (NaN until the (n+1)th value)."""
    c = np.asarray(close, float)
    m = len(c)
    rsi = np.full(m, np.nan)
    if m < n + 1:
        return rsi
    delta = np.diff(c)                                    # delta[k] = c[k+1]-c[k], length m-1
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = gain[:n].mean(); al = loss[:n].mean()            # seed = SMA of first n changes
    def _rsi(ag, al):
        if al == 0:
            return 100.0 if ag > 0 else 50.0
        rs = ag / al
        return 100.0 - 100.0 / (1.0 + rs)
    rsi[n] = _rsi(ag, al)                                  # first RSI at index n (the (n+1)th close)
    for i in range(n + 1, m):
        ag = (ag * (n - 1) + gain[i - 1]) / n
        al = (al * (n - 1) + loss[i - 1]) / n
        rsi[i] = _rsi(ag, al)
    return rsi


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(CSV)
    ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
    raw = raw.assign(date=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)
    n_all = len(raw)
    raw = raw[(raw["hm"] >= 555) & (raw["hm"] < 930)].copy()   # regular session only (drop pre-open/evening)
    print(f"15-min rows: {n_all:,} total -> {len(raw):,} in regular session 09:15-15:30")
    bk = raw["hm"].map(bucket_of)
    raw["cidx"] = [b[0] for b in bk]; raw["bucket"] = [b[1] for b in bk]

    # aggregate 15-min -> hourly, ordered by (date, candle index)
    rows = []
    for (d, ci), g in raw.sort_values("hm").groupby(["date", "cidx"]):
        rows.append({"date": d, "candle": int(ci), "hour_bucket": g["bucket"].iloc[0],
                     "open": float(g["open"].iloc[0]), "high": float(g["high"].max()),
                     "low": float(g["low"].min()), "close": float(g["close"].iloc[-1]),
                     "volume": float(g["volume"].sum()), "n_15min": len(g)})
    H = pd.DataFrame(rows).sort_values(["date", "candle"]).reset_index(drop=True)
    H["rsi_14"] = np.round(wilder_rsi(H["close"].values, RSI_N), 4)

    H.to_csv(OUTDIR / "nifty_hourly_rsi.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "nifty_hourly_rsi.xlsx", engine="openpyxl") as w:
        H.to_excel(w, sheet_name="hourly_rsi", index=False)

    # ── sanity checks ──
    per_day = H.groupby("date").size()
    bad_days = per_day[per_day != 7]
    rsi_valid = H["rsi_14"].dropna()
    out_of_range = int(((rsi_valid < 0) | (rsi_valid > 100)).sum())
    first_rsi_idx = int(H["rsi_14"].notna().idxmax()) if H["rsi_14"].notna().any() else -1

    pd.set_option("display.width", 200)
    print("\n" + "=" * 82 + "\nNIFTY HOURLY CANDLES + WILDER RSI-14  (SPOT; candle-1 open = 09:15, see flag d)\n" + "=" * 82)
    print(f"hourly candles: {len(H):,} over {H['date'].nunique():,} days "
          f"[{H['date'].min()} -> {H['date'].max()}]")
    print("\nFIRST ~30 HOURLY CANDLES (eyeball bucketing + RSI seeding):")
    print(H.head(30).to_string(index=False))

    print("\n--- SANITY CHECKS ---")
    print(f"  candles/day distribution: {per_day.value_counts().sort_index().to_dict()} (expect 7 on normal days)")
    print(f"  days with != 7 candles (half-days/special): {len(bad_days)}")
    if len(bad_days):
        print("    e.g.:", dict(list(bad_days.items())[:8]))
    print(f"  RSI out-of-range [0,100] violations: {out_of_range}")
    print(f"  first RSI stamped at row index {first_rsi_idx} (candle #{first_rsi_idx+1}; expect the 15th)")
    print(f"  RSI range: min {rsi_valid.min():.2f}  max {rsi_valid.max():.2f}  mean {rsi_valid.mean():.2f}")
    print(f"  candle-1 (09:07-10) opens: do they equal 09:15? (they must — no 9:07 in data)")
    c1 = H[H["candle"] == 1].head(3)
    print("    first 3 candle-1 opens:", list(c1["open"]))
    print(f"\nSaved -> {OUTDIR}/nifty_hourly_rsi.csv (+ .xlsx)")


if __name__ == "__main__":
    main()
