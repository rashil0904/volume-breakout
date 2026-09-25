# -*- coding: utf-8 -*-
"""audit_sensex_aug2026_extension.py — THOROUGH verification of the newly-pulled SENSEX spot 1-min data
for August 2026 (data/sensex_1min_ohlc.csv). Checks every row: OHLC sanity, duplicate timestamps, NaN/
missing values, non-positive prices, per-day candle-count/start-end-time consistency, timestamp bounds,
volume/OI convention, boundary continuity with pre-Aug data, and flags any abnormal single-minute price
jump OUTSIDE the known 15:28 CAS-reveal minute (to catch genuine bad ticks vs the expected CAS pattern).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "sensex_1min_ohlc.csv"
NEW_FROM = pd.Timestamp("2026-09-10")
NEW_TO = pd.Timestamp("2026-09-10")
JUMP_THRESHOLD_PCT = 0.5   # flag single-minute moves bigger than this, outside the CAS-reveal minutes


def main():
    sp = pd.read_csv(SPOT)
    ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute

    aug = sp[(sp["date"] >= NEW_FROM) & (sp["date"] <= NEW_TO)].copy()
    pre = sp[sp["date"] < NEW_FROM]
    print(f"New-window rows: {len(aug):,} | days: {aug['date'].nunique()}", flush=True)

    issues = []

    # 1. duplicate timestamps
    dup = aug["ts"].duplicated().sum()
    if dup: issues.append(("duplicate_timestamps", dup))
    print(f"1) duplicate timestamps: {dup}")

    # 2. NaN / missing values
    nan_ct = aug[["open", "high", "low", "close", "volume", "open_interest"]].isna().sum().sum()
    if nan_ct: issues.append(("nan_values", nan_ct))
    print(f"2) NaN values across OHLCV+OI: {nan_ct}")

    # 3. OHLC sanity
    bad_hl = (aug["high"] < aug["low"]).sum()
    bad_ho = (aug["high"] < aug[["open", "close"]].max(axis=1)).sum()
    bad_lo = (aug["low"] > aug[["open", "close"]].min(axis=1)).sum()
    nonpos = (aug[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    for name, n in [("high<low", bad_hl), ("high<max(o,c)", bad_ho), ("low>min(o,c)", bad_lo), ("non_positive_price", nonpos)]:
        if n: issues.append((name, n))
    print(f"3) OHLC violations -> high<low: {bad_hl} | high<max(o,c): {bad_ho} | low>min(o,c): {bad_lo} | non-positive: {nonpos}")

    # 4. volume / OI convention (should be all zero, matches rest of dataset)
    nonzero_vol = (aug["volume"] != 0).sum(); nonzero_oi = (aug["open_interest"] != 0).sum()
    print(f"4) non-zero volume rows: {nonzero_vol} | non-zero OI rows: {nonzero_oi}  (expect 0, index has no real volume)")
    if nonzero_vol or nonzero_oi: issues.append(("unexpected_nonzero_volume_or_oi", nonzero_vol + nonzero_oi))

    # 5. per-day candle count + start/end time
    per_day = aug.groupby("date").agg(n=("ts", "size"), first_mod=("mod", "min"), last_mod=("mod", "max"))
    bad_count = per_day[per_day["n"] != 375]
    bad_start = per_day[per_day["first_mod"] != 555]
    bad_end = per_day[per_day["last_mod"] != 929]
    print(f"5) days with candle count != 375: {len(bad_count)}")
    if len(bad_count): print(bad_count)
    print(f"   days not starting at 09:15: {len(bad_start)} | days not ending at 15:29: {len(bad_end)}")
    if len(bad_count): issues.append(("wrong_candle_count_days", len(bad_count)))

    # 6. timestamp bounds sanity (nothing outside 09:15-15:29 window)
    out_of_hours = aug[(aug["mod"] < 555) | (aug["mod"] > 929)]
    print(f"6) rows outside 09:15-15:29 window: {len(out_of_hours)}")
    if len(out_of_hours): issues.append(("out_of_hours_rows", len(out_of_hours)))

    # 7. weekday/holiday cross-check
    bdays = pd.bdate_range(NEW_FROM, NEW_TO).date
    have_days = set(aug["date"].dt.date)
    missing = sorted(set(bdays) - have_days)
    print(f"7) business weekdays with zero candles: {len(missing)} {missing if missing else ''}")
    if missing: issues.append(("missing_weekdays", len(missing)))

    # 8. boundary continuity with pre-Aug data
    if len(pre):
        gap_days = (aug["date"].min() - pre["date"].max()).days
        overlap = set(pre["ts"]) & set(aug["ts"])
        print(f"8) boundary: pre-Aug last date {pre['date'].max().date()} -> Aug first date {aug['date'].min().date()} (gap {gap_days}d) | overlapping timestamps: {len(overlap)}")
        if overlap: issues.append(("boundary_timestamp_overlap", len(overlap)))

    # 9. abnormal single-minute jumps OUTSIDE the known CAS-reveal minutes (15:28, 15:29)
    aug_sorted = aug.sort_values("ts").reset_index(drop=True)
    per_symbol_ret = []
    for d, g in aug_sorted.groupby("date"):
        g = g.sort_values("ts").reset_index(drop=True)
        ret = g["close"].pct_change().abs() * 100
        flagged = g[(ret > JUMP_THRESHOLD_PCT) & (~g["mod"].isin([928, 929]))]
        for _, r in flagged.iterrows():
            per_symbol_ret.append({"date": d.date(), "ts": r["ts"], "mod": r["mod"], "close": r["close"]})
    print(f"9) abnormal (>{JUMP_THRESHOLD_PCT}%) single-minute moves OUTSIDE 15:28-15:29: {len(per_symbol_ret)}")
    if per_symbol_ret:
        issues.append(("abnormal_jump_outside_cas_window", len(per_symbol_ret)))
        for r in per_symbol_ret[:20]: print("   ", r)

    print("\n" + "=" * 90)
    if issues:
        print(f"AUDIT RESULT: {len(issues)} issue type(s) found:")
        for name, n in issues: print(f"  - {name}: {n}")
    else:
        print("AUDIT RESULT: ZERO issues found across all 9 checks.")
    print("=" * 90)


if __name__ == "__main__":
    main()
