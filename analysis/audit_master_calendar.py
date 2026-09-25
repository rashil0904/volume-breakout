# -*- coding: utf-8 -*-
"""
audit_master_calendar.py
========================
Calendar / structural-integrity audit of the refreshed 1-min master_data (2022 onward).
Reports only — never modifies data.

(1) Unique trading days per month  — the headline table.
(2) Expected-vs-actual missing-date + discrepancy checks.

DEFINITIONS / FLAGS (see verdict):
  • "trading day present" = a date on which the MASTER has data. We count a date as a real
    trading day only if ≥ MIN_STOCKS_FRAC × peak-daily-stock-count stocks have data that day
    (flag b) — a date with only 1-2 stocks is reported separately as low-coverage (candidate
    bad/partial date), not a full trading day.
  • The master's own calendar is AUTHORITATIVE for market-open days: a genuine NSE trading day
    would have ~all 1,600 stocks, so a real trading day cannot be absent from the master. Hence
    "candidate missing dates" at the calendar level are almost always unlisted holidays, not data
    gaps; true missing DAYS surface per-stock (STEP 3 coverage deviation).
  • NSE holiday calendar: NONE exists in the project (flag a). NSE_HOLIDAYS below is a BEST-EFFORT
    hardcoded reference (2022-2026) — VERIFY against the official NSE list. It is used only to
    LABEL market-closed weekdays; the master defines which days were actually open.
"""
import sys, time
from pathlib import Path
from datetime import date, timedelta
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_loading as dl

IST = "Asia/Kolkata"
MASTER_DIR = dl.MASTER_DIR
OUTDIR = dl.BASE / "results" / "master_calendar_audit"
MIN_STOCKS_FRAC = 0.20          # date counts as a trading day if ≥ frac × peak stocks present
LOW_MONTH_FLAG = 15             # months with fewer trading days than this = suspicious

# ── BEST-EFFORT NSE equity trading holidays (VERIFY against official NSE list) ──
NSE_HOLIDAYS = {
    # 2022
    "2022-01-26", "2022-03-01", "2022-03-18", "2022-04-14", "2022-04-15", "2022-05-03",
    "2022-08-09", "2022-08-15", "2022-08-31", "2022-10-05", "2022-10-24", "2022-10-26", "2022-11-08",
    # 2023
    "2023-01-26", "2023-03-07", "2023-03-30", "2023-04-04", "2023-04-07", "2023-04-14",
    "2023-05-01", "2023-06-28", "2023-08-15", "2023-09-19", "2023-10-02", "2023-10-24",
    "2023-11-14", "2023-11-27", "2023-12-25",
    # 2024
    "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29", "2024-04-11", "2024-04-17",
    "2024-05-01", "2024-05-20", "2024-06-17", "2024-07-17", "2024-08-15", "2024-10-02",
    "2024-11-01", "2024-11-15", "2024-12-25",
    # 2025
    "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14", "2025-04-18",
    "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02", "2025-10-21", "2025-10-22",
    "2025-11-05", "2025-12-25",
    # 2026 (partial / less certain — VERIFY)
    "2026-01-26", "2026-04-03",
}
NSE_HOLIDAYS = {pd.Timestamp(h).date() for h in NSE_HOLIDAYS}


def scan_dates():
    """Per symbol -> set of IST dates; global date -> n_stocks; also weekend/dup flags."""
    per_symbol = {}
    date_count = Counter()
    weekend_hits = Counter()             # weekend date -> n_stocks
    files = sorted(MASTER_DIR.glob("*.parquet"))
    t0 = time.time()
    for i, pq in enumerate(files, 1):
        sym = pq.stem
        try:
            raw = pd.read_parquet(pq, columns=["timestamp"])
        except Exception as e:
            print(f"  !! unreadable {sym}: {e}", flush=True); continue
        if len(raw) == 0:
            continue
        ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        dts = pd.unique(ts.dt.date.values)
        s = set(dts)
        per_symbol[sym] = s
        for d in s:
            date_count[d] += 1
            if d.weekday() >= 5:
                weekend_hits[d] += 1
        if i % 300 == 0:
            print(f"  …{i}/{len(files)} parquets ({time.time()-t0:.0f}s)", flush=True)
    return per_symbol, date_count, weekend_hits


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("MASTER CALENDAR / STRUCTURAL-INTEGRITY AUDIT  (report-only)")
    print("=" * 78)
    print(f"Scanning {len(list(MASTER_DIR.glob('*.parquet')))} parquets (dates only) …")
    per_symbol, date_count, weekend_hits = scan_dates()
    if not date_count:
        print("No data."); return

    all_dates = sorted(date_count)
    dmin, dmax = all_dates[0], all_dates[-1]
    peak = max(date_count.values())
    thresh = MIN_STOCKS_FRAC * peak
    trading_days = sorted(d for d, n in date_count.items() if n >= thresh)
    td_set = set(trading_days)
    low_cov = sorted((d, n) for d, n in date_count.items() if 0 < n < thresh and d.weekday() < 5)
    print(f"\nmaster span: {dmin} → {dmax} | peak stocks/day = {peak} | "
          f"trading-day threshold = {thresh:.0f} stocks")
    print(f"trading days (≥threshold): {len(trading_days)} | low-coverage weekday dates: {len(low_cov)}")

    # ── STEP 1 — unique trading days per month ──
    def months_between(a, b):
        out, y, m = [], a.year, a.month
        while (y, m) <= (b.year, b.month):
            out.append((y, m)); y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return out

    def weekdays_in(y, m):
        d = date(y, m, 1); end = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        wk = []
        while d < end:
            if d.weekday() < 5:
                wk.append(d)
            d += timedelta(days=1)
        return wk

    td_by_month = Counter(); any_by_month = Counter(); low_by_month = Counter()
    for d in trading_days:
        td_by_month[(d.year, d.month)] += 1
    for d in all_dates:
        any_by_month[(d.year, d.month)] += 1
    for d, _ in low_cov:
        low_by_month[(d.year, d.month)] += 1

    missing_rows = []
    month_rows = []
    for (y, m) in months_between(dmin, dmax):
        wkdays = weekdays_in(y, m)
        hols = [d for d in wkdays if d in NSE_HOLIDAYS]
        exp_trading = [d for d in wkdays if d not in NSE_HOLIDAYS]
        actual = td_by_month.get((y, m), 0)
        # candidate missing = expected trading day (weekday, non-holiday) absent from master
        cand_missing = [d for d in exp_trading if d not in td_set]
        for d in cand_missing:
            n = date_count.get(d, 0)
            missing_rows.append({"date": d, "weekday": d.strftime("%a"),
                                 "n_stocks_present": n,
                                 "note": "no data (likely unlisted holiday)" if n == 0
                                 else "PARTIAL data — candidate bad/missing date"})
        verdict = "complete"
        if len(cand_missing):
            verdict = f"{len(cand_missing)} candidate missing day(s)"
        if actual < LOW_MONTH_FLAG:
            verdict = f"LOW ({actual} trading days) — {verdict}"
        if low_by_month.get((y, m), 0):
            verdict += f"; {low_by_month[(y,m)]} low-coverage date(s)"
        month_rows.append({
            "year": y, "month": m, "month_label": f"{y}-{m:02d}",
            "n_unique_trading_days": actual,
            "n_dates_any_stock": any_by_month.get((y, m), 0),
            "expected_trading_days": len(exp_trading),
            "known_holidays": len(hols),
            "diff_expected_minus_actual": len(exp_trading) - actual,
            "n_low_coverage_dates": low_by_month.get((y, m), 0),
            "verdict": verdict,
        })
    monthly = pd.DataFrame(month_rows)
    missing = pd.DataFrame(missing_rows)

    # ── STEP 3 — discrepancy checks ──
    disc = []
    # weekend data
    for d, n in sorted(weekend_hits.items()):
        disc.append({"type": "weekend_data", "date": d, "detail": f"{d.strftime('%a')}, {n} stocks have data"})
    # holidays with data (>= threshold => a 'closed' day that looks open)
    for d in sorted(NSE_HOLIDAYS):
        if dmin <= d <= dmax and date_count.get(d, 0) >= thresh:
            disc.append({"type": "holiday_with_data", "date": d,
                         "detail": f"listed NSE holiday but {date_count[d]} stocks have data (>=threshold)"})
    # low-coverage weekday dates (candidate partial/bad trading dates)
    for d, n in low_cov:
        disc.append({"type": "low_coverage_date", "date": d, "detail": f"{d.strftime('%a')}, only {n} stocks"})
    # date-sequence gaps: consecutive trading days with intervening NON-holiday weekdays
    for a, b in zip(trading_days, trading_days[1:]):
        gap = (b - a).days
        if gap > 1:
            inter = [a + timedelta(days=k) for k in range(1, gap)]
            unexplained = [d for d in inter if d.weekday() < 5 and d not in NSE_HOLIDAYS]
            if unexplained:
                disc.append({"type": "sequence_gap", "date": a,
                             "detail": f"{a} -> {b}: {len(unexplained)} unexplained weekday(s) missing "
                                       f"({unexplained[0]}..{unexplained[-1]})"})
    # date-range sanity
    if dmin > date(2022, 1, 10):
        disc.append({"type": "range_start", "date": dmin, "detail": f"master starts {dmin} (expected ~2022-01)"})
    if dmax < date.today() - timedelta(days=10):
        disc.append({"type": "range_end", "date": dmax, "detail": f"master ends {dmax} (looks truncated)"})
    discrepancies = pd.DataFrame(disc)

    # ── per-stock coverage deviation (whole missing DAYS within each stock's life) ──
    cov_rows = []
    for sym, s in per_symbol.items():
        if not s:
            continue
        f, l = min(s), max(s)
        exp = [d for d in trading_days if f <= d <= l]
        miss = [d for d in exp if d not in s]
        cov_rows.append({"symbol": sym, "first_date": f, "last_date": l,
                         "expected_trading_days": len(exp), "present": len(exp) - len(miss),
                         "missing_days": len(miss),
                         "coverage_pct": round((len(exp) - len(miss)) / len(exp) * 100, 3) if exp else 0.0,
                         "listed_after_start": f > trading_days[0],
                         "delisted_before_end": l < trading_days[-1]})
    per_stock = pd.DataFrame(cov_rows).sort_values("missing_days", ascending=False)

    # ── save ──
    monthly.to_csv(OUTDIR / "monthly_trading_days.csv", index=False)
    missing.to_csv(OUTDIR / "missing_dates.csv", index=False)
    discrepancies.to_csv(OUTDIR / "discrepancies.csv", index=False)
    per_stock.to_csv(OUTDIR / "per_stock_date_coverage.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "master_calendar_audit.xlsx", engine="openpyxl") as w:
        monthly.to_excel(w, sheet_name="monthly_trading_days", index=False)
        (missing if len(missing) else pd.DataFrame(columns=["date"])).to_excel(w, sheet_name="missing_dates", index=False)
        (discrepancies if len(discrepancies) else pd.DataFrame(columns=["type"])).to_excel(w, sheet_name="discrepancies", index=False)
        per_stock.to_excel(w, sheet_name="per_stock_date_coverage", index=False)

    # ── console report ──
    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print("STEP 1 — UNIQUE TRADING DAYS PER MONTH")
    print("=" * 78)
    print(monthly[["month_label", "n_unique_trading_days", "expected_trading_days",
                   "diff_expected_minus_actual", "verdict"]].to_string(index=False))
    print(f"\n  yearly totals (trading days):")
    yr = monthly.groupby("year")["n_unique_trading_days"].sum()
    for y, n in yr.items():
        print(f"    {y}: {n}")

    print("\n" + "=" * 78)
    print("STEP 2/3 — DISCREPANCIES")
    print("=" * 78)
    n_cand = len(missing); n_partial = int((missing["n_stocks_present"] > 0).sum()) if len(missing) else 0
    tcounts = discrepancies["type"].value_counts().to_dict() if len(discrepancies) else {}
    print(f"  candidate missing dates              : {n_cand}  (of which PARTIAL-data suspects: {n_partial})")
    print(f"  weekend dates with data              : {tcounts.get('weekend_data', 0)}")
    print(f"  listed holidays with data            : {tcounts.get('holiday_with_data', 0)}")
    print(f"  low-coverage weekday dates           : {tcounts.get('low_coverage_date', 0)}")
    print(f"  date-sequence gaps (unexplained)     : {tcounts.get('sequence_gap', 0)}")
    print(f"  date-range: {dmin} → {dmax}")
    if len(missing):
        print("\n  candidate missing dates (verify vs official NSE holidays):")
        print(missing.to_string(index=False))
    stk = per_stock[(per_stock["missing_days"] > 0) & (~per_stock["listed_after_start"]) &
                    (~per_stock["delisted_before_end"])]
    print(f"\n  stocks with interior missing DAYS (bounded by data, full-life): {len(stk)} "
          f"| worst: {', '.join(stk.head(5)['symbol'] + '(' + stk.head(5)['missing_days'].astype(str) + ')') if len(stk) else '—'}")

    n_clean = int((monthly["verdict"] == "complete").sum())
    print("\n" + "=" * 78)
    print(f"VERDICT: {n_clean}/{len(monthly)} months clean ('complete'); "
          f"{len(monthly)-n_clean} flagged (see monthly table + discrepancies.csv).")
    print("  FLAGS: (a) no NSE holiday file in project — NSE_HOLIDAYS is best-effort, VERIFY;")
    print("         candidate-missing dates with n_stocks=0 are almost certainly unlisted holidays.")
    print(f"         (b) trading day = date with ≥{thresh:.0f} stocks; low-coverage dates listed separately.")
    print(f"\nSaved 4 CSVs + Excel → {OUTDIR}")


if __name__ == "__main__":
    main()
