# -*- coding: utf-8 -*-
"""verify_new_pull_data.py — verification (audit only, no filling) scoped ONLY to the two pulls just run:
(1) the 1609 existing stocks' EXTENSION (2026-08-01 onward, appended by extend_1609_to_latest.py), and
(2) the 953 new-listing stocks' full pulled history (master_data_new_listings/, by
pull_new_listed_stocks.py). Does NOT re-scan the pre-existing 2022-2026-07-31 history.

Checks per stock: duplicate timestamps, non-monotonic ordering, out-of-session candles, and per-day
candle-count anomalies relative to the cross-sectional "normal" for that date (peer-day comparison, since
holidays/half-days affect all stocks the same way that day).
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = "Asia/Kolkata"
HM_OPEN, HM_CLOSE = 555, 929   # 09:15..15:29
EXT_FLOOR = pd.Timestamp("2026-08-01").date()
OUTDIR = rb.RESULTS / "new_pull_verification"; OUTDIR.mkdir(parents=True, exist_ok=True)


def load_and_slice(fn, floor=None):
    df = pd.read_parquet(fn, columns=["timestamp", "open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    df = df.assign(ts=ts, date=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)
    if floor is not None:
        df = df[df["date"] >= floor]
    return df


def per_symbol_checks(sym, df):
    issues = []
    dup = df["ts"].duplicated().sum()
    if dup:
        issues.append({"symbol": sym, "issue": "duplicate_timestamps", "detail": int(dup)})
    if not df["ts"].is_monotonic_increasing:
        issues.append({"symbol": sym, "issue": "non_monotonic", "detail": ""})
    oos = (~df["hm"].between(HM_OPEN, HM_CLOSE)).sum()
    if oos:
        issues.append({"symbol": sym, "issue": "out_of_session_candles", "detail": int(oos)})
    return issues


def main():
    # ---- PART 1: verify the 1609 extension (Aug-1-2026 onward only) ----
    print("=== PART 1: verifying 1609 extension (2026-08-01 onward) ===", flush=True)
    files_1609 = sorted(rb.MASTER_DIR.glob("*.parquet"))
    all_issues_1 = []
    day_counts_1 = {}
    t0 = time.time()
    for i, fn in enumerate(files_1609, 1):
        sym = fn.stem
        df = load_and_slice(fn, floor=EXT_FLOOR)
        if df.empty:
            all_issues_1.append({"symbol": sym, "issue": "no_extension_data_at_all", "detail": ""})
            continue
        all_issues_1.extend(per_symbol_checks(sym, df))
        for d, n in df.groupby("date").size().items():
            day_counts_1.setdefault(d, []).append((sym, n))
        if i % 400 == 0:
            print(f"  ...{i}/{len(files_1609)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"scan complete ({time.time()-t0:.0f}s). issues found: {len(all_issues_1)}", flush=True)

    # cross-sectional per-day anomaly check: flag stocks whose candle count that day deviates
    # sharply from the day's own peer median (catches partial/truncated pulls, not real holidays)
    anomalies_1 = []
    for d, lst in sorted(day_counts_1.items()):
        counts = np.array([n for _, n in lst])
        med = np.median(counts)
        for sym, n in lst:
            if n < med * 0.5 and med >= 300:   # this stock got <50% of peers' candles on a normal-looking day
                anomalies_1.append({"date": d, "symbol": sym, "n_candles": n, "peer_median": med})
    print(f"cross-sectional anomalies (stock got <50% of peer median candles on an otherwise-normal day): {len(anomalies_1)}", flush=True)

    I1 = pd.DataFrame(all_issues_1); A1 = pd.DataFrame(anomalies_1)
    I1.to_csv(OUTDIR / "part1_1609_extension_issues.csv", index=False)
    A1.to_csv(OUTDIR / "part1_1609_extension_anomalies.csv", index=False)
    if len(I1):
        print("\n--- Part 1 issues by type ---"); print(I1["issue"].value_counts().to_string())
    if len(A1):
        print("\n--- Part 1 anomalies (sample) ---"); print(A1.head(20).to_string(index=False))

    # boundary continuity check: confirm no stock has a duplicated or overlapping date at Jul31/Aug1 seam
    boundary_issues = []
    for fn in files_1609:
        df_full = pd.read_parquet(fn, columns=["timestamp"])
        ts = pd.to_datetime(df_full["timestamp"], utc=True).dt.tz_convert(IST).dt.tz_localize(None)
        d = ts.dt.date
        around = sorted(set(d[(d >= pd.Timestamp("2026-07-28").date()) & (d <= pd.Timestamp("2026-08-04").date())]))
        if len(around) != len(set(around)):
            boundary_issues.append({"symbol": fn.stem, "dates_around_boundary": around})
    print(f"\nboundary continuity: {len(boundary_issues)} stocks with any duplicate-date issue around Jul31/Aug1 seam", flush=True)

    # ---- PART 2: verify new-listings full pull ----
    print("\n=== PART 2: verifying new-listings pull (master_data_new_listings/) ===", flush=True)
    new_dir = rb.BASE / "master_data_new_listings"
    files_new = sorted(new_dir.glob("*.parquet"))
    all_issues_2 = []
    day_counts_2 = {}
    t0 = time.time()
    for i, fn in enumerate(files_new, 1):
        sym = fn.stem
        df = load_and_slice(fn, floor=None)
        if df.empty:
            all_issues_2.append({"symbol": sym, "issue": "empty_file", "detail": ""})
            continue
        all_issues_2.extend(per_symbol_checks(sym, df))
        for d, n in df.groupby("date").size().items():
            day_counts_2.setdefault(d, []).append((sym, n))
        if i % 200 == 0:
            print(f"  ...{i}/{len(files_new)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"scan complete ({time.time()-t0:.0f}s). issues found: {len(all_issues_2)}", flush=True)

    anomalies_2 = []
    for d, lst in sorted(day_counts_2.items()):
        counts = np.array([n for _, n in lst])
        if len(counts) < 5:
            continue   # too few stocks trading that day to get a reliable peer median (early listings)
        med = np.median(counts)
        for sym, n in lst:
            if n < med * 0.5 and med >= 300:
                anomalies_2.append({"date": d, "symbol": sym, "n_candles": n, "peer_median": med})
    print(f"cross-sectional anomalies: {len(anomalies_2)}", flush=True)

    I2 = pd.DataFrame(all_issues_2); A2 = pd.DataFrame(anomalies_2)
    I2.to_csv(OUTDIR / "part2_new_listings_issues.csv", index=False)
    A2.to_csv(OUTDIR / "part2_new_listings_anomalies.csv", index=False)
    if len(I2):
        print("\n--- Part 2 issues by type ---"); print(I2["issue"].value_counts().to_string())
    if len(A2):
        print("\n--- Part 2 anomalies (sample) ---"); print(A2.head(20).to_string(index=False))

    with pd.ExcelWriter(OUTDIR / "new_pull_verification_summary.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"part": "1 (1609 extension, Aug-1-2026 onward)", "n_stocks_scanned": len(files_1609),
             "n_issues": len(I1), "n_cross_sectional_anomalies": len(A1), "n_boundary_issues": len(boundary_issues)},
            {"part": "2 (953 new listings, full history)", "n_stocks_scanned": len(files_new),
             "n_issues": len(I2), "n_cross_sectional_anomalies": len(A2), "n_boundary_issues": "n/a"},
        ]).to_excel(w, sheet_name="Overview", index=False)
        I1.to_excel(w, sheet_name="Part1_Issues", index=False)
        A1.to_excel(w, sheet_name="Part1_Anomalies", index=False)
        pd.DataFrame(boundary_issues).to_excel(w, sheet_name="Part1_Boundary", index=False)
        I2.to_excel(w, sheet_name="Part2_Issues", index=False)
        A2.to_excel(w, sheet_name="Part2_Anomalies", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)

    print(f"\nSaved -> {OUTDIR}/new_pull_verification_summary.xlsx")


if __name__ == "__main__":
    main()
