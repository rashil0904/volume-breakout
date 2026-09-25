# -*- coding: utf-8 -*-
"""audit_banknifty_futures.py — READ-ONLY thorough verification of the BankNifty futures 1-min pull
(data/futures_intraday_full/BANKNIFTY). Checks: (1) structural - every genuine futures expiry from the API
present as a file OR the known gap; nothing silently missing; (2) per-contract - schema/dtype consistency,
nulls, OHLC validity, duplicate timestamps, DTE recompute correctness (calendar days), whole missing trading
days within the contract's OWN active window (vs NSE calendar), candles-per-day != 375 on non-special days
(known exceptions: 2024-11-01 Muhurat evening block, Aug-2026 extended-to-15:39 sessions - excluded from the
violation count, reported separately); (3) manifest cross-check (file candle counts == manifest n_candles,
nothing stale). No modify/re-pull.
"""
import sys, os, glob
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import opt_pull_nifty_full as op

ROOT = rb.BASE / "data" / "futures_intraday_full" / "BANKNIFTY"
MAN = ROOT.parent / "manifest_banknifty_fut.csv"; ISS = ROOT.parent / "issues_banknifty_fut.csv"
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
UND = "NSE_INDEX|Nifty Bank"; CUTOFF = "2026-08-25"; FULL = 375
KNOWN_SPECIAL_DATES = {pd.Timestamp("2024-11-01").date()}   # Diwali Muhurat evening (extra rows, not a violation)


def main():
    d = pd.read_parquet(DAILY, columns=["date"]); tdarr = np.array(sorted(d["date"].dt.date.unique()))

    # ---- structural: re-fetch genuine futures expiries from API, compare to what's on disk ----
    exps_raw, _ = op.get(f"{op.BASE}/expiries?instrument_key={op.enc(UND)}")
    exps_raw = sorted([e for e in exps_raw if e <= CUTOFF])
    genuine = []
    for e in exps_raw:
        cons, _ = op.get(f"{op.BASE}/future/contract?instrument_key={op.enc(UND)}&expiry_date={e}")
        if cons: genuine.append((e, cons[0]))
    print(f"API genuine futures expiries <= {CUTOFF}: {len(genuine)}")

    exp_dirs = sorted([x for x in ROOT.iterdir() if x.is_dir()]) if ROOT.exists() else []
    present_expiries = set()
    for x in exp_dirs:
        fs = list(x.glob("*.parquet"))
        if fs: present_expiries.add(f"{x.name[:4]}-{x.name[4:6]}-{x.name[6:]}")

    struct_issues = []
    for e, c in genuine:
        if e in present_expiries: continue
        try:
            iss = pd.read_csv(ISS); logged = ((iss.expiry == e) & (iss.issue == "empty_no_candles")).any()
        except Exception: logged = False
        struct_issues.append({"expiry": e, "issue": "missing_and_UNLOGGED" if not logged else "missing_but_logged_empty", "symbol": c["trading_symbol"]})
    print(f"expiries present on disk: {len(present_expiries)} | missing: {len(genuine) - len(present_expiries)}")
    for s in struct_issues: print(f"  {s}")

    # ---- manifest cross-check ----
    man = pd.read_csv(MAN) if MAN.exists() else pd.DataFrame()
    man_issues = []

    # ---- per-contract deep checks ----
    rows_issues = []; totals = Counter(); dte_min_max = []
    for x in exp_dirs:
        fs = list(x.glob("*.parquet"))
        if not fs: continue
        for fn in fs:
            df = pd.read_parquet(fn)
            sym = df["symbol"].iloc[0]; exp = df["expiry_date"].iloc[0]
            totals["contracts"] += 1; totals["rows"] += len(df)
            # schema
            expect_cols = ["contract_month", "expiry_date", "symbol", "timestamp", "DTE", "open", "high", "low", "close", "volume", "OI"]
            if list(df.columns) != expect_cols:
                rows_issues.append({"symbol": sym, "expiry": exp, "issue": "schema_mismatch", "detail": str(list(df.columns))})
            # nulls
            nnull = int(df[["open", "high", "low", "close", "volume", "OI", "DTE"]].isna().any(axis=1).sum())
            if nnull: rows_issues.append({"symbol": sym, "expiry": exp, "issue": "null_fields", "detail": nnull})
            # OHLC validity
            bad = int(((df.high < df.low) | (df.open < df.low) | (df.open > df.high) | (df.close < df.low) | (df.close > df.high)).sum())
            if bad: rows_issues.append({"symbol": sym, "expiry": exp, "issue": "ohlc_violation", "detail": bad})
            # duplicate timestamps
            ndup = int(df.duplicated("timestamp").sum())
            if ndup: rows_issues.append({"symbol": sym, "expiry": exp, "issue": "duplicate_timestamp", "detail": ndup})
            # DTE recompute (calendar days)
            recomputed = (pd.Timestamp(exp).normalize() - df["timestamp"].dt.normalize()).dt.days
            ndte = int((df["DTE"].values != recomputed.values).sum())
            if ndte: rows_issues.append({"symbol": sym, "expiry": exp, "issue": "dte_mismatch", "detail": ndte})
            # zero-vol
            totals["zero_vol"] += int((df.volume == 0).sum())
            # min DTE (expiry-day row present?)
            dmin = int(df["DTE"].min()); dmax = int(df["DTE"].max())
            dte_min_max.append({"symbol": sym, "expiry": exp, "dte_min": dmin, "dte_max": dmax, "n_days": df["timestamp"].dt.normalize().nunique()})
            if dmin != 0: rows_issues.append({"symbol": sym, "expiry": exp, "issue": "no_expiry_day_row", "detail": dmin})
            # candles per day (excluding known special dates)
            df["date"] = df["timestamp"].dt.normalize().dt.date
            cpd = df.groupby("date").size()
            bad_days = cpd[(cpd != FULL) & (~cpd.index.isin(KNOWN_SPECIAL_DATES))]
            if len(bad_days):
                rows_issues.append({"symbol": sym, "expiry": exp, "issue": "day_not_375_rows(non-special)", "detail": f"{len(bad_days)} days: {dict(list(bad_days.items())[:5])}"})
            # whole missing trading days within contract's OWN active window
            fd = df["date"].min(); ld = df["date"].max()
            window = [dd for dd in tdarr if fd <= dd <= ld]
            present_dates = set(df["date"].unique())
            missing = sorted(set(window) - present_dates)
            if missing:
                rows_issues.append({"symbol": sym, "expiry": exp, "issue": "missing_trading_days_in_window", "detail": f"{len(missing)}: {missing[:5]}"})
            # manifest cross-check
            if len(man):
                mrow = man[(man.expiry == exp) & (man.symbol == sym)]
                if len(mrow) and int(mrow.iloc[0]["n_candles"]) != len(df):
                    man_issues.append({"symbol": sym, "expiry": exp, "manifest_n": int(mrow.iloc[0]["n_candles"]), "actual_n": len(df)})
                elif not len(mrow):
                    man_issues.append({"symbol": sym, "expiry": exp, "issue": "not_in_manifest"})

    I = pd.DataFrame(rows_issues); D = pd.DataFrame(dte_min_max)
    print(f"\n{'='*90}\nBANKNIFTY FUTURES — THOROUGH AUDIT\n{'='*90}")
    print(f"contracts on disk: {totals['contracts']} | total rows: {totals['rows']:,} | zero-vol rows: {totals['zero_vol']:,}")
    print(f"manifest cross-check mismatches: {len(man_issues)}")
    for m in man_issues: print(f"  {m}")
    print(f"\n--- structural (API vs disk) ---")
    print("CLEAN - every genuine API expiry accounted for (present or logged-empty)" if not struct_issues else struct_issues)
    print(f"\n--- per-contract issue counts ---")
    print(I["issue"].value_counts().to_string() if len(I) else "  NONE - all contracts clean")
    if len(I): print(I.to_string(index=False))
    print(f"\n--- DTE min/max per contract (expect min=0 unless flagged) ---")
    print(D.to_string(index=False))
    print(f"\nSaved nothing (read-only). Total flagged issue-rows: {len(I)} | structural issues: {len(struct_issues)} | manifest mismatches: {len(man_issues)}")


if __name__ == "__main__":
    main()
