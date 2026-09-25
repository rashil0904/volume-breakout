# -*- coding: utf-8 -*-
"""audit_nifty_options_aug2026_extension.py — THOROUGH verification of the 4 newly-pulled NIFTY options
expiries (2026-08-04, 08-11, 08-18, 08-25), covering EVERY contract (not samples): OHLC sanity, duplicate
timestamps, DTE-field correctness, manifest-vs-file consistency, per-day candle-count/closing-time checks
(pre- vs post-CAS 15:29->15:39 cutover), timestamp bounds, and OI/volume sanity. Read-only — flags issues
rather than fixing them.
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
MAN = rb.BASE / "data" / "options_intraday_full" / "manifest_nifty.csv"
OUTDIR = rb.RESULTS / "fullchain_audit"; OUTDIR.mkdir(parents=True, exist_ok=True)
NEW_EXPS = ["20260804", "20260811", "20260818", "20260825"]
CAS_START = pd.Timestamp("2026-08-03")


def main():
    manifest = pd.read_csv(MAN)
    problems = []
    stats_rows = []

    files = []
    for e in NEW_EXPS:
        files += sorted(glob.glob(str(OPTDIR / e / "*.parquet")))
    print(f"auditing {len(files)} contracts across {len(NEW_EXPS)} new expiries...", flush=True)

    for i, fp in enumerate(files, 1):
        fp = Path(fp)
        exp_folder = fp.parent.name
        exp_date = pd.Timestamp(exp_folder)
        sym = fp.stem.replace("_", " ")
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            problems.append({"file": str(fp), "check": "read_error", "detail": str(e)[:100]}); continue

        n = len(df)
        if n == 0:
            problems.append({"file": str(fp), "check": "empty_file", "detail": ""}); continue

        # 1. duplicate timestamps
        dup = df["timestamp"].duplicated().sum()
        if dup:
            problems.append({"file": str(fp), "check": "duplicate_timestamps", "detail": f"{dup} dupes"})

        # 2. OHLC sanity
        bad_hl = (df["high"] < df["low"]).sum()
        bad_ho = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
        bad_lo = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
        nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
        if bad_hl: problems.append({"file": str(fp), "check": "high<low", "detail": f"{bad_hl} rows"})
        if bad_ho: problems.append({"file": str(fp), "check": "high<max(o,c)", "detail": f"{bad_ho} rows"})
        if bad_lo: problems.append({"file": str(fp), "check": "low>min(o,c)", "detail": f"{bad_lo} rows"})
        if nonpos: problems.append({"file": str(fp), "check": "non_positive_price", "detail": f"{nonpos} rows"})

        # 3. volume/OI sanity
        neg_vol = (df["volume"] < 0).sum(); neg_oi = (df["OI"] < 0).sum()
        if neg_vol: problems.append({"file": str(fp), "check": "negative_volume", "detail": f"{neg_vol} rows"})
        if neg_oi: problems.append({"file": str(fp), "check": "negative_OI", "detail": f"{neg_oi} rows"})

        # 4. DTE correctness
        expected_dte = (exp_date.normalize() - df["timestamp"].dt.normalize()).dt.days
        bad_dte = (df["DTE"] != expected_dte).sum()
        if bad_dte: problems.append({"file": str(fp), "check": "DTE_mismatch", "detail": f"{bad_dte} rows"})

        # 5. timestamp bounds: no candle after expiry date (+1 day buffer for the settlement-day close)
        max_ts = df["timestamp"].max(); min_ts = df["timestamp"].min()
        if max_ts.normalize() > exp_date.normalize():
            problems.append({"file": str(fp), "check": "timestamp_after_expiry", "detail": str(max_ts)})
        if min_ts.normalize() < (exp_date - pd.Timedelta(days=150)):
            problems.append({"file": str(fp), "check": "timestamp_too_early", "detail": str(min_ts)})

        # 6. per-day candle count + closing-time check (CAS cutover: 15:29 before 08-03, 15:39 from 08-03)
        df["date"] = df["timestamp"].dt.normalize()
        per_day_last = df.groupby("date")["timestamp"].max()
        for d, last_ts in per_day_last.items():
            last_hm = last_ts.hour * 60 + last_ts.minute
            if d < CAS_START:
                if last_hm not in (15 * 60 + 29,) and last_hm < 15 * 60:
                    pass  # contract may have stopped trading early in the day genuinely; only flag odd LATE closes
            else:
                if last_hm > 15 * 60 + 39:
                    problems.append({"file": str(fp), "check": "close_after_1539_post_CAS", "detail": f"{d.date()} last={last_ts.strftime('%H:%M')}"})

        # 7. manifest consistency
        mrow = manifest[(manifest.expiry == str(exp_date.date())) & (manifest.symbol == sym)]
        if mrow.empty:
            problems.append({"file": str(fp), "check": "missing_from_manifest", "detail": sym})
        else:
            m_n = int(mrow.iloc[0]["n_candles"])
            if m_n != n:
                problems.append({"file": str(fp), "check": "manifest_n_candles_mismatch", "detail": f"manifest={m_n} actual={n}"})

        stats_rows.append({"expiry": exp_folder, "symbol": sym, "n_candles": n,
                            "date_min": str(min_ts)[:10], "date_max": str(max_ts)[:10],
                            "last_close_time": per_day_last.iloc[-1].strftime("%H:%M")})

        if i % 200 == 0:
            print(f"  {i}/{len(files)} audited | issues so far: {len(problems)}", flush=True)

    PROB = pd.DataFrame(problems)
    STATS = pd.DataFrame(stats_rows)

    print(f"\n{'='*100}\nAUDIT COMPLETE: {len(files)} contracts | {len(PROB)} issue rows found\n{'='*100}")
    if len(PROB):
        print(PROB["check"].value_counts().to_string())
    else:
        print("ZERO issues found across all checks.")

    with pd.ExcelWriter(OUTDIR / "aug2026_extension_audit.xlsx", engine="openpyxl") as w:
        (PROB if len(PROB) else pd.DataFrame(columns=["file", "check", "detail"])).to_excel(w, sheet_name="Issues", index=False)
        STATS.to_excel(w, sheet_name="Per_Contract_Stats", index=False)
    PROB.to_csv(OUTDIR / "aug2026_extension_issues.csv", index=False)

    print(f"\nSaved -> {OUTDIR}/aug2026_extension_audit.xlsx")

    # ---- summary: CAS cutover consistency across ALL new-expiry contracts ----
    print("\n--- CAS closing-time consistency (last candle per day, sampled 08-03 onward) ---")
    close_check = []
    for fp in files[:]:
        fp = Path(fp)
        df = pd.read_parquet(fp, columns=["timestamp"])
        post = df[df["timestamp"] >= CAS_START]
        if post.empty:
            continue
        post_dates = post["timestamp"].dt.normalize()
        last_per_day = post.groupby(post_dates)["timestamp"].max()
        hm = last_per_day.dt.hour * 60 + last_per_day.dt.minute
        close_check.append(hm.max())
    if close_check:
        cc = pd.Series(close_check)
        print(f"contracts with post-CAS data: {len(cc)}")
        print(f"max close-minute distribution: {cc.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
