# -*- coding: utf-8 -*-
"""surface_recheck.py — fast READ-ONLY sanity sweep of the densified full-chain NIFTY options dataset
after the Muhurat fixes. Stratified sample (~15 files/expiry + all Muhurat-affected expiries fully) +
structural counts. Checks: timestamps in-session (09:15-15:29 or the 2 Muhurat special windows), OHLC
validity, DTE == calendar recompute & >=0 & reaches 0, nulls, dup timestamps, real zero-vol O=H=L=C,
is_synthetic present. Verifies both Muhurat sessions hold real rows. Report only.
"""
import sys, time, glob, random
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

ROOT = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
REG = (555, 929)                                              # 09:15-15:29
SPECIAL = {pd.Timestamp("2024-11-01").date(): (1080, 1139),   # Diwali-24 Muhurat 18:00-18:59
           pd.Timestamp("2025-10-21").date(): (825, 884)}     # Diwali-25 Muhurat 13:45-14:44


def check(fn):
    d = pd.read_parquet(fn)
    exp = pd.Timestamp(d["expiry_date"].iloc[0]).normalize()
    ts = d["timestamp"]; dts = ts.dt.date; mn = ts.dt.hour * 60 + ts.dt.minute
    f = {}
    f["nrows"] = len(d)
    f["null"] = int(d[["open", "high", "low", "close", "volume", "OI", "DTE"]].isna().any(axis=1).sum())
    f["ohlc"] = int(((d.high < d.low) | (d.open < d.low) | (d.open > d.high) | (d.close < d.low) | (d.close > d.high)).sum())
    f["neg_price"] = int(((d[["open", "high", "low", "close"]] < 0).any(axis=1)).sum())
    exp_dte = (exp - ts.dt.normalize()).dt.days
    f["dte_mis"] = int((d["DTE"].values != exp_dte.values).sum())
    f["dte_neg"] = int((d["DTE"] < 0).sum())
    f["reach0"] = int(d["DTE"].min() == 0)
    # timestamp in a valid session window
    ok = (mn >= REG[0]) & (mn <= REG[1])
    for sd, (lo, hi) in SPECIAL.items():
        ok = ok | ((dts == sd) & (mn >= lo) & (mn <= hi))
    f["ts_out"] = int((~ok).sum())
    f["dup"] = int(ts.duplicated().sum())
    flat = (d.open == d.high) & (d.high == d.low) & (d.low == d.close)
    issyn = d["is_synthetic"].astype(bool) if "is_synthetic" in d.columns else pd.Series(False, index=d.index)
    f["zv_bad"] = int(((d.volume == 0) & (~issyn) & (~flat)).sum())
    f["has_syn_col"] = "is_synthetic" in d.columns
    return f


def main():
    t0 = time.time()
    exp_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    muh_exps = set()  # expiries touched by Muhurat fixes
    for d in exp_dirs:
        if "20241101" <= d.name <= "20250401" or "20251021" <= d.name <= "20260401":
            muh_exps.add(d.name)
    files = []
    for d in exp_dirs:
        fs = glob.glob(str(d / "*.parquet"))
        files += fs if d.name in muh_exps else random.sample(fs, min(15, len(fs)))
    print(f"surface recheck: {len(exp_dirs)} expiries | sampling {len(files):,} files ...", flush=True)

    agg = {"null": 0, "ohlc": 0, "neg_price": 0, "dte_mis": 0, "dte_neg": 0, "ts_out": 0, "dup": 0, "zv_bad": 0}
    nrows = clean = nosyn = noreach = 0; flagged = []
    for i, fn in enumerate(files, 1):
        f = check(fn); nrows += f["nrows"]
        if not f["has_syn_col"]: nosyn += 1
        if not f["reach0"]: noreach += 1
        bad = {k: f[k] for k in agg if f[k]}
        for k in agg: agg[k] += f[k]
        if bad: flagged.append((Path(fn).stem, bad))
        else: clean += 1
        if i % 2000 == 0: print(f"  {i}/{len(files)} | {time.time()-t0:.0f}s", flush=True)

    # Muhurat presence check
    muh_ok = {}
    for sd in SPECIAL:
        ed = "20241107" if sd.year == 2024 else "20251028"
        fs = glob.glob(str(ROOT / ed / "*.parquet"))[:20]
        got = sum(1 for f in fs if (pd.read_parquet(f, columns=["timestamp"]).timestamp.dt.date == sd).any())
        muh_ok[str(sd)] = f"{got}/{len(fs)} sampled contracts have {sd} rows"

    print("\n" + "=" * 84 + "\nSURFACE RECHECK — densified full-chain NIFTY options (post-Muhurat-fix)\n" + "=" * 84)
    print(f"  files sampled          : {len(files):,} (all {len(muh_exps)} Muhurat-affected expiries in full)")
    print(f"  rows examined          : {nrows:,}")
    print(f"  contracts fully clean  : {clean:,} / {len(files):,} ({clean/len(files)*100:.1f}%)")
    print(f"  is_synthetic missing   : {nosyn}")
    print("\n  --- field flag totals (across sampled rows) ---")
    for k, v in agg.items(): print(f"    {k:12s}: {v:,}")
    print(f"    (reach0=0 i.e. min DTE>0 / dormant deep strike): {noreach} contracts  [benign]")
    print("\n  --- Muhurat session presence ---")
    for k, v in muh_ok.items(): print(f"    {k}: {v}")
    if flagged:
        print(f"\n  --- sample of flagged contracts ({len(flagged)}) ---")
        for s, b in flagged[:10]: print(f"    {s}: {b}")
    else:
        print("\n  NO field-level anomalies in the sample (nulls/OHLC/DTE/timestamps/dups all clean).")
    print(f"\n{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
