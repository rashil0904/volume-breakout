# -*- coding: utf-8 -*-
"""full_chain_pull.py — PHASE 1: full-option-chain NIFTY 1-min OHLCV+OI raw pull (ALL strikes, CE+PE, no
moneyness/ATM restriction), all expiries Oct-2024..2026-07-31. EXTENDS the existing dataset: skips any
contract whose parquet already exists (the ~11,463 daily-rolling ones), so only the ~8,459 additional
full-chain strikes are fetched. Same schema/DTE/chunking/partitioning as before. Resumable, manifest +
issue log (no silent skips). Densification to a 375/day grid is a SEPARATE phase-2 pass after this.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import opt_pull_nifty_full as op

OUT, MAN, CUTOFF, FLOOR = op.OUT, op.MAN, op.CUTOFF, op.FLOOR
ISSUES = OUT.parent / "issues_nifty.csv"


def main():
    exps, _ = op.get(f"{op.BASE}/expiries?instrument_key={op.enc(op.NIFTY)}"); exps = exps or []
    exps = sorted([e for e in exps if FLOOR <= e <= CUTOFF])
    print(f"FULL-CHAIN pull | {len(exps)} expiries | ALL strikes (no band) | extends existing (skips pulled)", flush=True)
    man_rows = []; iss_rows = []; t0 = time.time(); n_fetch = n_skip = n_empty = 0
    for ei, exp in enumerate(exps, 1):
        cons, _ = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(op.NIFTY)}&expiry_date={exp}"); cons = cons or []
        if not cons:
            iss_rows.append({"expiry": exp, "symbol": "", "issue": "no_contracts", "detail": ""}); continue
        to_date = min(pd.Timestamp(exp).date(), pd.Timestamp(CUTOFF).date())
        edir = OUT / exp.replace("-", ""); edir.mkdir(parents=True, exist_ok=True)
        for c in cons:                                          # ALL contracts, full chain
            ck = c["instrument_key"]; sym = c["trading_symbol"]
            fn = edir / (sym.replace(" ", "_").replace("/", "-") + ".parquet")
            if fn.exists():
                n_skip += 1; continue
            rows, iss = op.pull_contract(ck, exp, to_date)
            for tag, rng, code in iss:
                iss_rows.append({"expiry": exp, "symbol": sym, "issue": tag, "detail": f"{rng} http={code}"})
            if not rows:
                n_empty += 1
                iss_rows.append({"expiry": exp, "symbol": sym, "issue": "empty_no_candles", "detail": f"strike={c['strike_price']} {c['instrument_type']}"}); continue
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "OI"])
            ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            keep = ts <= pd.Timestamp(CUTOFF) + pd.Timedelta(days=1); df = df[keep].copy(); ts = ts[keep]; df["timestamp"] = ts
            df["symbol"] = sym; df["strike"] = c["strike_price"]; df["option_type"] = c["instrument_type"]; df["expiry_date"] = exp
            df["DTE"] = (pd.Timestamp(exp).normalize() - ts.dt.normalize()).dt.days
            df = df[["symbol", "strike", "option_type", "expiry_date", "timestamp", "DTE", "open", "high", "low", "close", "volume", "OI"]].drop_duplicates("timestamp").sort_values("timestamp")
            df.to_parquet(fn, index=False); n_fetch += 1
            man_rows.append({"expiry": exp, "symbol": sym, "strike": c["strike_price"], "option_type": c["instrument_type"],
                             "n_candles": len(df), "date_min": str(df["timestamp"].iloc[0])[:10], "date_max": str(df["timestamp"].iloc[-1])[:10],
                             "dte_min": int(df["DTE"].min()), "dte_max": int(df["DTE"].max()), "file": str(fn.relative_to(OUT.parent))})
        if man_rows:
            m = pd.DataFrame(man_rows)
            if MAN.exists():
                m = pd.concat([pd.read_csv(MAN), m], ignore_index=True).drop_duplicates(["expiry", "symbol"], keep="last")
            m.to_csv(MAN, index=False); man_rows = []
        if iss_rows:
            df2 = pd.DataFrame(iss_rows)
            if ISSUES.exists():
                df2 = pd.concat([pd.read_csv(ISSUES), df2], ignore_index=True)
            df2.to_csv(ISSUES, index=False); iss_rows = []
        print(f"[{ei}/{len(exps)}] {exp}: {len(cons)} full-chain | cum new {n_fetch:,} skip {n_skip:,} empty {n_empty} | {time.time()-t0:.0f}s", flush=True)
    print(f"\nCOMPLETE: new {n_fetch:,} | skipped(existing) {n_skip:,} | empty {n_empty} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
