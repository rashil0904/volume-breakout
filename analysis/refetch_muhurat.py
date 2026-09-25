# -*- coding: utf-8 -*-
"""refetch_muhurat.py — targeted fix for the 493 contracts whose densified files have all-NaN daytime
grids on Diwali Muhurat dates (evening-only session dropped by the daytime grid). Re-pulls each raw
(recovers the ~18:00-19:00 Muhurat candles the in-place densify had overwritten) and re-densifies with
the FIXED densify_df (daytime -> 375 grid; evening/Muhurat -> real rows kept as-is). 1 worker, resumable
(skips contracts whose file no longer has NaN OHLC). Reads the audit's null_fields flag list.
"""
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import opt_pull_nifty_full as op
import densify_nifty_options as dn
import run_backtest as rb

OUT, CUTOFF = op.OUT, op.CUTOFF
FLAGGED = rb.RESULTS / "fullchain_audit" / "flagged_contracts.csv"
_cc = {}


def contracts(exp):
    if exp not in _cc:
        c, _ = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(op.NIFTY)}&expiry_date={exp}")
        _cc[exp] = {(round(x["strike_price"], 2), x["instrument_type"]): x for x in (c or [])}
    return _cc[exp]


def main():
    fl = pd.read_csv(FLAGGED)
    tgt = fl[fl.issue == "null_fields"][["expiry", "symbol", "strike", "type"]].drop_duplicates()
    print(f"Muhurat-affected contracts to fix: {len(tgt)}", flush=True)
    t0 = time.time(); fixed = skip = fail = 0; evening_rows = 0
    for i, r in enumerate(tgt.itertuples(), 1):
        exp = r.expiry; edir = OUT / exp.replace("-", "")
        fn = edir / (r.symbol.replace(" ", "_").replace("/", "-") + ".parquet")
        if fn.exists():
            cur = pd.read_parquet(fn, columns=["close"])
            if not cur["close"].isna().any():                 # already fixed (no NaN)
                skip += 1; continue
        c = contracts(exp).get((round(float(r.strike), 2), r.type))
        if not c:
            fail += 1; continue
        ck = c["instrument_key"]; sym = c["trading_symbol"]
        to_date = min(pd.Timestamp(exp).date(), pd.Timestamp(CUTOFF).date())
        rows, iss = op.pull_contract(ck, exp, to_date)
        if not rows:
            fail += 1; continue
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "OI"])
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        keep = ts <= pd.Timestamp(CUTOFF) + pd.Timedelta(days=1); df = df[keep].copy(); ts = ts[keep]; df["timestamp"] = ts
        df["symbol"] = sym; df["strike"] = float(r.strike); df["option_type"] = r.type; df["expiry_date"] = exp
        df["DTE"] = 0
        df = df[["symbol", "strike", "option_type", "expiry_date", "timestamp", "DTE", "open", "high", "low", "close", "volume", "OI"]]
        dense = dn.densify_df(df)                              # FIXED densifier (Muhurat-aware)
        m = dense["timestamp"].dt.hour * 60 + dense["timestamp"].dt.minute
        evening_rows += int(((m < 555) | (m > 929)).sum())
        edir.mkdir(parents=True, exist_ok=True); dense.to_parquet(fn, index=False)
        fixed += 1
        if i % 25 == 0 or i == len(tgt):
            print(f"  {i}/{len(tgt)} | fixed {fixed} skip {skip} fail {fail} | evening(Muhurat) rows recovered {evening_rows:,} | {time.time()-t0:.0f}s", flush=True)
    print(f"\nCOMPLETE: fixed {fixed} | already-ok {skip} | failed {fail} | Muhurat rows recovered {evening_rows:,} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
