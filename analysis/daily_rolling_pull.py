# -*- coding: utf-8 -*-
"""daily_rolling_pull.py — upgrade the NIFTY options pull from a FIXED expiry-day ATM+-15 band to a
DAILY-ROLLING union: for each expiry, strikes = UNION over the contract's active window of each day's
ATM+-15 (anchored to NIFTY's daily close that day). The fixed band is a subset, so existing parquet are
kept and skipped; this pulls only the additional union strikes. Same schema/DTE/chunking as the base
pull. Resumable via file-existence skip. Writes to the same data/options_intraday_full/NIFTY location.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import opt_pull_nifty_full as op

OUT, MAN, CUTOFF = op.OUT, op.MAN, op.CUTOFF
FLOOR = op.FLOOR; ATM_W = op.ATM_W


def nifty_daily():
    """returns date -> (high, low) so the ATM band can cover intraday movement, not just the close."""
    c, _ = op.get(f"https://api.upstox.com/v2/historical-candle/{op.enc(op.NIFTY)}/day/2026-07-31/2024-06-01")
    d = pd.DataFrame(c, columns=["ts", "o", "high", "low", "close", "v", "oi"])
    d["date"] = pd.to_datetime(d["ts"]).dt.tz_localize(None).dt.normalize().dt.date
    return {row.date: (row.high, row.low) for row in d.itertuples()}


def main():
    nclose = nifty_daily(); tdays = sorted(nclose)
    exps, _ = op.get(f"{op.BASE}/expiries?instrument_key={op.enc(op.NIFTY)}"); exps = exps or []
    exps = sorted([e for e in exps if FLOOR <= e <= CUTOFF])
    man = pd.read_csv(MAN)
    man["dm"] = pd.to_datetime(man["date_min"], errors="coerce")
    wstart = man.dropna(subset=["dm"]).groupby("expiry")["dm"].min().dt.date.to_dict()
    print(f"daily-rolling union pull | {len(exps)} expiries | ATM+-{ATM_W} rolling | skips existing", flush=True)

    man_rows = []; t0 = time.time(); n_fetch = n_skip = n_empty = 0
    for ei, exp in enumerate(exps, 1):
        cons, _ = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(op.NIFTY)}&expiry_date={exp}"); cons = cons or []
        if not cons:
            print(f"[{ei}/{len(exps)}] {exp}: no contracts", flush=True); continue
        strikes = sorted(set(c["strike_price"] for c in cons))
        step = float(np.median(np.diff(strikes))) if len(strikes) > 1 else 50.0
        ws = wstart.get(exp, (pd.Timestamp(exp) - pd.Timedelta(days=150)).date())
        expd = pd.Timestamp(exp).date()
        window = [d for d in tdays if ws <= d <= expd and d in nclose]
        if window:
            min_low = min(nclose[d][1] for d in window)                   # lowest intraday low over life
            max_high = max(nclose[d][0] for d in window)                  # highest intraday high over life
        else:
            min_low = max_high = float(np.median(strikes))
        atm_lo = min(strikes, key=lambda s: abs(s - min_low))
        atm_hi = min(strikes, key=lambda s: abs(s - max_high))
        lo_s, hi_s = atm_lo - ATM_W * step, atm_hi + ATM_W * step         # UNION band, ATM+-15 from intraday hi/lo
        pick = [c for c in cons if lo_s <= c["strike_price"] <= hi_s]
        to_date = min(expd, pd.Timestamp(CUTOFF).date())
        edir = OUT / exp.replace("-", ""); edir.mkdir(parents=True, exist_ok=True)
        for c in pick:
            ck = c["instrument_key"]; sym = c["trading_symbol"]
            fn = edir / (sym.replace(" ", "_").replace("/", "-") + ".parquet")
            if fn.exists():
                n_skip += 1; continue
            rows, iss = op.pull_contract(ck, exp, to_date)
            if not rows:
                n_empty += 1; continue
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
        print(f"[{ei}/{len(exps)}] {exp}: union {len(pick)} strikes-band | cum new {n_fetch:,} skip {n_skip:,} empty {n_empty} | {time.time()-t0:.0f}s", flush=True)
    print(f"\nCOMPLETE: new {n_fetch:,} | skipped(existing) {n_skip:,} | empty {n_empty} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
