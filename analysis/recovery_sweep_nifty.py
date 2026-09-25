# -*- coding: utf-8 -*-
"""recovery_sweep_nifty.py — broad RECOVERY sweep over every NIFTY option contract still flagged by the
audit (missing_trading_days + stopped_before_expiry). Root cause of cut-offs: the original pull stopped
scanning backward when a chunk transiently returned empty (rate-limit blip). This sweep uses a ROBUST
pull (all ~30-day chunks across the full 150-day lookback, NO early-break) so transient empties can't
truncate, and KEEP-MAX (overwrite only when the re-pull yields strictly MORE candles -> never loses
data). Resumable per-contract. Reports improved / unchanged, and extra candles recovered.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import opt_pull_nifty_full as op

OUT, MAN, CUTOFF = op.OUT, op.MAN, op.CUTOFF
AUD = op.rb.RESULTS / "options_audit"
FLAGGED = AUD / "flagged_contracts.csv"
PROC = AUD / "recovery_processed.csv"; RES = AUD / "recovery_results.csv"
LOOKBACK, CHUNK = 150, 30
_ccache = {}


def contracts(exp):
    if exp not in _ccache:
        c, _ = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(op.NIFTY)}&expiry_date={exp}")
        _ccache[exp] = {(round(x["strike_price"], 2), x["instrument_type"]): x for x in (c or [])}
    return _ccache[exp]


def robust_pull(ck, exp, to_date):
    frames = []; hi = to_date
    floor = (pd.Timestamp(exp) - pd.Timedelta(days=LOOKBACK)).date()
    for _ in range(LOOKBACK // CHUNK + 1):
        lo = hi - pd.Timedelta(days=CHUNK)
        cd, sc = op.get(f"{op.BASE}/historical-candle/{op.enc(ck)}/1minute/{hi.isoformat()}/{lo.isoformat()}")
        time.sleep(op.THROTTLE)
        if cd:
            frames.append(cd)
        hi = lo
        if hi < floor:
            break
    return [r for f in frames for r in f]                      # NO early-break on empty


def main():
    t0 = time.time()
    fl = pd.read_csv(FLAGGED)
    tgt = fl[fl["issue"].isin(["missing_trading_days", "stopped_before_expiry"])][["expiry", "symbol", "strike", "type"]].drop_duplicates()
    tgt = tgt[tgt["symbol"].astype(str).str.len() > 0]
    man = pd.read_csv(MAN); man["key"] = man["expiry"].astype(str) + "|" + man["symbol"].astype(str)
    old_n = dict(zip(man["key"], man["n_candles"]))
    done = set()
    if PROC.exists():
        done = set(pd.read_csv(PROC)["key"])
    todo = [r for r in tgt.itertuples() if f"{r.expiry}|{r.symbol}" not in done]
    print(f"flagged recoverable contracts: {len(tgt)} | already processed {len(done)} | to do {len(todo)}", flush=True)

    proc_rows = []; res_rows = []; n_improved = 0; extra_total = 0
    for i, r in enumerate(todo, 1):
        exp = r.expiry; key = f"{exp}|{r.symbol}"
        cmap = contracts(exp); c = cmap.get((round(float(r.strike), 2), r.type))
        if not c:
            res_rows.append({"expiry": exp, "symbol": r.symbol, "status": "no_contract", "old_n": old_n.get(key, 0), "new_n": 0, "extra": 0})
            proc_rows.append({"key": key}); continue
        ck = c["instrument_key"]; sym = c["trading_symbol"]
        edir = OUT / exp.replace("-", ""); fn = edir / (sym.replace(" ", "_").replace("/", "-") + ".parquet")
        on = old_n.get(key, 0)
        try:
            on = len(pd.read_parquet(fn, columns=["timestamp"])) if fn.exists() else 0
        except Exception:
            on = 0
        to_date = min(pd.Timestamp(exp).date(), pd.Timestamp(CUTOFF).date())
        rows = robust_pull(ck, exp, to_date)
        nn = 0; status = "empty"
        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "OI"])
            ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            keep = ts <= pd.Timestamp(CUTOFF) + pd.Timedelta(days=1); df = df[keep].copy(); ts = ts[keep]; df["timestamp"] = ts
            df["symbol"] = sym; df["strike"] = float(r.strike); df["option_type"] = r.type; df["expiry_date"] = exp
            df["DTE"] = (pd.Timestamp(exp).normalize() - ts.dt.normalize()).dt.days
            df = df[["symbol", "strike", "option_type", "expiry_date", "timestamp", "DTE", "open", "high", "low", "close", "volume", "OI"]].drop_duplicates("timestamp").sort_values("timestamp")
            nn = len(df)
            if nn > on:                                         # KEEP-MAX: overwrite only if strictly more data
                edir.mkdir(parents=True, exist_ok=True); df.to_parquet(fn, index=False)
                status = "improved"; n_improved += 1; extra_total += (nn - on)
            else:
                status = "unchanged"
        res_rows.append({"expiry": exp, "symbol": sym, "status": status, "old_n": on, "new_n": nn, "extra": max(nn - on, 0)})
        proc_rows.append({"key": key})
        if i % 50 == 0 or i == len(todo):
            pd.DataFrame(proc_rows).to_csv(PROC, mode="a", header=not PROC.exists(), index=False)
            pd.DataFrame(res_rows).to_csv(RES, mode="a", header=not RES.exists(), index=False)
            proc_rows, res_rows = [], []
            print(f"  {i}/{len(todo)} | improved {n_improved} (+{extra_total:,} candles) | {time.time()-t0:.0f}s", flush=True)
    if proc_rows:
        pd.DataFrame(proc_rows).to_csv(PROC, mode="a", header=not PROC.exists(), index=False)
        pd.DataFrame(res_rows).to_csv(RES, mode="a", header=not RES.exists(), index=False)
    print(f"\nCOMPLETE: improved {n_improved} contracts (+{extra_total:,} candles recovered) | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
