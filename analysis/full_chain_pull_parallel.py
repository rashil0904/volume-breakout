# -*- coding: utf-8 -*-
"""full_chain_pull_parallel.py — PHASE 1 full-chain NIFTY 1-min pull, PARALLELIZED with a worker pool.
ALL strikes/CE/PE, all expiries Oct-2024..2026-07-31, extends existing (skips pulled contracts). Each
worker pulls one contract (its chunks sequential inside the task) so distinct parquet files never
collide; the main thread aggregates manifest/issue rows and flushes periodically. Per-request 429
backoff (in opt_pull_nifty_full.get) still paces us if the API pushes back. Resumable via file-skip.
"""
import sys, time
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import opt_pull_nifty_full as op

OUT, MAN, CUTOFF, FLOOR = op.OUT, op.MAN, op.CUTOFF, op.FLOOR
ISSUES = OUT.parent / "issues_nifty.csv"
WORKERS = 1
CHUNK, LOOKBACK = 30, 150


def pull_fast(ck, exp, to_date):
    """newest->oldest chunks; break after 2 empty chunks near expiry (these full-chain strikes are never
    near-money, so empty-near-expiry => never traded). Cuts empty-contract cost 5 req -> 2."""
    frames = []; iss = []; hi = to_date; got = False; consec = 0
    floor = (pd.Timestamp(exp) - pd.Timedelta(days=LOOKBACK)).date()
    for _ in range(LOOKBACK // CHUNK + 1):
        lo = hi - timedelta(days=CHUNK)
        if hi < floor:
            break
        cd, sc = op.get(f"{op.BASE}/historical-candle/{op.enc(ck)}/1minute/{hi.isoformat()}/{lo.isoformat()}")
        time.sleep(op.THROTTLE)
        if cd is None:
            iss.append(("http_error", f"{lo}..{hi}", sc))
        elif len(cd):
            frames.append(cd); got = True; consec = 0
        else:
            consec += 1
            if got or consec >= 2:                             # data ended, OR 2 empty chunks -> deep strike
                break
        hi = lo
    return [r for f in frames for r in f], iss


def pull_one(exp, c):
    ck = c["instrument_key"]; sym = c["trading_symbol"]
    edir = OUT / exp.replace("-", ""); fn = edir / (sym.replace(" ", "_").replace("/", "-") + ".parquet")
    to_date = min(pd.Timestamp(exp).date(), pd.Timestamp(CUTOFF).date())
    rows, iss = pull_fast(ck, exp, to_date)
    iss_rows = [{"expiry": exp, "symbol": sym, "issue": tag, "detail": f"{rng} http={code}"} for tag, rng, code in iss]
    if not rows:
        iss_rows.append({"expiry": exp, "symbol": sym, "issue": "empty_no_candles", "detail": f"strike={c['strike_price']} {c['instrument_type']}"})
        return "empty", None, iss_rows
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "OI"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    keep = ts <= pd.Timestamp(CUTOFF) + pd.Timedelta(days=1); df = df[keep].copy(); ts = ts[keep]; df["timestamp"] = ts
    df["symbol"] = sym; df["strike"] = c["strike_price"]; df["option_type"] = c["instrument_type"]; df["expiry_date"] = exp
    df["DTE"] = (pd.Timestamp(exp).normalize() - ts.dt.normalize()).dt.days
    df = df[["symbol", "strike", "option_type", "expiry_date", "timestamp", "DTE", "open", "high", "low", "close", "volume", "OI"]].drop_duplicates("timestamp").sort_values("timestamp")
    edir.mkdir(parents=True, exist_ok=True); df.to_parquet(fn, index=False)
    man = {"expiry": exp, "symbol": sym, "strike": c["strike_price"], "option_type": c["instrument_type"], "n_candles": len(df),
           "date_min": str(df["timestamp"].iloc[0])[:10], "date_max": str(df["timestamp"].iloc[-1])[:10],
           "dte_min": int(df["DTE"].min()), "dte_max": int(df["DTE"].max()), "file": str(fn.relative_to(OUT.parent))}
    return "fetch", man, iss_rows


def main():
    exps, _ = op.get(f"{op.BASE}/expiries?instrument_key={op.enc(op.NIFTY)}"); exps = exps or []
    exps = sorted([e for e in exps if FLOOR <= e <= CUTOFF])
    tasks = []; n_skip = 0
    for exp in exps:                                            # build task list of contracts NOT yet on disk
        cons, _ = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(op.NIFTY)}&expiry_date={exp}"); cons = cons or []
        edir = OUT / exp.replace("-", "")
        for c in cons:
            fn = edir / (c["trading_symbol"].replace(" ", "_").replace("/", "-") + ".parquet")
            if fn.exists():
                n_skip += 1
            else:
                tasks.append((exp, c))
    print(f"FULL-CHAIN PARALLEL | {len(exps)} expiries | already on disk {n_skip:,} | to fetch {len(tasks):,} | {WORKERS} workers", flush=True)

    man_rows = []; iss_rows = []; t0 = time.time(); n_fetch = n_empty = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(pull_one, exp, c): (exp, c) for exp, c in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                status, man, iss = fut.result()
            except Exception as e:
                iss_rows.append({"expiry": "?", "symbol": "?", "issue": "task_error", "detail": str(e)[:80]}); status = "err"; man = None; iss = []
            if status == "fetch":
                n_fetch += 1; man_rows.append(man)
            elif status == "empty":
                n_empty += 1
            if iss:
                iss_rows += iss
            if i % 200 == 0 or i == len(tasks):
                if man_rows:
                    m = pd.DataFrame(man_rows)
                    if MAN.exists():
                        m = pd.concat([pd.read_csv(MAN), m], ignore_index=True).drop_duplicates(["expiry", "symbol"], keep="last")
                    m.to_csv(MAN, index=False); man_rows = []
                if iss_rows:
                    d2 = pd.DataFrame(iss_rows)
                    if ISSUES.exists():
                        d2 = pd.concat([pd.read_csv(ISSUES), d2], ignore_index=True)
                    d2.to_csv(ISSUES, index=False); iss_rows = []
                rate = i / max(time.time() - t0, 1)
                print(f"  {i}/{len(tasks)} | new {n_fetch:,} empty {n_empty} | {rate:.1f} contracts/s | {time.time()-t0:.0f}s", flush=True)
    print(f"\nCOMPLETE: new {n_fetch:,} | empty {n_empty} | skipped(existing) {n_skip:,} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
