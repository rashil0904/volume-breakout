# -*- coding: utf-8 -*-
"""full_chain_pull_banknifty_options.py — BankNifty OPTIONS full-chain 1-min pull, MONTHLY EXPIRIES ONLY.
Monthly = last real expiry within each calendar month (derived from the ACTUAL expiry list at runtime, not
hardcoded/assumed weekday - confirmed pre-pull: 22 monthly expiries Oct-2024..Jul-2026, weekday changed
Wed->Tue->Thu->Tue->Mon->Tue across the window; 6 weekly expiries in Oct-Nov 2024 explicitly EXCLUDED). ALL
strikes/CE/PE per monthly expiry. Reuses opt_pull_nifty_full.get()/enc()/BASE/THROTTLE (retry+backoff on
429/5xx built in). Resumable (file-skip). DTE = calendar days. Zero-volume candles KEPT. Output:
data/options_intraday_full/BANKNIFTY/{YYYYMMDD}/{symbol}.parquet + manifest_banknifty_opt.csv +
issues_banknifty_opt.csv. DATA-PULL ONLY.
"""
import sys, time
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import opt_pull_nifty_full as op
import run_backtest as rb

UNDERLYING = "NSE_INDEX|Nifty Bank"
OUT = rb.BASE / "data" / "options_intraday_full" / "BANKNIFTY"; OUT.mkdir(parents=True, exist_ok=True)
MAN = OUT.parent / "manifest_banknifty_opt.csv"; ISSUES = OUT.parent / "issues_banknifty_opt.csv"
FLOOR, CUTOFF = "2024-10-01", "2026-07-31"
WORKERS = 1; CHUNK, LOOKBACK = 30, 150


def derive_monthly_expiries():
    """Fetch all real option expiries in window, keep only the LAST one per calendar month."""
    raw, _ = op.get(f"{op.BASE}/expiries?instrument_key={op.enc(UNDERLYING)}")
    raw = sorted([e for e in raw if FLOOR <= e <= CUTOFF])
    real = []
    for e in raw:
        cons, _ = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(UNDERLYING)}&expiry_date={e}")
        if cons: real.append(e)
        time.sleep(0.1)
    df = pd.DataFrame({"date": pd.to_datetime(real)}); df["month"] = df["date"].dt.to_period("M")
    monthly = sorted(df.groupby("month")["date"].max().dt.strftime("%Y-%m-%d").tolist())
    excluded = sorted(set(real) - set(monthly))
    return monthly, excluded


def pull_fast(ck, exp, to_date):
    """newest->oldest chunks; break after 2 empty chunks near expiry (deep never-traded strike)."""
    frames = []; iss = []; hi = to_date; got = False; consec = 0
    floor = (pd.Timestamp(exp) - pd.Timedelta(days=LOOKBACK)).date()
    for _ in range(LOOKBACK // CHUNK + 1):
        lo = hi - timedelta(days=CHUNK)
        if hi < floor: break
        cd, sc = op.get(f"{op.BASE}/historical-candle/{op.enc(ck)}/1minute/{hi.isoformat()}/{lo.isoformat()}")
        time.sleep(op.THROTTLE)
        if cd is None:
            iss.append(("http_error", f"{lo}..{hi}", sc))
        elif len(cd):
            frames.append(cd); got = True; consec = 0
        else:
            consec += 1
            if got or consec >= 2: break
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
    monthly, excluded = derive_monthly_expiries()
    print(f"MONTHLY expiries confirmed: {len(monthly)} | excluded (weekly): {len(excluded)} {excluded}", flush=True)
    print(f"  monthly list: {monthly}", flush=True)

    tasks = []; n_skip = 0
    for exp in monthly:
        cons, _ = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(UNDERLYING)}&expiry_date={exp}"); cons = cons or []
        edir = OUT / exp.replace("-", "")
        for c in cons:
            fn = edir / (c["trading_symbol"].replace(" ", "_").replace("/", "-") + ".parquet")
            if fn.exists(): n_skip += 1
            else: tasks.append((exp, c))
    print(f"BANKNIFTY OPTIONS MONTHLY-ONLY | {len(monthly)} expiries | already on disk {n_skip:,} | to fetch {len(tasks):,} | {WORKERS} workers", flush=True)

    man_rows = []; iss_rows = []; t0 = time.time(); n_fetch = n_empty = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(pull_one, exp, c): (exp, c) for exp, c in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                status, man, iss = fut.result()
            except Exception as e:
                iss_rows.append({"expiry": "?", "symbol": "?", "issue": "task_error", "detail": str(e)[:80]}); status = "err"; man = None; iss = []
            if status == "fetch": n_fetch += 1; man_rows.append(man)
            elif status == "empty": n_empty += 1
            if iss: iss_rows += iss
            if i % 200 == 0 or i == len(tasks):
                if man_rows:
                    m = pd.DataFrame(man_rows)
                    if MAN.exists(): m = pd.concat([pd.read_csv(MAN), m], ignore_index=True).drop_duplicates(["expiry", "symbol"], keep="last")
                    m.to_csv(MAN, index=False); man_rows = []
                if iss_rows:
                    d2 = pd.DataFrame(iss_rows)
                    if ISSUES.exists(): d2 = pd.concat([pd.read_csv(ISSUES), d2], ignore_index=True)
                    d2.to_csv(ISSUES, index=False); iss_rows = []
                rate = i / max(time.time() - t0, 1)
                print(f"  {i}/{len(tasks)} | new {n_fetch:,} empty {n_empty} | {rate:.2f} contracts/s | {time.time()-t0:.0f}s", flush=True)
    # final flush
    if man_rows:
        m = pd.DataFrame(man_rows)
        if MAN.exists(): m = pd.concat([pd.read_csv(MAN), m], ignore_index=True).drop_duplicates(["expiry", "symbol"], keep="last")
        m.to_csv(MAN, index=False)
    if iss_rows:
        d2 = pd.DataFrame(iss_rows)
        if ISSUES.exists(): d2 = pd.concat([pd.read_csv(ISSUES), d2], ignore_index=True)
        d2.to_csv(ISSUES, index=False)
    print(f"\nCOMPLETE: new {n_fetch:,} | empty {n_empty} | skipped(existing) {n_skip:,} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
