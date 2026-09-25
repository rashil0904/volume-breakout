# -*- coding: utf-8 -*-
"""full_pull_banknifty_futures.py — BankNifty FUTURES 1-min OHLCV+OI pull via Upstox expired-instruments.
Monthly contracts, earliest available expiry 2024-10-30 (contract life starts 2024-09-30, ONE DAY EARLIER than
the NIFTY/SENSEX options floor of 2024-10-01 - confirmed via probe, not assumed) through July-2026 expiry
(inclusive; later excluded). One contract per expiry (FUT, no strikes/CE-PE). Reuses opt_pull_nifty_full's
get()/enc()/BASE/THROTTLE (retry+backoff on 429/5xx already built in). Chunked ~30-day windows, newest->oldest,
LOOKBACK=120 (safety margin; contract typically lives ~30d but this avoids truncating any that list earlier).
DTE = calendar days (expiry_date - candle_date), matching the NIFTY/SENSEX options convention. Zero-volume
candles KEPT (not dropped). Output: data/futures_intraday_full/BANKNIFTY/{YYYYMMDD}/{contract}.parquet
(partitioned by expiry month) + manifest_banknifty_fut.csv + issues_banknifty_fut.csv. Resumable (file-skip).
"""
import sys, time
from pathlib import Path
from datetime import timedelta
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import opt_pull_nifty_full as op
import run_backtest as rb

UNDERLYING = "NSE_INDEX|Nifty Bank"
OUT = rb.BASE / "data" / "futures_intraday_full" / "BANKNIFTY"; OUT.mkdir(parents=True, exist_ok=True)
MAN = OUT.parent / "manifest_banknifty_fut.csv"; ISSUES = OUT.parent / "issues_banknifty_fut.csv"
CUTOFF = "2026-08-25"                                   # extended: latest genuine futures expiry available on the platform
CHUNK, LOOKBACK = 30, 120


def pull_contract(ck, exp, to_date):
    """~30-day chunks, newest->oldest, stop after 2 empty chunks once we've seen data (or floor reached)."""
    frames = []; issues = []; hi = to_date; got = False; consec = 0
    floor = (pd.Timestamp(exp) - pd.Timedelta(days=LOOKBACK)).date()
    for _ in range(LOOKBACK // CHUNK + 2):
        lo = hi - timedelta(days=CHUNK)
        if hi < floor: break
        cd, sc = op.get(f"{op.BASE}/historical-candle/{op.enc(ck)}/1minute/{hi.isoformat()}/{lo.isoformat()}")
        time.sleep(op.THROTTLE)
        if cd is None:
            issues.append(("http_error", f"{lo}..{hi}", sc))
        elif len(cd):
            frames.append(cd); got = True; consec = 0
        else:
            consec += 1
            if got or consec >= 2: break
        hi = lo
    return [r for f in frames for r in f], issues


def main():
    exps_raw, sc = op.get(f"{op.BASE}/expiries?instrument_key={op.enc(UNDERLYING)}")
    if not exps_raw:
        print("expiries fetch failed", sc); return
    exps_raw = sorted([e for e in exps_raw if e <= CUTOFF])
    print(f"candidate expiry dates (incl. leftover weekly-options dates) <= {CUTOFF}: {len(exps_raw)}", flush=True)

    # resolve which are GENUINE futures expiries (non-empty future/contract)
    contracts = []; man_rows = []; issue_rows = []
    for e in exps_raw:
        cons, csc = op.get(f"{op.BASE}/future/contract?instrument_key={op.enc(UNDERLYING)}&expiry_date={e}")
        if cons:
            contracts.append((e, cons[0]))
        time.sleep(0.15)
    print(f"genuine FUTURES expiries found: {len(contracts)} | earliest {contracts[0][0]} | latest {contracts[-1][0]}", flush=True)

    t0 = time.time(); n_fetch = n_skip = n_empty = 0
    for i, (exp, c) in enumerate(contracts, 1):
        ck = c["instrument_key"]; sym = c["trading_symbol"]
        edir = OUT / exp.replace("-", ""); edir.mkdir(parents=True, exist_ok=True)
        fn = edir / (sym.replace(" ", "_").replace("/", "-") + ".parquet")
        if fn.exists():
            n_skip += 1
            print(f"  [{i}/{len(contracts)}] {exp} {sym}: SKIP (exists)", flush=True); continue
        to_date = min(pd.Timestamp(exp).date(), pd.Timestamp(CUTOFF).date())
        rows, iss = pull_contract(ck, exp, to_date)
        for tag, rng, code in iss:
            issue_rows.append({"expiry": exp, "symbol": sym, "issue": tag, "detail": f"{rng} http={code}"})
        if not rows:
            n_empty += 1
            issue_rows.append({"expiry": exp, "symbol": sym, "issue": "empty_no_candles", "detail": f"contract={ck}"})
            print(f"  [{i}/{len(contracts)}] {exp} {sym}: EMPTY", flush=True); continue
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "OI"])
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        keep = ts <= pd.Timestamp(CUTOFF) + pd.Timedelta(days=1); df = df[keep].copy(); ts = ts[keep]; df["timestamp"] = ts
        df["contract_month"] = pd.Timestamp(exp).strftime("%Y-%m"); df["expiry_date"] = exp; df["symbol"] = sym
        df["DTE"] = (pd.Timestamp(exp).normalize() - ts.dt.normalize()).dt.days
        df = df[["contract_month", "expiry_date", "symbol", "timestamp", "DTE", "open", "high", "low", "close", "volume", "OI"]]
        df = df.drop_duplicates("timestamp").sort_values("timestamp")
        df.to_parquet(fn, index=False); n_fetch += 1
        man_rows.append({"expiry": exp, "symbol": sym, "instrument_key": ck, "lot_size": c.get("lot_size"), "n_candles": len(df),
                         "date_min": str(df["timestamp"].iloc[0])[:10], "date_max": str(df["timestamp"].iloc[-1])[:10],
                         "dte_min": int(df["DTE"].min()), "dte_max": int(df["DTE"].max()), "zero_vol_rows": int((df.volume == 0).sum()),
                         "trading_days": df["timestamp"].dt.normalize().nunique(), "file": str(fn.relative_to(OUT.parent))})
        zv = int((df.volume == 0).sum())
        print(f"  [{i}/{len(contracts)}] {exp} {sym}: {len(df):,} candles | {df['timestamp'].dt.normalize().nunique()} days | zero-vol {zv} | {df['timestamp'].iloc[0]}..{df['timestamp'].iloc[-1]} | {time.time()-t0:.0f}s", flush=True)
        # flush progressively
        if man_rows:
            m = pd.DataFrame(man_rows)
            if MAN.exists(): m = pd.concat([pd.read_csv(MAN), m], ignore_index=True).drop_duplicates(["expiry", "symbol"], keep="last")
            m.to_csv(MAN, index=False); man_rows = []
        if issue_rows:
            d2 = pd.DataFrame(issue_rows)
            if ISSUES.exists(): d2 = pd.concat([pd.read_csv(ISSUES), d2], ignore_index=True)
            d2.to_csv(ISSUES, index=False); issue_rows = []

    print(f"\nCOMPLETE: fetched {n_fetch} | skipped(existing) {n_skip} | empty/flagged {n_empty} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
