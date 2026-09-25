# -*- coding: utf-8 -*-
"""full_pull_nifty_futures.py — NIFTY FUTURES 1-min OHLCV+OI pull via Upstox expired-instruments.
Monthly contracts, earliest genuine futures expiry confirmed via probe = 2024-10-31 (that contract's own
trading life starts 2024-07-26 -- ~2 months EARLIER than the NIFTY options/BankNifty-futures floor of
Oct-2024, confirmed via probe_nifty_futures_expired_history.py, not assumed), through the Aug-2026 expiry
(inclusive; later excluded). One contract per expiry (FUT, no strikes/CE-PE). Reuses opt_pull_nifty_full's
get()/enc()/BASE/THROTTLE (retry+backoff on 429/5xx already built in) -- exact same pattern as
full_pull_banknifty_futures.py, adapted to the NIFTY underlying and its longer per-contract life.
DTE = calendar days (expiry_date - candle_date), matching the NIFTY/SENSEX options convention. Zero-volume
candles KEPT (not dropped). Output: data/futures_intraday_full/NIFTY/{YYYYMMDD}/{contract}.parquet
(partitioned by expiry month) + manifest_nifty_fut.csv + issues_nifty_fut.csv. Resumable (file-skip).
"""
import sys, time
from pathlib import Path
from datetime import timedelta
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
import opt_pull_nifty_full as op
import run_backtest as rb

UNDERLYING = "NSE_INDEX|Nifty 50"
OUT = rb.BASE / "data" / "futures_intraday_full" / "NIFTY"; OUT.mkdir(parents=True, exist_ok=True)
MAN = OUT.parent / "manifest_nifty_fut.csv"; ISSUES = OUT.parent / "issues_nifty_fut.csv"
CUTOFF = "2026-08-31"          # capture the Aug-2026 expiry (~2026-08-25), exclude Sep-2026+
CHUNK, LOOKBACK = 30, 120      # NIFTY futures contracts can run ~3 calendar months (confirmed via probe)


def pull_contract(ck, exp, to_date):
    """~30-day chunks, newest->oldest, stop after 2 empty chunks once we've seen data (or floor reached)."""
    frames = []; issues = []; hi = to_date; got = False; consec = 0
    floor = (pd.Timestamp(exp) - pd.Timedelta(days=LOOKBACK)).date()
    for _ in range(LOOKBACK // CHUNK + 2):
        lo = hi - timedelta(days=CHUNK)
        if hi < floor:
            break
        cd, sc = op.get(f"{op.BASE}/historical-candle/{op.enc(ck)}/1minute/{hi.isoformat()}/{lo.isoformat()}")
        time.sleep(op.THROTTLE)
        if cd is None:
            issues.append(("http_error", f"{lo}..{hi}", sc))
        elif len(cd):
            frames.append(cd); got = True; consec = 0
        else:
            consec += 1
            if got or consec >= 2:
                break
        hi = lo
    return [r for f in frames for r in f], issues


def main():
    # ---- STEP 1: confirm starting point (re-verified fresh, same result as the earlier probe) ----
    exps_raw, sc = op.get(f"{op.BASE}/expiries?instrument_key={op.enc(UNDERLYING)}")
    if not exps_raw:
        print("expiries fetch failed", sc); return
    exps_raw = sorted([e for e in exps_raw if e <= CUTOFF])
    print(f"STEP 1: candidate expiry dates (incl. leftover weekly-options dates) <= {CUTOFF}: {len(exps_raw)}", flush=True)

    contracts = []
    for e in exps_raw:
        cons, csc = op.get(f"{op.BASE}/future/contract?instrument_key={op.enc(UNDERLYING)}&expiry_date={e}")
        if cons:
            contracts.append((e, cons[0]))
        time.sleep(0.15)
    print(f"genuine FUTURES expiries found: {len(contracts)} | earliest {contracts[0][0]} | latest {contracts[-1][0]}", flush=True)
    known_floor = "2024-10-01"
    print(f"  earliest futures expiry {contracts[0][0]} vs known NIFTY-options/BankNifty-futures floor {known_floor} "
          f"-> {'later' if contracts[0][0] > known_floor else 'matches/earlier'}", flush=True)

    # ---- STEP 2: full pull ----
    t0 = time.time(); n_fetch = n_skip = n_empty = 0
    man_rows, issue_rows = [], []
    for i, (exp, c) in enumerate(contracts, 1):
        ck = c["instrument_key"]; sym = c["trading_symbol"]; lot = c.get("lot_size")
        edir = OUT / exp.replace("-", ""); edir.mkdir(parents=True, exist_ok=True)
        fn = edir / (sym.replace(" ", "_").replace("/", "-") + ".parquet")
        if fn.exists():
            n_skip += 1
            print(f"  [{i}/{len(contracts)}] {exp} {sym}: SKIP (exists)", flush=True); continue
        to_date = min(pd.Timestamp(exp).date(), pd.Timestamp(CUTOFF).date())
        rows, iss = pull_contract(ck, exp, to_date)
        for tag, rng, code in iss:
            issue_rows.append({"expiry": exp, "symbol": sym, "instrument_key": ck, "issue": tag, "detail": f"{rng} http={code}"})
        if not rows:
            n_empty += 1
            issue_rows.append({"expiry": exp, "symbol": sym, "instrument_key": ck, "issue": "empty_no_candles", "detail": f"contract={ck}"})
            print(f"  [{i}/{len(contracts)}] {exp} {sym}: EMPTY -- FLAGGED", flush=True); continue

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "OI"])
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        keep = ts <= pd.Timestamp(CUTOFF) + pd.Timedelta(days=1)
        df = df[keep].copy(); ts = ts[keep]; df["timestamp"] = ts
        df["contract_month"] = pd.Timestamp(exp).strftime("%Y-%m"); df["expiry_date"] = exp; df["symbol"] = sym
        df["DTE"] = (pd.Timestamp(exp).normalize() - ts.dt.normalize()).dt.days
        df = df[["contract_month", "expiry_date", "symbol", "timestamp", "DTE", "open", "high", "low", "close", "volume", "OI"]]
        df = df.drop_duplicates("timestamp").sort_values("timestamp")

        has_oi = df["OI"].notna().any() and (df["OI"] != 0).any()
        zv = int((df.volume == 0).sum())
        n_days = df["timestamp"].dt.normalize().nunique()
        # flag any day whose candle count looks abnormally short (potential mid-day data issue), not just
        # missing days entirely -- a normal full NIFTY session is 375 one-min candles (09:15-15:29)
        by_day_n = df.groupby(df["timestamp"].dt.date).size()
        short_days = by_day_n[by_day_n < 370]  # small tolerance below 375 for holiday-shortened/CAS-affected sessions
        if len(short_days):
            for d, n in short_days.items():
                issue_rows.append({"expiry": exp, "symbol": sym, "instrument_key": ck, "issue": "short_session_candle_count",
                                   "detail": f"date={d} n_candles={n} (normal=375)"})

        df.to_parquet(fn, index=False); n_fetch += 1
        man_rows.append({"expiry": exp, "symbol": sym, "instrument_key": ck, "lot_size": lot, "n_candles": len(df),
                         "date_min": str(df["timestamp"].iloc[0])[:10], "date_max": str(df["timestamp"].iloc[-1])[:10],
                         "dte_min": int(df["DTE"].min()), "dte_max": int(df["DTE"].max()), "zero_vol_rows": zv,
                         "has_oi": has_oi, "trading_days": n_days, "n_short_sessions": len(short_days),
                         "file": str(fn.relative_to(OUT.parent))})
        print(f"  [{i}/{len(contracts)}] {exp} {sym}: {len(df):,} candles | {n_days} days | zero-vol {zv} | "
              f"OI={has_oi} | short_sessions={len(short_days)} | {df['timestamp'].iloc[0]}..{df['timestamp'].iloc[-1]} | {time.time()-t0:.0f}s", flush=True)

        if man_rows:
            pd.DataFrame(man_rows).to_csv(MAN, index=False)
        if issue_rows:
            pd.DataFrame(issue_rows).to_csv(ISSUES, index=False)

    print(f"\nDONE: fetched={n_fetch} skipped(existing)={n_skip} empty={n_empty} total_contracts={len(contracts)}", flush=True)
    print(f"Manifest -> {MAN}\nIssues -> {ISSUES}")


if __name__ == "__main__":
    main()
