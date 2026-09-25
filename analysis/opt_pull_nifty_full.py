# -*- coding: utf-8 -*-
"""opt_pull_nifty_full.py — NIFTY options 1-min OHLCV+OI via Upstox expired-instruments, ATM+-15 strikes,
ALL expiries Oct-2024 .. 2026-07-31 (inclusive; expiries after cutoff excluded). Per contract, 1-min
candles pulled in ~30-day chunks (newest->oldest, stop when a chunk is empty after data), capped at the
cutoff. Adds calendar-day DTE. Resumable (skip if parquet exists). Error/empty logging (no silent skip),
429 backoff. Output: data/options_intraday_full/NIFTY/{YYYYMMDD}/{symbol}.parquet + manifest + issue log.
Schema: symbol, strike, option_type, expiry_date, timestamp, DTE, open, high, low, close, volume, OI.
DATA-PULL ONLY — no backtest logic.
"""
import sys, time
from pathlib import Path
from datetime import datetime, timedelta
import requests, pandas as pd, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb, data_loading as dl

OUT = rb.BASE / "data" / "options_intraday_full" / "NIFTY"; OUT.mkdir(parents=True, exist_ok=True)
MAN = OUT.parent / "manifest_nifty.csv"; ISSUES = OUT.parent / "issues_nifty.csv"
H = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}
BASE = "https://api.upstox.com/v2/expired-instruments"
NIFTY = "NSE_INDEX|Nifty 50"
ATM_W, THROTTLE = 15, 0.12
FLOOR, CUTOFF = "2024-10-01", "2026-09-08"
CHUNK_DAYS, MAX_LOOKBACK = 30, 120
def enc(k): return k.replace("|", "%7C")


def get(url):
    for a in range(6):
        try:
            r = requests.get(url, headers=H, timeout=45)
        except Exception:
            time.sleep(1.0 * (a + 1)); continue
        if r.status_code == 200:
            d = r.json().get("data", [])
            return (d.get("candles", []) if isinstance(d, dict) else d), 200
        if r.status_code in (429, 500, 502, 503):
            time.sleep(1.5 * (a + 1)); continue
        return None, r.status_code
    return None, -1


def spot_on(day):
    c, _ = get(f"https://api.upstox.com/v2/historical-candle/{enc(NIFTY)}/day/{day}/{day}")
    return c[0][4] if c else None


def pull_contract(ck, exp, to_date):
    """~30-day chunks, newest->oldest, stop when empty after data. returns (candles_list, issues)."""
    frames = []; issues = []; got = False
    hi = to_date
    floor = (pd.Timestamp(exp) - pd.Timedelta(days=MAX_LOOKBACK)).date()
    for _ in range(MAX_LOOKBACK // CHUNK_DAYS + 1):
        lo = hi - timedelta(days=CHUNK_DAYS)
        if hi < floor:
            break
        cd, sc = get(f"{BASE}/historical-candle/{enc(ck)}/1minute/{hi.isoformat()}/{lo.isoformat()}")
        time.sleep(THROTTLE)
        if cd is None:
            issues.append(("http_error", f"{lo}..{hi}", sc))
        elif len(cd):
            frames.append(cd); got = True
        elif got:
            break                                            # data ended going back -> stop
        hi = lo
    rows = [r for f in frames for r in f]
    return rows, issues


def main():
    exps, sc = get(f"{BASE}/expiries?instrument_key={enc(NIFTY)}")
    if not exps:
        print("expiries fetch failed", sc); return
    exps = sorted([e for e in exps if FLOOR <= e <= CUTOFF])
    print(f"NIFTY expiries {len(exps)} ({exps[0]}..{exps[-1]}) | ATM+-{ATM_W} | cutoff {CUTOFF} | DTE=calendar days", flush=True)

    man_rows = []; issue_rows = []; t0 = time.time(); n_fetch = n_skip = n_empty = 0
    for ei, exp in enumerate(exps, 1):
        cons, sc = get(f"{BASE}/option/contract?instrument_key={enc(NIFTY)}&expiry_date={exp}")
        if not cons:
            issue_rows.append({"expiry": exp, "symbol": "", "issue": f"no_contracts_{sc}", "detail": ""}); continue
        strikes = sorted(set(c["strike_price"] for c in cons))
        step = float(np.median(np.diff(strikes))) if len(strikes) > 1 else 50.0
        sp = spot_on(exp) or float(np.median(strikes))
        atm = min(strikes, key=lambda s: abs(s - sp))
        lo_s, hi_s = atm - ATM_W * step, atm + ATM_W * step
        pick = [c for c in cons if lo_s <= c["strike_price"] <= hi_s]
        to_date = min(pd.Timestamp(exp).date(), pd.Timestamp(CUTOFF).date())
        edir = OUT / exp.replace("-", ""); edir.mkdir(parents=True, exist_ok=True)
        for c in pick:
            ck = c["instrument_key"]; sym = c["trading_symbol"]
            fn = edir / (sym.replace(" ", "_").replace("/", "-") + ".parquet")
            if fn.exists():
                n_skip += 1; continue
            rows, iss = pull_contract(ck, exp, to_date)
            for tag, rng, code in iss:
                issue_rows.append({"expiry": exp, "symbol": sym, "issue": tag, "detail": f"{rng} http={code}"})
            if not rows:
                n_empty += 1
                issue_rows.append({"expiry": exp, "symbol": sym, "issue": "empty_no_candles", "detail": f"strike={c['strike_price']} {c['instrument_type']}"})
                continue
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "OI"])
            ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            df = df[ts <= pd.Timestamp(CUTOFF) + pd.Timedelta(days=1)].copy(); ts = ts[ts <= pd.Timestamp(CUTOFF) + pd.Timedelta(days=1)]
            df["timestamp"] = ts
            df["symbol"] = sym; df["strike"] = c["strike_price"]; df["option_type"] = c["instrument_type"]; df["expiry_date"] = exp
            df["DTE"] = (pd.Timestamp(exp).normalize() - ts.dt.normalize()).dt.days
            df = df[["symbol", "strike", "option_type", "expiry_date", "timestamp", "DTE", "open", "high", "low", "close", "volume", "OI"]]
            df = df.drop_duplicates("timestamp").sort_values("timestamp")
            df.to_parquet(fn, index=False); n_fetch += 1
            man_rows.append({"expiry": exp, "symbol": sym, "strike": c["strike_price"], "option_type": c["instrument_type"],
                             "n_candles": len(df), "date_min": str(df["timestamp"].iloc[0])[:10], "date_max": str(df["timestamp"].iloc[-1])[:10],
                             "dte_min": int(df["DTE"].min()), "dte_max": int(df["DTE"].max()), "file": str(fn.relative_to(OUT.parent))})
        # flush per expiry
        if man_rows:
            m = pd.DataFrame(man_rows)
            if MAN.exists():
                m = pd.concat([pd.read_csv(MAN), m], ignore_index=True).drop_duplicates(["expiry", "symbol"], keep="last")
            m.to_csv(MAN, index=False); man_rows = []
        if issue_rows:
            iss = pd.DataFrame(issue_rows)
            if ISSUES.exists():
                iss = pd.concat([pd.read_csv(ISSUES), iss], ignore_index=True)
            iss.to_csv(ISSUES, index=False); issue_rows = []
        print(f"[{ei}/{len(exps)}] {exp}: {len(pick)} picked (ATM {atm}) | cum fetched {n_fetch:,} skip {n_skip:,} empty {n_empty} | {time.time()-t0:.0f}s", flush=True)
    print(f"\nCOMPLETE: fetched {n_fetch:,} | skipped {n_skip:,} | empty/flagged {n_empty} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
