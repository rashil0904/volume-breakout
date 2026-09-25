# -*- coding: utf-8 -*-
"""opt_pull.py — resumable, paced, checkpointed puller for Upstox-Plus EXPIRED option 1-min OHLCV+OI.
Validation slice: NIFTY, N recent expired expiries, ATM+-W strikes, CE+PE. Partitioned parquet + manifest.
"""
import sys, time, json
from pathlib import Path
import requests, pandas as pd, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb, data_loading as dl

OUT = rb.BASE / "data" / "options_intraday"; OUT.mkdir(parents=True, exist_ok=True)
MAN = OUT / "manifest.csv"
H = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}
BASE = "https://api.upstox.com/v2/expired-instruments"
def enc(k): return k.replace("|", "%7C")
def get(url, tries=4):
    for a in range(tries):
        r = requests.get(url, headers=H, timeout=45)
        if r.status_code == 200: return r.json().get("data", [])
        if r.status_code == 429: time.sleep(1.5*(a+1)); continue
        return None
    return None

def spot_on(day):  # NIFTY close on a date for ATM (v3 index endpoint -> dict)
    r = requests.get(f"https://api.upstox.com/v3/historical-candle/NSE_INDEX%7CNifty%2050/days/1/{day}/{day}", headers=H, timeout=30)
    if r.status_code == 200:
        c = r.json().get("data", {}).get("candles", [])
        return c[0][4] if c else None
    return None

def run(underlying_key, name, n_expiries, atm_w, throttle=0.12):
    done = set()
    if MAN.exists(): done = set(pd.read_csv(MAN)["contract_key"])
    exps = get(f"{BASE}/expiries?instrument_key={enc(underlying_key)}") or []
    exps = sorted([e for e in exps if e < time.strftime("%Y-%m-%d")], reverse=True)[:n_expiries]   # most-recent EXPIRED
    print(f"{name}: testing {len(exps)} recent expired expiries: {exps}", flush=True)
    rows_man = []; nrows_tot = 0
    for exp in exps:
        cons = get(f"{BASE}/option/contract?instrument_key={enc(underlying_key)}&expiry_date={exp}") or []
        sp = spot_on(exp) or (np.median([c["strike_price"] for c in cons]) if cons else 0)
        strikes = sorted(set(c["strike_price"] for c in cons))
        atm = min(strikes, key=lambda s: abs(s-sp)) if strikes else 0
        step = np.median(np.diff(strikes)) if len(strikes) > 1 else 50
        lo, hi = atm - atm_w*step, atm + atm_w*step
        pick = [c for c in cons if lo <= c["strike_price"] <= hi]
        print(f"  {exp}: spot~{sp:.0f} ATM {atm:.0f} | {len(cons)} contracts -> ATM+-{atm_w} = {len(pick)}", flush=True)
        edir = OUT / name / exp.replace("-", ""); edir.mkdir(parents=True, exist_ok=True)
        for c in pick:
            ck = c["instrument_key"]
            if ck in done: continue
            frm = (pd.Timestamp(exp) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
            cd = get(f"{BASE}/historical-candle/{enc(ck)}/1minute/{exp}/{frm}")
            if isinstance(cd, dict): cd = cd.get("candles", [])
            time.sleep(throttle)
            if cd:
                df = pd.DataFrame(cd, columns=["ts","open","high","low","close","volume","oi"])
                fn = edir / (c["trading_symbol"].replace(" ","_")+".parquet"); df.to_parquet(fn, index=False)
                rows_man.append({"contract_key": ck, "underlying": name, "expiry": exp, "symbol": c["trading_symbol"],
                                 "strike": c["strike_price"], "type": c["instrument_type"], "n_candles": len(df),
                                 "date_min": str(df["ts"].iloc[-1])[:10], "date_max": str(df["ts"].iloc[0])[:10], "file": str(fn.relative_to(OUT))})
                nrows_tot += len(df)
            else:
                rows_man.append({"contract_key": ck, "underlying": name, "expiry": exp, "symbol": c["trading_symbol"],
                                 "strike": c["strike_price"], "type": c["instrument_type"], "n_candles": 0, "date_min":"","date_max":"","file":"NO_DATA"})
    m = pd.DataFrame(rows_man)
    if MAN.exists(): m = pd.concat([pd.read_csv(MAN), m], ignore_index=True).drop_duplicates("contract_key", keep="last")
    m.to_csv(MAN, index=False)
    print(f"\nDONE: {len(rows_man)} contracts fetched | {nrows_tot:,} candle-rows | manifest {len(m)} total", flush=True)
    print(m[m.n_candles>0][["symbol","strike","type","n_candles","date_min","date_max"]].head(10).to_string(index=False))

run("NSE_INDEX|Nifty 50", "NIFTY", n_expiries=2, atm_w=15)
