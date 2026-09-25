# -*- coding: utf-8 -*-
"""opt_pull_full.py — FULL Upstox-Plus expired-option 1-min OHLCV+OI pull, ATM+-15, all expired expiries
(>=2024-10, the data floor) for 5 index options + all F&O stocks. Resumable: skips any contract whose
parquet already exists; manifest appended per expiry. Paced + 429 backoff. Indices first, then stocks.
"""
import sys, time, io, gzip, json
from pathlib import Path
import requests, pandas as pd, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb, data_loading as dl

OUT = rb.BASE / "data" / "options_intraday"; OUT.mkdir(parents=True, exist_ok=True)
MAN = OUT / "manifest.csv"
H = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}
BASE = "https://api.upstox.com/v2/expired-instruments"
ATM_W, THROTTLE, FLOOR = 15, 0.10, "2024-10-01"
INDEX_ORDER = ["NIFTY"]          # only NIFTY among indices; all other index options excluded
def enc(k): return k.replace("|", "%7C")
TODAY = time.strftime("%Y-%m-%d")


def get(url, tries=5):
    for a in range(tries):
        try:
            r = requests.get(url, headers=H, timeout=45)
        except Exception:
            time.sleep(1.0 * (a + 1)); continue
        if r.status_code == 200:
            d = r.json().get("data", [])
            return d.get("candles", []) if isinstance(d, dict) else d
        if r.status_code in (429, 500, 502, 503):
            time.sleep(1.5 * (a + 1)); continue
        return None
    return None


def spot_on(underlying_key, day, is_index):
    seg = "days/1"
    r = requests.get(f"https://api.upstox.com/v3/historical-candle/{enc(underlying_key)}/{seg}/{day}/{day}", headers=H, timeout=30)
    if r.status_code == 200:
        c = r.json().get("data", {}).get("candles", [])
        return c[0][4] if c else None
    return None


def underlyings():
    raw = json.load(gzip.GzipFile(fileobj=io.BytesIO(requests.get(
        "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz", timeout=90, headers={"User-Agent": "M"}).content)))
    opts = [o for o in raw if o.get("instrument_type") in ("CE", "PE") and str(o.get("segment", "")).startswith("NSE_FO")]
    umap = {}
    for o in opts:
        nm = o.get("name"); uk = o.get("underlying_key")
        if nm and uk and nm not in umap:
            umap[nm] = (uk, o.get("underlying_type") == "INDEX")
    idx = [(n, umap[n][0], True) for n in INDEX_ORDER if n in umap]          # NIFTY only
    stk = sorted([(n, uk, False) for n, (uk, isidx) in umap.items() if not isidx], key=lambda x: x[0])  # actual stocks (non-index)
    return idx + stk


def main():
    unds = underlyings()
    print(f"underlyings with options: {len(unds)} ({len(INDEX_ORDER)} index + {len(unds)-len(INDEX_ORDER)} stock)", flush=True)
    man_rows = []; t0 = time.time(); n_fetched = n_skip = n_nodata = 0

    def flush_manifest():
        if not man_rows:
            return
        m = pd.DataFrame(man_rows)
        if MAN.exists():
            m = pd.concat([pd.read_csv(MAN), m], ignore_index=True).drop_duplicates("contract_key", keep="last")
        m.to_csv(MAN, index=False)

    for ui, (name, ukey, is_idx) in enumerate(unds, 1):
        exps = get(f"{BASE}/expiries?instrument_key={enc(ukey)}") or []
        exps = sorted([e for e in exps if FLOOR <= e < TODAY], reverse=True)
        print(f"[{ui}/{len(unds)}] {name}: {len(exps)} expired expiries", flush=True)
        for exp in exps:
            cons = get(f"{BASE}/option/contract?instrument_key={enc(ukey)}&expiry_date={exp}") or []
            if not cons:
                continue
            strikes = sorted(set(c["strike_price"] for c in cons))
            sp = spot_on(ukey, exp, is_idx) or float(np.median(strikes))
            atm = min(strikes, key=lambda s: abs(s - sp))
            step = float(np.median(np.diff(strikes))) if len(strikes) > 1 else 1
            lo, hi = atm - ATM_W * step, atm + ATM_W * step
            pick = [c for c in cons if lo <= c["strike_price"] <= hi]
            edir = OUT / name / exp.replace("-", ""); edir.mkdir(parents=True, exist_ok=True)
            for c in pick:
                ck = c["instrument_key"]; fn = edir / (c["trading_symbol"].replace(" ", "_").replace("/", "-") + ".parquet")
                if fn.exists():
                    n_skip += 1; continue
                frm = (pd.Timestamp(exp) - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
                cd = get(f"{BASE}/historical-candle/{enc(ck)}/1minute/{exp}/{frm}")
                time.sleep(THROTTLE)
                if cd:
                    df = pd.DataFrame(cd, columns=["ts", "open", "high", "low", "close", "volume", "oi"])
                    df.to_parquet(fn, index=False); n_fetched += 1
                    man_rows.append({"contract_key": ck, "underlying": name, "expiry": exp, "symbol": c["trading_symbol"],
                                     "strike": c["strike_price"], "type": c["instrument_type"], "n_candles": len(df),
                                     "date_min": str(df["ts"].iloc[-1])[:10], "date_max": str(df["ts"].iloc[0])[:10],
                                     "file": str(fn.relative_to(OUT))})
                else:
                    n_nodata += 1
            flush_manifest(); man_rows = []
            print(f"    {name} {exp}: {len(pick)} picked | cum fetched {n_fetched:,} skip {n_skip:,} nodata {n_nodata:,} | {time.time()-t0:.0f}s", flush=True)
    flush_manifest()
    print(f"\nCOMPLETE: fetched {n_fetched:,} | skipped(existing) {n_skip:,} | nodata {n_nodata:,} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
