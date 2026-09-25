# -*- coding: utf-8 -*-
"""reverify_expired_history.py — thorough re-verification of how far back Upstox serves data.
(A) raw /expiries response (hidden cap? metadata?)  (B) index 1-min baseline back to 2022?
(C) probe OLD monthly-expiry dates DIRECTLY on the contract endpoint (bypassing the expiries list)
(D) if an old expiry yields contracts, pull its 1-min candles.  Tries v2 and v3 paths.
"""
import sys, time, json
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_loading as dl

H = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}
NIFTY = "NSE_INDEX|Nifty 50"
def enc(k): return k.replace("|", "%7C")


def raw(url):
    try:
        r = requests.get(url, headers=H, timeout=40)
        try:
            j = r.json()
        except Exception:
            j = {"_text": r.text[:200]}
        return r.status_code, j
    except Exception as e:
        return -1, {"_err": str(e)[:150]}


def main():
    print("=" * 90 + "\n(A) RAW /expiries response (v2)\n" + "=" * 90)
    for base in ["https://api.upstox.com/v2/expired-instruments/expiries",
                 "https://api.upstox.com/v3/expired-instruments/expiries"]:
        sc, j = raw(f"{base}?instrument_key={enc(NIFTY)}")
        data = j.get("data", j) if isinstance(j, dict) else j
        n = len(data) if isinstance(data, list) else "?"
        oldest = min(data) if isinstance(data, list) and data else "-"
        print(f"  {base.split('/')[3]}: http={sc} n_expiries={n} oldest={oldest} keys={list(j)[:4] if isinstance(j,dict) else '-'}")

    print("\n" + "=" * 90 + "\n(B) INDEX 1-min baseline — does general history reach 2022?\n" + "=" * 90)
    for path in [f"https://api.upstox.com/v2/historical-candle/{enc(NIFTY)}/1minute/2022-02-01/2022-01-03",
                 f"https://api.upstox.com/v3/historical-candle/{enc(NIFTY)}/minutes/1/2022-02-01/2022-01-03"]:
        sc, j = raw(path)
        c = (j.get("data", {}) or {}).get("candles", []) if isinstance(j, dict) else []
        print(f"  {path.split('upstox.com')[1][:55]}: http={sc} candles={len(c)}{' first='+str(c[-1][0])[:10] if c else ''}")

    print("\n" + "=" * 90 + "\n(C) OLD monthly expiries probed DIRECTLY (bypass the /expiries list)\n" + "=" * 90)
    old_exps = ["2024-09-26", "2024-06-27", "2024-03-28", "2023-12-28", "2023-06-29", "2022-12-29", "2022-06-30"]
    found = []
    for exp in old_exps:
        for ver, url in [("v2", f"https://api.upstox.com/v2/expired-instruments/option/contract?instrument_key={enc(NIFTY)}&expiry_date={exp}"),
                         ("v3", f"https://api.upstox.com/v3/expired-instruments/option/contract?instrument_key={enc(NIFTY)}&expiry_date={exp}")]:
            sc, j = raw(url)
            data = j.get("data", []) if isinstance(j, dict) else []
            ncons = len(data) if isinstance(data, list) else 0
            msg = ""
            if isinstance(j, dict) and j.get("errors"):
                msg = str(j["errors"])[:90]
            print(f"  {exp} [{ver}]: http={sc} contracts={ncons} {msg}")
            if ncons:
                found.append((exp, data)); break
        time.sleep(0.2)

    print("\n" + "=" * 90 + "\n(D) 1-min pull for any OLD expiry that yielded contracts\n" + "=" * 90)
    if not found:
        print("  none of the pre-Oct-2024 expiries returned contracts -> not available.")
    for exp, cons in found[:2]:
        strikes = sorted(set(c["strike_price"] for c in cons)); atm = strikes[len(strikes) // 2]
        ck = next(c["instrument_key"] for c in cons if c["strike_price"] == atm)
        frm = time.strftime("%Y-%m-%d", time.gmtime(time.mktime(time.strptime(exp, "%Y-%m-%d")) - 40 * 86400))
        sc, j = raw(f"https://api.upstox.com/v2/expired-instruments/historical-candle/{enc(ck)}/1minute/{exp}/{frm}")
        c = (j.get("data", {}) or {}).get("candles", []) if isinstance(j, dict) else []
        print(f"  {exp} ATM {atm} ({ck}): http={sc} candles={len(c)}")


if __name__ == "__main__":
    main()
