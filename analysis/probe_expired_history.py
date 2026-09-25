# -*- coding: utf-8 -*-
"""probe_expired_history.py — determine how far back Upstox 'expired-instruments' serves 1-minute option
data for NIFTY. Enumerates expired expiries, then binary-searches (availability is monotonic: recent =
served, old = not) for the EARLIEST expiry whose ATM contract returns candles.
"""
import sys, time
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_loading as dl

H = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}
BASE = "https://api.upstox.com/v2/expired-instruments"
NIFTY = "NSE_INDEX|Nifty 50"
def enc(k): return k.replace("|", "%7C")


def get(url):
    for a in range(4):
        try:
            r = requests.get(url, headers=H, timeout=40)
        except Exception:
            time.sleep(1.2 * (a + 1)); continue
        if r.status_code == 200:
            d = r.json().get("data", [])
            return (d.get("candles", []) if isinstance(d, dict) else d), 200
        if r.status_code in (429, 500, 502, 503):
            time.sleep(1.5 * (a + 1)); continue
        return None, r.status_code
    return None, -1


def expiry_has_data(exp):
    cons, sc = get(f"{BASE}/option/contract?instrument_key={enc(NIFTY)}&expiry_date={exp}")
    if not cons:
        return None, f"no-contracts({sc})"
    strikes = sorted(set(c["strike_price"] for c in cons))
    atm = strikes[len(strikes) // 2]
    ck = next(c["instrument_key"] for c in cons if c["strike_price"] == atm)
    frm = time.strftime("%Y-%m-%d", time.gmtime(time.mktime(time.strptime(exp, "%Y-%m-%d")) - 40 * 86400))
    cd, sc2 = get(f"{BASE}/historical-candle/{enc(ck)}/1minute/{exp}/{frm}")
    n = len(cd) if cd else 0
    return (n > 0), f"candles={n} http={sc2} atm={atm}"


def main():
    exps, sc = get(f"{BASE}/expiries?instrument_key={enc(NIFTY)}")
    if not exps:
        print(f"expiries call failed (http {sc}). data:", exps); return
    today = time.strftime("%Y-%m-%d")
    exps = sorted([e for e in exps if e < today])
    print(f"NIFTY expired expiries returned: {len(exps)} | oldest {exps[0]} | newest {exps[-1]}", flush=True)

    # quick scan at a few checkpoints
    print("\ncheckpoint probes (oldest -> newest):")
    idxs = sorted(set([0, len(exps)//6, len(exps)//3, len(exps)//2, 2*len(exps)//3, 5*len(exps)//6, len(exps)-1]))
    for i in idxs:
        ok, info = expiry_has_data(exps[i]); print(f"  [{i:3d}] {exps[i]}: {'DATA' if ok else 'none'} | {info}", flush=True); time.sleep(0.2)

    # binary search for earliest expiry WITH data (monotonic: old=none, recent=data)
    lo, hi, first = 0, len(exps) - 1, None
    print("\nbinary search for earliest expiry with 1-min data:")
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, info = expiry_has_data(exps[mid])
        print(f"  test [{mid:3d}] {exps[mid]}: {'DATA' if ok else 'none'} | {info}", flush=True)
        if ok:
            first = exps[mid]; hi = mid - 1
        else:
            lo = mid + 1
        time.sleep(0.2)
    print(f"\n==> EARLIEST NIFTY expiry serving 1-min data: {first}")


if __name__ == "__main__":
    main()
