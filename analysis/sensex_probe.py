# -*- coding: utf-8 -*-
"""sensex_probe.py — PRE-PULL PROBE for SENSEX (BSE) expired options via Upstox expired-instruments API.
Confirms: (1) token live, (2) which SENSEX instrument_key the API accepts, (3) expiries list + weekday
(flags weekday changes), (4) contract naming/format for one expiry, (5) 1-min candle flow + earliest date
available (empirical BSE floor). NO bulk pull — a few requests only."""
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import opt_pull_nifty_full as op   # reuse get(), enc(), BASE, headers

CANDIDATES = ["BSE_INDEX|SENSEX", "BSE_INDEX|Sensex", "BSE_INDEX|BSE SENSEX", "BSE_INDEX|SENSEX 30"]


def main():
    # 0) token liveness via a known-good NIFTY call
    ex_n, sc_n = op.get(f"{op.BASE}/expiries?instrument_key={op.enc(op.NIFTY)}")
    print(f"[token check] NIFTY expiries -> http {sc_n} | n={len(ex_n) if ex_n else 0}")
    if not ex_n:
        print("  !! token appears DEAD or API down — cannot proceed. Re-auth needed."); return

    # 1) find the SENSEX key the API accepts
    good = None
    for k in CANDIDATES:
        ex, sc = op.get(f"{op.BASE}/expiries?instrument_key={op.enc(k)}")
        print(f"[key probe] {k!r:28} -> http {sc} | expiries {len(ex) if ex else 0}")
        if ex:
            good = (k, ex); break
        time.sleep(0.3)
    if not good:
        print("  !! No SENSEX key accepted by expired-instruments API. BSE may not be served — FLAG."); return
    key, exps = good
    print(f"\n>>> USABLE SENSEX KEY: {key!r}")

    exps = sorted(exps)
    ed = pd.to_datetime(exps)
    print(f"\n[expiries] total {len(exps)} | first {exps[0]} | last {exps[-1]}")
    inwin = [e for e in exps if "2024-10-01" <= e <= "2026-07-31"]
    print(f"[expiries in Oct2024..Jul2026] {len(inwin)} | first {inwin[0] if inwin else '-'} | last {inwin[-1] if inwin else '-'}")
    # weekday distribution + change points
    wk = ed.day_name()
    dfw = pd.DataFrame({"date": exps, "wd": wk})
    print("\n[weekday distribution of ALL expiries]"); print(dfw["wd"].value_counts().to_string())
    # detect change points (chronological weekday transitions)
    changes = []
    prev = None
    for d, w in zip(exps, wk):
        if prev and w != prev:
            changes.append((d, prev, w))
        prev = w
    print("\n[weekday change points] (date, from->to):")
    for d, a, b in changes:
        print(f"   {d}: {a} -> {b}")
    if not changes:
        print("   none (single weekday throughout)")

    # 2) contract naming for a recent in-window expiry
    probe_exp = inwin[-1] if inwin else exps[-1]
    cons, sc = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(key)}&expiry_date={probe_exp}")
    print(f"\n[contracts] expiry {probe_exp} -> http {sc} | n={len(cons) if cons else 0}")
    if cons:
        strikes = sorted(set(c["strike_price"] for c in cons))
        print(f"  strikes {len(strikes)} | min {strikes[0]} max {strikes[-1]} | step~{strikes[1]-strikes[0] if len(strikes)>1 else '?'}")
        print(f"  CE {sum(c['instrument_type']=='CE' for c in cons)} PE {sum(c['instrument_type']=='PE' for c in cons)}")
        c0 = cons[len(cons)//2]
        print("  sample contract keys:", list(c0.keys()))
        print(f"  sample: trading_symbol={c0.get('trading_symbol')!r} instrument_key={c0.get('instrument_key')!r} strike={c0.get('strike_price')} type={c0.get('instrument_type')}")
        # 3) 1-min candle flow on the ATM-ish contract, and earliest available date
        ck = c0["instrument_key"]
        to_d = pd.Timestamp(probe_exp).date(); frm = (pd.Timestamp(probe_exp) - pd.Timedelta(days=30)).date()
        cd, sc = op.get(f"{op.BASE}/historical-candle/{op.enc(ck)}/1minute/{to_d.isoformat()}/{frm.isoformat()}")
        print(f"\n[1-min candle test] {c0.get('trading_symbol')} last 30d -> http {sc} | candles {len(cd) if cd else 0}")
        if cd:
            print("  first candle:", cd[-1]); print("  last  candle:", cd[0])

    # 4) empirical BSE floor: earliest expiry that returns candles (probe a few early in-window expiries)
    print("\n[floor probe] earliest in-window expiries with any option data:")
    for e in inwin[:6]:
        cons2, _ = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(key)}&expiry_date={e}")
        if not cons2:
            print(f"   {e}: no contracts listed"); continue
        cc = sorted(cons2, key=lambda c: c["strike_price"])[len(cons2)//2]
        to_d = pd.Timestamp(e).date(); frm = (pd.Timestamp(e) - pd.Timedelta(days=20)).date()
        cd, sc = op.get(f"{op.BASE}/historical-candle/{op.enc(cc['instrument_key'])}/1minute/{to_d.isoformat()}/{frm.isoformat()}")
        emin = pd.to_datetime([r[0] for r in cd]).min() if cd else None
        print(f"   {e}: contracts {len(cons2)} | mid-strike candles {len(cd) if cd else 0} | earliest ts {emin}")
        time.sleep(0.3)


if __name__ == "__main__":
    main()
