# -*- coding: utf-8 -*-
"""refetch_nifty_options.py — targeted re-fetch of the audit's actionable items ONLY:
(1) empty expiry 2024-12-26 (probe contracts; if present, pull ATM+-15), (2) the 1 corrupted contract,
(3) the 5 stopped-before-expiry contracts (re-pull to confirm genuine dormancy vs a fetch cut-off).
Reuses opt_pull_nifty_full functions so schema/DTE are identical. Reports old-vs-new last-date per contract
and updates the manifest for re-fetched contracts. Does NOT touch anything else.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import opt_pull_nifty_full as op

OUT, MAN, CUTOFF = op.OUT, op.MAN, op.CUTOFF
SPECIFIC = [                                     # (expiry, strike, option_type, reason)
    ("2025-08-21", 25600.0, "CE", "corrupted_file"),
    ("2024-10-24", 23650.0, "CE", "stopped_early"),
    ("2025-04-17", 23650.0, "CE", "stopped_early"),
    ("2025-04-17", 24550.0, "PE", "stopped_early"),
    ("2025-11-25", 26650.0, "CE", "stopped_early"),
    ("2026-05-12", 22650.0, "CE", "stopped_early"),
]
_ccache = {}


def contracts(exp):
    if exp not in _ccache:
        c, _ = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(op.NIFTY)}&expiry_date={exp}")
        _ccache[exp] = c or []
    return _ccache[exp]


def old_last(fn):
    if not fn.exists():
        return "-"
    try:
        return str(pd.read_parquet(fn, columns=["timestamp"])["timestamp"].max())[:10]
    except Exception:
        return "CORRUPT"


def fetch_one(exp, strike, otype, reason):
    cons = contracts(exp)
    match = [c for c in cons if c["strike_price"] == strike and c["instrument_type"] == otype]
    if not match:
        return {"expiry": exp, "strike": strike, "type": otype, "reason": reason, "status": "no_contract_in_chain", "old_last": "-", "new_last": "-", "n": 0}
    c = match[0]; ck = c["instrument_key"]; sym = c["trading_symbol"]
    edir = OUT / exp.replace("-", ""); edir.mkdir(parents=True, exist_ok=True)
    fn = edir / (sym.replace(" ", "_").replace("/", "-") + ".parquet")
    ol = old_last(fn)
    if fn.exists():
        fn.unlink()                                            # remove stale/corrupt before re-pull
    to_date = min(pd.Timestamp(exp).date(), pd.Timestamp(CUTOFF).date())
    rows, iss = op.pull_contract(ck, exp, to_date)
    if not rows:
        return {"expiry": exp, "strike": strike, "type": otype, "reason": reason, "status": "still_empty", "old_last": ol, "new_last": "-", "n": 0}
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "OI"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    keep = ts <= pd.Timestamp(CUTOFF) + pd.Timedelta(days=1)
    df = df[keep].copy(); ts = ts[keep]; df["timestamp"] = ts
    df["symbol"] = sym; df["strike"] = strike; df["option_type"] = otype; df["expiry_date"] = exp
    df["DTE"] = (pd.Timestamp(exp).normalize() - ts.dt.normalize()).dt.days
    df = df[["symbol", "strike", "option_type", "expiry_date", "timestamp", "DTE", "open", "high", "low", "close", "volume", "OI"]].drop_duplicates("timestamp").sort_values("timestamp")
    df.to_parquet(fn, index=False)
    nl = str(df["timestamp"].iloc[-1])[:10]
    return {"expiry": exp, "strike": strike, "type": otype, "reason": reason,
            "status": "improved" if (ol not in ("-",) and nl > ol) else ("rewritten" if ol == "CORRUPT" else "unchanged" if nl == ol else "fetched"),
            "old_last": ol, "new_last": nl, "n": len(df), "min_DTE": int(df["DTE"].min()), "file": sym}


def main():
    t0 = time.time(); results = []
    # (1) empty expiry 2024-12-26
    print("=== 2024-12-26 (empty expiry) ===", flush=True)
    exp = "2024-12-26"; cons = contracts(exp)
    print(f"  contract endpoint returned {len(cons)} contracts for {exp}", flush=True)
    if cons:
        strikes = sorted(set(c["strike_price"] for c in cons)); step = float(np.median(np.diff(strikes))) if len(strikes) > 1 else 50.0
        sp = op.spot_on(exp) or float(np.median(strikes)); atm = min(strikes, key=lambda s: abs(s - sp))
        pick = [c for c in cons if atm - op.ATM_W * step <= c["strike_price"] <= atm + op.ATM_W * step]
        print(f"  ATM {atm} -> {len(pick)} contracts to pull", flush=True)
        for c in pick:
            results.append(fetch_one(exp, c["strike_price"], c["instrument_type"], "empty_expiry_2024-12-26"))
    else:
        results.append({"expiry": exp, "strike": "-", "type": "-", "reason": "empty_expiry", "status": "GENUINELY_UNAVAILABLE (0 contracts on Upstox)", "old_last": "-", "new_last": "-", "n": 0})
    # (2)+(3) specific contracts
    print("\n=== specific contracts (corrupted + stopped-early) ===", flush=True)
    for exp, strike, otype, reason in SPECIFIC:
        r = fetch_one(exp, strike, otype, reason); results.append(r)
        print(f"  {exp} {int(strike)} {otype} [{reason}]: {r['status']} | last {r['old_last']}->{r.get('new_last','-')} | n={r['n']}", flush=True)

    R = pd.DataFrame(results)
    # update manifest for re-fetched contracts
    if MAN.exists() and (R["n"] > 0).any():
        man = pd.read_csv(MAN)
        add = []
        for r in results:
            if r["n"] > 0:
                add.append({"expiry": r["expiry"], "symbol": r.get("file", ""), "strike": r["strike"], "option_type": r["type"],
                            "n_candles": r["n"], "date_min": "", "date_max": r["new_last"], "dte_min": r.get("min_DTE", 0), "dte_max": "", "file": ""})
        if add:
            man = pd.concat([man, pd.DataFrame(add)], ignore_index=True).drop_duplicates(["expiry", "symbol"], keep="last")
            man.to_csv(MAN, index=False)
    R.to_csv(op.OUT.parent.parent.parent / "results" / "options_audit" / "refetch_results.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 88 + "\nRE-FETCH RESULTS\n" + "=" * 88)
    print(R[["expiry", "strike", "type", "reason", "status", "old_last", "new_last", "n"]].to_string(index=False))
    print(f"\n  2024-12-26 contracts now: {int((R.reason.str.contains('2024-12-26') & (R.n>0)).sum())} pulled")
    print(f"  stopped_early confirmed dormant (unchanged): {int(((R.reason=='stopped_early') & (R.status=='unchanged')).sum())} / 5")
    print(f"  stopped_early recovered more data (improved): {int(((R.reason=='stopped_early') & (R.status=='improved')).sum())} / 5")
    print(f"  corrupted file rewritten: {int((R.reason=='corrupted_file').sum())}")
    print(f"\n  NOTE: to refresh the audit for affected expiries, delete their part files in results/options_audit/parts/ and re-run audit_nifty_options.py")
    print(f"\nDone | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
