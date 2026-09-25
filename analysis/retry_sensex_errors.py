# -*- coding: utf-8 -*-
"""retry_sensex_errors.py — re-pull the SENSEX contracts that logged http_error during the full pull (code -1
= get() exhausted retries in a transient window). Resolves: (a) 14 contracts with data but a failed chunk
(possible interior gap) and (b) 125 contracts logged empty_no_candles that may be FALSE empties (the chunk
errored rather than genuinely no-trade). Re-pulls each, overwrites its parquet if data now returns, updates
manifest_sensex.csv, and rewrites issues_sensex.csv dropping the resolved http_error/empty rows. Report only
otherwise; genuine deep-strike empties stay flagged empty."""
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import opt_pull_nifty_full as op
import full_chain_pull_sensex as fs

MAN, ISS, UND = fs.MAN, fs.ISSUES, fs.UNDERLYING


def main():
    iss = pd.read_csv(ISS)
    err = iss[iss.issue == "http_error"][["expiry", "symbol"]].drop_duplicates()
    print(f"errored contracts to retry: {len(err)} across {err.expiry.nunique()} expiries", flush=True)
    # contract map per expiry (trading_symbol -> contract dict)
    cmap = {}
    for exp in sorted(err.expiry.unique()):
        cons, _ = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(UND)}&expiry_date={exp}"); cons = cons or []
        cmap[exp] = {c["trading_symbol"]: c for c in cons}
        time.sleep(0.15)

    resolved_data, still_empty, still_err, notfound = [], [], [], []
    new_man = []
    for _, r in err.iterrows():
        exp, sym = r["expiry"], r["symbol"]
        c = cmap.get(exp, {}).get(sym)
        if c is None:
            notfound.append((exp, sym)); continue
        status, man, _iss = fs.pull_one(exp, c)     # re-pulls, overwrites parquet if data
        if status == "fetch":
            resolved_data.append((exp, sym)); new_man.append(man)
        elif status == "empty":
            still_empty.append((exp, sym))
        else:
            still_err.append((exp, sym))
        time.sleep(op.THROTTLE)

    # update manifest with newly resolved contracts
    if new_man:
        m = pd.DataFrame(new_man)
        if MAN.exists():
            m = pd.concat([pd.read_csv(MAN), m], ignore_index=True).drop_duplicates(["expiry", "symbol"], keep="last")
        m.to_csv(MAN, index=False)
    # rewrite issues: drop all http_error rows; drop empty_no_candles rows for contracts now resolved-with-data
    resolved_syms = set((e, s) for e, s in resolved_data)
    keep = iss[iss.issue != "http_error"].copy()
    mask_drop_empty = keep.apply(lambda x: x.issue == "empty_no_candles" and (x.expiry, x.symbol) in resolved_syms, axis=1)
    keep = keep[~mask_drop_empty]
    keep.to_csv(ISS, index=False)

    print("\n" + "=" * 80 + "\nSENSEX http_error RETRY RESULT\n" + "=" * 80)
    print(f"  retried            : {len(err)}")
    print(f"  resolved WITH data : {len(resolved_data)}  (parquet written/updated, manifest updated)")
    print(f"  genuinely empty    : {len(still_empty)}  (deep strike, no trades - legitimately empty)")
    print(f"  STILL errored      : {len(still_err)}  (need another retry / investigate)")
    print(f"  contract not found : {len(notfound)}")
    if still_err:
        print("  still-errored (first 10):", still_err[:10])
    if resolved_data:
        print("  resolved sample:", resolved_data[:5])
    print("\nissues_sensex.csv rewritten (http_error rows cleared; false-empties removed for resolved).")


if __name__ == "__main__":
    main()
