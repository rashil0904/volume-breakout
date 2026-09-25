# -*- coding: utf-8 -*-
"""retry_banknifty_opt_errors.py — re-pull BankNifty option contracts that logged http_error during the
monthly-only full pull (transient window, code -1/429/5xx). Resolves contracts either lacking data (possible
interior gap) or logged empty_no_candles (possible false-empty). Re-pulls each, overwrites parquet if data
now returns, updates manifest, rewrites issues dropping resolved http_error/empty rows."""
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import opt_pull_nifty_full as op
import full_chain_pull_banknifty_options as fp

MAN, ISS, UND = fp.MAN, fp.ISSUES, fp.UNDERLYING


def main():
    iss = pd.read_csv(ISS)
    err = iss[iss.issue == "http_error"][["expiry", "symbol"]].drop_duplicates()
    print(f"errored contracts to retry: {len(err)} across {err.expiry.nunique()} expiries", flush=True)
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
        status, man, _iss = fp.pull_one(exp, c)
        if status == "fetch": resolved_data.append((exp, sym)); new_man.append(man)
        elif status == "empty": still_empty.append((exp, sym))
        else: still_err.append((exp, sym))
        time.sleep(op.THROTTLE)

    if new_man:
        m = pd.DataFrame(new_man)
        if MAN.exists(): m = pd.concat([pd.read_csv(MAN), m], ignore_index=True).drop_duplicates(["expiry", "symbol"], keep="last")
        m.to_csv(MAN, index=False)
    resolved_syms = set((e, s) for e, s in resolved_data)
    keep = iss[iss.issue != "http_error"].copy()
    mask_drop_empty = keep.apply(lambda x: x.issue == "empty_no_candles" and (x.expiry, x.symbol) in resolved_syms, axis=1)
    keep = keep[~mask_drop_empty]
    keep.to_csv(ISS, index=False)

    print("\n" + "=" * 80 + "\nBANKNIFTY OPTIONS http_error RETRY RESULT\n" + "=" * 80)
    print(f"  retried            : {len(err)}")
    print(f"  resolved WITH data : {len(resolved_data)}")
    print(f"  genuinely empty    : {len(still_empty)}")
    print(f"  STILL errored      : {len(still_err)}")
    print(f"  contract not found : {len(notfound)}")
    if still_err: print("  still-errored (first 10):", still_err[:10])
    print("\nissues_banknifty_opt.csv rewritten (http_error cleared; false-empties removed for resolved).")


if __name__ == "__main__":
    main()
