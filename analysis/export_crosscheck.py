# -*- coding: utf-8 -*-
"""export_crosscheck.py — export Excel workbooks for 5 RANDOM expiries so option prices can be manually
cross-checked. Per expiry: (1) 'daily_all_strikes' sheet = per contract/day OHLCV+OI + real/synthetic
counts (scan every strike's daily levels); (2) full 1-minute sheets for ~6 ATM-region contracts (CE/PE)
for minute-level checks. Read-only. -> results/crosscheck/crosscheck_{YYYYMMDD}.xlsx
"""
import sys, time, glob, random
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb, opt_pull_nifty_full as op

ROOT = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "crosscheck"; OUTDIR.mkdir(parents=True, exist_ok=True)
COLS1 = ["symbol", "strike", "option_type", "expiry_date", "timestamp", "DTE", "open", "high", "low", "close", "volume", "OI", "is_synthetic"]


def spot_on(day):
    c, _ = op.get(f"https://api.upstox.com/v2/historical-candle/{op.enc(op.NIFTY)}/day/{day}/{day}")
    return c[0][4] if c else None


def daily_summary(fn):
    d = pd.read_parquet(fn); d["date"] = d["timestamp"].dt.date
    g = d.groupby("date")
    out = pd.DataFrame({
        "day_open": g["open"].first(), "day_high": g["high"].max(), "day_low": g["low"].min(),
        "day_close": g["close"].last(), "volume": g["volume"].sum(), "OI_close": g["OI"].last(),
        "n_real": g["is_synthetic"].apply(lambda s: int((~s).sum())), "n_synthetic": g["is_synthetic"].sum(), "DTE": g["DTE"].first()})
    out.insert(0, "symbol", d["symbol"].iloc[0]); out.insert(1, "strike", d["strike"].iloc[0]); out.insert(2, "type", d["option_type"].iloc[0])
    return out.reset_index()


def main():
    random.seed()
    exp_dirs = [d for d in sorted(ROOT.iterdir()) if d.is_dir() and len(glob.glob(str(d / "*.parquet"))) > 20]
    picks = random.sample(exp_dirs, 5)
    print("picked expiries:", [d.name for d in picks], flush=True)
    for d in picks:
        exp = f"{d.name[:4]}-{d.name[4:6]}-{d.name[6:]}"; t0 = time.time()
        files = sorted(glob.glob(str(d / "*.parquet")))
        # daily summary across ALL strikes
        daily = pd.concat([daily_summary(f) for f in files], ignore_index=True).sort_values(["strike", "type", "date"])
        strikes = sorted(daily["strike"].unique()); step = float(np.median(np.diff(strikes))) if len(strikes) > 1 else 50.0
        sp = spot_on(exp) or float(np.median(strikes)); atm = min(strikes, key=lambda s: abs(s - sp))
        # pick ~6 ATM-region contracts (CE & PE at ATM, +/-2, +/-4 steps)
        want = [(atm, "CE"), (atm, "PE"), (atm + 2 * step, "CE"), (atm - 2 * step, "PE"), (atm + 4 * step, "CE"), (atm - 4 * step, "PE")]
        fn_by = {}
        for f in files:
            h = pd.read_parquet(f, columns=["strike", "option_type"]).iloc[0]; fn_by[(h["strike"], h["option_type"])] = f
        out = OUTDIR / f"crosscheck_{d.name}.xlsx"
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            info = pd.DataFrame([{"expiry": exp, "NIFTY_spot_on_expiry": round(sp) if sp else "n/a", "ATM_strike": atm,
                                  "n_contracts_in_chain": len(files), "strike_range": f"{strikes[0]}-{strikes[-1]}",
                                  "note": "daily_all_strikes=per-strike daily OHLCV; 1min sheets=ATM-region; is_synthetic=True -> reconstructed minute (Upstox omitted)"}])
            info.T.reset_index().to_excel(w, sheet_name="INFO", index=False, header=False)
            daily.to_excel(w, sheet_name="daily_all_strikes", index=False)
            for stk, ot in want:
                f = fn_by.get((round(stk, 2), ot)) or fn_by.get((float(stk), ot))
                if not f:
                    continue
                dd = pd.read_parquet(f)[COLS1]
                sheet = f"{int(stk)}{ot}"[:31]
                dd.to_excel(w, sheet_name=sheet, index=False)
        print(f"  {exp}: {len(files)} contracts, spot~{round(sp) if sp else '?'}, ATM {atm} -> {out.name} ({daily.shape[0]:,} daily rows) | {time.time()-t0:.0f}s", flush=True)
    print(f"\nSaved 5 workbooks -> {OUTDIR}")


if __name__ == "__main__":
    main()
