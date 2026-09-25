# -*- coding: utf-8 -*-
"""
reversal_bounce_trade_list.py
=============================
Export the 2-day reversal-bounce trade list with D0 / D1 / D2 prices (and entry + best-exit
09:30 outcome) to Excel. Same signal logic as reversal_bounce_exit_sweep.py.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "reversal_bounce"
PER_TRADE, EXPENSE = 100_000, 0.0023
DAY_OPEN_HM, DAY_CLOSE_HM, EXIT_HM = 555, 915, 570      # 09:15 open, 15:15 close, 09:30 exit
SESSION_HMS = list(range(555, 916, 15))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["symbol", "date"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    eligible = set(zip(diag["symbol"].values, diag["date"].values))
    symbols = sorted(diag["symbol"].unique())
    print(f"Scanning {len(symbols)} symbols for reversal-bounce signals …")

    rec = []
    t0 = time.time()
    for si, sym in enumerate(symbols, 1):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        raw = raw[raw["hm"].isin(SESSION_HMS)]
        op = raw.pivot_table(index="date", columns="hm", values="open", aggfunc="last").reindex(columns=SESSION_HMS)
        cl = raw.pivot_table(index="date", columns="hm", values="close", aggfunc="last").reindex(columns=SESSION_HMS)
        dates = sorted(op.index)
        day_open = op[DAY_OPEN_HM]
        day_close = cl[DAY_CLOSE_HM].where(cl[DAY_CLOSE_HM].notna(), cl.ffill(axis=1).iloc[:, -1])
        for i in range(2, len(dates) - 1):
            d0, d1, d2, d3 = dates[i-2], dates[i-1], dates[i], dates[i+1]
            c0, c1 = day_close.get(d0, np.nan), day_close.get(d1, np.nan)
            o1 = day_open.get(d1, np.nan)
            c2, o2 = day_close.get(d2, np.nan), day_open.get(d2, np.nan)
            o0 = day_open.get(d0, np.nan)
            if not (c1 < c0):
                continue
            if not (c2 > o1 and c2 > o2):
                continue
            if (sym, d2) not in eligible:
                continue
            entry = c2
            if not (entry == entry and entry > 0):
                continue
            shares = float(np.floor(PER_TRADE / entry))
            if shares <= 0:
                continue
            d3_0930 = op.loc[d3, EXIT_HM] if d3 in op.index else np.nan
            cap = shares * entry
            gpnl = shares * (d3_0930 - entry) if d3_0930 == d3_0930 else np.nan
            gret = (d3_0930 - entry) / entry * 100 if d3_0930 == d3_0930 else np.nan
            rec.append({
                "symbol": sym,
                "D0_date": d0, "D0_open": round(o0, 2) if o0 == o0 else np.nan, "D0_close": round(c0, 2),
                "D1_date": d1, "D1_open": round(o1, 2) if o1 == o1 else np.nan, "D1_close": round(c1, 2),
                "D2_date": d2, "D2_open": round(o2, 2) if o2 == o2 else np.nan,
                "D2_close_ENTRY": round(entry, 2),
                "shares": int(shares), "capital_deployed": round(cap, 0),
                "D3_date": d3, "D3_open_0930_EXIT": round(d3_0930, 2) if d3_0930 == d3_0930 else np.nan,
                "gross_return_pct": round(gret, 4) if gret == gret else np.nan,
                "gross_pnl": round(gpnl, 2) if gpnl == gpnl else np.nan,
                "net_return_pct": round(gret - EXPENSE * 100, 4) if gret == gret else np.nan,
                "net_pnl": round(gpnl - cap * EXPENSE, 2) if gpnl == gpnl else np.nan,
            })
        if si % 150 == 0:
            print(f"  …{si}/{len(symbols)} ({len(rec)} trades, {time.time()-t0:.0f}s)")

    df = pd.DataFrame(rec).sort_values(["D2_date", "symbol"]).reset_index(drop=True)
    out = OUTDIR / "reversal_bounce_trades.xlsx"
    print(f"Writing {len(df):,} trades -> {out.name} …")
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="trades", index=False)
    df.to_csv(OUTDIR / "reversal_bounce_trades.csv", index=False)
    print(f"Saved -> {out}  ({len(df):,} rows)")
    print("\nColumns:", list(df.columns))
    print("\nFirst 5 trades:")
    pd.set_option("display.width", 240)
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
