# -*- coding: utf-8 -*-
"""
entry_uc_pct_from_uc.py
=======================
UC-hitters: % price differences FROM UC (denominator = uc_price) to the 3:15pm entry and to
the entry-day close. Reuses build_trades() (NOT recomputed).

uc_price   = prev_close * 1.1995 (theoretical ~+20% circuit; plain prior-day 15:15 close)
entry_price= 15:15 (3:15pm) candle OPEN ; close_price = 15:15 candle CLOSE
pct_uc_to_entry = (uc - entry)/uc*100 ; pct_uc_to_close = (uc - close)/uc*100
Split by locked-at-entry (entry >= uc*0.999) vs opened-by-entry.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import final_performance_report as fpr

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "entry_day_uc"
UC_MULT = 1.1995
ENTRY_HM, DAY_CLOSE_HM = 915, 915
SESSION_HMS = list(range(555, 916, 15))
PIN = 0.999


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Building main-strategy trades (reused) …")
    T = fpr.build_trades()[["symbol", "entry_date"]].copy().reset_index(drop=True)
    T["entry_date"] = pd.to_datetime(T["entry_date"]).dt.date
    n_all = len(T); print(f"  trades: {n_all}")

    recs = []
    t0 = time.time()
    for si, (sym, grp) in enumerate(T.groupby("symbol"), 1):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        raw = raw[raw["hm"].isin(SESSION_HMS)]
        po = raw.pivot_table(index="date", columns="hm", values="open", aggfunc="last").reindex(columns=SESSION_HMS)
        ph = raw.pivot_table(index="date", columns="hm", values="high", aggfunc="max").reindex(columns=SESSION_HMS)
        pc = raw.pivot_table(index="date", columns="hm", values="close", aggfunc="last").reindex(columns=SESSION_HMS)
        day_close = pc[DAY_CLOSE_HM].where(pc[DAY_CLOSE_HM].notna(), pc.ffill(axis=1).iloc[:, -1])
        dates = sorted(po.index)
        prev_close = {dates[k]: day_close.get(dates[k-1], np.nan) for k in range(1, len(dates))}
        for ed in grp["entry_date"].values:
            pcl = prev_close.get(ed, np.nan)
            if not (pcl == pcl and pcl > 0) or ed not in po.index:
                continue
            uc = pcl * UC_MULT
            if not (np.nanmax(ph.loc[ed].values) >= uc):
                continue
            entry = po.loc[ed, ENTRY_HM]; close = pc.loc[ed, ENTRY_HM]
            if not (entry == entry and entry > 0 and close == close):
                continue
            recs.append({"symbol": sym, "entry_date": ed, "uc_price": round(float(uc), 2),
                         "entry_price": round(float(entry), 2), "close_price": round(float(close), 2),
                         "pct_uc_to_entry": round(float((uc - entry) / uc * 100), 4),
                         "pct_uc_to_close": round(float((uc - close) / uc * 100), 4),
                         "locked_at_entry": bool(entry >= uc * PIN)})
        if si % 100 == 0:
            print(f"  …{si} symbols ({time.time()-t0:.0f}s)")

    D = pd.DataFrame(recs); n = len(D)
    locked = D[D["locked_at_entry"]]; opened = D[~D["locked_at_entry"]]

    def row(df, label):
        return {"group": label, "n": len(df), "pct_of_all_trades": round(len(df)/n_all*100, 2),
                "avg_pct_uc_to_entry": round(float(df["pct_uc_to_entry"].mean()), 4),
                "median_pct_uc_to_entry": round(float(df["pct_uc_to_entry"].median()), 4),
                "avg_pct_uc_to_close": round(float(df["pct_uc_to_close"].mean()), 4),
                "median_pct_uc_to_close": round(float(df["pct_uc_to_close"].median()), 4)}
    summ = pd.DataFrame([row(D, "ALL uc-hitters"), row(locked, "locked at entry"), row(opened, "opened by entry")])

    D.to_csv(OUTDIR / "uc_pct_from_uc_detail.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "entry_uc_pct_from_uc.xlsx", engine="openpyxl") as w:
        summ.to_excel(w, sheet_name="summary", index=False)
        D.to_excel(w, sheet_name="trade_detail", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 72 + "\nUC-HITTERS — % DIFFERENCES FROM UC (denominator = uc_price)\n" + "=" * 72)
    print(f"  n_uc_hitters = {n} of {n_all} trades ({n/n_all*100:.2f}%)")
    print(f"  locked at 3:15 = {len(locked)} ({len(locked)/n*100:.1f}% of hitters) | "
          f"opened by 3:15 = {len(opened)} ({len(opened)/n*100:.1f}%)\n")
    print(summ.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
