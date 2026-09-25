# -*- coding: utf-8 -*-
"""
entry_uc_price_diffs.py
=======================
For the main strategy's entry-day UC-hitters, average price differences between the UC level,
the 3:15pm entry, and the entry-day close. Reuses build_trades() (NOT recomputed).

uc_price   = prev_close * 1.1995  (theoretical ~+20% circuit level; plain prior-day close)
entry_price= 15:15 (3:15pm) candle OPEN (strategy entry)
close_price= 15:15 candle CLOSE (entry-day close)
Diffs (₹ + %): (1) uc-entry, (2) uc-close, (3) entry-close.
Split by locked-at-entry (entry >= uc*0.999) vs opened-by-entry (entry < uc).
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    print(f"  trades: {len(T)}")

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
            day_hi = np.nanmax(ph.loc[ed].values)
            if not (day_hi >= uc):                               # UC-hitter set only
                continue
            entry = po.loc[ed, ENTRY_HM]; close = pc.loc[ed, ENTRY_HM]
            if not (entry == entry and entry > 0 and close == close):
                continue
            recs.append({"symbol": sym, "entry_date": ed,
                         "uc_price": round(float(uc), 2), "entry_price": round(float(entry), 2),
                         "close_price": round(float(close), 2), "day_high": round(float(day_hi), 2),
                         "diff_uc_entry_abs": round(float(uc - entry), 2),
                         "diff_uc_entry_pct": round(float((uc - entry) / entry * 100), 4),
                         "diff_uc_close_abs": round(float(uc - close), 2),
                         "diff_uc_close_pct": round(float((uc - close) / close * 100), 4),
                         "diff_entry_close_abs": round(float(close - entry), 2),
                         "diff_entry_close_pct": round(float((close - entry) / entry * 100), 4),
                         "locked_at_entry": bool(entry >= uc * PIN)})
        if si % 100 == 0:
            print(f"  …{si} symbols ({time.time()-t0:.0f}s)")

    D = pd.DataFrame(recs)
    n = len(D)
    METRICS = [("diff_uc_entry", "UC vs 3:15 entry"), ("diff_uc_close", "UC vs close"),
               ("diff_entry_close", "3:15 entry vs close")]

    def stats(df, label):
        rows = []
        for k, name in METRICS:
            rows.append({"group": label, "difference": name, "n": len(df),
                         "avg_abs_inr": round(float(df[f"{k}_abs"].mean()), 2),
                         "median_abs_inr": round(float(df[f"{k}_abs"].median()), 2),
                         "avg_pct": round(float(df[f"{k}_pct"].mean()), 4),
                         "median_pct": round(float(df[f"{k}_pct"].median()), 4)})
        return pd.DataFrame(rows)

    locked = D[D["locked_at_entry"]]; opened = D[~D["locked_at_entry"]]
    allst = stats(D, "ALL uc-hitters")
    lst = stats(locked, "locked at entry")
    ost = stats(opened, "opened by entry")
    full = pd.concat([allst, lst, ost], ignore_index=True)

    # histogram of diff_uc_entry_pct
    fig, ax = plt.subplots(figsize=(9, 5))
    x = D["diff_uc_entry_pct"].clip(-1, 25)
    ax.hist(x, bins=40, color="#1f77b4", edgecolor="white")
    ax.axvline(D["diff_uc_entry_pct"].median(), ls="--", color="#d62728",
               label=f"median {D['diff_uc_entry_pct'].median():.2f}%")
    ax.set_xlabel("diff_uc_entry_pct  (how far below UC the 3:15 entry was, %)")
    ax.set_ylabel("count of trades")
    ax.set_title("UC-hitters: distance of 3:15pm entry below the circuit price")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUTDIR / "diff_uc_entry_hist.png", dpi=120); plt.close(fig)

    D.to_csv(OUTDIR / "uc_price_diffs_detail.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "entry_uc_price_diffs.xlsx", engine="openpyxl") as w:
        full.to_excel(w, sheet_name="diffs_summary", index=False)
        D.to_excel(w, sheet_name="trade_detail", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 78 + "\nUC-HITTERS — PRICE DIFFERENCES (UC / 3:15 entry / close)\n" + "=" * 78)
    print(f"  n_uc_hitters = {n} | locked at entry = {len(locked)} ({len(locked)/n*100:.1f}%) | "
          f"opened by entry = {len(opened)} ({len(opened)/n*100:.1f}%)\n")
    print(full.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
