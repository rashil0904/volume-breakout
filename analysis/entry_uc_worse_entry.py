# -*- coding: utf-8 -*-
"""
entry_uc_worse_entry.py
=======================
Counterfactual: buy at the UC price instead of the actual 3:15pm entry, for ALL entry-day
UC-hitters. Same shares, same exits (reused) -> additional_loss = shares × (actual_entry − uc),
exit-independent. Reuses build_trades() (NOT recomputed).

uc_entry_price = prev_close × 1.1995 (theoretical circuit; plain prior-day 15:15 close).
Share convention (default): SAME shares as the actual trade (isolates the price-gap impact).
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
BASE_POOL = 500_000
ENTRY_HM, DAY_CLOSE_HM = 915, 915
SESSION_HMS = list(range(555, 916, 15))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Building main-strategy trades (reused) …")
    T = fpr.build_trades()[["symbol", "entry_date", "entry_price", "shares", "exit_price",
                            "gross_pnl"]].copy().reset_index(drop=True)
    T["entry_date"] = pd.to_datetime(T["entry_date"]).dt.date
    n_all = len(T); print(f"  trades: {n_all}")

    # prev_close + UC-hit flag per (symbol, entry_date)
    uc_lvl = np.full(n_all, np.nan); hit = np.zeros(n_all, bool)
    t0 = time.time()
    for si, (sym, g) in enumerate(T.groupby("symbol"), 1):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        raw = raw[raw["hm"].isin(SESSION_HMS)]
        ph = raw.pivot_table(index="date", columns="hm", values="high", aggfunc="max").reindex(columns=SESSION_HMS)
        pc = raw.pivot_table(index="date", columns="hm", values="close", aggfunc="last").reindex(columns=SESSION_HMS)
        day_close = pc[DAY_CLOSE_HM].where(pc[DAY_CLOSE_HM].notna(), pc.ffill(axis=1).iloc[:, -1])
        dates = sorted(ph.index)
        prev_close = {dates[k]: day_close.get(dates[k-1], np.nan) for k in range(1, len(dates))}
        for i, ed in zip(g.index, g["entry_date"]):
            pcl = prev_close.get(ed, np.nan)
            if not (pcl == pcl and pcl > 0) or ed not in ph.index:
                continue
            uc = pcl * UC_MULT
            uc_lvl[i] = uc
            hit[i] = bool(np.nanmax(ph.loc[ed].values) >= uc)
        if si % 100 == 0:
            print(f"  …{si} symbols ({time.time()-t0:.0f}s)")

    T["uc_entry_price"] = np.round(uc_lvl, 2); T["hit_uc"] = hit
    U = T[T["hit_uc"] & T["uc_entry_price"].notna()].copy()
    n = len(U)

    U["actual_pnl"] = U["gross_pnl"]
    U["uc_pnl"] = U["shares"] * (U["exit_price"] - U["uc_entry_price"])
    U["additional_loss"] = U["uc_pnl"] - U["actual_pnl"]          # = shares × (actual_entry − uc)
    U["additional_loss_pct"] = (U["entry_price"] - U["uc_entry_price"]) / U["entry_price"] * 100
    U["entry_gap_pct"] = (U["uc_entry_price"] - U["entry_price"]) / U["entry_price"] * 100

    tot_add = float(U["additional_loss"].sum())
    tot_actual = float(U["actual_pnl"].sum()); tot_uc = float(U["uc_pnl"].sum())
    flips = U[(U["actual_pnl"] > 0) & (U["uc_pnl"] <= 0)]

    summary = pd.DataFrame([{
        "n_uc_hitters": n, "pct_of_all_trades": round(n / n_all * 100, 2),
        "total_additional_loss_inr": round(tot_add, 0),
        "avg_additional_loss_inr": round(float(U["additional_loss"].mean()), 2),
        "median_additional_loss_inr": round(float(U["additional_loss"].median()), 2),
        "avg_additional_loss_pct": round(float(U["additional_loss_pct"].mean()), 4),
        "median_additional_loss_pct": round(float(U["additional_loss_pct"].median()), 4),
        "total_actual_pnl_inr": round(tot_actual, 0),
        "total_actual_return_fixedbase_pct": round(tot_actual / BASE_POOL * 100, 4),
        "total_uc_pnl_inr": round(tot_uc, 0),
        "total_uc_return_fixedbase_pct": round(tot_uc / BASE_POOL * 100, 4),
        "n_flip_winner_to_loser": len(flips),
        "pct_flip_of_hitters": round(len(flips) / n * 100, 2),
    }])

    detail = U[["symbol", "entry_date", "entry_price", "uc_entry_price", "entry_gap_pct", "shares",
                "exit_price", "actual_pnl", "uc_pnl", "additional_loss", "additional_loss_pct"]].sort_values("additional_loss")
    detail.to_csv(OUTDIR / "uc_worse_entry_detail.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "entry_uc_worse_entry.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        detail.to_excel(w, sheet_name="trade_detail", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 72 + "\nWORSE-ENTRY (buy at UC) COUNTERFACTUAL — UC-hitters\n" + "=" * 72)
    print(summary.T.to_string(header=False))
    print(f"\n  {n} UC-hitters | same shares, same exits (only entry price -> UC).")
    print(f"  TOTAL additional loss: ₹{tot_add:,.0f} (avg ₹{U['additional_loss'].mean():,.0f}/trade, "
          f"{U['additional_loss_pct'].mean():.2f}% avg entry penalty).")
    print(f"  UC subset P&L: actual ₹{tot_actual:,.0f} ({tot_actual/BASE_POOL*100:.2f}% fixedbase) "
          f"-> UC-entry ₹{tot_uc:,.0f} ({tot_uc/BASE_POOL*100:.2f}% fixedbase).")
    print(f"  Winner->loser flips under UC entry: {len(flips)} of {n} ({len(flips)/n*100:.1f}%).")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
