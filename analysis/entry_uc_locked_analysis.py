# -*- coding: utf-8 -*-
"""
entry_uc_locked_analysis.py
===========================
How many of the main strategy's trades were ACTUALLY at upper circuit (locked) at the 3:15pm
entry. Reuses final_performance_report.build_trades() (NOT recomputed).

uc_level = prev_close * 1.1995 (plain prior-day 15:15 close; exchange circuit convention).
STRICT-LOCKED proxy (fill-realism relevant): entry-day 15:15 candle is FLAT (high~low, ~zero
range) AND pinned at/above uc_level -> genuinely circuit-locked, unfillable at the assumed price.
LOOSER: 15:15 candle high >= uc_level (touched UC at the entry candle).
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
FLAT_TOL = 0.002        # 15:15 candle range <= 0.2% of price -> "flat" (locked)
PIN_TOL = 0.999         # price >= uc_level * 0.999 -> pinned at/above UC


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Building main-strategy trades (reused) …")
    T = fpr.build_trades()
    T = T[["symbol", "entry_date", "entry_price", "capital_deployed", "gross_ret",
           "gross_pnl", "net_pnl", "net_ret"]].copy().reset_index(drop=True)
    T["entry_date"] = pd.to_datetime(T["entry_date"]).dt.date
    n_all = len(T)
    print(f"  trades: {n_all}")

    o15 = np.full(n_all, np.nan); h15 = np.full(n_all, np.nan)
    l15 = np.full(n_all, np.nan); c15 = np.full(n_all, np.nan); pcv = np.full(n_all, np.nan)
    idx_by_sym = {s: g.index.values for s, g in T.groupby("symbol")}
    t0 = time.time()
    for si, (sym, rows) in enumerate(idx_by_sym.items(), 1):
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
        pl = raw.pivot_table(index="date", columns="hm", values="low", aggfunc="min").reindex(columns=SESSION_HMS)
        pc = raw.pivot_table(index="date", columns="hm", values="close", aggfunc="last").reindex(columns=SESSION_HMS)
        day_close = pc[DAY_CLOSE_HM].where(pc[DAY_CLOSE_HM].notna(), pc.ffill(axis=1).iloc[:, -1])
        dates = sorted(po.index)
        prev_close = {dates[k]: day_close.get(dates[k-1], np.nan) for k in range(1, len(dates))}
        for i in rows:
            ed = T.at[i, "entry_date"]
            if ed not in po.index:
                continue
            o15[i] = po.at[ed, ENTRY_HM]; h15[i] = ph.at[ed, ENTRY_HM]
            l15[i] = pl.at[ed, ENTRY_HM]; c15[i] = pc.at[ed, ENTRY_HM]
            pcv[i] = prev_close.get(ed, np.nan)
        if si % 100 == 0:
            print(f"  …{si} symbols ({time.time()-t0:.0f}s)")

    T["prev_close"] = pcv; T["uc_level"] = pcv * UC_MULT
    T["o1515"], T["h1515"], T["l1515"], T["c1515"] = o15, h15, l15, c15
    valid = T["uc_level"].notna() & T["c1515"].notna() & (T["prev_close"] > 0)
    V = T[valid].copy(); n = len(V)

    rng = (V["h1515"] - V["l1515"]) / V["c1515"]
    flat = rng <= FLAT_TOL
    pinned = (V["c1515"] >= V["uc_level"] * PIN_TOL) & (V[["o1515", "l1515"]].min(axis=1) >= V["uc_level"] * PIN_TOL)
    V["strict_locked"] = flat & pinned
    V["touched_uc_1515"] = V["h1515"] >= V["uc_level"]

    locked = V[V["strict_locked"]]
    touched = V[V["touched_uc_1515"]]
    nl, nt = len(locked), len(touched)

    tot_gross = float(V["gross_pnl"].sum()); tot_net = float(V["net_pnl"].sum())
    lk_gross = float(locked["gross_pnl"].sum()); lk_net = float(locked["net_pnl"].sum())

    summary = pd.DataFrame([{
        "n_trades": n,
        "n_at_uc_strict_locked": nl, "pct_strict_locked": round(nl / n * 100, 2),
        "n_touched_uc_1515": nt, "pct_touched": round(nt / n * 100, 2),
        "locked_total_gross_pnl_inr": round(lk_gross, 0),
        "locked_total_net_pnl_inr": round(lk_net, 0),
        "locked_avg_gross_ret_pct": round(float(locked["gross_ret"].mean()), 4) if nl else np.nan,
        "locked_avg_net_ret_pct": round(float(locked["net_ret"].mean()), 4) if nl else np.nan,
        "locked_share_of_total_gross_pnl_pct": round(lk_gross / tot_gross * 100, 2) if tot_gross else np.nan,
        "locked_share_of_total_net_pnl_pct": round(lk_net / tot_net * 100, 2) if tot_net else np.nan,
        "overall_avg_gross_ret_pct": round(float(V["gross_ret"].mean()), 4),
        "overall_total_net_pnl_inr": round(tot_net, 0),
    }])

    locked_out = locked[["entry_date", "symbol", "prev_close", "uc_level", "o1515", "h1515",
                         "l1515", "c1515", "gross_ret", "net_ret", "gross_pnl", "net_pnl"]].sort_values("entry_date")
    locked_out.to_csv(OUTDIR / "uc_locked_at_entry_trades.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "entry_uc_locked_analysis.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        locked_out.to_excel(w, sheet_name="locked_trades", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 66 + "\nUC-LOCKED-AT-3:15pm-ENTRY ANALYSIS (main strategy)\n" + "=" * 66)
    print(summary.T.to_string(header=False))
    print(f"\n  Strict-locked (flat 15:15 candle pinned at UC): {nl} of {n} ({nl/n*100:.2f}%)")
    print(f"  Touched UC at 15:15 (high>=uc, looser):         {nt} of {n} ({nt/n*100:.2f}%)")
    print(f"  Fill-realism exposure: locked trades hold ₹{lk_net:,.0f} net pnl = "
          f"{lk_net/tot_net*100:.1f}% of the strategy's ₹{tot_net:,.0f} total net pnl")
    print(f"  Locked-group avg net return {float(locked['net_ret'].mean()):.3f}% vs overall "
          f"{float(V['net_ret'].mean()):.3f}% per trade")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
