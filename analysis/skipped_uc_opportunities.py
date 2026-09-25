# -*- coding: utf-8 -*-
"""
skipped_uc_opportunities.py
===========================
Fresh FULL-universe screen for UC opportunities the strategy SKIPPED via the +5% filter:
  mcap ₹1,500-5,000 Cr (diagnostic_table = in-band days) + passes_volume (VM6/LB36)
  + hit UC on entry day + return_at_1515 < +5% (strategy's VWAP prev-close reference)
  + NOT in the traded set (not passes_all_three)  -> disjoint from the 3,494 trades.

uc_level = plain prev-close × 1.1995 (exchange circuit; matches prior UC analyses -> reconciles
the 254 traded UC-hitters). return_at_1515 = (entry_315 - vwap_prev)/vwap_prev × 100.
UC-timing split: first UC hit BEFORE 3:15 (faded, correctly skipped) vs AT/AFTER 3:15
(the 15:15 candle -> late run you missed).
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "entry_day_uc"
UC_MULT = 1.1995
ENTRY_HM, DAY_CLOSE_HM = 915, 915
SESSION_HMS = list(range(555, 916, 15))
PRE = [h for h in SESSION_HMS if h < ENTRY_HM]                 # before 15:15


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "passes_volume", "passes_all_three",
                                "prev_day_vwap_close", "entry_price_315pm", "next_day_930_open"],
                       parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    DV = diag[diag["passes_volume"] == True].copy().reset_index(drop=True)
    n_univ = len(DV)
    print(f"Full-universe mcap+volume screen (passes_volume): {n_univ:,} stock-days")

    n = n_univ
    plain_pc = np.full(n, np.nan); hit = np.zeros(n, bool); fhm = np.full(n, np.nan)
    peak_pre = np.full(n, np.nan); peak_post = np.full(n, np.nan)
    idx_by_sym = {s: g.index.values for s, g in DV.groupby("symbol")}
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
        ph = raw.pivot_table(index="date", columns="hm", values="high", aggfunc="max").reindex(columns=SESSION_HMS)
        pc = raw.pivot_table(index="date", columns="hm", values="close", aggfunc="last").reindex(columns=SESSION_HMS)
        day_close = pc[DAY_CLOSE_HM].where(pc[DAY_CLOSE_HM].notna(), pc.ffill(axis=1).iloc[:, -1])
        dates = sorted(ph.index)
        prev_close = {dates[k]: day_close.get(dates[k-1], np.nan) for k in range(1, len(dates))}
        colpre = [SESSION_HMS.index(h) for h in PRE]
        for i in rows:
            ed = DV.at[i, "date"]
            pcl = prev_close.get(ed, np.nan)
            if not (pcl == pcl and pcl > 0) or ed not in ph.index:
                continue
            plain_pc[i] = pcl; uc = pcl * UC_MULT
            hv = ph.loc[ed].values
            hm_mask = hv >= uc
            if hm_mask.any():
                hit[i] = True; fhm[i] = SESSION_HMS[int(np.argmax(hm_mask))]
            pre_hi = np.nanmax(hv[colpre]) if len(colpre) else np.nan
            peak_pre[i] = (pre_hi - pcl) / pcl * 100 if pre_hi == pre_hi else np.nan
            peak_post[i] = (hv[SESSION_HMS.index(ENTRY_HM)] - pcl) / pcl * 100
        if si % 100 == 0:
            print(f"  …{si} symbols ({time.time()-t0:.0f}s, {int(hit.sum())} UC hits so far)")

    DV["plain_prev_close"] = plain_pc; DV["uc_price"] = plain_pc * UC_MULT
    DV["hit_uc"] = hit; DV["uc_first_hit_hm"] = fhm
    DV["peak_return_pre_1515"] = np.round(peak_pre, 3); DV["peak_return_post_1515"] = np.round(peak_post, 3)
    DV["return_at_1515"] = (DV["entry_price_315pm"] - DV["prev_day_vwap_close"]) / DV["prev_day_vwap_close"] * 100
    DV["pct_uc_to_1515entry"] = (DV["uc_price"] - DV["entry_price_315pm"]) / DV["uc_price"] * 100

    # screen: hit UC, < +5% at 3:15, and NOT traded (disjoint from the 3,494)
    S = DV[DV["hit_uc"] & (DV["return_at_1515"] < 5) & (DV["passes_all_three"] != True)
           & DV["plain_prev_close"].notna()].copy()
    ns = len(S)
    S["uc_hit_after_315"] = S["uc_first_hit_hm"] >= ENTRY_HM      # first hit at the 15:15 candle
    S["uc_first_hit_time"] = S["uc_first_hit_hm"].map(lambda h: hm_lbl(h) if h == h else "")
    after = S[S["uc_hit_after_315"]]; before = S[~S["uc_hit_after_315"]]

    # hypothetical: entered 15:15, exit next-day 9:30 open (rough BTST proxy, flagged)
    S["hypo_next930_ret_pct"] = (S["next_day_930_open"] - S["entry_price_315pm"]) / S["entry_price_315pm"] * 100
    after_h = S[S["uc_hit_after_315"]]["hypo_next930_ret_pct"].dropna()

    N_TRADED_UC = 254
    summary = pd.DataFrame([{
        "n_skipped_uc_opportunities": ns,
        "pct_of_full_volume_universe": round(ns / n_univ * 100, 3),
        "ratio_vs_254_traded_uc_hitters": round(ns / N_TRADED_UC, 2),
        "n_hit_uc_AFTER_315_genuine_miss": len(after),
        "n_hit_uc_BEFORE_315_faded_correctly_skipped": len(before),
        "avg_pct_uc_to_1515entry_all": round(float(S["pct_uc_to_1515entry"].mean()), 3),
        "median_pct_uc_to_1515entry_all": round(float(S["pct_uc_to_1515entry"].median()), 3),
        "avg_return_at_1515_all": round(float(S["return_at_1515"].mean()), 3),
        "median_return_at_1515_all": round(float(S["return_at_1515"].median()), 3),
        "AFTER315_avg_pct_uc_to_1515entry": round(float(after["pct_uc_to_1515entry"].mean()), 3) if len(after) else np.nan,
        "AFTER315_avg_return_at_1515": round(float(after["return_at_1515"].mean()), 3) if len(after) else np.nan,
        "AFTER315_hypo_next930_avg_ret_pct": round(float(after_h.mean()), 3) if len(after_h) else np.nan,
        "AFTER315_hypo_next930_win_rate": round(float((after_h > 0).mean() * 100), 1) if len(after_h) else np.nan,
    }])

    detail = S[["symbol", "date", "plain_prev_close", "prev_day_vwap_close", "entry_price_315pm",
                "return_at_1515", "uc_price", "uc_first_hit_time", "uc_hit_after_315",
                "pct_uc_to_1515entry", "peak_return_pre_1515", "peak_return_post_1515",
                "hypo_next930_ret_pct"]].sort_values(["uc_hit_after_315", "pct_uc_to_1515entry"], ascending=[False, False])
    detail.to_csv(OUTDIR / "skipped_uc_opportunities_detail.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "skipped_uc_opportunities.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        detail.to_excel(w, sheet_name="skipped_detail", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 74 + "\nSKIPPED UC OPPORTUNITIES (mcap+volume+hitUC+ <5% at 3:15, not traded)\n" + "=" * 74)
    print(summary.T.to_string(header=False))
    ud = after["uc_first_hit_time"].value_counts()
    print(f"\n  UC-timing split: {len(after)} hit UC AT/AFTER 3:15 (genuine late miss) | "
          f"{len(before)} hit UC BEFORE 3:15 then faded <5% (correctly skipped)")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
