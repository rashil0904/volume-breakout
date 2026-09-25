# -*- coding: utf-8 -*-
"""
entry_uc_below5_at_entry.py
===========================
UC-hitters that were still BELOW +5% at the 3:15pm entry (entered before the run to UC).
Reuses build_trades() + diagnostic_table (NOT recomputed).

RECONCILIATION: the strategy's +5% filter uses the 15:00 (3pm) candle OPEN vs the VWAP-based
prev-close (prepare_data). But entry is the 15:15 (3:15pm) open. So a qualifying trade can be
< +5% at 3:15 vs the plain prev-close. We report return_at_entry under BOTH references.

uc_level = plain_prev_close * 1.1995. hit_uc = some entry-day candle high >= uc_level.
pct_uc_to_entry = (uc - entry_315)/uc*100.
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


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Building trades (reused) + merging diagnostic reference …")
    T = fpr.build_trades()[["symbol", "entry_date", "entry_price", "gross_ret", "gross_pnl",
                            "capital_deployed"]].copy().reset_index(drop=True)
    T["entry_date"] = pd.to_datetime(T["entry_date"]).dt.date
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "prev_day_vwap_close", "return_pct_vs_prev_close",
                                "entry_price_3pm"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    T = T.merge(diag, left_on=["symbol", "entry_date"], right_on=["symbol", "date"], how="left")
    n_all = len(T)

    plain_pc = np.full(n_all, np.nan); hit = np.zeros(n_all, bool); uc_hit_hm = np.full(n_all, np.nan)
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
            plain_pc[i] = pcl; uc = pcl * UC_MULT
            hm_mask = ph.loc[ed].values >= uc
            if hm_mask.any():
                hit[i] = True; uc_hit_hm[i] = SESSION_HMS[int(np.argmax(hm_mask))]
        if si % 100 == 0:
            print(f"  …{si} symbols ({time.time()-t0:.0f}s)")

    T["plain_prev_close"] = plain_pc; T["uc_level"] = plain_pc * UC_MULT
    T["hit_uc"] = hit; T["uc_first_hit_hm"] = uc_hit_hm
    T["ret315_vs_plain"] = (T["entry_price"] - T["plain_prev_close"]) / T["plain_prev_close"] * 100
    T["ret315_vs_vwap"] = (T["entry_price"] - T["prev_day_vwap_close"]) / T["prev_day_vwap_close"] * 100

    hitters = T[T["hit_uc"] & T["plain_prev_close"].notna()]
    nh = len(hitters)
    # discrepancy: among UC-hitters, how many < +5% under each reference
    below5_plain = hitters[hitters["ret315_vs_plain"] < 5]
    below5_vwap = hitters[hitters["ret315_vs_vwap"] < 5]

    # subset per definition: hit UC AND < +5% vs plain prev close
    S = below5_plain.copy()
    S["pct_uc_to_entry"] = (S["uc_level"] - S["entry_price"]) / S["uc_level"] * 100
    S["uc_first_hit_time"] = S["uc_first_hit_hm"].map(lambda h: hm_lbl(h) if h == h else "")
    S["uc_hit_after_315"] = S["uc_first_hit_hm"] >= 915
    ns = len(S)

    print("\n" + "=" * 74)
    print("UC-HITTERS BELOW +5% AT 3:15 ENTRY (entered before the run)")
    print("=" * 74)
    print(f"  strategy filter: +5% at 15:00 open vs VWAP prev-close; ENTRY at 15:15 open.")
    print(f"  UC-hitters: {nh} | < +5% at 3:15 vs PLAIN prev-close: {len(below5_plain)} "
          f"| < +5% vs VWAP prev-close: {len(below5_vwap)}")
    if ns == 0:
        print("  -> no UC-hitters were < +5% at 3:15 vs plain prev-close.")
        return
    ret_overall = T["gross_ret"].mean()
    summary = pd.DataFrame([{
        "n_subset": ns, "pct_of_all_trades": round(ns / n_all * 100, 2),
        "pct_of_uc_hitters": round(ns / nh * 100, 2),
        "avg_pct_uc_to_entry": round(float(S["pct_uc_to_entry"].mean()), 4),
        "median_pct_uc_to_entry": round(float(S["pct_uc_to_entry"].median()), 4),
        "avg_return_at_entry_plain": round(float(S["ret315_vs_plain"].mean()), 4),
        "median_return_at_entry_plain": round(float(S["ret315_vs_plain"].median()), 4),
        "n_uc_hit_after_315": int(S["uc_hit_after_315"].sum()),
        "n_uc_hit_before_315": int((~S["uc_hit_after_315"]).sum()),
        "subset_avg_trade_return_pct": round(float(S["gross_ret"].mean()), 4),
        "subset_median_trade_return_pct": round(float(S["gross_ret"].median()), 4),
        "subset_win_rate_pct": round(float((S["gross_pnl"] > 0).mean() * 100), 2),
        "overall_avg_trade_return_pct": round(float(ret_overall), 4),
    }])

    S["actual_trade_return_pct"] = S["gross_ret"]
    detail = S[["symbol", "entry_date", "plain_prev_close", "prev_day_vwap_close", "entry_price_3pm",
                "entry_price", "ret315_vs_plain", "ret315_vs_vwap", "return_pct_vs_prev_close",
                "uc_level", "uc_first_hit_time", "uc_hit_after_315", "pct_uc_to_entry",
                "actual_trade_return_pct"]].rename(columns={"entry_price": "entry_price_315pm"}).sort_values("pct_uc_to_entry", ascending=False)
    detail.to_csv(OUTDIR / "uc_below5_at_entry_detail.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "entry_uc_below5_at_entry.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        detail.to_excel(w, sheet_name="subset_detail", index=False)

    ud = S["uc_first_hit_time"].value_counts().reindex([hm_lbl(h) for h in SESSION_HMS]).fillna(0).astype(int)
    ud = ud[ud > 0]
    pd.set_option("display.width", 200)
    print(summary.T.to_string(header=False))
    print(f"\n  UC first-hit time distribution (subset):")
    print(ud.to_string())
    print(f"\n  subset avg trade return {float(S['gross_ret'].mean()):.3f}% vs overall {ret_overall:.3f}% "
          f"(win rate {float((S['gross_pnl']>0).mean()*100):.1f}%)")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
