# -*- coding: utf-8 -*-
"""
uc_c_sweep_by_year.py — robustness check on the Category-C entry-time sweep.
Splits the 15:15..15:25 sweep BY YEAR. If a later entry (e.g. 15:21) is a real edge it
should win (or at least rank high) consistently every year; if it's noise, the best minute
should jump around randomly year to year. Reuses uc_c_entry_time_sweep's cache + trade calc.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import uc_staggered_dd_report as R
import uc_c_entry_time_sweep as SW

OUTDIR = rb.RESULTS / "uc_c_entry_time_sweep"


def main():
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "passes_all_three",
                                "prev_day_vwap_close", "entry_price_315pm"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    Q = diag[diag["passes_all_three"] == True]
    recs = []
    for s in sorted(Q["symbol"].unique()):
        recs += R.scan_symbol(s, Q[Q["symbol"] == s])
    T_def, dd = R.size_and_price(recs)
    perC = {d: v["per_C"] for d, v in dd.items()}
    c = [r for r in recs if r["category"] == "C" and r["entered"] and perC.get(r["entry_date"], 0) > 0]
    cap = SW.capture_c_windows(c)

    # per (year, T): mean net_A per-trade return
    recs_out = []
    for T in SW.GRID:
        for r in c:
            w = cap.get((r["symbol"], r["entry_date"]))
            if w is None:
                continue
            res = SW.c_trade_at(w["opens"].get(T, np.nan), perC[r["entry_date"]], w)
            if res is None:
                continue
            recs_out.append((r["entry_date"].year, SW.hm_lbl(T), res["netA"] / res["cap"] * 100))
    df = pd.DataFrame(recs_out, columns=["year", "T", "netA_ret"])
    piv = df.groupby(["year", "T"])["netA_ret"].mean().unstack("T").round(4)
    piv = piv[[SW.hm_lbl(t) for t in SW.GRID]]                      # keep chronological cols

    # which minute wins each year + where 15:21 ranks
    winners = piv.idxmax(axis=1)
    rank21 = piv.rank(axis=1, ascending=False)["15:21"].astype(int)
    summary = pd.DataFrame({"best_minute": winners, "rank_of_15:21_of_11": rank21})

    piv.to_csv(OUTDIR / "by_year_C_avg_return_netA.csv")
    summary.to_csv(OUTDIR / "by_year_winner.csv")

    pd.set_option("display.width", 240)
    print("=" * 90 + "\nCategory-C avg net_A return per trade (%), by YEAR x entry minute\n" + "=" * 90)
    print(piv.to_string())
    print("\nBest minute each year, and where 15:21 ranks (of 11):")
    print(summary.to_string())
    n_years = len(piv)
    print(f"\n15:21 was the BEST minute in {int((winners == '15:21').sum())} of {n_years} years; "
          f"avg rank of 15:21 = {rank21.mean():.1f} of 11.")
    print(f"Saved -> {OUTDIR}")


if __name__ == "__main__":
    main()
