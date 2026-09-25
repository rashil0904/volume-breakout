# -*- coding: utf-8 -*-
"""sma36_filter_compare.py — add a 36-day daily-SMA trend filter to the baseline and compare.
NEW filter: trade's avg entry price > sma_36, where sma_36 = SMA of DAILY CLOSE over the prior 36
trading days THROUGH THE PREVIOUS DAY (no look-ahead). Everything else about the baseline unchanged.

Flags: (a) sma_36 = 36-day SMA of PLAIN daily close (last-traded ~15:29) through prev day — the project's
VWAP-close exists only for signal days, not a full daily series, so plain close is used (standard for a
trend SMA); (b) condition tests the trade's AVG entry price > sma_36 (Cat-A multi-leg tested on avg, not
per-leg); (c) needs >=36 prior daily closes else the trade is excluded and flagged separately;
(d) both configs identical everything-else; combined pnl gross/net_A/net_B.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
import baseline_and_cross_final as B

BASE_XLSX = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
OUTDIR = rb.RESULTS / "sma36_filter"
BASE_POOL = B.BASE_POOL


def headline(T, label):
    dd = T.groupby("exit_date").agg(g=("gross_pnl", "sum"), a=("netA_pnl", "sum"),
                                    b=("netB_pnl", "sum"), c=("capital_deployed", "sum")).reset_index()
    rfd = B.RF / 252 * 100
    def shp(x):
        return round((x.mean() - rfd) / x.std(ddof=1) * np.sqrt(252), 3) if len(x) > 1 else 0.0
    row = {"config": label, "n_trades": len(T)}
    for s, tag in [("gross", "gross"), ("netA", "net_A"), ("netB", "net_B")]:
        row[f"tot_ret_{tag}_pct"] = round(T[f"{s}_pnl"].sum() / BASE_POOL * 100, 2)
        row[f"tot_pnl_{tag}_inr"] = round(T[f"{s}_pnl"].sum(), 0)
        row[f"win_{tag}_pct"] = round((T[f"{s}_pnl"] > 0).mean() * 100, 2)
        row[f"avg_ret_{tag}_pct"] = round(T[f"{s}_ret"].mean(), 4)
        row[f"med_ret_{tag}_pct"] = round(T[f"{s}_ret"].median(), 4)
    row["sharpe_gross"] = shp(dd["g"] / dd["c"] * 100)
    row["sharpe_net_A"] = shp(dd["a"] / dd["c"] * 100)
    row["sharpe_net_B"] = shp(dd["b"] / dd["c"] * 100)
    return row


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = pd.read_excel(BASE_XLSX, sheet_name="all_trades")
    T["entry_date"] = pd.to_datetime(T["entry_date"])
    T["exit_date"] = pd.to_datetime(T["exit_date"])

    # ── 36-day SMA of daily close, THROUGH previous day (rolling(36).shift(1)) ──
    d = pd.read_parquet(DAILY, columns=["symbol", "date", "close"])
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["symbol", "date"])
    d["sma36_prevday"] = d.groupby("symbol")["close"].transform(lambda s: s.rolling(36).mean().shift(1))
    smap = d.set_index(["symbol", "date"])["sma36_prevday"].to_dict()
    T["sma36"] = [smap.get((s, dt), np.nan) for s, dt in zip(T["symbol"], T["entry_date"])]

    no_sma = T["sma36"].isna()
    above = (~no_sma) & (T["avg_entry"] > T["sma36"])
    below = (~no_sma) & (T["avg_entry"] <= T["sma36"])
    Tf = T[above].reset_index(drop=True)          # CONFIG 2 survivors
    Rem = T[below].reset_index(drop=True)         # removed by the > SMA condition
    print(f"baseline {len(T):,} | above-SMA(kept) {int(above.sum()):,} | below-SMA(removed) {int(below.sum()):,} | "
          f"no-SMA(<36d history, excluded) {int(no_sma.sum()):,}")

    # ── 1. headline comparison ──
    CMP = pd.DataFrame([headline(T, "baseline"), headline(Tf, "sma36_filtered")])
    delta = {"config": "delta(filtered-baseline)"}
    for k in CMP.columns:
        if k == "config":
            continue
        delta[k] = round(CMP.iloc[1][k] - CMP.iloc[0][k], 4)
    CMP = pd.concat([CMP, pd.DataFrame([delta])], ignore_index=True)

    # ── 2. removed-trades standalone contribution + category split ──
    def contrib(df, lbl):
        return {"set": lbl, "n_trades": len(df),
                "gross_pnl_inr": round(df["gross_pnl"].sum(), 0), "netA_pnl_inr": round(df["netA_pnl"].sum(), 0),
                "gross_ret_on_5L_pct": round(df["gross_pnl"].sum() / BASE_POOL * 100, 2),
                "netA_ret_on_5L_pct": round(df["netA_pnl"].sum() / BASE_POOL * 100, 2),
                "avg_netA_ret_per_trade_pct": round(df["netA_ret"].mean(), 4) if len(df) else 0.0,
                "median_netA_ret_pct": round(df["netA_ret"].median(), 4) if len(df) else 0.0,
                "win_rate_netA_pct": round((df["netA_pnl"] > 0).mean() * 100, 2) if len(df) else 0.0}
    QUAL = pd.DataFrame([contrib(Tf, "SURVIVING (above SMA)"), contrib(Rem, "REMOVED (below SMA)"),
                         contrib(T[no_sma], "EXCLUDED (<36d history)")])

    # ── 3. category split of the filter effect ──
    cats = []
    for c in ["A", "B", "C"]:
        nb = int((T["category"] == c).sum()); nr = int((Rem["category"] == c).sum())
        cats.append({"category": c, "baseline_n": nb, "removed_n": nr,
                     "kept_n": int((Tf["category"] == c).sum()), "pct_removed": round(nr / nb * 100, 1) if nb else 0.0})
    CATS = pd.DataFrame(cats)

    with pd.ExcelWriter(OUTDIR / "sma36_filter_comparison.xlsx", engine="openpyxl") as w:
        CMP.to_excel(w, sheet_name="headline_compare", index=False)
        QUAL.to_excel(w, sheet_name="surviving_vs_removed", index=False)
        CATS.to_excel(w, sheet_name="category_split", index=False)

    # ── 3(opt). full filtered report if favorable (net_A avg ret per trade improves) ──
    fav = CMP.iloc[1]["avg_ret_net_A_pct"] > CMP.iloc[0]["avg_ret_net_A_pct"]
    if fav:
        diag = {}
        for dd0, g in Tf.groupby("entry_date"):
            ab = g[g["category"].isin(["A", "B"])]; cc = g[g["category"] == "C"]; a2 = g[(g["category"] == "A") & (g["n_legs"] == 2)]
            diag[dd0] = {"X": float(ab["capital_deployed"].sum()), "per_c": float(cc["capital_deployed"].mean()) if len(cc) else 0.0,
                         "nC": int(len(cc)), "p2": float(len(a2) * 0.5 * B.BASE_ALLOC)}
        B.full_report(Tf.drop(columns=["sma36"]), diag, OUTDIR / "baseline_sma36_filtered_report.xlsx")

    pd.set_option("display.width", 250)
    print("\n" + "=" * 100)
    print("36-DAY SMA TREND FILTER (entry price > 36d SMA) vs BASELINE  [combined long+short]")
    print("=" * 100)
    show = ["config", "n_trades", "tot_ret_gross_pct", "tot_ret_net_A_pct", "tot_ret_net_B_pct",
            "win_net_A_pct", "avg_ret_net_A_pct", "med_ret_net_A_pct", "sharpe_gross", "sharpe_net_A", "sharpe_net_B"]
    print(CMP[show].to_string(index=False))
    print("\n--- SURVIVING vs REMOVED vs EXCLUDED (trade quality) ---")
    print(QUAL.to_string(index=False))
    print("\n--- CATEGORY SPLIT of the filter effect ---")
    print(CATS.to_string(index=False))
    print(f"\n  full filtered report written: {fav}  (net_A avg return per trade "
          f"{'improved' if fav else 'did NOT improve'} vs baseline)")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
