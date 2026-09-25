# -*- coding: utf-8 -*-
"""extended_sma_test.py — test 'entry >= 50% above 36-day SMA = net loser'.
TABLE 1: full baseline vs baseline EXCLUDING extended (>=50%) trades (gross/net_A/net_B + delta).
TABLE 2: standalone performance + per-trade detail of the extended trades.
(a) sma36 = 36d SMA of daily close thru prev day (no look-ahead); (b) threshold >= 50 inclusive;
(c) entry_price = avg_entry (Cat-A multi-leg = blended avg); (d) combined long+short pnl.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

BASE_XLSX = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
FEAT = rb.RESULTS / "pattern_mining" / "trade_features.parquet"
OUTDIR = rb.RESULTS / "extended_sma_test"
BP = 500_000
RF = 0.075


def metrics(T, label):
    dd = T.groupby("exit_date").agg(g=("gross_pnl", "sum"), a=("netA_pnl", "sum"),
                                    b=("netB_pnl", "sum"), c=("capital_deployed", "sum")).reset_index()
    rfd = RF / 252 * 100
    def shp(x):
        return round((x.mean() - rfd) / x.std(ddof=1) * np.sqrt(252), 3) if len(x) > 1 else 0.0
    row = {"set": label, "n_trades": len(T)}
    for s, tag in [("gross", "gross"), ("netA", "net_A"), ("netB", "net_B")]:
        row[f"tot_ret_{tag}_pct"] = round(T[f"{s}_pnl"].sum() / BP * 100, 2)
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
    T["entry_date"] = pd.to_datetime(T["entry_date"]); T["exit_date"] = pd.to_datetime(T["exit_date"])
    F = pd.read_parquet(FEAT, columns=["symbol", "entry_date", "sma36", "dist_sma36_pct", "avg_entry"])
    F["entry_date"] = pd.to_datetime(F["entry_date"])
    T = T.merge(F[["symbol", "entry_date", "sma36", "dist_sma36_pct"]], on=["symbol", "entry_date"], how="left")

    ext = T["dist_sma36_pct"] >= 50.0                       # (b) inclusive
    E = T[ext].copy(); K = T[~ext].copy()
    print(f"total trades {len(T):,} | extended (>=50% above SMA) {len(E)} | surviving {len(K):,} | "
          f"n missing sma {int(T['dist_sma36_pct'].isna().sum())}")

    # ── TABLE 1 ──
    tbl1 = pd.DataFrame([metrics(T, "FULL baseline"), metrics(K, "EXCLUDING extended (<50%)")])
    delta = {"set": "delta (excl - full)"}
    for c in tbl1.columns:
        if c == "set":
            continue
        delta[c] = round(tbl1.iloc[1][c] - tbl1.iloc[0][c], 4)
    tbl1 = pd.concat([tbl1, pd.DataFrame([delta])], ignore_index=True)

    # ── TABLE 2: standalone + per-trade ──
    win = E["gross_pnl"] > 0
    t2 = {"n_trades": len(E),
          "total_pnl_gross_inr": round(E["gross_pnl"].sum(), 0), "total_pnl_netA_inr": round(E["netA_pnl"].sum(), 0),
          "total_pnl_netB_inr": round(E["netB_pnl"].sum(), 0),
          "tot_ret_gross_pct": round(E["gross_pnl"].sum() / BP * 100, 3), "tot_ret_netA_pct": round(E["netA_pnl"].sum() / BP * 100, 3),
          "tot_ret_netB_pct": round(E["netB_pnl"].sum() / BP * 100, 3),
          "win_rate_pct": round(win.mean() * 100, 2), "n_win": int(win.sum()), "n_loss": int((~win).sum()),
          "avg_ret_pct": round(E["gross_ret"].mean(), 3), "median_ret_pct": round(E["gross_ret"].median(), 3),
          "avg_ret_winners_pct": round(E.loc[win, "gross_ret"].mean(), 3) if win.any() else np.nan,
          "avg_ret_losers_pct": round(E.loc[~win, "gross_ret"].mean(), 3) if (~win).any() else np.nan,
          "n_catA": int((E["category"] == "A").sum()), "n_catB": int((E["category"] == "B").sum()),
          "n_catC": int((E["category"] == "C").sum()),
          "pct_of_baseline_gross_pnl": round(E["gross_pnl"].sum() / T["gross_pnl"].sum() * 100, 3)}
    TBL2 = pd.DataFrame([t2])
    detail = E[["symbol", "entry_date", "avg_entry", "sma36", "dist_sma36_pct", "category",
                "gross_pnl", "gross_ret", "netA_pnl", "long_pnl", "short_pnl"]].copy()
    detail["entry_date"] = detail["entry_date"].dt.strftime("%Y-%m-%d")
    detail = detail.rename(columns={"avg_entry": "entry_price", "gross_pnl": "combined_pnl", "gross_ret": "return_pct"})
    detail = detail.sort_values("dist_sma36_pct", ascending=False).round(3)

    with pd.ExcelWriter(OUTDIR / "extended_sma_test.xlsx", engine="openpyxl") as w:
        tbl1.to_excel(w, sheet_name="TABLE1_excl_vs_full", index=False)
        TBL2.to_excel(w, sheet_name="TABLE2_extended_standalone", index=False)
        detail.to_excel(w, sheet_name="extended_per_trade", index=False)

    pd.set_option("display.width", 250)
    print("\n" + "=" * 100 + "\nTABLE 1 — FULL baseline vs EXCLUDING extended (>=50% above 36-SMA)\n" + "=" * 100)
    show1 = ["set", "n_trades", "tot_ret_gross_pct", "tot_ret_net_A_pct", "tot_ret_net_B_pct",
             "win_net_A_pct", "avg_ret_net_A_pct", "med_ret_net_A_pct", "sharpe_gross", "sharpe_net_A", "sharpe_net_B"]
    print(tbl1[show1].to_string(index=False))
    print("\n" + "=" * 100 + "\nTABLE 2 — STANDALONE: the extended (>=50%-above-SMA) trades\n" + "=" * 100)
    for k, v in t2.items():
        print(f"  {k:28s}: {v}")
    print("\n--- PER-TRADE DETAIL (all extended trades) ---")
    print(detail.to_string(index=False))

    tot = E["gross_pnl"].sum()
    print("\n" + "=" * 100 + "\nVERDICT\n" + "=" * 100)
    print(f"  extended trades: {len(E)} | aggregate combined pnl: Rs{tot:,.0f}  "
          f"({'NET NEGATIVE - confirmed losers' if tot < 0 else 'NET POSITIVE - NOT losers'})")
    print(f"  they are {t2['pct_of_baseline_gross_pnl']:.2f}% of baseline gross pnl (a drag if negative)")
    print(f"  total return gross: {tbl1.iloc[0]['tot_ret_gross_pct']:.2f}% -> {tbl1.iloc[1]['tot_ret_gross_pct']:.2f}% "
          f"(delta {tbl1.iloc[2]['tot_ret_gross_pct']:+.2f} pp) when excluded")
    print(f"  net_A: {tbl1.iloc[0]['tot_ret_net_A_pct']:.2f}% -> {tbl1.iloc[1]['tot_ret_net_A_pct']:.2f}% "
          f"(delta {tbl1.iloc[2]['tot_ret_net_A_pct']:+.2f} pp)")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
