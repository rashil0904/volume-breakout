# -*- coding: utf-8 -*-
"""full_deployment_compare.py — CONFIG1 capped baseline (1L/trade) vs CONFIG2 full 5L deployment
(Cat-C 3:21 split uncapped, A/B priority kept). Combined long+short, gross/net_A/net_B. Same signals.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
import baseline_and_cross_final as B

OUTDIR = rb.RESULTS / "full_deployment"
BP, RF = B.BASE_POOL, B.RF


def metrics(T, label):
    dd = T.groupby("exit_date").agg(g=("gross_pnl", "sum"), a=("netA_pnl", "sum"),
                                    b=("netB_pnl", "sum"), c=("capital_deployed", "sum")).reset_index()
    rfd = RF / 252 * 100
    def shp(x):
        return round((x.mean() - rfd) / x.std(ddof=1) * np.sqrt(252), 3) if len(x) > 1 else 0.0
    daycap = T.groupby("entry_date")["capital_deployed"].sum()
    row = {"config": label, "n_trades": len(T),
           "avg_capital_deployed_per_trade": round(T["capital_deployed"].mean(), 0),
           "avg_daily_capital_deployed": round(daycap.mean(), 0),
           "avg_daily_utilization_pct": round(daycap.mean() / BP * 100, 1)}
    for s, tag in [("gross", "gross"), ("netA", "net_A"), ("netB", "net_B")]:
        row[f"tot_ret_{tag}_pct"] = round(T[f"{s}_pnl"].sum() / BP * 100, 2)
        row[f"tot_pnl_{tag}_inr"] = round(T[f"{s}_pnl"].sum(), 0)
        row[f"win_{tag}_pct"] = round((T[f"{s}_pnl"] > 0).mean() * 100, 2)
        row[f"avg_ret_{tag}_pct"] = round(T[f"{s}_ret"].mean(), 4)
        row[f"med_ret_{tag}_pct"] = round(T[f"{s}_ret"].median(), 4)
    row["sharpe_gross"] = shp(dd["g"] / dd["c"] * 100)
    row["sharpe_net_A"] = shp(dd["a"] / dd["c"] * 100)
    row["sharpe_net_B"] = shp(dd["b"] / dd["c"] * 100)
    # risk on daily netA pnl
    dpnl = T.groupby("exit_date")["netA_pnl"].sum().sort_index()
    eq = dpnl.cumsum().values
    row["max_dd_daily_netA_inr"] = round(float((eq - np.maximum.accumulate(eq)).min()), 0)
    row["worst_day_netA_inr"] = round(float(dpnl.min()), 0)
    return row


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache = B.build_cache()
    T1, _ = B.run_config("baseline", cache)                    # capped
    T2, _ = B.run_config("baseline", cache, full_deploy=True)  # full deployment
    for T in (T1, T2):
        T["entry_date"] = pd.to_datetime(T["entry_date"]); T["exit_date"] = pd.to_datetime(T["exit_date"])
    print(f"Config1 (capped) {len(T1)} trades | Config2 (full) {len(T2)} trades")

    comp = pd.DataFrame([metrics(T1, "1_capped_1L"), metrics(T2, "2_full_5L_deploy")])
    delta = {"config": "delta (2-1)"}
    for c in comp.columns:
        if c != "config":
            delta[c] = round(comp.iloc[1][c] - comp.iloc[0][c], 4)
    comp = pd.concat([comp, pd.DataFrame([delta])], ignore_index=True)

    # low vs high signal days
    def sig_analysis(T, label):
        nday = T.groupby("entry_date").size()
        T = T.merge(nday.rename("n_signals_day"), left_on="entry_date", right_index=True)
        rows = []
        for lbl, mask in [("low (1-3 trades)", T["n_signals_day"] <= 3), ("mid (4-8)", (T["n_signals_day"] >= 4) & (T["n_signals_day"] <= 8)),
                          ("high (9+)", T["n_signals_day"] >= 9)]:
            g = T[mask]
            rows.append({"config": label, "day_bucket": lbl, "n_trades": len(g),
                         "netA_pnl_inr": round(g["netA_pnl"].sum(), 0), "avg_netA_ret_pct": round(g["netA_ret"].mean(), 3),
                         "avg_cap_per_trade": round(g["capital_deployed"].mean(), 0)})
        return pd.DataFrame(rows)
    SIG = pd.concat([sig_analysis(T1, "capped"), sig_analysis(T2, "full")], ignore_index=True)

    # single-stock concentration (max one-trade capital / 5L per day) — full deploy
    conc2 = T2.groupby("entry_date")["capital_deployed"].max() / BP * 100
    conc1 = T1.groupby("entry_date")["capital_deployed"].max() / BP * 100
    CONC = pd.DataFrame([
        {"config": "capped", "median_max1stock_pct": round(conc1.median(), 1), "p90": round(conc1.quantile(.9), 1),
         "p99": round(conc1.quantile(.99), 1), "max": round(conc1.max(), 1),
         "n_days_1stock_ge_50pct": int((conc1 >= 50).sum()), "n_days_1stock_eq_100pct": int((conc1 >= 99.9).sum())},
        {"config": "full", "median_max1stock_pct": round(conc2.median(), 1), "p90": round(conc2.quantile(.9), 1),
         "p99": round(conc2.quantile(.99), 1), "max": round(conc2.max(), 1),
         "n_days_1stock_ge_50pct": int((conc2 >= 50).sum()), "n_days_1stock_eq_100pct": int((conc2 >= 99.9).sum())}])

    # per-year (netA total return)
    py = []
    for lbl, T in [("capped", T1), ("full", T2)]:
        T = T.copy(); T["yr"] = T["entry_date"].dt.year
        for y, g in T.groupby("yr"):
            py.append({"config": lbl, "year": y, "n": len(g), "netA_total_ret_pct": round(g["netA_pnl"].sum() / BP * 100, 2),
                       "avg_daily_util_pct": round(g.groupby("entry_date")["capital_deployed"].sum().mean() / BP * 100, 1)})
    PY = pd.DataFrame(py)

    with pd.ExcelWriter(OUTDIR / "full_deployment_compare.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="headline_compare", index=False)
        SIG.to_excel(w, sheet_name="low_vs_high_signal_days", index=False)
        CONC.to_excel(w, sheet_name="concentration_risk", index=False)
        PY.to_excel(w, sheet_name="per_year", index=False)

    pd.set_option("display.width", 260)
    print("\n" + "=" * 110 + "\nFULL 5L DEPLOYMENT vs 1L-CAPPED BASELINE (combined long+short)\n" + "=" * 110)
    show = ["config", "n_trades", "avg_capital_deployed_per_trade", "avg_daily_utilization_pct",
            "tot_ret_gross_pct", "tot_ret_net_A_pct", "tot_ret_net_B_pct", "win_net_A_pct", "avg_ret_net_A_pct",
            "sharpe_net_A", "max_dd_daily_netA_inr", "worst_day_netA_inr"]
    print(comp[show].to_string(index=False))
    print("\n--- LOW vs HIGH signal-day analysis (where full-deploy concentrates) ---")
    print(SIG.to_string(index=False))
    print("\n--- SINGLE-STOCK CONCENTRATION (max 1-trade capital as % of 5L, per day) ---")
    print(CONC.to_string(index=False))
    print("\n--- PER YEAR (net_A total return + daily utilization) ---")
    print(PY.pivot(index="year", columns="config", values=["netA_total_ret_pct", "avg_daily_util_pct"]).to_string())
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
