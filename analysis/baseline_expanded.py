# -*- coding: utf-8 -*-
"""baseline_expanded.py — expand universe to mcap 1,500-7,500 Cr (diagnostic_table_mcap1p5k7p5k.csv);
produce TWO full reports: Config1 (1L cap) and Config2 (full 5L deploy). Plus comparison sheet,
5000-7500 vs 1500-5000 band contribution, and full-deploy risk read. Combined long+short, gross/netA/netB.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
import baseline_and_cross_final as B

DIAG_WIDE = rb.RESULTS / "diagnostic_table_mcap1p5k7p5k.csv"
OUTDIR = rb.RESULTS / "baseline_expanded"
BP, RF = B.BASE_POOL, B.RF


def add_mcap(T, mcap_map):
    T = T.copy()
    T["market_cap_value"] = [mcap_map.get((s, d), np.nan) for s, d in zip(T["symbol"], pd.to_datetime(T["entry_date"]).dt.date)]
    T["mcap_band"] = np.where(T["market_cap_value"] > 5000, "5000-7500", "1500-5000")
    return T


def cmp_metrics(T, label):
    T = T.copy(); T["entry_date"] = pd.to_datetime(T["entry_date"]); T["exit_date"] = pd.to_datetime(T["exit_date"])
    dd = T.groupby("exit_date").agg(g=("gross_pnl", "sum"), a=("netA_pnl", "sum"), b=("netB_pnl", "sum"), c=("capital_deployed", "sum")).reset_index()
    rfd = RF / 252 * 100
    def shp(x):
        return round((x.mean() - rfd) / x.std(ddof=1) * np.sqrt(252), 3) if len(x) > 1 else 0.0
    daycap = T.groupby("entry_date")["capital_deployed"].sum()
    dpnl = T.groupby("exit_date")["netA_pnl"].sum().sort_index(); eq = dpnl.cumsum().values
    row = {"config": label, "n_trades": len(T), "avg_cap_per_trade": round(T["capital_deployed"].mean(), 0),
           "avg_daily_utilization_pct": round(daycap.mean() / BP * 100, 1)}
    for s, tag in [("gross", "gross"), ("netA", "net_A"), ("netB", "net_B")]:
        row[f"tot_ret_{tag}_pct"] = round(T[f"{s}_pnl"].sum() / BP * 100, 2)
        row[f"tot_pnl_{tag}_inr"] = round(T[f"{s}_pnl"].sum(), 0)
        row[f"win_{tag}_pct"] = round((T[f"{s}_pnl"] > 0).mean() * 100, 2)
        row[f"avg_ret_{tag}_pct"] = round(T[f"{s}_ret"].mean(), 4)
    row["sharpe_net_A"] = shp(dd["a"] / dd["c"] * 100)
    row["max_dd_daily_netA_inr"] = round(float((eq - np.maximum.accumulate(eq)).min()), 0)
    row["worst_day_netA_inr"] = round(float(dpnl.min()), 0)
    return row


def band_contrib(T, label):
    rows = []
    for band, g in T.groupby("mcap_band"):
        rows.append({"config": label, "mcap_band": band, "n_trades": len(g),
                     "gross_pnl_inr": round(g["gross_pnl"].sum(), 0), "netA_pnl_inr": round(g["netA_pnl"].sum(), 0),
                     "netA_ret_on_5L_pct": round(g["netA_pnl"].sum() / BP * 100, 2),
                     "avg_netA_ret_pct": round(g["netA_ret"].mean(), 3), "win_rate_pct": round((g["netA_pnl"] > 0).mean() * 100, 2)})
    return rows


def concentration(T, label):
    T = T.copy(); T["entry_date"] = pd.to_datetime(T["entry_date"])
    conc = T.groupby("entry_date")["capital_deployed"].max() / BP * 100
    return {"config": label, "median_max1stock_pct": round(conc.median(), 1), "p90": round(conc.quantile(.9), 1),
            "p99": round(conc.quantile(.99), 1), "max": round(conc.max(), 1),
            "n_days_1stock_ge_50pct": int((conc >= 50).sum()), "n_days_1stock_eq_100pct": int((conc >= 99.9).sum())}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    mc = pd.read_csv(DIAG_WIDE, usecols=["symbol", "date", "market_cap_value"], parse_dates=["date"])
    mc["d"] = mc["date"].dt.date
    mcap_map = {(s, d): v for s, d, v in zip(mc["symbol"], mc["d"], mc["market_cap_value"])}

    cache = B.build_cache(diag_path=DIAG_WIDE)
    T1, d1 = B.run_config("baseline", cache)                    # 1L cap
    T2, d2 = B.run_config("baseline", cache, full_deploy=True)  # full 5L
    T1 = add_mcap(T1, mcap_map); T2 = add_mcap(T2, mcap_map)
    print(f"EXPANDED universe: Config1 {len(T1)} trades | Config2 {len(T2)} trades")
    n_wide = int((T1["mcap_band"] == "5000-7500").sum())
    print(f"  5000-7500 band adds {n_wide} trades ({round(n_wide/len(T1)*100,1)}%) vs {len(T1)-n_wide} in 1500-5000")

    # two full reports
    B.full_report(T1, d1, OUTDIR / "baseline_expanded_1Lcap_performance.xlsx")
    B.full_report(T2, d2, OUTDIR / "baseline_expanded_fulldeploy_performance.xlsx")

    # comparison
    comp = pd.DataFrame([cmp_metrics(T1, "1_1Lcap"), cmp_metrics(T2, "2_fulldeploy")])
    delta = {"config": "delta (2-1)"}
    for c in comp.columns:
        if c != "config":
            delta[c] = round(comp.iloc[1][c] - comp.iloc[0][c], 4)
    comp = pd.concat([comp, pd.DataFrame([delta])], ignore_index=True)
    BAND = pd.DataFrame(band_contrib(T1, "1Lcap") + band_contrib(T2, "fulldeploy"))
    CONC = pd.DataFrame([concentration(T1, "1Lcap"), concentration(T2, "fulldeploy")])

    with pd.ExcelWriter(OUTDIR / "expanded_comparison.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="1Lcap_vs_fulldeploy", index=False)
        BAND.to_excel(w, sheet_name="band_contribution", index=False)
        CONC.to_excel(w, sheet_name="concentration_risk", index=False)

    pd.set_option("display.width", 260)
    print("\n" + "=" * 110 + "\nEXPANDED (mcap 1500-7500) — 1L CAP vs FULL 5L DEPLOY\n" + "=" * 110)
    show = ["config", "n_trades", "avg_cap_per_trade", "avg_daily_utilization_pct", "tot_ret_gross_pct",
            "tot_ret_net_A_pct", "tot_ret_net_B_pct", "win_net_A_pct", "avg_ret_net_A_pct", "sharpe_net_A",
            "max_dd_daily_netA_inr", "worst_day_netA_inr"]
    print(comp[show].to_string(index=False))
    print("\n--- 5000-7500 vs 1500-5000 BAND CONTRIBUTION ---")
    print(BAND.to_string(index=False))
    print("\n--- CONCENTRATION RISK (max 1-stock % of 5L per day) ---")
    print(CONC.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")
    print("  baseline_expanded_1Lcap_performance.xlsx / baseline_expanded_fulldeploy_performance.xlsx / expanded_comparison.xlsx")


if __name__ == "__main__":
    main()
