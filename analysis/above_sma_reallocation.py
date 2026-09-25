# -*- coding: utf-8 -*-
"""above_sma_reallocation.py — 'keep entry > 36-day SMA' (remove below-SMA trades) with capital
reallocation on affected days, vs full baseline (1) and passive exclusion (2). Reuses the engine
run_config(..., exclude=set). Validates whether the removed below-SMA trades are net-negative
(they are expected to be net POSITIVE -> the filter cuts profit) in-sample and out-of-sample.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
import baseline_and_cross_final as B

FEAT = rb.RESULTS / "pattern_mining" / "trade_features.parquet"
LIQ = rb.RESULTS / "liquidity_proxies" / "per_trade_liquidity.csv"
OUTDIR = rb.RESULTS / "above_sma_reallocation"
BP = B.BASE_POOL
RF = B.RF
IS_YEARS = {2022, 2023, 2024}


def metrics(T, label):
    dd = T.groupby("exit_date").agg(g=("gross_pnl", "sum"), a=("netA_pnl", "sum"),
                                    b=("netB_pnl", "sum"), c=("capital_deployed", "sum")).reset_index()
    rfd = RF / 252 * 100
    def shp(x):
        return round((x.mean() - rfd) / x.std(ddof=1) * np.sqrt(252), 3) if len(x) > 1 else 0.0
    row = {"config": label, "n_trades": len(T), "avg_capital_deployed": round(T["capital_deployed"].mean(), 0)}
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
    F = pd.read_parquet(FEAT, columns=["symbol", "entry_date", "sma36", "dist_sma36_pct", "avg_entry", "yr"])
    below = F[F["dist_sma36_pct"] <= 0].copy()            # entry <= sma36
    below["ed"] = pd.to_datetime(below["entry_date"]).dt.date
    exclude = set(zip(below["symbol"], below["ed"]))
    print(f"below-SMA trades to remove (entry<=sma36): {len(exclude)}")

    cache = B.build_cache()
    T1, _ = B.run_config("baseline", cache)
    T3, _ = B.run_config("baseline", cache, exclude=exclude)
    for T in (T1, T3):
        T["entry_date"] = pd.to_datetime(T["entry_date"]); T["exit_date"] = pd.to_datetime(T["exit_date"])
        T["ed"] = T["entry_date"].dt.date
    key1 = list(zip(T1["symbol"], T1["ed"]))
    T2 = T1[[k not in exclude for k in key1]].copy()
    removed = T1[[k in exclude for k in key1]].copy()
    print(f"Config1 {len(T1)} | Config2 {len(T2)} | Config3 {len(T3)} | removed {len(removed)}")

    # ── three-way ──
    comp = pd.DataFrame([metrics(T1, "1_full_baseline"), metrics(T2, "2_passive_exclude"),
                         metrics(T3, "3_exclude_reallocate")])
    for lbl, i, j in [("delta 2-1 (removal)", 1, 0), ("delta 3-2 (reallocation)", 2, 1), ("delta 3-1 (total)", 2, 0)]:
        d = {"config": lbl}
        for c in comp.columns:
            if c != "config":
                d[c] = round(comp.iloc[i][c] - comp.iloc[j][c], 4)
        comp = pd.concat([comp, pd.DataFrame([d])], ignore_index=True)

    # ── validation: removed set net-negative? IS vs OOS ──
    removed["yr"] = removed["entry_date"].dt.year
    def agg(df):
        return {"n": len(df), "gross_pnl": round(df["gross_pnl"].sum(), 0), "netA_pnl": round(df["netA_pnl"].sum(), 0),
                "avg_gross_ret": round(df["gross_ret"].mean(), 3), "win_rate": round((df["gross_pnl"] > 0).mean() * 100, 1)}
    VAL = pd.DataFrame([{"period": "ALL", **agg(removed)},
                        {"period": "IS 2022-24", **agg(removed[removed["yr"].isin(IS_YEARS)])},
                        {"period": "OOS 2025+", **agg(removed[~removed["yr"].isin(IS_YEARS)])}])

    # ── reallocation diagnostics ──
    aff_days = set(below["ed"])
    c1a = T1[T1["ed"].isin(aff_days)].set_index(["symbol", "ed"])
    c3a = T3.set_index(["symbol", "ed"])
    rows = []
    for k in c3a.index:
        if k in c1a.index and k[1] in aff_days:
            cap1 = float(np.ravel(c1a.loc[k, "capital_deployed"])[0]); cap3 = float(np.ravel(c3a.loc[k, "capital_deployed"])[0])
            rows.append({"symbol": k[0], "ed": k[1], "cap_inc": cap3 - cap1})
    RB = pd.DataFrame(rows); topped = RB[RB["cap_inc"] > 1.0]
    day_rows = []
    for d, g in below.groupby("ed"):
        freed = float(T1[(T1["ed"] == d) & (T1["symbol"].isin(g["symbol"]))]["capital_deployed"].sum())
        redep = float(RB[RB["ed"] == d]["cap_inc"].sum()) if len(RB) else 0.0
        tup = RB[(RB["ed"] == d) & (RB["cap_inc"] > 1.0)]
        day_rows.append({"date": str(d), "n_removed": len(g), "capital_freed": round(freed, 0),
                         "capital_redeployed": round(redep, 0), "capital_idle": round(freed - redep, 0),
                         "n_topped_up": len(tup), "avg_topup_inr": round(tup["cap_inc"].mean(), 0) if len(tup) else 0.0,
                         "day_pnl_C1": round(float(T1[T1["ed"] == d]["gross_pnl"].sum()), 0),
                         "day_pnl_C3": round(float(T3[T3["ed"] == d]["gross_pnl"].sum()), 0)})
    DAYS = pd.DataFrame(day_rows).sort_values("date")
    freed_tot, redep_tot = DAYS["capital_freed"].sum(), DAYS["capital_redeployed"].sum()
    idle_tot = freed_tot - redep_tot

    lq = pd.read_csv(LIQ)[["symbol", "entry_date", "shares_pct_of_volume"]]
    lq["ed"] = pd.to_datetime(lq["entry_date"]).dt.date
    lmap = lq.set_index(["symbol", "ed"])["shares_pct_of_volume"].to_dict()
    rem_liq = np.nanmean([lmap.get((s, d), np.nan) for s, d in zip(below["symbol"], below["ed"])])
    top_liq = np.nanmean([lmap.get((s, d), np.nan) for s, d in zip(topped["symbol"], topped["ed"])]) if len(topped) else np.nan

    with pd.ExcelWriter(OUTDIR / "above_sma_reallocation.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="three_way_compare", index=False)
        VAL.to_excel(w, sheet_name="removed_set_validation", index=False)
        DAYS.to_excel(w, sheet_name="per_affected_day", index=False)

    pd.set_option("display.width", 250)
    print("\n" + "=" * 110 + "\nABOVE-36SMA FILTER: baseline (1) vs passive exclude (2) vs exclude+reallocate (3)\n" + "=" * 110)
    show = ["config", "n_trades", "avg_capital_deployed", "tot_ret_gross_pct", "tot_ret_net_A_pct",
            "tot_ret_net_B_pct", "win_net_A_pct", "avg_ret_net_A_pct", "sharpe_gross", "sharpe_net_A", "sharpe_net_B"]
    print(comp[show].to_string(index=False))
    print("\n--- REMOVED-SET VALIDATION (below-SMA trades: net negative?) ---")
    print(VAL.to_string(index=False))
    print("\n--- REALLOCATION DIAGNOSTICS ---")
    print(f"  affected days {len(DAYS)} | freed Rs{freed_tot:,.0f} | redeployed Rs{redep_tot:,.0f} | "
          f"idle Rs{idle_tot:,.0f} ({idle_tot/freed_tot*100:.0f}% idle)")
    print(f"  trades topped up {len(topped)} | avg topup Rs{topped['cap_inc'].mean():,.0f}" if len(topped) else "  none topped up")
    print(f"  liquidity: removed below-SMA avg shares_pct {rem_liq:.2f} vs topped-up {top_liq:.2f}")
    g0, g1, g2 = comp.iloc[0]['tot_ret_net_A_pct'], comp.iloc[1]['tot_ret_net_A_pct'], comp.iloc[2]['tot_ret_net_A_pct']
    print("\nVERDICT (net_A total return):")
    print(f"  baseline {g0:.2f}% -> passive {g1:.2f}% ({g1-g0:+.2f}) -> reallocated {g2:.2f}% ({g2-g0:+.2f} vs baseline)")
    print(f"  removal effect (2-1): {g1-g0:+.2f} pp | reallocation effect (3-2): {g2-g1:+.2f} pp")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
