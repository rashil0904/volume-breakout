# -*- coding: utf-8 -*-
"""extended_reallocation.py — exclude the 38 extended (>=50%-above-36SMA) trades WITH capital
reallocation on the affected days (re-run the daily allocation sequence without them), vs the full
baseline (Config 1) and passive exclusion (Config 2). Engine reused: run_config(..., exclude=set)
drops the 38 BEFORE sequencing so freed capital reflows to same-day trades under the hard 1L cap.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
import baseline_and_cross_final as B

EXT_XLSX = rb.RESULTS / "extended_sma_test" / "extended_sma_test.xlsx"
LIQ = rb.RESULTS / "liquidity_proxies" / "per_trade_liquidity.csv"
OUTDIR = rb.RESULTS / "extended_reallocation"
BP = B.BASE_POOL
RF = B.RF


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
    ext = pd.read_excel(EXT_XLSX, sheet_name="extended_per_trade")
    ext["ed"] = pd.to_datetime(ext["entry_date"]).dt.date
    exclude = set(zip(ext["symbol"], ext["ed"]))
    print(f"extended trades to remove: {len(exclude)}")

    cache = B.build_cache()
    T1, d1 = B.run_config("baseline", cache)                     # Config 1 — full baseline
    T3, d3 = B.run_config("baseline", cache, exclude=exclude)    # Config 3 — exclude + reallocate
    T1["entry_date"] = pd.to_datetime(T1["entry_date"]); T1["exit_date"] = pd.to_datetime(T1["exit_date"])
    T3["entry_date"] = pd.to_datetime(T3["entry_date"]); T3["exit_date"] = pd.to_datetime(T3["exit_date"])
    T1["ed"] = T1["entry_date"].dt.date; T3["ed"] = T3["entry_date"].dt.date

    # Config 2 — passive exclusion (drop the 38 rows, others UNCHANGED)
    key1 = list(zip(T1["symbol"], T1["ed"]))
    T2 = T1[[k not in exclude for k in key1]].copy()

    print(f"Config1 {len(T1)} | Config2 {len(T2)} | Config3 {len(T3)}")

    # ── STEP 3: three-way ──
    comp = pd.DataFrame([metrics(T1, "1_full_baseline"), metrics(T2, "2_passive_exclude"),
                         metrics(T3, "3_exclude_reallocate")])
    d21 = {"config": "delta 2-1 (loser removal)"}
    d32 = {"config": "delta 3-2 (reallocation boost)"}
    d31 = {"config": "delta 3-1 (total)"}
    for c in comp.columns:
        if c == "config":
            continue
        d21[c] = round(comp.iloc[1][c] - comp.iloc[0][c], 4)
        d32[c] = round(comp.iloc[2][c] - comp.iloc[1][c], 4)
        d31[c] = round(comp.iloc[2][c] - comp.iloc[0][c], 4)
    comp = pd.concat([comp, pd.DataFrame([d21, d32, d31])], ignore_index=True)

    # ── STEP 4: reallocation diagnostics (compare C1 vs C3 on affected days) ──
    aff_days = set(ext["ed"])
    c1a = T1[T1["ed"].isin(aff_days)].set_index(["symbol", "ed"])
    c3a = T3[T3["ed"].isin(aff_days)].set_index(["symbol", "ed"])
    surv_keys = [k for k in c3a.index if k in c1a.index]           # trades present in both (non-removed)
    rows = []
    for k in surv_keys:
        cap1 = float(c1a.loc[k, "capital_deployed"]); cap3 = float(c3a.loc[k, "capital_deployed"])
        if isinstance(cap1, pd.Series):
            cap1 = cap1.iloc[0]; cap3 = c3a.loc[k, "capital_deployed"].iloc[0]
        rows.append({"symbol": k[0], "ed": k[1], "cap_c1": cap1, "cap_c3": cap3, "cap_inc": cap3 - cap1,
                     "pnl_c1": float(np.ravel(c1a.loc[k, "gross_pnl"])[0]), "pnl_c3": float(np.ravel(c3a.loc[k, "gross_pnl"])[0])})
    RB = pd.DataFrame(rows)
    topped = RB[RB["cap_inc"] > 1.0]

    # per-day: freed / redeployed / idle
    day_rows = []
    for d, g in ext.groupby("ed"):
        freed = float(g["entry_price"].notna().sum() and T1[(T1["ed"] == d) & (T1["symbol"].isin(g["symbol"]))]["capital_deployed"].sum())
        redep = float(RB[RB["ed"] == d]["cap_inc"].sum()) if len(RB) else 0.0
        tup = RB[(RB["ed"] == d) & (RB["cap_inc"] > 1.0)]
        pnl_before = float(T1[(T1["ed"] == d)]["gross_pnl"].sum())
        pnl_after = float(T3[(T3["ed"] == d)]["gross_pnl"].sum())
        day_rows.append({"date": str(d), "removed_syms": ",".join(g["symbol"]), "n_removed": len(g),
                         "capital_freed": round(freed, 0), "capital_redeployed": round(redep, 0),
                         "capital_idle": round(freed - redep, 0), "n_topped_up": len(tup),
                         "avg_topup_inr": round(tup["cap_inc"].mean(), 0) if len(tup) else 0.0,
                         "day_pnl_C1": round(pnl_before, 0), "day_pnl_C3": round(pnl_after, 0),
                         "day_pnl_change": round(pnl_after - pnl_before, 0)})
    DAYS = pd.DataFrame(day_rows).sort_values("date")

    freed_tot = DAYS["capital_freed"].sum(); redep_tot = DAYS["capital_redeployed"].sum()
    idle_tot = DAYS["capital_idle"].sum()
    n_days_idle = int((DAYS["capital_idle"] > 1000).sum())

    # liquidity shift: removed extended vs topped-up beneficiaries
    lq = pd.read_csv(LIQ)[["symbol", "entry_date", "shares_pct_of_volume"]]
    lq["ed"] = pd.to_datetime(lq["entry_date"]).dt.date
    lmap = lq.set_index(["symbol", "ed"])["shares_pct_of_volume"].to_dict()
    rem_liq = np.nanmean([lmap.get((s, d), np.nan) for s, d in zip(ext["symbol"], ext["ed"])])
    top_liq = np.nanmean([lmap.get((s, d), np.nan) for s, d in zip(topped["symbol"], topped["ed"])]) if len(topped) else np.nan

    with pd.ExcelWriter(OUTDIR / "extended_reallocation.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="three_way_compare", index=False)
        DAYS.to_excel(w, sheet_name="per_affected_day", index=False)
        topped.sort_values("cap_inc", ascending=False).to_excel(w, sheet_name="topped_up_trades", index=False)

    pd.set_option("display.width", 250)
    print("\n" + "=" * 110 + "\nTHREE-WAY: full baseline (1) vs passive exclude (2) vs exclude+reallocate (3)\n" + "=" * 110)
    show = ["config", "n_trades", "avg_capital_deployed", "tot_ret_gross_pct", "tot_ret_net_A_pct",
            "tot_ret_net_B_pct", "win_net_A_pct", "avg_ret_net_A_pct", "sharpe_gross", "sharpe_net_A", "sharpe_net_B"]
    print(comp[show].to_string(index=False))
    print("\n--- REALLOCATION DIAGNOSTICS ---")
    print(f"  affected days: {len(DAYS)} | capital freed Rs{freed_tot:,.0f} | redeployed Rs{redep_tot:,.0f} "
          f"| idle Rs{idle_tot:,.0f} ({idle_tot/freed_tot*100:.0f}% idle)")
    print(f"  days where freed capital stayed (largely) idle (>Rs1000): {n_days_idle} of {len(DAYS)}")
    print(f"  trades topped up: {len(topped)} | avg topup Rs{topped['cap_inc'].mean():,.0f}" if len(topped) else "  trades topped up: 0")
    print(f"  liquidity: removed-extended avg shares_pct_of_vol {rem_liq:.2f}  vs topped-up {top_liq:.2f}")
    print("\n--- PER-AFFECTED-DAY (topped-up days first) ---")
    print(DAYS[DAYS["n_topped_up"] > 0].to_string(index=False) if (DAYS["n_topped_up"] > 0).any()
          else "  NO day had a topped-up trade — all freed capital stayed idle (see verdict).")
    print(f"\nVERDICT: baseline gross {comp.iloc[0]['tot_ret_gross_pct']:.2f}% -> passive {comp.iloc[1]['tot_ret_gross_pct']:.2f}% "
          f"-> reallocated {comp.iloc[2]['tot_ret_gross_pct']:.2f}%")
    print(f"  net_A: {comp.iloc[0]['tot_ret_net_A_pct']:.2f}% -> {comp.iloc[1]['tot_ret_net_A_pct']:.2f}% -> {comp.iloc[2]['tot_ret_net_A_pct']:.2f}%")
    print(f"  loser-removal (2-1) net_A {comp.iloc[4]['tot_ret_net_A_pct']:+.2f} pp | reallocation (3-2) net_A {comp.iloc[5]['tot_ret_net_A_pct']:+.2f} pp")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
