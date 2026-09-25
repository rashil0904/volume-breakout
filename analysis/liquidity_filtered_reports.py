# -*- coding: utf-8 -*-
"""liquidity_filtered_reports.py — rebuild the baseline performance reports on a LIQUIDITY-FILTERED
trade set: exclude every SYMBOL whose AVG shares_pct_of_volume (entry window 15:00-15:22, across all
its trades) exceeds 10%. Symbol-level exclusion (drops ALL that symbol's trades). Produce BOTH the
combined (long+short) and long-only reports on the survivors, mirroring baseline_final_performance.

Flags: (a) SYMBOL-level exclusion by avg shares_pct_of_volume>10% ; (b) window = 15:00-15:22 (same as
liquidity work) ; (c) combined + long-only both rebuilt ; (d) dropped symbols' standalone contribution
reported. NOTE: surviving trades keep their ORIGINAL capital/pnl — the day-level capital SEQUENCING is
not re-run; the allocation-diagnostics block is reconstructed from the surviving trades.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
import baseline_and_cross_final as B          # reuse full_report + constants (BASE_POOL, RF, BASE_ALLOC, R)

BASE_XLSX = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
LIQ_XLSX = rb.RESULTS / "liquidity_proxies" / "liquidity_proxies.xlsx"
OUTDIR = rb.RESULTS / "liquidity_filtered"
THRESH = 10.0
BASE_POOL, BASE_ALLOC = B.BASE_POOL, B.BASE_ALLOC


def to_long_only(T):
    """Strip the short leg: gross=long_pnl; net_A=long-0.23% cost; net_B=long-0.38% cost; no short cost."""
    L = T.copy()
    L["short_pnl"] = 0.0; L["short_cost"] = 0.0
    L["gross_pnl"] = L["long_pnl"]
    L["combined_pnl"] = L["long_pnl"]
    L["netA_pnl"] = L["long_pnl"] - L["long_cost_023"]
    L["netB_pnl"] = L["long_pnl"] - L["long_cost_038"]
    cap = L["capital_deployed"].replace(0, np.nan)
    L["gross_ret"] = (L["gross_pnl"] / cap * 100).round(6)
    L["netA_ret"] = (L["netA_pnl"] / cap * 100).round(6)
    L["netB_ret"] = (L["netB_pnl"] / cap * 100).round(6)
    return L


def reconstruct_diag(T):
    """Rebuild the per-entry-day allocation dict {date: {X, per_c, nC, p2}} from surviving trades."""
    diag = {}
    for d, g in T.groupby("entry_date"):
        ab = g[g["category"].isin(["A", "B"])]
        cc = g[g["category"] == "C"]
        a2 = g[(g["category"] == "A") & (g["n_legs"] == 2)]
        diag[d] = {"X": float(ab["capital_deployed"].sum()),
                   "per_c": float(cc["capital_deployed"].mean()) if len(cc) else 0.0,
                   "nC": int(len(cc)),
                   "p2": float(len(a2) * 0.5 * BASE_ALLOC)}
    return diag


def headline(T, label):
    """Key metrics for the filtered-vs-unfiltered comparison."""
    dd = T.groupby("exit_date").agg(g=("gross_pnl", "sum"), a=("netA_pnl", "sum"),
                                    b=("netB_pnl", "sum"), c=("capital_deployed", "sum")).reset_index()
    rfd = B.RF / 252 * 100
    def shp(x):
        return round((x.mean() - rfd) / x.std(ddof=1) * np.sqrt(252), 3) if len(x) > 1 else 0.0
    return {"set": label, "n_trades": len(T),
            "tot_ret_gross_pct": round(T["gross_pnl"].sum() / BASE_POOL * 100, 2),
            "tot_ret_netA_pct": round(T["netA_pnl"].sum() / BASE_POOL * 100, 2),
            "tot_ret_netB_pct": round(T["netB_pnl"].sum() / BASE_POOL * 100, 2),
            "win_rate_netA_pct": round((T["netA_pnl"] > 0).mean() * 100, 2),
            "avg_ret_netA_pct": round(T["netA_ret"].mean(), 4),
            "sharpe_gross": shp(dd["g"] / dd["c"] * 100),
            "sharpe_netA": shp(dd["a"] / dd["c"] * 100),
            "sharpe_netB": shp(dd["b"] / dd["c"] * 100)}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T_full = pd.read_excel(BASE_XLSX, sheet_name="all_trades")
    T_full["entry_date"] = pd.to_datetime(T_full["entry_date"])
    T_full["exit_date"] = pd.to_datetime(T_full["exit_date"])

    # STEP 1 — excluded symbols (avg shares_pct_of_volume > 10%, entry-window basis)
    PS = pd.read_excel(LIQ_XLSX, sheet_name="per_symbol_profile")
    excl = PS[PS["avg_shares_pct_of_volume"] > THRESH].copy()
    excl_syms = set(excl["symbol"])
    # attach n_trades + total pnl of each excluded symbol (combined basis)
    contr = T_full[T_full["symbol"].isin(excl_syms)].groupby("symbol").agg(
        n_trades=("symbol", "size"), gross_pnl=("gross_pnl", "sum"), netA_pnl=("netA_pnl", "sum"),
        avg_netA_ret=("netA_ret", "mean")).reset_index()
    EXCL = excl[["symbol", "avg_shares_pct_of_volume"]].merge(contr, on="symbol", how="left") \
        .sort_values("avg_shares_pct_of_volume", ascending=False).reset_index(drop=True)

    surv = T_full[~T_full["symbol"].isin(excl_syms)].reset_index(drop=True)
    dropped = T_full[T_full["symbol"].isin(excl_syms)].reset_index(drop=True)
    print(f"symbols excluded (avg shares%>{THRESH}): {len(excl_syms)} | "
          f"trades before {len(T_full):,} -> after {len(surv):,} (dropped {len(dropped)})")

    # STEP 3 — rebuild reports on survivors (combined + long-only)
    diag_comb = reconstruct_diag(surv)
    B.full_report(surv, diag_comb, OUTDIR / "baseline_liquidity_filtered_combined.xlsx")

    surv_long = to_long_only(surv)
    B.full_report(surv_long, reconstruct_diag(surv_long), OUTDIR / "baseline_liquidity_filtered_long_only.xlsx")

    # STEP 4 — comparison filtered vs unfiltered (combined + long-only)
    full_long = to_long_only(T_full)
    cmp_comb = pd.DataFrame([headline(T_full, "COMBINED unfiltered"), headline(surv, "COMBINED filtered")])
    cmp_long = pd.DataFrame([headline(full_long, "LONG-ONLY unfiltered"), headline(surv_long, "LONG-ONLY filtered")])

    # dropped-symbols standalone contribution (combined + long-only)
    drop_long = to_long_only(dropped)
    drop_contrib = pd.DataFrame([
        {"basis": "combined", "n_dropped_trades": len(dropped),
         "gross_pnl_inr": round(dropped["gross_pnl"].sum(), 0), "netA_pnl_inr": round(dropped["netA_pnl"].sum(), 0),
         "gross_ret_on_5L_pct": round(dropped["gross_pnl"].sum() / BASE_POOL * 100, 2),
         "netA_ret_on_5L_pct": round(dropped["netA_pnl"].sum() / BASE_POOL * 100, 2),
         "avg_netA_ret_per_trade_pct": round(dropped["netA_ret"].mean(), 4),
         "win_rate_netA_pct": round((dropped["netA_pnl"] > 0).mean() * 100, 2)},
        {"basis": "long_only", "n_dropped_trades": len(drop_long),
         "gross_pnl_inr": round(drop_long["gross_pnl"].sum(), 0), "netA_pnl_inr": round(drop_long["netA_pnl"].sum(), 0),
         "gross_ret_on_5L_pct": round(drop_long["gross_pnl"].sum() / BASE_POOL * 100, 2),
         "netA_ret_on_5L_pct": round(drop_long["netA_pnl"].sum() / BASE_POOL * 100, 2),
         "avg_netA_ret_per_trade_pct": round(drop_long["netA_ret"].mean(), 4),
         "win_rate_netA_pct": round((drop_long["netA_pnl"] > 0).mean() * 100, 2)}])

    with pd.ExcelWriter(OUTDIR / "liquidity_filter_comparison.xlsx", engine="openpyxl") as w:
        EXCL.to_excel(w, sheet_name="excluded_symbols", index=False)
        cmp_comb.to_excel(w, sheet_name="compare_combined", index=False)
        cmp_long.to_excel(w, sheet_name="compare_long_only", index=False)
        drop_contrib.to_excel(w, sheet_name="dropped_contribution", index=False)

    pd.set_option("display.width", 240)
    print("\n=== EXCLUDED SYMBOLS (avg shares_pct_of_volume > 10%) ===")
    print(EXCL.to_string(index=False))
    print("\n=== COMBINED: filtered vs unfiltered ===")
    print(cmp_comb.to_string(index=False))
    print("\n=== LONG-ONLY: filtered vs unfiltered ===")
    print(cmp_long.to_string(index=False))
    print("\n=== DROPPED SYMBOLS' STANDALONE CONTRIBUTION ===")
    print(drop_contrib.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")
    print("  baseline_liquidity_filtered_combined.xlsx / _long_only.xlsx / liquidity_filter_comparison.xlsx")


if __name__ == "__main__":
    main()
