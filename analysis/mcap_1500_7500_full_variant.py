# -*- coding: utf-8 -*-
"""
mcap_1500_7500_full_variant.py
==============================
Full write-up of the single-continuous-band ₹1,500–7,500 Cr variant (a fresh entry
filter over [1500,7500], NOT a merge of the two sub-band trade sets), vs the existing
₹1,500–5,000 Cr strategy.

Steps 1–4 of the spec:
  1. ₹1,500–7,500 Cr trade set (reuses diagnostic_table_mcap1p5k7p5k.csv + canonical
     entry->09:45/12:00 split + 14% target exit; pool-split sizing, cap ₹1L/trade).
  2. Gross + net returns.
  3. Quarterly compounding — FULL COMPOUNDING (pool & alloc carry the prior quarter's
     ENDING value every quarter: scale up on a gain, down on a loss / loss baked in; no
     reset, no flat carry). Gross and net are separate self-contained chains.
  4. Side-by-side vs ₹1,500–5,000 Cr (its base metrics are the canonical existing
     figures; compounding computed with the identical carry-forward function for a
     like-for-like comparison).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import final_performance_report as fpr
import mcap_band_comparison as MC

OUTDIR = rb.RESULTS / "mcap_1500_7500_variant"
DIAG_WIDE = rb.RESULTS / "diagnostic_table_mcap1p5k7p5k.csv"
BP, BA = fpr.BASE_POOL, fpr.BASE_ALLOC     # 500,000 / 100,000
EXPENSE = fpr.EXPENSE


def compound_chain(T, use_net):
    """Full-compounding quarterly chain (gross or net). Pool & alloc carry the prior
    quarter's ENDING value every quarter: scale UP on a gain, DOWN on a loss (loss baked
    in via pool & alloc *= 1+ret/100). No reset to base, no flat carry-forward."""
    quarters = sorted(T["quarter"].unique())
    pool, alloc = BP, BA
    prev_ret = prev_q = None
    rows = []
    for i, q in enumerate(quarters):
        g = T[T["quarter"] == q]
        cg, cn = fpr.scaled(g, alloc)
        val = cn if use_net else cg
        ret_q = val / pool * 100
        ending = pool + val
        if i == 0:
            trig, scale = "N/A — first quarter", ""
        else:
            trig, scale = (prev_ret > 0), round(1 + prev_ret / 100, 6)   # up or down
        rows.append({
            "quarter": q,
            "prior_quarter": "N/A" if i == 0 else prev_q,
            "prior_quarter_compounded_return_pct": "" if i == 0 else round(prev_ret, 4),
            "compounding_trigger_met": trig,
            "scale_factor_applied": scale,
            "pool_for_this_quarter_inr": round(pool, 0),
            "per_trade_allocation_used_inr": round(alloc, 0),
            "n_trades_this_quarter": len(g),
            ("net_pnl" if use_net else "gross_pnl") + "_this_quarter_inr": round(val, 0),
            "compounded_return_pct_this_quarter": round(ret_q, 4),
            "ending_pool_value_inr": round(ending, 0),
            "cumulative_growth_multiple": round(ending / BP, 4),
        })
        prev_ret, prev_q = ret_q, q
        f = 1 + ret_q / 100; pool *= f; alloc *= f     # always compound ending pool (up or down)
    df = pd.DataFrame(rows)
    return df, df["ending_pool_value_inr"].iloc[-1], df["cumulative_growth_multiple"].iloc[-1]


def base_and_net_metrics(T):
    n = len(T)
    gp, npl = T["gross_pnl"].sum(), T["net_pnl"].sum()
    return {
        "n_trades": n,
        "gross_total_return_fixedbase_pct": round(gp / BP * 100, 4),
        "gross_total_pnl_inr": round(gp, 0),
        "net_total_return_fixedbase_pct": round(npl / BP * 100, 4),
        "net_total_pnl_inr": round(npl, 0),
        "gross_win_rate_pct": round((T["gross_pnl"] > 0).mean() * 100, 2),
        "net_win_rate_pct": round((T["net_pnl"] > 0).mean() * 100, 2),
        "gross_avg_return_per_trade_pct": round(T["gross_ret"].mean(), 4),
        "net_avg_return_per_trade_pct": round(T["net_ret"].mean(), 4),
        "gross_median_return_per_trade_pct": round(T["gross_ret"].median(), 4),
        "net_median_return_per_trade_pct": round(T["net_ret"].median(), 4),
        "avg_capital_deployed_per_trade": round(T["capital_deployed"].mean(), 0),
    }


def full_column(T, label):
    m = {"strategy": label, **base_and_net_metrics(T)}
    gdet, g_final, g_mult = compound_chain(T, use_net=False)
    ndet, n_final, n_mult = compound_chain(T, use_net=True)
    m["gross_compounded_final_pool_value_inr"] = round(g_final, 0)
    m["gross_compounded_growth_multiple"] = g_mult
    m["net_compounded_final_pool_value_inr"] = round(n_final, 0)
    m["net_compounded_growth_multiple"] = n_mult
    return m, gdet, ndet


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not DIAG_WIDE.exists():
        raise SystemExit(f"Missing {DIAG_WIDE.name} — build ₹1,500–7,500 Cr table first.")

    # ── Band B: ₹1,500–7,500 Cr (fresh single filter) ──
    print("Band ₹1,500–7,500 Cr (single fresh filter) …")
    T_wide, _ = MC.trades_for(DIAG_WIDE)
    cols = ["entry_date", "symbol", "entry_price", "shares", "capital_deployed",
            "exit_date", "exit_price", "exit_type", "exit_time",
            "gross_ret", "gross_pnl", "net_ret", "net_pnl"]
    T_wide[cols].to_csv(OUTDIR / "trades_mcap1p5k7p5k.csv", index=False)
    print(f"  trades: {len(T_wide):,}")

    # ── Band A: ₹1,500–5,000 Cr canonical (existing results, not recomputed) ──
    print("Band ₹1,500–5,000 Cr (canonical existing) …")
    T_base = fpr.build_trades()
    print(f"  trades: {len(T_base):,}")

    mB, gdetB, ndetB = full_column(T_wide, "mcap_1500_7500_cr")
    mA, _, _ = full_column(T_base, "mcap_1500_5000_cr")
    comp = pd.DataFrame([mA, mB]).set_index("strategy").T

    # ── Excel: comparison + wide-band compounding detail (gross + net) ──
    xlsx = OUTDIR / "mcap_1500_7500_vs_1500_5000.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="comparison")
        gdetB.to_excel(w, sheet_name="quarterly_compounding_gross", index=False)
        ndetB.to_excel(w, sheet_name="quarterly_compounding_net", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print("COMPARISON — ₹1,500–5,000 Cr  vs  ₹1,500–7,500 Cr  (carry-forward compounding)")
    print("=" * 100)
    print(comp.to_string())
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
