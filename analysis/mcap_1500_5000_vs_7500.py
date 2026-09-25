# -*- coding: utf-8 -*-
"""
mcap_1500_5000_vs_7500.py
=========================
Focused side-by-side: ₹1,500–5,000 Cr (existing baseline) vs ₹1,500–7,500 Cr (widened
upper bound). Reuses trades_for()/metrics() from mcap_band_comparison — no logic dup.

Nested note: ₹1,500–5,000 Cr is a SUBSET of ₹1,500–7,500 Cr. The wider band simply
extends the upper bound from 5,000 to 7,500 Cr, so the comparison shows what adding the
₹5,000–7,500 Cr names (with correct combined per-day signal counts / pool-split sizing)
does to the baseline. Everything else identical; pool-split sizing; only the band differs.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import mcap_band_comparison as MC

OUTDIR = rb.RESULTS / "mcap_band_comparison"
BANDS = [
    ("mcap_1500_5000_cr", rb.RESULTS / "diagnostic_table.csv"),
    ("mcap_1500_7500_cr", rb.RESULTS / "diagnostic_table_mcap1p5k7p5k.csv"),
]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for _, p in BANDS:
        if not p.exists():
            raise SystemExit(f"Missing {p.name} — run build_diag_band.py 1500 7500 mcap1p5k7p5k")

    cols = ["entry_date", "symbol", "entry_price", "shares", "capital_deployed",
            "exit_date", "exit_price", "exit_type", "exit_time",
            "gross_ret", "gross_pnl", "net_ret", "net_pnl"]

    results = []
    for label, diag in BANDS:
        print(f"Band {label} …")
        T, base = MC.trades_for(diag)
        print(f"  base positions: {len(base):,} | final trades: {len(T):,} | "
              f"entry days {T['entry_date'].nunique():,} | avg cap/trade "
              f"₹{T['capital_deployed'].mean():,.0f} | avg entry px ₹{T['entry_price'].mean():,.1f}")
        results.append((label, T, base))
        if label == "mcap_1500_7500_cr":
            T[cols].to_csv(OUTDIR / "trades_mcap1p5k7p5k.csv", index=False)

    comp = pd.DataFrame([MC.metrics(T, label) for label, T, _ in results])
    comp.to_csv(OUTDIR / "comparison_1500_5000_vs_7500.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "comparison_1500_5000_vs_7500.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="comparison", index=False)
        results[1][1][cols].to_excel(w, sheet_name="trades_mcap1500_7500", index=False)

    pd.set_option("display.width", 220)
    gross_cols = ["strategy", "n_trades", "win_rate_pct", "avg_return_per_trade_pct",
                  "median_return_per_trade_pct", "total_return_fixedbase_pct",
                  "total_pnl_inr", "avg_return_per_month_pct",
                  "avg_capital_deployed_per_trade_inr"]
    net_cols = ["strategy", "net_win_rate_pct", "net_avg_return_per_trade_pct",
                "net_median_return_per_trade_pct", "net_total_return_fixedbase_pct",
                "net_total_pnl_inr", "net_avg_return_per_month_pct"]
    print("\n" + "=" * 128)
    print("SIDE-BY-SIDE — GROSS   ₹1,500–5,000 Cr  vs  ₹1,500–7,500 Cr")
    print("  NOTE: ₹1,500–5,000 ⊂ ₹1,500–7,500 (wider band just raises the upper bound to 7,500).")
    print("=" * 128)
    print(comp[gross_cols].to_string(index=False))
    print("\n" + "=" * 128)
    print("SIDE-BY-SIDE — NET of 0.23% expense")
    print("=" * 128)
    print(comp[net_cols].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
