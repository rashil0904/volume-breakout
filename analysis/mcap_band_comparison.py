# -*- coding: utf-8 -*-
"""
mcap_band_comparison.py
=======================
Side-by-side comparison of the locked baseline strategy across two market-cap bands:
  A) ₹1,500–5,000 Cr  (existing)   — diagnostic_table.csv
  B) ₹5,000–10,000 Cr (new variant) — diagnostic_table_mcap5k10k.csv

EVERYTHING is identical between the two except which stocks qualify (mcap band):
  entry 15:15, lookback 36, volume 6x, +5% move; pool-split sizing (₹1L/trade when
  n<=5 signals that day, ₹5L/n when n>5); exit conditional_split_best_t2 (09:45/12:00)
  + 14% profit-target overlay. Both use the SAME pool-split sizing (per user: match the
  deployed baseline, which is NOT flat ₹1L/trade).

Exit logic is reused verbatim from final_performance_report.build_trades() by swapping
the base-position source. Nothing existing is overwritten; outputs carry _mcap5k10k /
comparison suffixes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import exit_time_sweep as ets
import final_performance_report as fpr

OUTDIR = rb.RESULTS / "mcap_band_comparison"
BASE_POOL = fpr.BASE_POOL
EXPENSE = fpr.EXPENSE

# ── all bands to compare (label, diagnostic table). Nested note: 5000–7500 ⊂ 5000–10000 ──
BANDS = [
    ("mcap_1500_5000_cr",  rb.RESULTS / "diagnostic_table.csv"),            # existing
    ("mcap_5000_7500_cr",  rb.RESULTS / "diagnostic_table_mcap5k7p5k.csv"), # new (lower half)
    ("mcap_5000_10000_cr", rb.RESULTS / "diagnostic_table_mcap5k10k.csv"),  # full upper band
]
DIAG_A = BANDS[0][1]                                          # baseline, for self-validation


def base_from_diag(diag_path):
    """Reproduce the baseline's pool-split base positions from a diagnostic table.
    Identical sizing to run_backtest.run_standard: per entry day count signals n,
    target = ₹1L if n<=5 else ₹5L/n, shares = floor(target/entry), drop zero-share."""
    diag = pd.read_csv(diag_path, parse_dates=["date"])
    sig = diag[diag["passes_all_three"]].dropna(subset=["entry_price_315pm"]).copy()
    rows = []
    for d, day in sig.groupby("date"):
        n = len(day)
        tgt = rb._day_target(n)                       # ₹1L (n<=5) else ₹5L/n
        for _, r in day.iterrows():
            ep = float(r["entry_price_315pm"])
            sh = int(tgt // ep)
            if sh == 0:
                continue
            rows.append({"date": d, "symbol": r["symbol"], "entry": ep,
                         "shares": sh, "cap": sh * ep})
    return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


def trades_for(diag_path):
    """Run the exact locked exit logic on a band's base positions."""
    base = base_from_diag(diag_path)
    orig = ets.load_base_positions
    ets.load_base_positions = lambda: base            # swap base source only
    try:
        T = fpr.build_trades()
    finally:
        ets.load_base_positions = orig                # always restore
    return T, base


def metrics(T, label):
    n = len(T)
    gp = T["gross_pnl"].sum(); npl = T["net_pnl"].sum()
    # per-month gross return (each month's Σpnl / fixed base), then mean of months
    mret = T.groupby("month")["gross_pnl"].sum() / BASE_POOL * 100
    mret_net = T.groupby("month")["net_pnl"].sum() / BASE_POOL * 100
    return {
        "strategy": label,
        "n_trades": n,
        "n_entry_days": T["entry_date"].nunique(),
        "win_rate_pct": round((T["gross_pnl"] > 0).mean() * 100, 2),
        "avg_return_per_trade_pct": round(T["gross_ret"].mean(), 4),
        "median_return_per_trade_pct": round(T["gross_ret"].median(), 4),
        "total_return_fixedbase_pct": round(gp / BASE_POOL * 100, 4),
        "total_pnl_inr": round(gp, 0),
        "avg_return_per_month_pct": round(mret.mean(), 4),
        "avg_capital_deployed_per_trade_inr": round(T["capital_deployed"].mean(), 0),
        # net-of-0.23% versions (existing net machinery)
        "net_win_rate_pct": round((T["net_pnl"] > 0).mean() * 100, 2),
        "net_avg_return_per_trade_pct": round(T["net_ret"].mean(), 4),
        "net_median_return_per_trade_pct": round(T["net_ret"].median(), 4),
        "net_total_return_fixedbase_pct": round(npl / BASE_POOL * 100, 4),
        "net_total_pnl_inr": round(npl, 0),
        "net_avg_return_per_month_pct": round(mret_net.mean(), 4),
    }


def suffix_for(diag_path):
    """trades_<suffix>.csv filename from a diagnostic_table_<suffix>.csv path."""
    stem = Path(diag_path).stem
    return stem.replace("diagnostic_table_", "") if stem != "diagnostic_table" else "mcap1500_5000"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    missing = [str(p.name) for _, p in BANDS if not p.exists()]
    if missing:
        raise SystemExit(f"Missing diagnostic tables: {missing} — "
                         f"run analysis/build_diag_band.py for each first.")

    cols = ["entry_date", "symbol", "entry_price", "shares", "capital_deployed",
            "exit_date", "exit_price", "exit_type", "exit_time",
            "gross_ret", "gross_pnl", "net_ret", "net_pnl"]

    results = []          # (label, T, base)
    for label, diag in BANDS:
        print(f"Band {label} …")
        T, base = trades_for(diag)
        print(f"  base positions: {len(base):,} | final trades: {len(T):,}")
        results.append((label, T, base))

        # self-validation only for the deployed baseline
        if diag == DIAG_A:
            try:
                std = ets.load_base_positions()
                print(f"  [validate] deployed baseline rows={len(std):,} vs reproduced="
                      f"{len(base):,} (Δ={len(base) - len(std):+d}); caps ₹{std['cap'].mean():,.0f}"
                      f" vs ₹{base['cap'].mean():,.0f}")
            except Exception as e:
                print(f"  [validate] skipped ({e})")

        # trade-level output per band (baseline is already in prior reports; write the new ones)
        if diag != DIAG_A:
            T[cols].to_csv(OUTDIR / f"trades_{suffix_for(diag)}.csv", index=False)

    comp = pd.DataFrame([metrics(T, label) for label, T, _ in results])
    comp.to_csv(OUTDIR / "comparison_table.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "mcap_band_comparison.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="comparison", index=False)
        for (label, T, _), (_, diag) in zip(results, BANDS):
            if diag != DIAG_A:
                T[cols].to_excel(w, sheet_name=f"trades_{suffix_for(diag)}"[:31], index=False)

    # ── prints ──
    pd.set_option("display.width", 220)
    gross_cols = ["strategy", "n_trades", "win_rate_pct", "avg_return_per_trade_pct",
                  "median_return_per_trade_pct", "total_return_fixedbase_pct",
                  "total_pnl_inr", "avg_return_per_month_pct",
                  "avg_capital_deployed_per_trade_inr"]
    net_cols = ["strategy", "net_win_rate_pct", "net_avg_return_per_trade_pct",
                "net_median_return_per_trade_pct", "net_total_return_fixedbase_pct",
                "net_total_pnl_inr", "net_avg_return_per_month_pct"]
    print("\n" + "=" * 128)
    print("THREE-BAND SIDE-BY-SIDE — GROSS  (all pool-split sized; only mcap band differs)")
    print("  NOTE: ₹5,000–7,500 Cr is a SUBSET of ₹5,000–10,000 Cr (isolates its lower half).")
    print("=" * 128)
    print(comp[gross_cols].to_string(index=False))
    print("\n" + "=" * 128)
    print("THREE-BAND SIDE-BY-SIDE — NET of 0.23% expense")
    print("=" * 128)
    print(comp[net_cols].to_string(index=False))

    print("\n--- SIGNAL / TRADE COUNTS & EFFECTIVE DEPLOYMENT ---")
    for label, T, base in results:
        print(f"  {label}: base {len(base):,} | trades {len(T):,} | "
              f"entry days {T['entry_date'].nunique():,} | "
              f"avg cap/trade ₹{T['capital_deployed'].mean():,.0f} | "
              f"avg entry px ₹{T['entry_price'].mean():,.1f}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
