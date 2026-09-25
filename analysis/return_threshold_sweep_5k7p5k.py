# -*- coding: utf-8 -*-
"""
return_threshold_sweep_5k7p5k.py
================================
Sweeps the daily-return entry threshold (X% vs prev close) from +3% to +8% for the
₹5,000–7,500 Cr variant. Everything else fixed: mcap 5,000–7,500 Cr, lookback 36,
volume 6x, 3:15pm entry, 09:45/12:00 split exit + 14% target, pool-split sizing
(₹1L/trade cap; ₹5L/n on >5-signal days) — identical to the established variant.

The diagnostic table stores return_pct_vs_prev_close + passes_volume per mcap-eligible
row, so each threshold's signal set is re-derived without rebuilding data. Exit logic is
reused verbatim via final_performance_report.build_trades() with a swapped base source.

X=5 is the original main-strategy threshold (reference row); its filter equals
passes_all_three, so it reconciles with the ₹5,000–7,500 variant's 916 trades / 117.28%.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import exit_time_sweep as ets
import final_performance_report as fpr

OUTDIR = rb.RESULTS / "return_threshold_sweep_5k7p5k"
DIAG = rb.RESULTS / "diagnostic_table_mcap5k7p5k.csv"
BASE_POOL = fpr.BASE_POOL
THRESHOLDS = [3, 4, 5, 6, 7, 8]
REF_X = 5


def base_at_threshold(diag, X):
    """Pool-split base positions for signals passing volume AND return >= X% (mcap already
    band-filtered in this table). Identical sizing to run_backtest.run_standard."""
    sig = diag[diag["passes_volume"] & (diag["return_pct_vs_prev_close"] >= X)].copy()
    sig = sig.dropna(subset=["entry_price_315pm"])
    rows = []
    for d, day in sig.groupby("date"):
        tgt = rb._day_target(len(day))                 # ₹1L (n<=5) else ₹5L/n
        for _, r in day.iterrows():
            ep = float(r["entry_price_315pm"])
            sh = int(tgt // ep)
            if sh == 0:
                continue
            rows.append({"date": d, "symbol": r["symbol"], "entry": ep,
                         "shares": sh, "cap": sh * ep})
    return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


def trades_at_threshold(base):
    orig = ets.load_base_positions
    ets.load_base_positions = lambda: base
    try:
        return fpr.build_trades()
    finally:
        ets.load_base_positions = orig


def metrics(T, X):
    n = len(T)
    gp = T["gross_pnl"].sum()
    win = T[T["gross_pnl"] > 0]; los = T[T["gross_pnl"] <= 0]
    return {
        "return_threshold_pct": X,
        "n_trades": n,
        "win_rate_pct": round((T["gross_pnl"] > 0).mean() * 100, 2) if n else 0,
        "avg_return_per_trade_pct": round(T["gross_ret"].mean(), 4) if n else 0,
        "median_return_per_trade_pct": round(T["gross_ret"].median(), 4) if n else 0,
        "avg_return_winning_trades_pct": round(win["gross_ret"].mean(), 4) if len(win) else 0,
        "avg_return_losing_trades_pct": round(los["gross_ret"].mean(), 4) if len(los) else 0,
        "total_return_fixedbase_pct": round(gp / BASE_POOL * 100, 4),
        "total_pnl_inr": round(gp, 0),
        "avg_capital_deployed_per_trade": round(T["capital_deployed"].mean(), 0) if n else 0,
        "is_reference": "<- X=5 main-strategy threshold" if X == REF_X else "",
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not DIAG.exists():
        raise SystemExit(f"Missing {DIAG.name} — build the ₹5,000–7,500 Cr table first.")
    diag = pd.read_csv(DIAG, parse_dates=["date"],
                       usecols=["symbol", "date", "entry_price_315pm",
                                "return_pct_vs_prev_close", "passes_volume"])

    rows = []
    for X in THRESHOLDS:
        base = base_at_threshold(diag, X)
        T = trades_at_threshold(base)
        m = metrics(T, X)
        rows.append(m)
        print(f"  X>={X}%: {m['n_trades']:,} trades | win {m['win_rate_pct']}% | "
              f"avg {m['avg_return_per_trade_pct']}% | total {m['total_return_fixedbase_pct']}%")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUTDIR / "return_threshold_sweep.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "return_threshold_sweep.xlsx", engine="openpyxl") as w:
        tbl.to_excel(w, sheet_name="threshold_sweep", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 128)
    print("RETURN-THRESHOLD SWEEP — ₹5,000–7,500 Cr variant  (gross; pool-split, cap ₹1L/trade)")
    print("  X=5 is the original main-strategy threshold (reference row).")
    print("=" * 128)
    print(tbl.to_string(index=False))

    # ── chart (a): total return by threshold, n annotated ──
    fig, ax = plt.subplots(figsize=(10, 6))
    x = tbl["return_threshold_pct"].values
    y = tbl["total_return_fixedbase_pct"].values
    colors = ["#b2182b" if xi == REF_X else "#1f4e79" for xi in x]
    bars = ax.bar(x, y, color=colors, edgecolor="white")
    for b, nt in zip(bars, tbl["n_trades"].values):
        ax.annotate(f"n={nt}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xlabel("daily return threshold  X  (>= X%)")
    ax.set_ylabel("total return (fixed ₹5L base) %")
    ax.set_title("Total return vs entry return-threshold — ₹5,000–7,500 Cr variant\n"
                 "(red = X=5 main-strategy reference; bar label = n_trades)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTDIR / "total_return_by_threshold.png", dpi=130); plt.close(fig)

    # ── chart (b): dual axis n_trades vs win_rate ──
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x, tbl["n_trades"].values, "o-", color="#1f4e79", lw=2, label="n_trades")
    ax1.set_xlabel("daily return threshold  X  (>= X%)")
    ax1.set_ylabel("n_trades", color="#1f4e79")
    ax1.tick_params(axis="y", labelcolor="#1f4e79")
    ax2 = ax1.twinx()
    ax2.plot(x, tbl["win_rate_pct"].values, "s--", color="#b2182b", lw=2, label="win_rate_pct")
    ax2.plot(x, tbl["avg_return_per_trade_pct"].values * 100, "^:", color="#2ca25f", lw=2,
             label="avg_return_per_trade_pct ×100")
    ax2.set_ylabel("win rate % / avg return ×100", color="#b2182b")
    ax2.tick_params(axis="y", labelcolor="#b2182b")
    ax1.axvline(REF_X, color="grey", ls=":", alpha=0.6)
    ax1.set_title("Trade count vs quality as the threshold tightens — ₹5,000–7,500 Cr variant",
                  fontweight="bold")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTDIR / "count_vs_quality_by_threshold.png", dpi=130); plt.close(fig)

    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
