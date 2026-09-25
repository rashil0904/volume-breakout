# -*- coding: utf-8 -*-
"""
selective_short_by_long_bucket.py
=================================
SELECTIVE double-down: bucket the SHORTED trades (early_target_pre_t1,
early_target_between_t1_t2, positive_at_t1) by LONG-leg return (1% steps) and report the
short-leg performance within each. Excludes exit_at_t2_no_target (long-only, no short).

short_return_pct = (short_exit − 3pm)/short_exit × 100  (positive = price fell = profitable
short). +14% target trades get their own bucket. Reuses the shared build; no recompute.
"""
import sys, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import double_down_short_variant as dd

OUTDIR = rb.RESULTS / "selective_double_down"
BASE_POOL = dd.BASE_POOL
TARGET_TYPES = {"early_target_pre_t1", "early_target_between_t1_t2"}
SHORT_TYPES = TARGET_TYPES | {"positive_at_t1"}
SMALL = 15


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = dd.build_long_with_ohlc()
    T = T[T["exit_type"].isin(SHORT_TYPES)].copy()             # shorted only
    T["long_ret"] = (T["long_exit_price"] - T["entry_price"]) / T["entry_price"] * 100
    T["short_pnl"] = T["shares"] * (T["long_exit_price"] - T["price_3pm_open"])
    T["short_ret"] = (T["long_exit_price"] - T["price_3pm_open"]) / T["long_exit_price"] * 100
    T["long_pnl"] = T["shares"] * (T["long_exit_price"] - T["entry_price"])
    T["combined_ret"] = (T["long_pnl"] + T["short_pnl"]) / T["capital_deployed"] * 100
    T["is_target"] = T["exit_type"].isin(TARGET_TYPES)

    def bucket(row):
        if row["is_target"]:
            return "+14% (target)"
        k = min(math.floor(max(row["long_ret"], 0)), 13)
        return f"{k} to {k+1}"
    T["bucket"] = T.apply(bucket, axis=1)

    order = [f"{k} to {k+1}" for k in range(0, 14)] + ["+14% (target)"]
    rows = []
    for b in order:
        g = T[T["bucket"] == b]
        if len(g) == 0:
            continue
        rows.append({
            "long_return_bucket": b, "n_trades": len(g),
            "n_positive_at_t1": int((~g["is_target"]).sum()), "n_target": int(g["is_target"].sum()),
            "avg_long_return_pct": round(g["long_ret"].mean(), 4),
            "median_long_return_pct": round(g["long_ret"].median(), 4),
            "avg_short_return_pct": round(g["short_ret"].mean(), 4),
            "median_short_return_pct": round(g["short_ret"].median(), 4),
            "short_win_rate_pct": round((g["short_pnl"] > 0).mean() * 100, 2),
            "avg_combined_return_pct": round(g["combined_ret"].mean(), 4),
            "short_total_pnl_inr": round(g["short_pnl"].sum(), 0),
            "small_sample_flag": "n_trades<15" if len(g) < SMALL else "",
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUTDIR / "selective_short_by_long_bucket.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "selective_short_by_long_bucket.xlsx", engine="openpyxl") as w:
        tbl.to_excel(w, sheet_name="short_by_long_return", index=False)

    # chart
    fig, ax1 = plt.subplots(figsize=(13, 6))
    x = np.arange(len(tbl))
    colors = ["#8856a7" if b == "+14% (target)" else ("#2ca25f" if v > 0 else "#b2182b")
              for b, v in zip(tbl["long_return_bucket"], tbl["avg_short_return_pct"])]
    ax1.bar(x, tbl["avg_short_return_pct"], color=colors, edgecolor="white")
    ax1.axhline(0, color="black", lw=0.7); ax1.set_ylabel("avg short-leg return %")
    ax1.set_xticks(x); ax1.set_xticklabels(tbl["long_return_bucket"], rotation=45, ha="right", fontsize=8)
    for i, nt in enumerate(tbl["n_trades"]):
        ax1.annotate(f"{nt}", (i, tbl['avg_short_return_pct'].iloc[i]), textcoords="offset points",
                     xytext=(0, 3 if tbl['avg_short_return_pct'].iloc[i] >= 0 else -11), ha="center", fontsize=7)
    ax2 = ax1.twinx()
    ax2.plot(x, tbl["short_win_rate_pct"], "o-", color="#1f4e79", lw=2)
    ax2.axhline(50, color="#1f4e79", ls=":", alpha=0.5)
    ax2.set_ylabel("short win rate %", color="#1f4e79"); ax2.tick_params(axis="y", labelcolor="#1f4e79")
    ax1.set_xlabel("LONG-leg return bucket (shorted trades only)")
    ax1.set_title("Selective double-down — short-leg performance by long-return bucket\n"
                  "(purple = +14% target; bar = avg short return; line = win rate; label = n)", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUTDIR / "selective_short_by_long_bucket.png", dpi=130); plt.close(fig)

    pd.set_option("display.width", 230)
    print("=" * 150)
    print("SELECTIVE DOUBLE-DOWN — SHORT-LEG BY LONG-RETURN BUCKET (shorted trades only)")
    print("=" * 150)
    print(tbl.to_string(index=False))

    # ── target vs positive_at_t1 head-to-head ──
    tg = T[T["is_target"]]; po = T[~T["is_target"]]
    print("\n--- TARGET vs POSITIVE_AT_T1 (short leg) ---")
    for name, g in [("+14% target", tg), ("positive_at_t1 (all)", po),
                    ("positive_at_t1 >=7% long", po[po["long_ret"] >= 7])]:
        if len(g):
            print(f"  {name:28s}: n={len(g):>4} | short win {round((g['short_pnl']>0).mean()*100,2):>6}% "
                  f"| avg short {round(g['short_ret'].mean(),4):>8}% | total ₹{g['short_pnl'].sum():>10,.0f}")
    print(f"\nSaved -> {OUTDIR / 'selective_short_by_long_bucket.xlsx'}")


if __name__ == "__main__":
    main()
