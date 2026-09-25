# -*- coding: utf-8 -*-
"""
double_down_short_by_long_bucket.py
===================================
Cross-tab: bucket double-down trades by LONG-leg return (1% steps), report SHORT-leg
performance within each bucket. Answers whether the exit-day short is more profitable
after strong long exits (stock ran up, room to fade) or weak/negative ones.

short_return_pct = (short_exit_price − price_3pm) / short_exit_price × 100   (per user;
positive = price FELL exit->3pm = profitable short). short_exit_price = long_exit_price.
+14% target trades (early_target_*) get their own bucket. Reuses the shared build.
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

OUTDIR = dd.OUTDIR
BASE_POOL = dd.BASE_POOL
TARGET_TYPES = {"early_target_pre_t1", "early_target_between_t1_t2"}
LOW_CUT = -5
SMALL = 15


def main():
    T = dd.build_long_with_ohlc()
    T["long_ret"] = (T["long_exit_price"] - T["entry_price"]) / T["entry_price"] * 100
    T["short_pnl"] = T["shares"] * (T["long_exit_price"] - T["price_3pm_open"])
    T["short_ret"] = (T["long_exit_price"] - T["price_3pm_open"]) / T["long_exit_price"] * 100  # user def
    T["long_pnl"] = T["shares"] * (T["long_exit_price"] - T["entry_price"])
    T["combined_pnl"] = T["long_pnl"] + T["short_pnl"]
    T["combined_ret"] = T["combined_pnl"] / T["capital_deployed"] * 100
    T["move_3pm_pct"] = (T["price_3pm_open"] - T["long_exit_price"]) / T["long_exit_price"] * 100

    def bucket(row):
        if row["exit_type"] in TARGET_TYPES:
            return "+14% (target)"
        r = row["long_ret"]
        if r < LOW_CUT:
            return "< -5%"
        k = math.floor(r)
        k = min(k, 13)                         # non-target caps below 14
        return f"{k} to {k+1}"

    T["bucket"] = T.apply(bucket, axis=1)

    # order: < -5, then -5 to -4 ... 13 to 14, then target
    order = ["< -5%"] + [f"{k} to {k+1}" for k in range(-5, 14)] + ["+14% (target)"]
    rows = []
    for b in order:
        g = T[T["bucket"] == b]
        if len(g) == 0:
            continue
        rows.append({
            "long_return_bucket": b, "n_trades": len(g),
            "avg_long_return_pct": round(g["long_ret"].mean(), 4),
            "median_long_return_pct": round(g["long_ret"].median(), 4),
            "avg_short_return_pct": round(g["short_ret"].mean(), 4),
            "median_short_return_pct": round(g["short_ret"].median(), 4),
            "short_win_rate_pct": round((g["short_pnl"] > 0).mean() * 100, 2),
            "avg_combined_return_pct": round(g["combined_ret"].mean(), 4),
            "short_total_pnl_inr": round(g["short_pnl"].sum(), 0),
            "avg_3pm_move_from_exit_pct": round(g["move_3pm_pct"].mean(), 4),
            "small_sample_flag": "n_trades<15" if len(g) < SMALL else "",
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUTDIR / "short_by_long_bucket.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "short_by_long_bucket.xlsx", engine="openpyxl") as w:
        tbl.to_excel(w, sheet_name="short_by_long_return", index=False)

    # ── chart: avg short return (bars) + short win rate (line) ──
    fig, ax1 = plt.subplots(figsize=(14, 6))
    x = np.arange(len(tbl))
    colors = ["#2ca25f" if (v > 0) else "#b2182b" for v in tbl["avg_short_return_pct"]]
    ax1.bar(x, tbl["avg_short_return_pct"], color=colors, edgecolor="white", label="avg short return %")
    ax1.set_ylabel("avg short-leg return %"); ax1.axhline(0, color="black", lw=0.7)
    ax1.set_xticks(x); ax1.set_xticklabels(tbl["long_return_bucket"], rotation=45, ha="right", fontsize=8)
    for i, nt in enumerate(tbl["n_trades"]):
        ax1.annotate(f"{nt}", (i, tbl['avg_short_return_pct'].iloc[i]), textcoords="offset points",
                     xytext=(0, 3 if tbl['avg_short_return_pct'].iloc[i] >= 0 else -11), ha="center", fontsize=6)
    ax2 = ax1.twinx()
    ax2.plot(x, tbl["short_win_rate_pct"], "o-", color="#1f4e79", lw=2, label="short win rate %")
    ax2.axhline(50, color="#1f4e79", ls=":", alpha=0.5)
    ax2.set_ylabel("short win rate %", color="#1f4e79"); ax2.tick_params(axis="y", labelcolor="#1f4e79")
    ax1.set_xlabel("LONG-leg return bucket")
    ax1.set_title("Short-leg performance by long-leg return bucket\n"
                  "(bar = avg short return; line = short win rate; label = n_trades)", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUTDIR / "short_by_long_bucket.png", dpi=130); plt.close(fig)

    pd.set_option("display.width", 230)
    print("=" * 150)
    print("SHORT-LEG PERFORMANCE BY LONG-LEG RETURN BUCKET")
    print("=" * 150)
    print(tbl.to_string(index=False))

    # ── highlight reliable vs drag ──
    big = tbl[tbl["small_sample_flag"] == ""]
    good = big[(big["short_win_rate_pct"] > 50) & (big["avg_short_return_pct"] > 0)]
    bad = big[(big["short_win_rate_pct"] <= 50) | (big["avg_short_return_pct"] <= 0)]
    print("\n--- SHORT RELIABLY PROFITABLE (win>50% AND avg>0, n>=15) ---")
    print("  " + ", ".join(f"[{r.long_return_bucket}: win {r.short_win_rate_pct}%, avg {r.avg_short_return_pct:+.2f}%]"
                           for _, r in good.iterrows()))
    print("\n--- SHORT A DRAG (win<=50% OR avg<=0, n>=15) ---")
    print("  " + ", ".join(f"[{r.long_return_bucket}: win {r.short_win_rate_pct}%, avg {r.avg_short_return_pct:+.2f}%]"
                           for _, r in bad.iterrows()))
    print(f"\nSaved -> {OUTDIR / 'short_by_long_bucket.xlsx'}")


if __name__ == "__main__":
    main()
