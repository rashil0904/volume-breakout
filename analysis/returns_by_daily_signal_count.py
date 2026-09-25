# -*- coding: utf-8 -*-
"""
returns_by_daily_signal_count.py
================================
Groups the LOCKED baseline strategy's trades by how many stocks signalled on their
ENTRY day (signal crowding / selectivity), and reports returns per group.

Locked config (no sweeping) — reused verbatim from final_performance_report.build_trades():
  Entry  : 15:15 open, lookback 36d, volume >= 6x 36-day trailing avg, day move >= +5%,
           market-cap band Rs 1,500-5,000 Cr.
  Exit   : conditional_split_best_t2 (t1=09:45, t2=12:00) + 14% profit-target overlay
           (exit_types: early_target_pre_t1 / positive_at_t1 /
            early_target_between_t1_t2 / exit_at_t2_no_target).

daily_signal_count(trade) = number of trades that shared its entry day. In the baseline,
this count is exactly what drives position sizing (n<=5 -> Rs 1L each; n>5 -> Rs 5L/n),
so every qualifying signal is a taken trade — the group label reflects ALL qualifying
signals that day, per ambiguity (b).
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
import final_performance_report as fpr

OUTDIR = rb.RESULTS / "signal_count_groups"
BASE_POOL = fpr.BASE_POOL                     # 500,000
SMALL_DAYS, SMALL_TRADES = 5, 15              # small-sample flags (ambiguity/spec point 4)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── reuse the exact locked-config trade list (no recompute of entries/exits) ──
    T = fpr.build_trades()
    print(f"Locked-config trades: {len(T):,}  over {T['entry_date'].nunique():,} entry days")

    # ── label every trade with the count of trades sharing its ENTRY day ──
    day_cnt = T.groupby("entry_date")["symbol"].transform("size")
    T = T.assign(daily_signal_count=day_cnt.values)

    # per-day rollup (needed for avg_return_per_day and n_days per group)
    day = (T.groupby("entry_date")
             .agg(daily_signal_count=("daily_signal_count", "first"),
                  day_pnl=("gross_pnl", "sum"),
                  day_cap=("capital_deployed", "sum"),
                  day_net_pnl=("net_pnl", "sum"))
             .reset_index())
    # per-day % return on that day's deployed capital (each day weighted equally)
    day["day_ret_pct"] = day["day_pnl"] / day["day_cap"] * 100

    max_cnt = int(T["daily_signal_count"].max())

    rows = []
    for c in range(1, max_cnt + 1):
        grp = T[T["daily_signal_count"] == c]
        if grp.empty:
            continue
        dgrp = day[day["daily_signal_count"] == c]
        n_days = len(dgrp)
        n_trades = len(grp)
        flags = []
        if n_days < SMALL_DAYS:
            flags.append(f"n_days<{SMALL_DAYS}")
        if n_trades < SMALL_TRADES:
            flags.append(f"n_trades<{SMALL_TRADES}")
        rows.append({
            "daily_signal_count": c,
            "n_days": n_days,
            "n_trades": n_trades,
            "win_rate_pct": round((grp["gross_pnl"] > 0).mean() * 100, 2),
            "avg_return_per_trade_pct": round(grp["gross_ret"].mean(), 4),
            "median_return_per_trade_pct": round(grp["gross_ret"].median(), 4),
            "total_return_fixedbase_pct": round(grp["gross_pnl"].sum() / BASE_POOL * 100, 4),
            "total_pnl_inr": round(grp["gross_pnl"].sum(), 0),
            "avg_return_per_day_pct": round(dgrp["day_ret_pct"].mean(), 4),
            "total_net_pnl_inr": round(grp["net_pnl"].sum(), 0),
            "small_sample_flag": ", ".join(flags) if flags else "",
        })

    tbl = pd.DataFrame(rows).sort_values("daily_signal_count").reset_index(drop=True)

    tbl.to_csv(OUTDIR / "returns_by_daily_signal_count.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "returns_by_daily_signal_count.xlsx", engine="openpyxl") as w:
        tbl.to_excel(w, sheet_name="by_signal_count", index=False)

    # ── prints ──
    pd.set_option("display.width", 220)
    print("\n" + "=" * 120)
    print("RETURNS BY DAILY SIGNAL COUNT  (locked baseline; gross unless noted)")
    print("=" * 120)
    show = tbl[["daily_signal_count", "n_days", "n_trades", "win_rate_pct",
                "avg_return_per_trade_pct", "median_return_per_trade_pct",
                "avg_return_per_day_pct", "total_return_fixedbase_pct",
                "total_pnl_inr", "small_sample_flag"]]
    print(show.to_string(index=False))

    # overall max signals-in-a-day and the date(s) that hit it
    max_days = day[day["daily_signal_count"] == max_cnt]["entry_date"].tolist()
    print(f"\nMax signals in a single day: {max_cnt}")
    print("  occurred on: " + ", ".join(str(d) for d in max_days))

    flagged = tbl[tbl["small_sample_flag"] != ""]
    print(f"\nSmall-sample groups (n_days<{SMALL_DAYS} or n_trades<{SMALL_TRADES}): "
          f"{len(flagged)} of {len(tbl)}")
    if len(flagged):
        print("  counts: " + ", ".join(f"{int(r.daily_signal_count)}"
                                        f"(days={int(r.n_days)},trades={int(r.n_trades)})"
                                        for _, r in flagged.iterrows()))

    # ── bar chart: avg return per trade by signal count, n_trades annotated ──
    fig, ax = plt.subplots(figsize=(max(11, len(tbl) * 0.5), 6))
    x = tbl["daily_signal_count"].values
    y = tbl["avg_return_per_trade_pct"].values
    small = tbl["small_sample_flag"].values != ""
    colors = ["#b0b0b0" if s else "#1f4e79" for s in small]
    bars = ax.bar(x, y, color=colors, edgecolor="white", linewidth=0.5)
    for b, nt, s in zip(bars, tbl["n_trades"].values, small):
        ax.annotate(f"{nt}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3 if b.get_height() >= 0 else -11),
                    ha="center", fontsize=7, color="#888" if s else "#333")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("daily signal count (stocks signalling together on entry day)")
    ax.set_ylabel("avg return per trade %  (gross)")
    ax.set_title("Return per trade vs signal crowding — locked baseline\n"
                 "(bar label = n_trades; grey = small sample)", fontweight="bold")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "avg_return_by_signal_count.png", dpi=130)
    plt.close(fig)

    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
