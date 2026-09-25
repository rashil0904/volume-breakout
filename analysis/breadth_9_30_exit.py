# -*- coding: utf-8 -*-
"""
breadth_9_30_exit.py
====================
Smallcap-100-breadth-triggered 9:30 exit overlay on the MAIN strategy
(mcap ₹1,500-5,000 Cr, LB36/VM6, 3:15 entry, +5%, conditional-split 9:45/12:00 + 14% target).

Reuses:
  - baseline trades from final_performance_report.build_trades() (NOT recomputed)
  - Smallcap 100 opening advance-decline series (results/smallcap_breadth/smallcap100_opening_ad_daily.csv),
    column decline_pct (9:30 opening breadth), keyed by trading date.

RULE (swept X in {50,55,60,65,70,75,80}, on decline_pct):
  On any EXIT DAY where Smallcap 100 decline_pct > X (STRICT), force-exit every position
  exiting that day at the stock's OWN 9:30 candle open, exit_type "breadth_9_30_exit",
  overriding the normal 9:45/12:00/14%-target logic. Trades on days with decline_pct <= X
  keep their baseline exit unchanged. 9:30 is earlier than the normal 9:45 first exit.

Metrics gross & net@0.23% (expense = capital_deployed × 0.0023). Fixed ₹5L base.
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

BASE_POOL = 500_000
EXPENSE = 0.0023
HM_930 = 570
THRESHOLDS = [50, 55, 60, 65, 70, 75, 80]
BREADTH_CSV = rb.RESULTS / "smallcap_breadth" / "smallcap100_opening_ad_daily.csv"
OUTDIR = rb.RESULTS / "breadth_9_30_exit"


def metrics(df):
    """Full gross & net@0.23% metrics over a trade frame (gross_ret, capital_deployed present)."""
    n = len(df)
    cap = df["capital_deployed"].values
    gret = df["gross_ret"].values
    gpnl = cap * gret / 100.0
    npnl = gpnl - cap * EXPENSE
    nret = gret - EXPENSE * 100
    gw = gpnl > 0; nl_g = ~gw
    nw = npnl > 0; nl_n = ~nw
    def sm(a): return round(float(a.mean()), 4) if len(a) else float("nan")
    return {
        "n_trades": n,
        "gross_total_return_fixedbase_pct": round(float(gpnl.sum()) / BASE_POOL * 100, 4),
        "gross_total_pnl_inr": round(float(gpnl.sum()), 0),
        "net_total_return_fixedbase_pct": round(float(npnl.sum()) / BASE_POOL * 100, 4),
        "net_total_pnl_inr": round(float(npnl.sum()), 0),
        "gross_win_rate_pct": round(float(gw.mean()) * 100, 2),
        "net_win_rate_pct": round(float(nw.mean()) * 100, 2),
        "gross_avg_return_per_trade_pct": round(float(gret.mean()), 4),
        "net_avg_return_per_trade_pct": round(float(nret.mean()), 4),
        "gross_median_return_per_trade_pct": round(float(np.median(gret)), 4),
        "net_median_return_per_trade_pct": round(float(np.median(nret)), 4),
        "gross_avg_return_winning_pct": sm(gret[gw]),
        "gross_avg_return_losing_pct": sm(gret[nl_g]),
        "net_avg_return_winning_pct": sm(nret[nw]),
        "net_avg_return_losing_pct": sm(nret[nl_n]),
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── baseline trades (reused, not recomputed) ──
    print("Building baseline trades via final_performance_report.build_trades() …")
    T = fpr.build_trades()
    T = T[["symbol", "entry_date", "entry_price", "shares", "capital_deployed",
           "exit_date", "exit_price", "exit_type", "gross_ret"]].copy().reset_index(drop=True)

    # ── each trade's OWN 9:30 open on its exit day (force-exit price) ──
    print("Fetching each trade's exit-day 9:30 open …")
    base = fpr.ets.load_base_positions()
    opens, _highs, exit_days = fpr.fetch_ohlc_and_exitday(base)
    open930 = opens[:, fpr.pts.HCOL[HM_930]]
    syms = base["symbol"].values
    edates = [pd.Timestamp(d).date() for d in base["date"].values]
    map930 = {(syms[i], edates[i]): open930[i] for i in range(len(base))}
    T["open_930_exit"] = [map930.get((s, d), np.nan) for s, d in zip(T["symbol"], T["entry_date"])]

    # ── Smallcap 100 9:30 decline_pct by exit date ──
    br = pd.read_csv(BREADTH_CSV)
    br["date"] = pd.to_datetime(br["date"]).dt.date
    decl_map = dict(zip(br["date"], br["decline_pct"]))
    T["exit_decl_pct"] = T["exit_date"].map(lambda d: decl_map.get(d, np.nan))
    no_breadth = int(T["exit_decl_pct"].isna().sum())
    print(f"  trades: {len(T)} | exit-days without a breadth reading: {no_breadth}")

    base_gret = T["gross_ret"].values.copy()          # immutable baseline per-trade return

    rows = [{"scenario": "baseline (no breadth exit)", "X_decline_pct": np.nan,
             "exit_days_triggered": 0, "trades_force_exited": 0,
             "triggered_no_930_data": 0, **metrics(T)}]
    paired_rows = []
    per_x_frames = {}

    for X in THRESHOLDS:
        df = T.copy()
        trig = (df["exit_decl_pct"] > X) & df["open_930_exit"].notna()          # strict >
        trig_no930 = (df["exit_decl_pct"] > X) & df["open_930_exit"].isna()     # would trigger but no 9:30 candle
        # apply force-exit
        df.loc[trig, "gross_ret"] = ((df.loc[trig, "open_930_exit"] - df.loc[trig, "entry_price"])
                                     / df.loc[trig, "entry_price"] * 100.0)
        df.loc[trig, "exit_price"] = df.loc[trig, "open_930_exit"]
        df.loc[trig, "exit_type"] = "breadth_9_30_exit"
        per_x_frames[X] = df

        n_days = df.loc[trig, "exit_date"].nunique()
        rows.append({"scenario": f"breadth_exit X>{X}", "X_decline_pct": X,
                     "exit_days_triggered": int(n_days), "trades_force_exited": int(trig.sum()),
                     "triggered_no_930_data": int(trig_no930.sum()), **metrics(df)})

        # ── paired comparison on the force-exited subset ──
        sub = df[trig]
        if len(sub):
            cap = sub["capital_deployed"].values
            new_ret = sub["gross_ret"].values                     # 9:30 exit
            old_ret = base_gret[trig.values]                      # same trades, baseline exit
            new_pnl = cap * new_ret / 100.0
            old_pnl = cap * old_ret / 100.0
            paired_rows.append({
                "X_decline_pct": X, "n_forced": int(trig.sum()),
                "subset_avg_ret_9_30_exit_pct": round(float(new_ret.mean()), 4),
                "subset_avg_ret_baseline_pct": round(float(old_ret.mean()), 4),
                "subset_avg_ret_delta_pct": round(float((new_ret - old_ret).mean()), 4),
                "subset_gross_pnl_9_30_exit_inr": round(float(new_pnl.sum()), 0),
                "subset_gross_pnl_baseline_inr": round(float(old_pnl.sum()), 0),
                "subset_gross_pnl_delta_inr": round(float((new_pnl - old_pnl).sum()), 0),
                "subset_winrate_9_30_exit_pct": round(float((new_ret > 0).mean()) * 100, 2),
                "subset_winrate_baseline_pct": round(float((old_ret > 0).mean()) * 100, 2),
                "verdict": "9:30 exit PROTECTED" if (new_pnl - old_pnl).sum() > 0 else "9:30 exit HURT",
            })
        else:
            paired_rows.append({"X_decline_pct": X, "n_forced": 0, "verdict": "no trades triggered"})

    table = pd.DataFrame(rows)
    paired = pd.DataFrame(paired_rows)

    # ── chart: net total return vs X, baseline line ──
    base_net = table.iloc[0]["net_total_return_fixedbase_pct"]
    tx = table[table["X_decline_pct"].notna()]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(tx["X_decline_pct"], tx["net_total_return_fixedbase_pct"], "o-", color="#1f77b4",
            label="net total return with breadth 9:30 exit")
    ax.axhline(base_net, ls="--", color="#d62728", label=f"baseline (no breadth exit) = {base_net:.2f}%")
    best = tx.loc[tx["net_total_return_fixedbase_pct"].idxmax()]
    ax.axvline(best["X_decline_pct"], ls=":", color="#2ca02c", alpha=.6)
    ax.set_xlabel("Smallcap 100 decline_pct threshold X (force-exit at 9:30 when decline_pct > X)")
    ax.set_ylabel("net total_return_fixedbase (%)")
    ax.set_title("Breadth-triggered 9:30 exit vs baseline (main strategy)")
    ax.set_xticks(THRESHOLDS); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUTDIR / "breadth_9_30_exit_net_vs_X.png", dpi=120); plt.close(fig)

    with pd.ExcelWriter(OUTDIR / "breadth_9_30_exit.xlsx", engine="openpyxl") as w:
        table.to_excel(w, sheet_name="threshold_sweep", index=False)
        paired.to_excel(w, sheet_name="paired_subset_comparison", index=False)
    table.to_csv(OUTDIR / "breadth_9_30_exit_sweep.csv", index=False)

    pd.set_option("display.width", 260)
    show = ["scenario", "X_decline_pct", "exit_days_triggered", "trades_force_exited", "n_trades",
            "gross_total_return_fixedbase_pct", "net_total_return_fixedbase_pct", "net_total_pnl_inr",
            "gross_win_rate_pct", "net_avg_return_per_trade_pct", "net_median_return_per_trade_pct",
            "gross_avg_return_winning_pct", "gross_avg_return_losing_pct"]
    print("\n" + "=" * 150)
    print("BREADTH-TRIGGERED 9:30 EXIT SWEEP — main strategy (Smallcap 100 decline_pct)")
    print("=" * 150)
    print(table[show].to_string(index=False))
    print("\n--- PAIRED subset (force-exited trades: 9:30 exit vs their baseline exit) ---")
    print(paired.to_string(index=False))
    print(f"\nBaseline net total = {base_net:.4f}% | best breadth-X net = "
          f"{best['net_total_return_fixedbase_pct']:.4f}% at X>{int(best['X_decline_pct'])} "
          f"({'BEATS' if best['net_total_return_fixedbase_pct'] > base_net else 'does NOT beat'} baseline)")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
