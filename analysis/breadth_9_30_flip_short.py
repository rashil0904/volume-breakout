# -*- coding: utf-8 -*-
"""
breadth_9_30_flip_short.py
==========================
Extends breadth_9_30_exit.py with an intraday SHORT overlay on weak-breadth days.

Base long strategy: mcap ₹1,500-5,000 Cr, LB36/VM6, 3:15 entry, +5%, conditional-split
9:45/12:00 + 14% target, ₹5L/₹1L. Reuses final_performance_report.build_trades() (NOT
recomputed) + the Smallcap 100 opening decline_pct series.

On each EXIT DAY, branch on Smallcap 100 9:30 decline_pct (X in {50..80}, STRICT >):
  decline_pct <= X : long follows NORMAL 9:45/12:00/target exit. No short.
  decline_pct >  X : flip every morning-open position —
      long_pnl  = shares × (open_930 − entry_price)      (long exited at its own 9:30 open)
      short_pnl = shares × (open_930 − open_300pm)        (short 9:30 → 3:00pm, + if it falls)

Configs compared per X (+ baseline):
  BASELINE : no breadth logic, all ride normal exits.
  CONFIG 1 : exit-only  — weak days exit long at 9:30, NO short. combined = long_pnl.
  CONFIG 2 : flip       — weak days exit long at 9:30 AND short to 3pm. combined = long_pnl+short_pnl.

Costs: long 0.23% of capital_deployed; short 0.10% of short notional (shares×open_930).
Fixed ₹5L base; total_return_fixedbase = Σ combined_pnl / 500000 × 100.
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
LONG_COST, SHORT_COST = 0.0023, 0.0010
HM_930, HM_300 = 570, 900                      # 09:30, 15:00 (3:00pm)
THRESHOLDS = [50, 55, 60, 65, 70, 75, 80]
BREADTH_CSV = rb.RESULTS / "smallcap_breadth" / "smallcap100_opening_ad_daily.csv"
OUTDIR = rb.RESULTS / "breadth_9_30_flip_short"


def metrics(gross_pnl, cost, cap):
    """gross & net metrics; net = gross_pnl - cost (per-trade, cost already blends long/short)."""
    n = len(gross_pnl)
    gret = gross_pnl / cap * 100.0
    net_pnl = gross_pnl - cost
    nret = net_pnl / cap * 100.0
    gw = gross_pnl > 0; nw = net_pnl > 0
    def sm(a): return round(float(a.mean()), 4) if len(a) else float("nan")
    return {
        "n_trades": n,
        "gross_total_return_fixedbase_pct": round(float(gross_pnl.sum()) / BASE_POOL * 100, 4),
        "gross_total_pnl_inr": round(float(gross_pnl.sum()), 0),
        "net_total_return_fixedbase_pct": round(float(net_pnl.sum()) / BASE_POOL * 100, 4),
        "net_total_pnl_inr": round(float(net_pnl.sum()), 0),
        "gross_win_rate_pct": round(float(gw.mean()) * 100, 2),
        "net_win_rate_pct": round(float(nw.mean()) * 100, 2),
        "gross_avg_return_per_trade_pct": round(float(gret.mean()), 4),
        "net_avg_return_per_trade_pct": round(float(nret.mean()), 4),
        "gross_median_return_per_trade_pct": round(float(np.median(gret)), 4),
        "net_median_return_per_trade_pct": round(float(np.median(nret)), 4),
        "gross_avg_return_winning_pct": sm(gret[gw]),
        "gross_avg_return_losing_pct": sm(gret[~gw]),
        "net_avg_return_winning_pct": sm(nret[nw]),
        "net_avg_return_losing_pct": sm(nret[~nw]),
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Building baseline trades (reused, not recomputed) …")
    T = fpr.build_trades()
    T = T[["symbol", "entry_date", "entry_price", "shares", "capital_deployed",
           "exit_date", "gross_ret"]].copy().reset_index(drop=True)

    print("Fetching each trade's exit-day 9:30 & 3:00pm opens …")
    base = fpr.ets.load_base_positions()
    opens, _highs, _ed = fpr.fetch_ohlc_and_exitday(base)
    o930 = opens[:, fpr.pts.HCOL[HM_930]]
    o300 = opens[:, fpr.pts.HCOL[HM_300]]
    syms = base["symbol"].values
    edates = [pd.Timestamp(d).date() for d in base["date"].values]
    m930 = {(syms[i], edates[i]): o930[i] for i in range(len(base))}
    m300 = {(syms[i], edates[i]): o300[i] for i in range(len(base))}
    T["open_930"] = [m930.get((s, d), np.nan) for s, d in zip(T["symbol"], T["entry_date"])]
    T["open_300"] = [m300.get((s, d), np.nan) for s, d in zip(T["symbol"], T["entry_date"])]

    br = pd.read_csv(BREADTH_CSV); br["date"] = pd.to_datetime(br["date"]).dt.date
    decl_map = dict(zip(br["date"], br["decline_pct"]))
    T["exit_decl"] = T["exit_date"].map(lambda d: decl_map.get(d, np.nan))
    print(f"  trades: {len(T)} | exit-days w/o breadth reading: {int(T['exit_decl'].isna().sum())}")

    entry = T["entry_price"].values.astype(float)
    shares = T["shares"].values.astype(float)
    cap = T["capital_deployed"].values.astype(float)
    op930 = T["open_930"].values.astype(float)
    op300 = T["open_300"].values.astype(float)
    decl = T["exit_decl"].values.astype(float)

    base_gross_pnl = cap * T["gross_ret"].values / 100.0            # baseline long, normal exits
    long930_pnl = shares * (op930 - entry)                          # long exited at 9:30
    short_pnl = shares * (op930 - op300)                            # short 9:30 -> 3pm
    short_notional = shares * op930

    # ── baseline row ──
    rows = [{"config": "baseline", "X_decline_pct": np.nan, "n_days_triggered": 0,
             "n_trades_affected": 0, **metrics(base_gross_pnl, cap * LONG_COST, cap)}]
    paired_rows, shortleg_rows = [], []
    cfg_curves = {"exit_only": {}, "flip_to_short": {}}

    for X in THRESHOLDS:
        trig = (decl > X) & ~np.isnan(op930)                       # strict >, need a 9:30 open
        short_ok = trig & ~np.isnan(op300)                         # short also needs 3pm open
        n_days = T.loc[trig, "exit_date"].nunique()
        n_aff = int(trig.sum())

        # CONFIG 1 — exit only (long exited at 9:30, no short)
        g1 = np.where(trig, long930_pnl, base_gross_pnl)
        c1 = cap * LONG_COST
        rows.append({"config": "exit_only", "X_decline_pct": X, "n_days_triggered": int(n_days),
                     "n_trades_affected": n_aff, **metrics(g1, c1, cap)})
        cfg_curves["exit_only"][X] = rows[-1]["net_total_return_fixedbase_pct"]

        # CONFIG 2 — flip to short (long exited at 9:30 + short 9:30->3pm)
        sp = np.where(short_ok, short_pnl, 0.0)                    # if no 3pm open, exit-only fallback
        g2 = np.where(trig, long930_pnl + sp, base_gross_pnl)
        c2 = cap * LONG_COST + np.where(short_ok, short_notional * SHORT_COST, 0.0)
        rows.append({"config": "flip_to_short", "X_decline_pct": X, "n_days_triggered": int(n_days),
                     "n_trades_affected": n_aff, **metrics(g2, c2, cap)})
        cfg_curves["flip_to_short"][X] = rows[-1]["net_total_return_fixedbase_pct"]

        # ── paired subset: same triggered trades, 3 outcomes ──
        m = trig
        cap_s = cap[m]
        base_s = base_gross_pnl[m]; c1_s = long930_pnl[m]; c2_s = (long930_pnl + np.where(short_ok, short_pnl, 0.0))[m]
        paired_rows.append({
            "X_decline_pct": X, "n_affected": n_aff,
            "baseline_hold_avg_ret_pct": round(float((base_s / cap_s * 100).mean()), 4),
            "cfg1_exit930_avg_ret_pct": round(float((c1_s / cap_s * 100).mean()), 4),
            "cfg2_flip_avg_ret_pct": round(float((c2_s / cap_s * 100).mean()), 4),
            "baseline_hold_gross_pnl_inr": round(float(base_s.sum()), 0),
            "cfg1_exit930_gross_pnl_inr": round(float(c1_s.sum()), 0),
            "cfg2_flip_gross_pnl_inr": round(float(c2_s.sum()), 0),
            "cfg1_vs_baseline_delta_inr": round(float(c1_s.sum() - base_s.sum()), 0),
            "cfg2_vs_baseline_delta_inr": round(float(c2_s.sum() - base_s.sum()), 0),
            "cfg2_vs_cfg1_delta_inr": round(float(c2_s.sum() - c1_s.sum()), 0),
        })

        # ── short-leg standalone detail (on flipped trades) ──
        sm_ = short_ok
        if sm_.any():
            spx = short_pnl[sm_]; snot = short_notional[sm_]
            sret = (op930[sm_] - op300[sm_]) / op930[sm_] * 100.0
            snet = spx - snot * SHORT_COST
            shortleg_rows.append({
                "X_decline_pct": X, "n_shorts": int(sm_.sum()),
                "short_gross_pnl_inr": round(float(spx.sum()), 0),
                "short_net_pnl_inr": round(float(snet.sum()), 0),
                "short_avg_return_pct": round(float(sret.mean()), 4),
                "short_median_return_pct": round(float(np.median(sret)), 4),
                "short_win_rate_pct": round(float((spx > 0).mean()) * 100, 2),
                "short_total_return_fixedbase_pct": round(float(spx.sum()) / BASE_POOL * 100, 4),
                "short_net_total_return_fixedbase_pct": round(float(snet.sum()) / BASE_POOL * 100, 4),
            })

    master = pd.DataFrame(rows)
    paired = pd.DataFrame(paired_rows)
    shortleg = pd.DataFrame(shortleg_rows)
    base_net = master.iloc[0]["net_total_return_fixedbase_pct"]
    master_sorted = master.sort_values("net_total_return_fixedbase_pct", ascending=False).reset_index(drop=True)

    # ── chart: Config1 & Config2 net vs X, baseline line ──
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    xs = THRESHOLDS
    ax.plot(xs, [cfg_curves["exit_only"][x] for x in xs], "o-", color="#1f77b4", label="Config 1: exit-only")
    ax.plot(xs, [cfg_curves["flip_to_short"][x] for x in xs], "s-", color="#2ca02c", label="Config 2: flip-to-short")
    ax.axhline(base_net, ls="--", color="#d62728", label=f"baseline (no breadth) = {base_net:.2f}%")
    ax.set_xlabel("Smallcap 100 decline_pct threshold X (act when decline_pct > X)")
    ax.set_ylabel("net total_return_fixedbase (%)")
    ax.set_title("Weak-breadth overlay: exit-only vs flip-to-short vs baseline")
    ax.set_xticks(xs); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUTDIR / "breadth_flip_net_vs_X.png", dpi=120); plt.close(fig)

    with pd.ExcelWriter(OUTDIR / "breadth_9_30_flip_short.xlsx", engine="openpyxl") as w:
        master_sorted.to_excel(w, sheet_name="master_comparison", index=False)
        paired.to_excel(w, sheet_name="breadth_triggered_paired", index=False)
        shortleg.to_excel(w, sheet_name="short_leg_detail", index=False)

    pd.set_option("display.width", 300)
    show = ["config", "X_decline_pct", "n_days_triggered", "n_trades_affected", "n_trades",
            "gross_total_return_fixedbase_pct", "net_total_return_fixedbase_pct", "net_total_pnl_inr",
            "gross_win_rate_pct", "net_avg_return_per_trade_pct", "net_median_return_per_trade_pct"]
    print("\n" + "=" * 170)
    print("MASTER — baseline + exit-only + flip-to-short (sorted by net total return)")
    print("=" * 170)
    print(master_sorted[show].to_string(index=False))
    print("\n--- PAIRED subset (same triggered trades: hold vs exit@9:30 vs flip) ---")
    print(paired.to_string(index=False))
    print("\n--- SHORT-LEG standalone (does shorting 9:30->3pm on weak days make money?) ---")
    print(shortleg.to_string(index=False))
    b1 = max(cfg_curves["exit_only"].items(), key=lambda kv: kv[1])
    b2 = max(cfg_curves["flip_to_short"].items(), key=lambda kv: kv[1])
    print(f"\nBaseline net = {base_net:.4f}%")
    print(f"Best Config 1 (exit-only)   : X>{b1[0]} -> net {b1[1]:.4f}%  ({'BEATS' if b1[1]>base_net else 'below'} baseline)")
    print(f"Best Config 2 (flip-to-short): X>{b2[0]} -> net {b2[1]:.4f}%  ({'BEATS' if b2[1]>base_net else 'below'} baseline)")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
