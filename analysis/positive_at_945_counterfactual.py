# -*- coding: utf-8 -*-
"""
positive_at_945_counterfactual.py
=================================
Main strategy's positive_at_t1 trades (positive at 09:45, exited 100% at the 09:45 open —
EXCLUDING early_target_pre_t1 which hit +14% by 09:45), bucketed by realized 09:45 return,
with the counterfactual return had each been HELD to the 12:00 exit instead.

Counterfactual respects the 14% target: if the high between 09:45 and 12:00 hit entry*1.14,
the held version caps at +14%; else it rides to the 12:00 open. Reuses fpr's base positions
+ OHLC fetch (same trade set); does not recompute the strategy.

Buckets by trade_return_945 = (open_945 - entry)/entry*100: [0,1.00], (1.00,2.00], ...,
(9.00,10.00], and ">10%".
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
import exit_time_sweep as ets
import profit_target_sweep as pts
import final_performance_report as fpr

OUTDIR = rb.RESULTS / "positive_at_945_cf"
BASE_POOL, TARGET = 500_000, 14.0
HM_0945, HM_1200 = pts.HM_0945, pts.HM_1200


def bucket_label(r):
    if r > 10:
        return ">10%"
    k = max(1, math.ceil(r - 1e-9))          # (k-1, k]; r in [0,1] -> 1
    k = min(k, 10)
    lo = "0" if k == 1 else f"{k-1}.01"
    return f"{lo}-{k}.00"


BUCKET_ORDER = ["0-1.00", "1.01-2.00", "2.01-3.00", "3.01-4.00", "4.01-5.00", "5.01-6.00",
                "6.01-7.00", "7.01-8.00", "8.01-9.00", "9.01-10.00", ">10%"]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = ets.load_base_positions()
    e = base["entry"].values.astype(float)
    sh = base["shares"].values.astype(float)
    cap = base["cap"].values.astype(float)
    print("Fetching next-day OHLC …")
    opens, highs, _ = fpr.fetch_ohlc_and_exitday(base)

    pct_high = (highs - e[:, None]) / e[:, None] * 100
    o945 = opens[:, pts.HCOL[HM_0945]]; o1200 = opens[:, pts.HCOL[HM_1200]]
    ret945 = (o945 - e) / e * 100
    pre_cols = [pts.HCOL[hm] for hm in pts.CANDLE_HMS if hm < HM_0945]
    bet_cols = [pts.HCOL[hm] for hm in pts.CANDLE_HMS if HM_0945 < hm < HM_1200]   # 10:00..11:45

    def wmax(cols):
        sub = pct_high[:, cols]
        return np.where(np.isnan(sub), -np.inf, sub).max(axis=1)
    pre_target = wmax(pre_cols) >= TARGET            # early_target_pre_t1
    p3max = wmax(bet_cols)                            # high between 09:45 and 12:00

    # positive_at_t1: NOT pre-target, valid 09:45 open, positive at 09:45
    pos = (~pre_target) & ~np.isnan(o945) & (ret945 > 0)
    # counterfactual held-to-12:00: cap at +14% if target hit between 09:45-12:00, else 12:00 open
    target_hit = p3max >= TARGET
    held_ret = np.where(target_hit, TARGET, (o1200 - e) / e * 100)
    held_px = np.where(target_hit, e * (1 + TARGET / 100), o1200)
    valid = pos & (target_hit | ~np.isnan(o1200))    # need a determinable held exit

    idx = np.where(valid)[0]
    r945 = ret945[idx]; rheld = held_ret[idx]; th = target_hit[idx]
    ei, shi, capi = e[idx], sh[idx], cap[idx]
    actual_pnl = shi * (o945[idx] - ei)
    held_pnl = shi * (held_px[idx] - ei)
    labels = np.array([bucket_label(r) for r in r945])
    print(f"positive_at_t1 trades (valid counterfactual): {len(idx):,}")

    rows = []
    for b in BUCKET_ORDER:
        m = labels == b
        nb = int(m.sum())
        if nb == 0:
            continue
        a, h = r945[m], rheld[m]
        better = h > a
        rows.append({
            "bucket_945_return_pct": b, "n_trades": nb,
            "avg_actual_945_return_pct": round(a.mean(), 4),
            "avg_held_1200_return_pct": round(h.mean(), 4),
            "delta_avg_held_minus_945": round(h.mean() - a.mean(), 4),
            "median_actual_945_pct": round(np.median(a), 4),
            "median_held_1200_pct": round(np.median(h), 4),
            "median_delta": round(np.median(h - a), 4),
            "pct_holding_better": round(better.mean() * 100, 2),
            "pct_holding_worse": round((~better).mean() * 100, 2),
            "n_hit_14target_in_held": int(th[m].sum()),
            "avg_capital_deployed_per_trade": round(capi[m].mean(), 0),
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUTDIR / "positive_at_945_counterfactual.csv", index=False)

    # overall summary
    tot_actual = actual_pnl.sum(); tot_held = held_pnl.sum()
    overall = pd.DataFrame([{
        "metric": "n_positive_at_t1_trades", "value": len(idx)},
        {"metric": "total_actual_945_pnl_inr", "value": round(tot_actual, 0)},
        {"metric": "total_actual_945_return_fixedbase_pct", "value": round(tot_actual / BASE_POOL * 100, 4)},
        {"metric": "total_counterfactual_held_1200_pnl_inr", "value": round(tot_held, 0)},
        {"metric": "total_counterfactual_held_1200_return_fixedbase_pct", "value": round(tot_held / BASE_POOL * 100, 4)},
        {"metric": "held_minus_actual_pnl_inr", "value": round(tot_held - tot_actual, 0)},
        {"metric": "held_minus_actual_return_pct", "value": round((tot_held - tot_actual) / BASE_POOL * 100, 4)},
        {"metric": "pct_trades_holding_wouldve_been_better", "value": round((rheld > r945).mean() * 100, 2)},
    ])

    with pd.ExcelWriter(OUTDIR / "positive_at_945_counterfactual.xlsx", engine="openpyxl") as w:
        tbl.to_excel(w, sheet_name="by_945_return_bucket", index=False)
        overall.to_excel(w, sheet_name="overall_summary", index=False)

    # ── chart 1: grouped bars actual vs held ──
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(tbl)); wdt = 0.4
    ax.bar(x - wdt/2, tbl["avg_actual_945_return_pct"], wdt, color="#1f4e79", label="avg actual 09:45 exit")
    ax.bar(x + wdt/2, tbl["avg_held_1200_return_pct"], wdt, color="#2ca25f", label="avg held-to-12:00 (14%-capped)")
    for i, nt in enumerate(tbl["n_trades"]):
        ax.annotate(f"n={nt}", (i, max(tbl['avg_actual_945_return_pct'].iloc[i], tbl['avg_held_1200_return_pct'].iloc[i])),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(tbl["bucket_945_return_pct"], rotation=35, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.6); ax.set_xlabel("realized 09:45 return bucket")
    ax.set_ylabel("avg return %"); ax.legend()
    ax.set_title("positive_at_t1 — actual 09:45 exit vs counterfactual held-to-12:00, by 09:45 return bucket",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(OUTDIR / "actual_vs_held_by_bucket.png", dpi=130); plt.close(fig)

    # ── chart 2: delta by bucket (crossover) ──
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["#2ca25f" if d > 0 else "#b2182b" for d in tbl["delta_avg_held_minus_945"]]
    ax.bar(x, tbl["delta_avg_held_minus_945"], color=colors, edgecolor="white")
    for i, d in enumerate(tbl["delta_avg_held_minus_945"]):
        ax.annotate(f"{d:+.2f}", (i, d), textcoords="offset points",
                    xytext=(0, 3 if d >= 0 else -12), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(tbl["bucket_945_return_pct"], rotation=35, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("realized 09:45 return bucket"); ax.set_ylabel("delta = avg_held − avg_945 (pts)")
    ax.set_title("Holding-to-12:00 advantage by 09:45 return bucket\n"
                 "(green = holding better; red = exiting at 09:45 better)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(OUTDIR / "delta_by_bucket.png", dpi=130); plt.close(fig)

    pd.set_option("display.width", 230)
    print("\n" + "=" * 140)
    print("positive_at_t1 — 09:45 exit vs counterfactual held-to-12:00 (14%-capped), by 09:45 return bucket")
    print("=" * 140)
    print(tbl.to_string(index=False))
    print("\n=== OVERALL ===")
    print(overall.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
