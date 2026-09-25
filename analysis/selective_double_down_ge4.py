# -*- coding: utf-8 -*-
"""
selective_double_down_ge4.py
============================
Selective double-down with a HIGHER short gate: short ONLY trades whose long-leg return
>= +4% (all 14%-target trades + positive_at_t1 with 9:45 return >= 4%). Everything else
(positive_at_t1 <4%, and all exit_at_t2_no_target noon exits) stays long-only. Short
squared off at the 3:00pm candle open. Reuses the canonical long trades; no recompute.

Costs: shorted = 2 round-trips (rate×long_notional + rate×short_notional); non-shorted =
1 round-trip. Net at 0.23% and 0.38%.

Flags: (a) gate long_return >= 4% inclusive. (b) 2 rt shorted / 1 non-shorted.
(c) short squared off at 3:00pm open.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import double_down_short_variant as dd

OUTDIR = rb.RESULTS / "selective_double_down"
BASE_POOL = dd.BASE_POOL
R023, R038 = 0.0023, 0.0038
GATE = 4.0


def row(pnl, ret, label):
    return {"strategy": label, "n_trades": len(pnl),
            "total_return_fixedbase_pct": round(pnl.sum() / BASE_POOL * 100, 4),
            "total_pnl_inr": round(pnl.sum(), 0),
            "win_rate_pct": round((pnl > 0).mean() * 100, 2),
            "avg_return_per_trade_pct": round(ret.mean(), 4),
            "median_return_per_trade_pct": round(ret.median(), 4)}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = dd.build_long_with_ohlc()
    cap = T["capital_deployed"]
    T["long_ret"] = (T["long_exit_price"] - T["entry_price"]) / T["entry_price"] * 100
    T["long_pnl"] = T["shares"] * (T["long_exit_price"] - T["entry_price"])
    short_pnl_full = T["shares"] * (T["long_exit_price"] - T["price_3pm_open"])
    short_notl_full = T["shares"] * T["long_exit_price"]
    shorted = T["long_ret"] >= GATE                       # >=4% gate

    B_pnl = T["long_pnl"] + np.where(shorted, short_pnl_full, 0.0)
    A_pnl = T["long_pnl"]
    A_ret, B_ret = A_pnl / cap * 100, B_pnl / cap * 100

    def net(pnl, sn, rr): return pnl - rr * (cap + sn)
    A_net = {r: A_pnl - rr * cap for r, rr in [("023", R023), ("038", R038)]}
    B_net = {r: net(B_pnl, np.where(shorted, short_notl_full, 0.0), rr)
             for r, rr in [("023", R023), ("038", R038)]}

    comp = pd.DataFrame([
        row(A_pnl, A_ret, "A_long_only (gross)"),
        row(A_net["023"], A_net["023"]/cap*100, "A_long_only (net@0.23%)"),
        row(A_net["038"], A_net["038"]/cap*100, "A_long_only (net@0.38%)"),
        row(B_pnl, B_ret, "B_dd_ge4pct (gross)"),
        row(B_net["023"], B_net["023"]/cap*100, "B_dd_ge4pct (net@0.23%)"),
        row(B_net["038"], B_net["038"]/cap*100, "B_dd_ge4pct (net@0.38%)"),
    ])

    # shorted-subset standalone
    g = T[shorted]
    n_sh = int(shorted.sum())
    sp = short_pnl_full[shorted]
    sr = ((T["long_exit_price"] - T["price_3pm_open"]) / T["long_exit_price"] * 100)[shorted]
    n_target = int(g["exit_type"].isin({"early_target_pre_t1", "early_target_between_t1_t2"}).sum())
    shorted_stats = pd.DataFrame([{
        "n_shorted": n_sh, "n_target_trades": n_target, "n_positive_at_t1_ge4pct": n_sh - n_target,
        "n_long_only": len(T) - n_sh,
        "short_win_rate_pct": round((sp > 0).mean() * 100, 2),
        "avg_short_return_pct": round(sr.mean(), 4),
        "median_short_return_pct": round(sr.median(), 4),
        "total_short_pnl_inr": round(sp.sum(), 0),
        "short_total_return_fixedbase_pct": round(sp.sum() / BASE_POOL * 100, 4),
    }])

    tl = T.loc[shorted, ["symbol", "entry_date", "entry_price", "shares", "capital_deployed",
                         "exit_type", "long_ret", "long_exit_price", "price_3pm_open"]].copy()
    tl["short_pnl"] = sp.values
    tl["short_ret_pct"] = sr.values

    with pd.ExcelWriter(OUTDIR / "selective_double_down_ge4.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="comparison", index=False)
        shorted_stats.to_excel(w, sheet_name="shorted_subset_stats", index=False)
        tl.to_excel(w, sheet_name="shorted_trades", index=False)

    pd.set_option("display.width", 220)
    print(f"Short gate long_return >= {GATE}%: shorted {n_sh:,} "
          f"(target {n_target}, positive_at_t1>=4% {n_sh-n_target}) | long-only {len(T)-n_sh:,}")
    print("\n" + "=" * 110 + "\nA (long-only) vs B (>=4%-gated selective double-down)\n" + "=" * 110)
    print(comp.to_string(index=False))
    print("\n--- SHORTED SUBSET (>=4% gate) STANDALONE ---")
    print(shorted_stats.T.to_string(header=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
