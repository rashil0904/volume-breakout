# -*- coding: utf-8 -*-
"""
full_double_down_splitcost.py
=============================
FULL double-down (short EVERY trade at its long exit, all three exit paths, no gate) with
a CORRECTED split-cost model: intraday short round-trip = 0.10%; delivery long round-trip =
0.23% (and 0.38% variant). Costs applied per leg on each leg's traded value. Reuses the
canonical long trades; no recompute.

  long_cost  = long_rate × capital_deployed        (long_rate = 0.23% or 0.38%)
  short_cost = 0.10% × (shares × long_exit_price)   (short notional at the short entry price)
  net_A = combined − long@0.23% − short@0.10%
  net_B = combined − long@0.38% − short@0.10%

Flags: (a) ALL trades shorted, all 3 exit paths, no return gate. (b) short 0.10%/intraday
round-trip, long 0.23%/0.38%/delivery round-trip, applied per leg. (c) short squared off at
the 3:00pm candle open same day.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import double_down_short_variant as dd

OUTDIR = rb.RESULTS / "full_double_down_splitcost"
BASE_POOL = dd.BASE_POOL
LONG_023, LONG_038, SHORT_RATE = 0.0023, 0.0038, 0.0010
EXIT_TYPES = ["early_target_pre_t1", "early_target_between_t1_t2", "positive_at_t1", "exit_at_t2_no_target"]


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
    T["long_pnl"] = T["shares"] * (T["long_exit_price"] - T["entry_price"])
    T["long_ret"] = (T["long_exit_price"] - T["entry_price"]) / T["entry_price"] * 100
    T["short_pnl"] = T["shares"] * (T["long_exit_price"] - T["price_3pm_open"])   # ALL trades
    T["short_ret"] = (T["long_exit_price"] - T["price_3pm_open"]) / T["long_exit_price"] * 100
    T["combined_pnl"] = T["long_pnl"] + T["short_pnl"]
    short_notl = T["shares"] * T["long_exit_price"]

    long_cost_023 = LONG_023 * cap; long_cost_038 = LONG_038 * cap
    short_cost = SHORT_RATE * short_notl

    A_pnl = T["long_pnl"]
    B_pnl = T["combined_pnl"]
    A_ret, B_ret = A_pnl / cap * 100, B_pnl / cap * 100
    # nets
    A_netA = A_pnl - long_cost_023;            A_netB = A_pnl - long_cost_038
    B_netA = B_pnl - long_cost_023 - short_cost; B_netB = B_pnl - long_cost_038 - short_cost

    comp = pd.DataFrame([
        row(A_pnl, A_ret, "A_long_only (gross)"),
        row(A_netA, A_netA/cap*100, "A_long_only (net long@0.23%)"),
        row(A_netB, A_netB/cap*100, "A_long_only (net long@0.38%)"),
        row(B_pnl, B_ret, "B_full_dd (gross)"),
        row(B_netA, B_netA/cap*100, "B_full_dd (net_A: long0.23%+short0.10%)"),
        row(B_netB, B_netB/cap*100, "B_full_dd (net_B: long0.38%+short0.10%)"),
    ])

    # ── decomposition ──
    leg = pd.DataFrame([
        {"leg": "long", "total_pnl_inr": round(T["long_pnl"].sum(), 0),
         "total_return_fixedbase_pct": round(T["long_pnl"].sum()/BASE_POOL*100, 4),
         "win_rate_pct": round((T["long_pnl"] > 0).mean()*100, 2),
         "avg_return_pct": round(T["long_ret"].mean(), 4)},
        {"leg": "short (all trades)", "total_pnl_inr": round(T["short_pnl"].sum(), 0),
         "total_return_fixedbase_pct": round(T["short_pnl"].sum()/BASE_POOL*100, 4),
         "win_rate_pct": round((T["short_pnl"] > 0).mean()*100, 2),
         "avg_return_pct": round(T["short_ret"].mean(), 4)},
        {"leg": "combined", "total_pnl_inr": round(T["combined_pnl"].sum(), 0),
         "total_return_fixedbase_pct": round(T["combined_pnl"].sum()/BASE_POOL*100, 4),
         "win_rate_pct": round((T["combined_pnl"] > 0).mean()*100, 2),
         "avg_return_pct": round(B_ret.mean(), 4)},
    ])

    # ── short by exit type ──
    st = []
    for etype in EXIT_TYPES:
        g = T[T["exit_type"] == etype]
        if len(g) == 0:
            continue
        st.append({"exit_type": etype, "n_trades": len(g),
                   "short_win_rate_pct": round((g["short_pnl"] > 0).mean()*100, 2),
                   "avg_short_return_pct": round(g["short_ret"].mean(), 4),
                   "median_short_return_pct": round(g["short_ret"].median(), 4),
                   "short_total_pnl_inr": round(g["short_pnl"].sum(), 0),
                   "short_net_pnl_at_0.10pct_inr": round((g["short_pnl"] - SHORT_RATE*g["shares"]*g["long_exit_price"]).sum(), 0)})
    short_by_exit = pd.DataFrame(st)

    tl = T[["symbol", "entry_date", "entry_price", "shares", "capital_deployed", "exit_type",
            "long_ret", "long_exit_price", "price_3pm_open", "long_pnl", "short_pnl",
            "short_ret", "combined_pnl"]].copy()

    with pd.ExcelWriter(OUTDIR / "full_double_down_splitcost.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="comparison", index=False)
        leg.to_excel(w, sheet_name="leg_decomposition", index=False)
        short_by_exit.to_excel(w, sheet_name="short_leg_by_exit_type", index=False)
        tl.to_excel(w, sheet_name="trade_level", index=False)

    pd.set_option("display.width", 220)
    print("Full double-down (ALL trades shorted). Short cost 0.10% intraday; long 0.23%/0.38% delivery.")
    print("\n" + "=" * 110 + "\nA (long-only) vs B (FULL double-down, split-cost)\n" + "=" * 110)
    print(comp.to_string(index=False))
    print("\n--- LEG DECOMPOSITION ---"); print(leg.to_string(index=False))
    print("\n--- SHORT LEG BY EXIT TYPE (short_net at 0.10%) ---"); print(short_by_exit.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
