# -*- coding: utf-8 -*-
"""
double_down_detailed.py
=======================
Detailed breakdown of the double-down (long + exit-day short) variant. Reuses the exact
trade construction from double_down_short_variant. Adds: long/short interaction quadrants,
exit->3pm fade distribution, short edge by hold duration (exit time), yearly stability,
and the exact cost breakeven rate. No strategy recompute beyond the shared build.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import double_down_short_variant as dd

OUTDIR = dd.OUTDIR
BASE_POOL = dd.BASE_POOL


def main():
    T = dd.build_long_with_ohlc()
    T["long_pnl"] = T["shares"] * (T["long_exit_price"] - T["entry_price"])
    T["long_ret"] = (T["long_exit_price"] - T["entry_price"]) / T["entry_price"] * 100
    T["short_pnl"] = T["shares"] * (T["long_exit_price"] - T["price_3pm_open"])
    T["short_ret"] = (T["long_exit_price"] - T["price_3pm_open"]) / T["entry_price"] * 100
    T["combined_pnl"] = T["long_pnl"] + T["short_pnl"]
    T["combined_ret"] = T["combined_pnl"] / T["capital_deployed"] * 100
    T["move_exit_to_3pm_pct"] = (T["price_3pm_open"] - T["long_exit_price"]) / T["long_exit_price"] * 100
    T["short_notional"] = T["shares"] * T["long_exit_price"]
    T["hold_min"] = 900 - T["exit_hm"]                # minutes short is held (exit -> 15:00)
    T["year"] = pd.to_datetime(T["entry_date"]).dt.year
    n = len(T)

    # ── 1. interaction quadrants ──
    lw, sw = T["long_pnl"] > 0, T["short_pnl"] > 0
    quad = {"long WIN + short WIN": lw & sw, "long WIN + short LOSS": lw & ~sw,
            "long LOSS + short WIN (rescue)": ~lw & sw, "long LOSS + short LOSS (double loss)": ~lw & ~sw}
    qrows = []
    for lab, m in quad.items():
        g = T[m]
        qrows.append({"quadrant": lab, "n_trades": len(g), "pct_of_trades": round(len(g)/n*100, 2),
                      "avg_long_ret_pct": round(g["long_ret"].mean(), 4) if len(g) else np.nan,
                      "avg_short_ret_pct": round(g["short_ret"].mean(), 4) if len(g) else np.nan,
                      "avg_combined_ret_pct": round(g["combined_ret"].mean(), 4) if len(g) else np.nan,
                      "combined_total_pnl_inr": round(g["combined_pnl"].sum(), 0)})
    quad_tbl = pd.DataFrame(qrows)

    # ── 2. exit->3pm fade distribution ──
    edges = [(-1e9, -3, "< -3%"), (-3, -1, "-3 to -1%"), (-1, -0.001, "-1 to 0%"),
             (-0.001, 1, "0 to +1%"), (1, 3, "+1 to +3%"), (3, 1e9, "> +3%")]
    frows = []
    mv = T["move_exit_to_3pm_pct"]
    for lo, hi, lab in edges:
        m = (mv > lo) & (mv <= hi)
        g = T[m]
        frows.append({"exit_to_3pm_move_bucket": lab, "n_trades": int(m.sum()),
                      "pct_of_trades": round(m.mean()*100, 2),
                      "avg_short_ret_pct": round(g["short_ret"].mean(), 4) if len(g) else np.nan,
                      "short_total_pnl_inr": round(g["short_pnl"].sum(), 0)})
    fade_tbl = pd.DataFrame(frows)

    # ── 3. short edge by exit time (hold duration) ──
    hrows = []
    for hm, g in T.groupby("exit_hm"):
        hh = int(hm)
        hrows.append({"exit_time": f"{hh//60:02d}:{hh%60:02d}", "hold_minutes": 900-hh,
                      "n_trades": len(g), "short_win_rate_pct": round((g["short_pnl"]>0).mean()*100, 2),
                      "short_avg_ret_pct": round(g["short_ret"].mean(), 4),
                      "avg_move_exit_to_3pm_pct": round(g["move_exit_to_3pm_pct"].mean(), 4),
                      "short_total_pnl_inr": round(g["short_pnl"].sum(), 0)})
    hold_tbl = pd.DataFrame(hrows).sort_values("hold_minutes")

    # ── 4. yearly long/short/combined ──
    yrows = []
    for y, g in T.groupby("year"):
        yrows.append({"year": int(y), "n_trades": len(g),
                      "long_total_pct": round(g["long_pnl"].sum()/BASE_POOL*100, 2),
                      "short_total_pct": round(g["short_pnl"].sum()/BASE_POOL*100, 2),
                      "combined_total_pct": round(g["combined_pnl"].sum()/BASE_POOL*100, 2),
                      "short_win_rate_pct": round((g["short_pnl"]>0).mean()*100, 2),
                      "short_avg_ret_pct": round(g["short_ret"].mean(), 4)})
    yearly_tbl = pd.DataFrame(yrows)

    # ── 5. cost breakeven ──
    short_pnl_tot = T["short_pnl"].sum(); short_notl_tot = T["short_notional"].sum()
    breakeven = short_pnl_tot / short_notl_tot * 100          # rate where short leg nets to zero
    cost_rows = []
    for r in [0.0, 0.10, 0.1533, 0.23, 0.2555, 0.30, 0.38, 0.50]:
        rr = r/100
        long_net = T["long_pnl"].sum() - rr*T["capital_deployed"].sum()
        dd_net = T["combined_pnl"].sum() - rr*(T["capital_deployed"].sum() + short_notl_tot)
        cost_rows.append({"cost_rate_per_roundtrip_pct": r,
                          "long_only_net_return_pct": round(long_net/BASE_POOL*100, 2),
                          "double_down_net_return_pct": round(dd_net/BASE_POOL*100, 2),
                          "double_down_advantage_pts": round((dd_net-long_net)/BASE_POOL*100, 2)})
    cost_tbl = pd.DataFrame(cost_rows)

    with pd.ExcelWriter(OUTDIR / "double_down_detailed.xlsx", engine="openpyxl") as w:
        quad_tbl.to_excel(w, sheet_name="interaction_quadrants", index=False)
        fade_tbl.to_excel(w, sheet_name="fade_distribution", index=False)
        hold_tbl.to_excel(w, sheet_name="short_by_hold_duration", index=False)
        yearly_tbl.to_excel(w, sheet_name="yearly_legs", index=False)
        cost_tbl.to_excel(w, sheet_name="cost_breakeven", index=False)

    pd.set_option("display.width", 220)
    print("=" * 120 + "\n1. LONG × SHORT INTERACTION QUADRANTS\n" + "=" * 120)
    print(quad_tbl.to_string(index=False))
    print("\n" + "=" * 120 + "\n2. EXIT -> 3PM FADE DISTRIBUTION\n" + "=" * 120)
    print(fade_tbl.to_string(index=False))
    print("\n" + "=" * 120 + "\n3. SHORT EDGE BY HOLD DURATION (exit time -> 3pm)\n" + "=" * 120)
    print(hold_tbl.to_string(index=False))
    print("\n" + "=" * 120 + "\n4. YEARLY — long / short / combined (% fixed-base)\n" + "=" * 120)
    print(yearly_tbl.to_string(index=False))
    print("\n" + "=" * 120 + "\n5. COST BREAKEVEN\n" + "=" * 120)
    print(f"  short-leg breakeven cost rate = Σshort_pnl / Σshort_notional = "
          f"₹{short_pnl_tot:,.0f} / ₹{short_notl_tot:,.0f} = {breakeven:.4f}% per round-trip")
    print(f"  -> above {breakeven:.3f}% cost, the short leg is net-negative and double-down loses to long-only")
    print(cost_tbl.to_string(index=False))
    print(f"\nSaved -> {OUTDIR / 'double_down_detailed.xlsx'}")


if __name__ == "__main__":
    main()
