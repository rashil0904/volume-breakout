# -*- coding: utf-8 -*-
"""
selective_double_down.py
========================
SELECTIVE double-down: short leg opened ONLY on the strong exits — the 14%-target trades
(early_target_pre_t1 / early_target_between_t1_t2) and positive_at_t1 (9:45 winners). The
non-positive noon exits (exit_at_t2_no_target) stay LONG-ONLY. Short squared off at the
3:00pm candle open. Reuses the canonical long trade set (double_down_short_variant.build).

Costs: shorted trades = 2 round-trips (rate×long_notional + rate×short_notional);
non-shorted = 1 round-trip (rate×long_notional). Net at 0.23% and 0.38%.

Flags: (a) short only on target + positive_at_t1; exit_at_t2_no_target long-only.
(b) 2 round-trips shorted / 1 non-shorted. (c) short squared off at 3:00pm open.
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
SHORT_TYPES = {"early_target_pre_t1", "early_target_between_t1_t2", "positive_at_t1"}
TARGET_TYPES = {"early_target_pre_t1", "early_target_between_t1_t2"}


def blk(pnl, ret, cap, label):
    w, l = pnl > 0, pnl <= 0
    return {"strategy": label, "n_trades": len(pnl),
            "total_return_fixedbase_pct": round(pnl.sum() / BASE_POOL * 100, 4),
            "total_pnl_inr": round(pnl.sum(), 0),
            "win_rate_pct": round((pnl > 0).mean() * 100, 2),
            "avg_return_per_trade_pct": round(ret.mean(), 4),
            "median_return_per_trade_pct": round(ret.median(), 4),
            "avg_return_winning_trades_pct": round(ret[w].mean(), 4) if w.any() else np.nan,
            "avg_return_losing_trades_pct": round(ret[l].mean(), 4) if l.any() else np.nan,
            "avg_capital_deployed_per_trade": round(cap.mean(), 0)}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = dd.build_long_with_ohlc()
    cap = T["capital_deployed"]
    T["long_pnl"] = T["shares"] * (T["long_exit_price"] - T["entry_price"])
    T["long_ret"] = (T["long_exit_price"] - T["entry_price"]) / T["entry_price"] * 100
    short_pnl_full = T["shares"] * (T["long_exit_price"] - T["price_3pm_open"])
    short_notl_full = T["shares"] * T["long_exit_price"]
    shorted = T["exit_type"].isin(SHORT_TYPES)             # selective mask

    # per-trade P&L for the three strategies
    A_pnl = T["long_pnl"]
    B_pnl = T["long_pnl"] + short_pnl_full
    C_pnl = T["long_pnl"] + np.where(shorted, short_pnl_full, 0.0)
    A_ret = A_pnl / cap * 100
    B_ret = B_pnl / cap * 100
    C_ret = C_pnl / cap * 100

    def nets(pnl, short_notl):
        return {r: pnl - rr * (cap + short_notl) for r, rr in [("023", R023), ("038", R038)]}
    A_net = {r: A_pnl - rr * cap for r, rr in [("023", R023), ("038", R038)]}
    B_net = nets(B_pnl, short_notl_full)
    C_net = nets(C_pnl, np.where(shorted, short_notl_full, 0.0))

    rows = []
    rows.append(blk(A_pnl, A_ret, cap, "A_long_only (gross)"))
    rows.append(blk(A_net["023"], A_net["023"]/cap*100, cap, "A_long_only (net@0.23%)"))
    rows.append(blk(A_net["038"], A_net["038"]/cap*100, cap, "A_long_only (net@0.38%)"))
    rows.append(blk(B_pnl, B_ret, cap, "B_doubledown_ALL (gross)"))
    rows.append(blk(B_net["023"], B_net["023"]/cap*100, cap, "B_doubledown_ALL (net@0.23%)"))
    rows.append(blk(B_net["038"], B_net["038"]/cap*100, cap, "B_doubledown_ALL (net@0.38%)"))
    rows.append(blk(C_pnl, C_ret, cap, "C_selective (gross)"))
    rows.append(blk(C_net["023"], C_net["023"]/cap*100, cap, "C_selective (net@0.23%)"))
    rows.append(blk(C_net["038"], C_net["038"]/cap*100, cap, "C_selective (net@0.38%)"))
    comp = pd.DataFrame(rows)

    # ── leg decomposition (C) ──
    n_short = int(shorted.sum())
    short_leg_total = short_pnl_full[shorted].sum()
    leg = pd.DataFrame([
        {"leg": "long (all trades)", "n": len(T), "total_pnl_inr": round(T["long_pnl"].sum(), 0),
         "total_return_fixedbase_pct": round(T["long_pnl"].sum()/BASE_POOL*100, 4)},
        {"leg": "short (selective: target + 9:45 only)", "n": n_short,
         "total_pnl_inr": round(short_leg_total, 0),
         "total_return_fixedbase_pct": round(short_leg_total/BASE_POOL*100, 4)},
        {"leg": "combined (C)", "n": len(T), "total_pnl_inr": round(C_pnl.sum(), 0),
         "total_return_fixedbase_pct": round(C_pnl.sum()/BASE_POOL*100, 4)},
    ])

    # ── short-leg split by the two shorted paths ──
    T["short_pnl"] = short_pnl_full
    T["short_ret"] = (T["long_exit_price"] - T["price_3pm_open"]) / T["long_exit_price"] * 100
    grp_target = T[T["exit_type"].isin(TARGET_TYPES)]
    grp_945 = T[T["exit_type"] == "positive_at_t1"]
    split = pd.DataFrame([
        {"shorted_path": "target trades (14% hit)", "n": len(grp_target),
         "short_win_rate_pct": round((grp_target["short_pnl"] > 0).mean()*100, 2),
         "avg_short_return_pct": round(grp_target["short_ret"].mean(), 4),
         "median_short_return_pct": round(grp_target["short_ret"].median(), 4),
         "short_total_pnl_inr": round(grp_target["short_pnl"].sum(), 0)},
        {"shorted_path": "positive_at_t1 (9:45 winners)", "n": len(grp_945),
         "short_win_rate_pct": round((grp_945["short_pnl"] > 0).mean()*100, 2),
         "avg_short_return_pct": round(grp_945["short_ret"].mean(), 4),
         "median_short_return_pct": round(grp_945["short_ret"].median(), 4),
         "short_total_pnl_inr": round(grp_945["short_pnl"].sum(), 0)},
        {"shorted_path": "ALL shorted (selective)", "n": n_short,
         "short_win_rate_pct": round((T.loc[shorted, "short_pnl"] > 0).mean()*100, 2),
         "avg_short_return_pct": round(T.loc[shorted, "short_ret"].mean(), 4),
         "median_short_return_pct": round(T.loc[shorted, "short_ret"].median(), 4),
         "short_total_pnl_inr": round(short_leg_total, 0)},
    ])

    tl = T[["symbol", "entry_date", "entry_price", "shares", "capital_deployed", "exit_type",
            "long_exit_price", "long_pnl", "price_3pm_open", "short_pnl", "short_ret"]].copy()
    tl["shorted_in_C"] = shorted.values
    tl["combined_pnl_C"] = C_pnl.values

    with pd.ExcelWriter(OUTDIR / "selective_double_down.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="comparison_A_B_C", index=False)
        leg.to_excel(w, sheet_name="leg_decomposition", index=False)
        split.to_excel(w, sheet_name="short_leg_by_exit_type", index=False)
        tl.to_excel(w, sheet_name="trade_level", index=False)

    pd.set_option("display.width", 230)
    print(f"Shorted trades (selective): {n_short:,} of {len(T):,} "
          f"(target {len(grp_target)}, positive_at_t1 {len(grp_945)}); "
          f"long-only noon exits: {int((~shorted).sum()):,}")
    print("\n" + "=" * 130 + "\nA / B / C COMPARISON\n" + "=" * 130)
    show = ["strategy", "n_trades", "total_return_fixedbase_pct", "total_pnl_inr", "win_rate_pct",
            "avg_return_per_trade_pct", "median_return_per_trade_pct",
            "avg_return_winning_trades_pct", "avg_return_losing_trades_pct"]
    print(comp[show].to_string(index=False))
    print("\n--- LEG DECOMPOSITION (C) ---"); print(leg.to_string(index=False))
    print("\n--- SHORT LEG BY SHORTED PATH ---"); print(split.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
