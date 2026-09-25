# -*- coding: utf-8 -*-
"""
capital_alloc_target_exit.py
============================
Day-high capital-allocation sweep, but with the EXIT = conditional_split_best_t2
(winners @09:45 / losers @12:00) + 14% intraday profit target — the combo we
validated earlier (overall best).

Only the capital-allocation step is swept (6 targets + equal-split baseline).
Reused as-is:
  - per-trade exit return %      -> profit_target_sweep.run_cond (09:45/12:00, X=14)
  - day-high allocation logic    -> run_backtest._day_alloc
  - both total-return metrics     -> run_backtest.compute_stats + fixed ₹5L base
  - day-high flag                 -> entry_price_315pm >= today_cumhigh_15 (from diag)

Per trade: pnl = shares x (exit_price - entry) = cap x ret%/100, where ret comes
from the profit-target exit and is independent of the share count.
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import exit_time_sweep as ets
import profit_target_sweep as pts

POOL = rb.DAILY_POOL
CB   = rb.CAPITAL_BASE
GRID = rb.CAPITAL_TARGET_GRID
OUTDIR = rb.RESULTS / "capital_alloc_sweep"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── Base trades + day-high flag ──
    base = ets.load_base_positions()                         # symbol, date, entry, shares, cap
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "today_cumhigh_15"], parse_dates=["date"])
    base = base.merge(diag, on=["symbol", "date"], how="left")
    entry = base["entry"].values.astype(float)
    ch = base["today_cumhigh_15"].values.astype(float)
    day_high = np.where(np.isnan(ch), False, entry >= ch)

    # ── Per-trade return from the profit-target exit (09:45/12:00, 14%) ──
    print("Fetching next-day OHLC (once) …")
    opens, highs = pts.fetch_nextday_ohlc(base)
    pct_high = (highs - entry[:, None]) / entry[:, None] * 100
    t1, t2 = pts.HM_0945, pts.HM_1200
    ot1 = opens[:, pts.HCOL[t1]]; ot2 = opens[:, pts.HCOL[t2]]
    ret_t1 = (ot1 - entry) / entry * 100
    p1max = pts._window_max(pct_high, [pts.HCOL[hm] for hm in pts.CANDLE_HMS if hm < t1])
    p3max = pts._window_max(pct_high, [pts.HCOL[hm] for hm in pts.CANDLE_HMS if t1 < hm < t2])
    _, ret, _, mask = pts.run_cond(14, entry, np.ones(len(base)), ot1, ot2, ret_t1, p1max, p3max)

    # ── Valid trades → per-day groups ──
    vi = np.where(mask)[0]
    dfv = pd.DataFrame({"date": pd.to_datetime(base["date"].values[vi]),
                        "entry": entry[vi], "is_high": day_high[vi].astype(bool),
                        "ret": ret[vi]})
    day_groups, nh_list, no_list = [], [], []
    for d, g in dfv.groupby("date"):
        e = g["entry"].values; ih = g["is_high"].values.astype(bool); r = g["ret"].values
        day_groups.append((d, e, ih, r))
        nh_list.append(int(ih.sum())); no_list.append(len(e) - int(ih.sum()))
    avg_nh, avg_no = float(np.mean(nh_list)), float(np.mean(no_list))
    days_ge5 = sum(1 for _, e, _, _ in day_groups if len(e) >= 5)
    days_lt5 = len(day_groups) - days_ge5

    def cap_run(mode):
        rows, cats = [], Counter()
        for d, ent, ish, r in day_groups:
            n = len(ent)
            if mode == "baseline":
                alloc, cat = np.full(n, rb._day_target(n)), "baseline"   # equal split, ₹1L/stock cap
            else:
                alloc, cat = rb._day_alloc(ent, ish, mode)
            cats[cat] += 1
            shares = np.floor(alloc / ent).astype(np.int64)
            for i in range(n):
                sh = int(shares[i])
                if sh == 0:
                    continue
                cap = sh * ent[i]
                rows.append({"date": d, "cap": cap, "pnl": cap * r[i] / 100, "ret": r[i]})
        return pd.DataFrame(rows), cats

    def metrics_row(label, trades, cats):
        s = rb.compute_stats(trades)
        return {
            "target_capital": label, "n_trades": len(trades),
            "win_rate_pct": s["win_rate"],
            "avg_return_per_trade_pct": s["avg_ret"],
            "median_return_per_trade_pct": s["median_ret"],
            "total_return_fixedbase_pct": round(float(trades["pnl"].sum()) / CB * 100, 4),
            "total_return_sumofdaily_pct": s["total_ret_pct"],
            "days_unscaled_lt5": cats.get("unscaled_lt5", 0),
            "days_scaledown_ge5": cats.get("boost5000_ge5", 0) + cats.get("flat_fallback", 0) + cats.get("flat_no_other", 0),
            "days_flat_fallback": cats.get("flat_fallback", 0) + cats.get("flat_no_other", 0),
        }

    rows = [metrics_row(X, *cap_run(X)) for X in GRID]
    summary = pd.DataFrame(rows).sort_values(
        "total_return_fixedbase_pct", ascending=False).reset_index(drop=True)
    base_row = metrics_row("baseline_equal_split", *cap_run("baseline"))
    for k in ("days_unscaled_lt5", "days_scaledown_ge5", "days_flat_fallback"):
        base_row[k] = "-"
    full = pd.concat([summary, pd.DataFrame([base_row])], ignore_index=True)

    full.to_csv(OUTDIR / "capital_alloc_target_exit.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "capital_alloc_target_exit.xlsx", engine="openpyxl") as w:
        full.to_excel(w, sheet_name="Capital_sweep_targetexit", index=False)

    print("\n" + "=" * 122)
    print("DAY-HIGH CAPITAL SWEEP  |  EXIT = 09:45/12:00 + 14% target  |  6 targets ranked + equal-split baseline")
    print("=" * 122)
    print(f"  Avg day-high/day : {avg_nh:.2f}   |   Avg other/day : {avg_no:.2f}")
    print(f"  Days n_total>=5 (scale-down) : {days_ge5:,}   |   Days n_total<5 (full-X) : {days_lt5:,}")
    print("-" * 122)
    print(f"  {'target':>20}  {'Trades':>7}  {'Win%':>7}  {'Avg%':>8}  {'Med%':>8}  "
          f"{'TotRet(5L)%':>12}  {'TotRet(daily)%':>15}  {'d<5':>5}  {'d>=5':>6}  {'flat':>5}")
    print("  " + "-" * 118)
    for _, r in full.iterrows():
        print(f"  {str(r['target_capital']):>20}  {r['n_trades']:>7,}  {r['win_rate_pct']:>7.2f}  "
              f"{r['avg_return_per_trade_pct']:>8.4f}  {r['median_return_per_trade_pct']:>8.4f}  "
              f"{r['total_return_fixedbase_pct']:>12.2f}  {r['total_return_sumofdaily_pct']:>15.2f}  "
              f"{str(r['days_unscaled_lt5']):>5}  {str(r['days_scaledown_ge5']):>6}  {str(r['days_flat_fallback']):>5}")
    print("=" * 122)
    print(f"  Saved → {OUTDIR}")


if __name__ == "__main__":
    main()
