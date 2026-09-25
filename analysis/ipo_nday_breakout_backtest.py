# -*- coding: utf-8 -*-
"""ipo_nday_breakout_backtest.py — generalises ipo_1week_breakout_backtest.py: instead of a FIXED 5-trading-day
reference high, sweeps the reference-window length itself, 3 to 10 trading days (8 values), each combined with
the T+1..T+20 holding-period sweep (20 values) -> 160 combinations. For a given reference length N_ref, there is
only ONE possible window per stock (the literal first N_ref trading days after listing) -- no window-search
ambiguity like the consolidation-breakout strategy had, so each N_ref is evaluated independently and directly.

Otherwise unchanged from the original: touch-based breakout (daily high as the intraday-touch proxy), 3-month
breakout deadline, gap-fill entry price, no stop-loss (plain version, matching the first "IPO 1 Week Breakout" run).
Reuses daily_ohlc() unmodified. New file; originals untouched.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
from ipo_consolidation_breakout_backtest import daily_ohlc

OUT = rb.RESULTS / "ipo_nday_breakout"; OUT.mkdir(parents=True, exist_ok=True)
REF_RANGE = list(range(3, 11))      # 3..10 trading days
HOLD_PERIODS = list(range(1, 21))   # T+1..T+20
MONTHS = 3


def scan_stock(sym, listing_date):
    D = daily_ohlc(sym)
    if D is None or len(D) < REF_RANGE[0] + 1:
        return [], [{"symbol": sym, "reason": "no_or_insufficient_price_data"}]
    tdays = list(D.index)
    hi = D["high"].values; lo = D["low"].values; op = D["open"].values; cl = D["close"].values
    end3 = (pd.Timestamp(listing_date) + pd.DateOffset(months=MONTHS)).date()

    rows, statuses = [], []
    for ref_n in REF_RANGE:
        if len(tdays) < ref_n + 1:
            statuses.append({"symbol": sym, "ref_days": ref_n, "reason": "insufficient_history_for_this_ref_length", "qualifies": False})
            continue
        ref_hi = float(hi[:ref_n].max()); ref_lo = float(lo[:ref_n].min())
        f = None
        for j in range(ref_n, len(tdays)):
            if tdays[j] >= end3:
                break
            if hi[j] > ref_hi + 1e-9:
                f = j; break
        base = {"symbol": sym, "listing_date": listing_date, "ref_days": ref_n, "ref_high": round(ref_hi, 3), "ref_low": round(ref_lo, 3)}
        if f is None:
            statuses.append({**base, "reason": "no_breakout_within_3_months", "qualifies": False}); continue
        entry_px = ref_hi if op[f] <= ref_hi + 1e-9 else float(op[f])
        statuses.append({**base, "reason": "qualifying_breakout", "qualifies": True, "breakout_date": tdays[f],
                          "days_listing_to_breakout": (tdays[f] - listing_date).days, "entry_price": round(entry_px, 3)})
        for N in HOLD_PERIODS:
            xi = f + N
            if xi >= len(tdays):
                continue
            xpx = float(cl[xi]); ret = (xpx - entry_px) / entry_px * 100.0
            rows.append({**base, "breakout_date": tdays[f], "days_listing_to_breakout": (tdays[f] - listing_date).days,
                         "gap_fill": bool(op[f] > ref_hi + 1e-9), "entry_price": round(entry_px, 3),
                         "holding_period": N, "exit_date": tdays[xi], "exit_price": round(xpx, 3), "return_pct": round(ret, 3)})
    return rows, statuses


def main():
    U = pd.read_csv(rb.RESULTS / "newly_listed_universe_combined.csv", parse_dates=["first_date"])
    U["first_date"] = U["first_date"].dt.date
    print(f"universe: {len(U)} stocks", flush=True)

    all_rows, all_status = [], []
    for i, r in enumerate(U.itertuples(), 1):
        rows, statuses = scan_stock(r.symbol, r.first_date)
        all_rows.extend(rows); all_status.extend(statuses)
        if i % 150 == 0: print(f"  ...{i}/{len(U)}", flush=True)
    R = pd.DataFrame(all_rows); ST = pd.DataFrame(all_status)

    print("\n=== PASS RATE by reference-window length ===")
    passr = ST.groupby("ref_days").apply(lambda g: pd.Series({"n_evaluable": len(g), "n_qualified": int((g["reason"]=="qualifying_breakout").sum()),
                                                                "pass_rate_pct": round((g["reason"]=="qualifying_breakout").mean()*100,1)}), include_groups=False).reset_index()
    print(passr.to_string(index=False), flush=True)

    def agg(g):
        return pd.Series({"n_trades": len(g), "avg_return_pct": round(g["return_pct"].mean(), 3), "median_return_pct": round(g["return_pct"].median(), 3),
                           "win_rate_pct": round((g["return_pct"] > 0).mean() * 100, 1), "std_pct": round(g["return_pct"].std(), 3)})
    G = R.groupby(["ref_days", "holding_period"]).apply(agg, include_groups=False).reset_index()

    grid_avg = G.pivot(index="ref_days", columns="holding_period", values="avg_return_pct")
    grid_med = G.pivot(index="ref_days", columns="holding_period", values="median_return_pct")
    grid_win = G.pivot(index="ref_days", columns="holding_period", values="win_rate_pct")
    grid_n = G.pivot(index="ref_days", columns="holding_period", values="n_trades")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 25)
    print("\n=== GRID: avg return % (ref_days rows x holding_period cols) ===\n" + grid_avg.round(2).to_string())
    print("\n=== GRID: median return % ===\n" + grid_med.round(2).to_string())
    print("\n=== GRID: win rate % ===\n" + grid_win.round(1).to_string())
    print("\n=== GRID: n_trades ===\n" + grid_n.astype(int).to_string())

    # stable 3x3 region search on the median grid (median, not average, given the known tail-outlier distortion)
    MIN_N = 15
    idxs, cols = list(grid_med.index), list(grid_med.columns)
    best = None
    for xi, X in enumerate(idxs):
        for ni, N in enumerate(cols):
            neigh = [(idxs[a], cols[b]) for a in range(max(0, xi - 1), min(len(idxs), xi + 2))
                     for b in range(max(0, ni - 1), min(len(cols), ni + 2))]
            vals = [grid_med.loc[a, b] for a, b in neigh if pd.notna(grid_med.loc[a, b])]
            ns = [grid_n.loc[a, b] for a, b in neigh if pd.notna(grid_n.loc[a, b])]
            if len(vals) < 5 or sum(ns) < MIN_N:
                continue
            rec = {"ref_days": X, "holding_period": N, "cell_median": grid_med.loc[X, N], "cell_avg": grid_avg.loc[X, N], "cell_n": int(grid_n.loc[X, N]),
                   "neigh_mean_median": np.mean(vals), "neigh_n_cells": len(vals), "neigh_total_trades": int(sum(ns))}
            if best is None or rec["neigh_mean_median"] > best["neigh_mean_median"]:
                best = rec
    print(f"\n=== BEST STABLE 3x3 REGION BY MEDIAN (min {MIN_N} trades in neighbourhood) ===\n{best}", flush=True)
    single_best_avg = G.loc[G["avg_return_pct"].idxmax()]
    single_best_med = G.loc[G["median_return_pct"].idxmax()]
    print(f"\nsingle best cell by AVG (reference only, tail-outlier risk): {dict(single_best_avg)}")
    print(f"single best cell by MEDIAN (reference only): {dict(single_best_med)}")

    fn = OUT / "ipo_nday_breakout.xlsx"
    with pd.ExcelWriter(fn, engine="openpyxl") as w:
        pd.DataFrame([{"note": "Reference window swept 3-10 trading days (was fixed at 5). For a given length there is only ONE "
                                "possible window per stock (the literal first N days), so no window-search ambiguity. Same touch-based "
                                "breakout, 3-month deadline, gap-fill entry, no SL. Stable-region search uses the MEDIAN grid, not "
                                "average, since prior runs showed the average is distorted by a handful of extreme penny-stock winners."}]).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        passr.to_excel(w, sheet_name="Pass_rate_by_ref_days", index=False)
        grid_avg.to_excel(w, sheet_name="Grid_avg_return_pct")
        grid_med.to_excel(w, sheet_name="Grid_median_return_pct")
        grid_win.to_excel(w, sheet_name="Grid_win_rate_pct")
        grid_n.to_excel(w, sheet_name="Grid_n_trades")
        pd.DataFrame([best]).to_excel(w, sheet_name="Best_stable_region_median", index=False)
        R.to_excel(w, sheet_name="All_trade_rows", index=False)
        ST.to_excel(w, sheet_name="All_stock_ref_status", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)
    print(f"\nSaved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
