# -*- coding: utf-8 -*-
"""newly_listed_first_results_entry_at_ath.py — CHANGES the entry rule for the "newly listed stocks - first
quarterly result" strategy: instead of buying T-1 close ahead of the result, wait and ONLY enter if/when the
stock prints a NEW ALL-TIME HIGH within 1 week (5 trading days) of the reaction day, then buy the ATH breakout
itself. Stocks whose result never produces a fresh ATH within that window are not traded at all -- this is a
strict subset of the 610-trade universe (the same 174 identified in newly_listed_first_results_ath_filter.py).

ENTRY (breakout convention, same as ipo_1week_breakout_backtest.py / ipo_consolidation_breakout_backtest.py):
  entry_price = the OLD all-time-high level (the breakout trigger) if the ATH day's open <= that level (a stop
  order fills exactly at the trigger); if the day GAPPED open above it, entry = that day's open instead.
EXIT: swept, close of trading day T+1..T+20 after the ATH/entry day.
Reuses newly_listed_first_results_hold_sweep.py's reaction-day logic and the ATH-detection from
newly_listed_first_results_ath_filter.py unmodified. New file; originals untouched.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import newly_listed_first_results_backtest as N
import newly_listed_first_results_hold_sweep as H
from ipo_consolidation_breakout_backtest import daily_ohlc

OUT = rb.RESULTS / "newly_listed_first_results_entry_at_ath"; OUT.mkdir(parents=True, exist_ok=True)
SRC = rb.RESULTS / "newly_listed_first_results"
ATH_WINDOW_DAYS = 5
HOLD_PERIODS = list(range(1, 21))


def ath_entry(sym, reaction_day):
    """Same ATH-detection window as the diagnostic, but now also returns the BREAKOUT entry price/day and the
    old ATH level it broke, plus the full tdays/close arrays for the exit sweep."""
    D = daily_ohlc(sym)
    if D is None or reaction_day not in D.index:
        return None
    tdays = list(D.index); hi = D["high"].values; op = D["open"].values; cl = D["close"].values
    r_idx = tdays.index(reaction_day)
    cummax = np.maximum.accumulate(hi)
    for j in range(r_idx, min(r_idx + ATH_WINDOW_DAYS, len(tdays) - 1) + 1):
        prior_max = cummax[j - 1] if j > 0 else -np.inf
        if hi[j] >= prior_max - 1e-9:
            entry_px = prior_max if op[j] <= prior_max + 1e-9 else float(op[j])
            return {"ath_date": tdays[j], "ath_idx": j, "old_ath_level": float(prior_max) if prior_max > -np.inf else None,
                    "gap_fill": bool(op[j] > prior_max + 1e-9), "entry_price": round(entry_px, 3), "tdays": tdays, "cl": cl}
    return None


def main():
    FR = pd.read_csv(SRC / "first_results_raw.csv", parse_dates=["announcement_datetime"])
    V1, _, _, _ = H.build(FR)     # reuse for symbol/reaction_day/listing_date/session/first_result_date only
    print(f"first-results universe: {len(V1)} trades", flush=True)

    rows, no_ath = [], 0
    for r in V1.itertuples():
        a = ath_entry(r.symbol, r.reaction_day)
        if a is None:
            no_ath += 1; continue
        base = {"symbol": r.symbol, "listing_date": r.listing_date, "first_result_date": r.first_result_date,
                "session": r.session, "reaction_day": r.reaction_day, "old_ath_level": a["old_ath_level"],
                "ath_breakout_date": a["ath_date"], "days_reaction_to_ath_entry": (a["ath_date"] - r.reaction_day).days,
                "gap_fill": a["gap_fill"], "entry_price": a["entry_price"]}
        for N_ in HOLD_PERIODS:
            xi = a["ath_idx"] + N_
            if xi >= len(a["tdays"]):
                continue
            xpx = float(a["cl"][xi])
            rows.append({**base, "holding_period": N_, "exit_date": a["tdays"][xi], "exit_price": xpx,
                         "return_pct": round((xpx - a["entry_price"]) / a["entry_price"] * 100, 3)})
    R = pd.DataFrame(rows)
    n_trades = R.drop_duplicates(["symbol", "reaction_day"]).shape[0]
    print(f"stocks with NO ATH within {ATH_WINDOW_DAYS}d of reaction (not traded): {no_ath}", flush=True)
    print(f"stocks trading THIS strategy (entered at the ATH breakout): {n_trades} of {len(V1)} ({n_trades/len(V1)*100:.1f}%)", flush=True)

    def stats(x):
        if len(x) == 0: return {"n": 0}
        return {"n": len(x), "avg_pct": round(x.mean(), 3), "median_pct": round(x.median(), 3), "win_pct": round((x > 0).mean() * 100, 1), "std_pct": round(x.std(), 3)}

    G = R.groupby("holding_period")["return_pct"].apply(lambda x: pd.Series(stats(x))).unstack().reset_index()
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 30)
    print("\n=== SWEEP TABLE: entry at ATH breakout, exit T+1..T+20 ===\n" + G.to_string(index=False), flush=True)

    WIN = 5; hp = G.set_index("holding_period")["avg_pct"]; hp_med = G.set_index("holding_period")["median_pct"]
    best_avg, best_avg_m = None, -1e18
    for i in range(1, 21 - WIN + 1):
        w = list(range(i, i + WIN)); m = hp.loc[w].mean()
        if m > best_avg_m: best_avg_m, best_avg = m, w
    best_med, best_med_m = None, -1e18
    for i in range(1, 21 - WIN + 1):
        w = list(range(i, i + WIN)); m = hp_med.loc[w].mean()
        if m > best_med_m: best_med_m, best_med = m, w
    print(f"\nbest stable {WIN}-day region by AVG: T+{best_avg[0]}..T+{best_avg[-1]} -> {best_avg_m:.2f}% "
          f"(individual: {[round(hp.loc[w],2) for w in best_avg]})", flush=True)
    print(f"best stable {WIN}-day region by MEDIAN: T+{best_med[0]}..T+{best_med[-1]} -> {best_med_m:.2f}% "
          f"(individual: {[round(hp_med.loc[w],2) for w in best_med]})", flush=True)

    # comparison vs the original T-1-close entry, same trade set, at matching holding periods
    orig = pd.read_excel(rb.RESULTS / "newly_listed_first_results_hold_sweep" / "newly_listed_first_results_hold_sweep.xlsx", "Sweep_table_T1_T20")
    cmp = G[["holding_period", "n", "avg_pct", "median_pct", "win_pct"]].merge(
        orig[["holding_period", "n", "avg_pct", "median_pct", "win_pct"]].rename(columns={"n": "n_orig", "avg_pct": "avg_pct_orig", "median_pct": "median_pct_orig", "win_pct": "win_pct_orig"}),
        on="holding_period")
    print("\n=== vs ORIGINAL (T-1-close entry, ALL 610, same N) for reference ===\n" + cmp.to_string(index=False), flush=True)

    centre = best_med[len(best_med) // 2]
    detail = R[R["holding_period"] == centre][["symbol", "listing_date", "first_result_date", "session", "reaction_day", "old_ath_level",
                                                "ath_breakout_date", "days_reaction_to_ath_entry", "gap_fill", "entry_price",
                                                "exit_date", "exit_price", "return_pct"]].sort_values("return_pct", ascending=False)
    print(f"\nper-trade detail at region-centre T+{centre} (n={len(detail)}):\n" + detail.to_string(index=False), flush=True)

    fn = OUT / "newly_listed_first_results_entry_at_ath.xlsx"
    with pd.ExcelWriter(fn, engine="openpyxl") as w:
        pd.DataFrame([{"note": f"Entry rule CHANGED: only trade a stock if it prints a new all-time high within {ATH_WINDOW_DAYS} trading "
                                "days of its first-quarterly-result reaction day; entry = the old ATH level (breakout trigger) or that "
                                "day's open if it gapped past the level. Stocks with no ATH in that window are not traded at all. Exit "
                                "swept T+1..T+20 trading days after the ATH entry day."}]).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        G.to_excel(w, sheet_name="Sweep_table", index=False)
        cmp.to_excel(w, sheet_name="Vs_original_T1close_entry", index=False)
        detail.to_excel(w, sheet_name="Trades_detail_centre_N", index=False)
        R.to_excel(w, sheet_name="All_trade_rows", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 34)
    print(f"\nSaved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
