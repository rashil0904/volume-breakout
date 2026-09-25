# -*- coding: utf-8 -*-
"""ipo_1week_breakout_backtest_sl.py — "IPO 1 Week Breakout" with a STOP-LOSS added: exit the trade the first day
its CLOSE falls back below the breakout level (the locked 5-day high), rather than riding out the full holding
period. The T+1..T+20 holding-period sweep is kept as a CAP on the hold -- actual exit is the EARLIER of the SL
trigger or the swept N-day time exit.

ASSUMPTION FLAGGED (not specified, so stated explicitly): the SL close-check starts on the BREAKOUT DAY ITSELF, not
the day after. A breakout that touches above the 5-day high intraday but closes back below it on that same day exits
immediately at that day's close (a same-day failed-breakout round trip). If you meant the check to start the day
AFTER entry instead, say so and I'll adjust -- it changes results only for trades that fail on day 0.

Everything else (universe, reference window, touch-based breakout entry, gap-fill entry price) is unchanged from
ipo_1week_breakout_backtest.py; only the exit rule differs. New file; original untouched.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
from ipo_consolidation_breakout_backtest import daily_ohlc

OUT = rb.RESULTS / "ipo_1week_breakout_sl"; OUT.mkdir(parents=True, exist_ok=True)
REF_DAYS = 5
HOLD_PERIODS = list(range(1, 21))
MONTHS = 3


def scan_stock(sym, listing_date):
    D = daily_ohlc(sym)
    if D is None or len(D) < REF_DAYS + 1:
        return None, {"symbol": sym, "reason": "no_or_insufficient_price_data"}
    tdays = list(D.index)
    hi = D["high"].values; lo = D["low"].values; op = D["open"].values; cl = D["close"].values
    ref_hi = float(hi[:REF_DAYS].max()); ref_lo = float(lo[:REF_DAYS].min())
    ref_start, ref_end = tdays[0], tdays[REF_DAYS - 1]

    end3 = (pd.Timestamp(listing_date) + pd.DateOffset(months=MONTHS)).date()
    f = None
    for j in range(REF_DAYS, len(tdays)):
        if tdays[j] >= end3:
            break
        if hi[j] > ref_hi + 1e-9:
            f = j; break

    base = {"symbol": sym, "listing_date": listing_date, "ref_window_start": ref_start, "ref_window_end": ref_end,
            "ref_high": round(ref_hi, 3), "ref_low": round(ref_lo, 3), "three_month_deadline": end3}
    if f is None:
        return None, {**base, "reason": "no_breakout_within_3_months", "qualifies": False}

    entry_px = ref_hi if op[f] <= ref_hi + 1e-9 else float(op[f])

    # first day (>= breakout day f) whose CLOSE falls back below the breakout level (ref_hi) -- SL trigger, if any
    sl_idx = None
    for k in range(f, len(tdays)):
        if cl[k] < ref_hi - 1e-9:
            sl_idx = k; break

    rows = []
    for N in HOLD_PERIODS:
        cap_idx = f + N
        if sl_idx is not None and sl_idx <= cap_idx:
            xi, exit_type = sl_idx, ("SL_day0_failed_breakout" if sl_idx == f else "SL_close_below_ref_high")
        elif cap_idx < len(tdays):
            xi, exit_type = cap_idx, "time_exit_Nday"
        else:
            continue          # neither SL (not yet triggered within cap) nor the time exit has data yet
        xpx = float(cl[xi]); ret = (xpx - entry_px) / entry_px * 100.0
        rows.append({**base, "breakout_date": tdays[f], "days_listing_to_breakout": (tdays[f] - listing_date).days,
                     "gap_fill": bool(op[f] > ref_hi + 1e-9), "entry_price": round(entry_px, 3), "holding_period_cap": N,
                     "exit_date": tdays[xi], "exit_price": round(xpx, 3), "actual_days_held": xi - f, "exit_type": exit_type,
                     "return_pct": round(ret, 3)})
    return rows, {**base, "reason": "qualifying_breakout", "qualifies": True, "breakout_date": tdays[f],
                  "days_listing_to_breakout": (tdays[f] - listing_date).days, "entry_price": round(entry_px, 3),
                  "sl_ever_triggered": sl_idx is not None, "sl_on_breakout_day": sl_idx == f if sl_idx is not None else False}


def main():
    U = pd.read_csv(rb.RESULTS / "newly_listed_universe_combined.csv", parse_dates=["first_date"])
    U["first_date"] = U["first_date"].dt.date
    print(f"universe: {len(U)} stocks", flush=True)

    all_rows, statuses = [], []
    for i, r in enumerate(U.itertuples(), 1):
        rows, status = scan_stock(r.symbol, r.first_date)
        if rows: all_rows.extend(rows)
        statuses.append(status)
        if i % 150 == 0: print(f"  ...{i}/{len(U)}", flush=True)
    R = pd.DataFrame(all_rows)
    ST = pd.DataFrame(statuses)

    n_data_issue = int((ST["reason"] == "no_or_insufficient_price_data").sum())
    n_no_breakout = int((ST["reason"] == "no_breakout_within_3_months").sum())
    n_qualified = int((ST["reason"] == "qualifying_breakout").sum())
    n_evaluable = len(U) - n_data_issue
    Q = ST[ST["qualifies"] == True]
    n_sl_ever = int(Q["sl_ever_triggered"].sum()); n_sl_day0 = int(Q["sl_on_breakout_day"].sum())
    print(f"\n=== PASS-RATE ===\nuniverse: {len(U)} | data issues: {n_data_issue} | evaluable: {n_evaluable} | "
          f"qualifying breakout: {n_qualified} ({n_qualified/n_evaluable*100:.1f}%) | never broke out: {n_no_breakout}\n"
          f"of the {n_qualified} qualifying: SL triggers AT SOME POINT (within full available history): {n_sl_ever} ({n_sl_ever/n_qualified*100:.1f}%) | "
          f"SL on the BREAKOUT DAY ITSELF (failed breakout, same-day close back under level): {n_sl_day0}", flush=True)

    def agg(g):
        return pd.Series({"n_trades": len(g), "avg_return_pct": round(g["return_pct"].mean(), 3), "median_return_pct": round(g["return_pct"].median(), 3),
                           "win_rate_pct": round((g["return_pct"] > 0).mean() * 100, 1), "std_pct": round(g["return_pct"].std(), 3),
                           "pct_exited_by_SL": round((g["exit_type"] != "time_exit_Nday").mean() * 100, 1),
                           "avg_actual_days_held": round(g["actual_days_held"].mean(), 2)})
    G = R.groupby("holding_period_cap").apply(agg, include_groups=False).reset_index()
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 25); pd.set_option("display.max_rows", 30)
    print("\n=== SWEEP TABLE (holding-period CAP T+1..T+20, WITH stop-loss) ===\n" + G.to_string(index=False), flush=True)

    WIN = 5; best_win, best_mean = None, -1e18
    hp = G.set_index("holding_period_cap")["avg_return_pct"]
    for i in range(1, 21 - WIN + 1):
        w = list(range(i, i + WIN))
        m = hp.loc[w].mean()
        if m > best_mean:
            best_mean, best_win = m, w
    print(f"\n=== BEST STABLE {WIN}-DAY HOLDING-CAP REGION ===\n  T+{best_win[0]}..T+{best_win[-1]} -> mean avg return {best_mean:.2f}% "
          f"(individual: {[round(hp.loc[w],2) for w in best_win]})", flush=True)
    single_best = int(hp.idxmax())
    print(f"  single best holding-period cap (reference only): T+{single_best} -> {hp.loc[single_best]:.2f}%", flush=True)

    # side-by-side vs the no-SL version, if its output exists
    try:
        NOSL = pd.read_excel(rb.RESULTS / "ipo_1week_breakout" / "ipo_1week_breakout.xlsx", "Sweep_table")
        cmp = G[["holding_period_cap", "n_trades", "avg_return_pct", "median_return_pct", "win_rate_pct"]].merge(
            NOSL[["holding_period", "avg_return_pct", "median_return_pct", "win_rate_pct"]].rename(
                columns={"holding_period": "holding_period_cap", "avg_return_pct": "avg_pct_NO_SL", "median_return_pct": "median_pct_NO_SL", "win_rate_pct": "win_pct_NO_SL"}),
            on="holding_period_cap")
        print("\n=== WITH-SL vs NO-SL comparison ===\n" + cmp.to_string(index=False), flush=True)
    except FileNotFoundError:
        cmp = pd.DataFrame()

    centre_N = best_win[len(best_win) // 2]
    detail = R[R["holding_period_cap"] == centre_N][["symbol", "listing_date", "ref_high", "ref_low", "breakout_date", "days_listing_to_breakout",
                                                       "gap_fill", "entry_price", "holding_period_cap", "exit_date", "actual_days_held", "exit_type",
                                                       "exit_price", "return_pct"]].sort_values("return_pct", ascending=False)
    print(f"\nper-trade detail at region-centre T+{centre_N} cap (n={len(detail)}):\n" + detail.to_string(index=False), flush=True)

    fn = OUT / "ipo_1week_breakout_sl.xlsx"
    with pd.ExcelWriter(fn, engine="openpyxl") as w:
        pd.DataFrame([{"note": "Stop-loss added: exit at the CLOSE of the first day (starting from the breakout day itself) that closes below "
                                "the locked 5-day high (the breakout level). The T+1..T+20 sweep is now a CAP -- actual exit is whichever comes "
                                "first, the SL or the cap. ASSUMPTION FLAGGED: SL check starts on the breakout day itself (same-day failed "
                                "breakouts can exit same-day) -- see module docstring if you meant day-after instead."}]).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        G.to_excel(w, sheet_name="Sweep_table", index=False)
        if len(cmp): cmp.to_excel(w, sheet_name="With_vs_NoSL", index=False)
        pd.DataFrame({"holding_period_cap": best_win, "avg_return_pct": [round(hp.loc[x], 3) for x in best_win]}).to_excel(w, sheet_name="Best_stable_region", index=False)
        detail.to_excel(w, sheet_name="Trades_detail_centre_N", index=False)
        R.to_excel(w, sheet_name="All_trade_holding_rows", index=False)
        ST.to_excel(w, sheet_name="All_stocks_status", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 34)
    print(f"\nSaved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
