# -*- coding: utf-8 -*-
"""ipo_1week_breakout_backtest.py — "IPO 1 Week Breakout": buy the breakout of a newly-listed stock's FIRST-5-
TRADING-DAY high, within 3 months of listing. Sweep the post-breakout holding period T+1..T+20.

CONFIRMED (not asked, since already fully specified with a clear default and matches the touch-based convention
used in the sibling ipo_consolidation_breakout_backtest.py): breakout confirmation is TOUCH-BASED, using the daily
high as the intraday-touch proxy (a day's high IS the exact highest price traded that day at 1-min resolution, so
reading it off daily bars gives the identical answer to scanning 1-min data, at far less compute). If you actually
wanted a CLOSE-based confirmation (breakout day's CLOSE above the 5-day high, not just an intraday touch), tell me
and I'll rerun -- flagging this explicitly per your instruction, proceeding with touch-based as specified.

UNIVERSE: the same 801-stock newly-listed universe (results/newly_listed_universe_combined.csv), listing date
proxied by each stock's own first available 1-min candle, all in master_data/.

REFERENCE WINDOW: the stock's first 5 trading days (index 0-4, day 1 = listing day itself). FIXED -- locked_high =
max(daily high) and locked_low = min(daily low) over exactly those 5 days; never extended.

ENTRY: from trading day 6 onward (index >= 5), first day with daily high > locked_high, provided that day falls
within [listing_date, listing_date + 3 calendar months). If the day's open already gapped above locked_high, entry
= that day's open (a stop order can't fill at the stale lower level); otherwise entry = locked_high itself.
Only the FIRST breakout counts -- one trade per stock, or none if the deadline passes with no breakout.

EXIT: close of the Nth trading day after the breakout day, for every N in 1..20 independently. Exits may fall
outside the 3-month window (only the BREAKOUT deadline is 3 months, not the exit) -- right-censored if the stock's
own data doesn't reach that far yet.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
from ipo_consolidation_breakout_backtest import daily_ohlc      # reused unmodified

OUT = rb.RESULTS / "ipo_1week_breakout"; OUT.mkdir(parents=True, exist_ok=True)
REF_DAYS = 5
HOLD_PERIODS = list(range(1, 21))       # T+1 .. T+20
MONTHS = 3


def scan_stock(sym, listing_date):
    D = daily_ohlc(sym)
    if D is None or len(D) < REF_DAYS + 1:
        return None, {"symbol": sym, "reason": "no_or_insufficient_price_data"}
    tdays = list(D.index)
    hi = D["high"].values; lo = D["low"].values; op = D["open"].values; cl = D["close"].values
    if tdays[0] != listing_date:
        # own data doesn't start exactly on the proxied listing date -- use the stock's own first available day as day 1
        pass
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
    rows = []
    for N in HOLD_PERIODS:
        xi = f + N
        if xi >= len(tdays):
            continue
        xpx = float(cl[xi]); ret = (xpx - entry_px) / entry_px * 100.0
        rows.append({**base, "breakout_date": tdays[f], "days_listing_to_breakout": (tdays[f] - listing_date).days,
                     "trading_days_listing_to_breakout": f, "gap_fill": bool(op[f] > ref_hi + 1e-9), "entry_price": round(entry_px, 3),
                     "holding_period": N, "exit_date": tdays[xi], "exit_price": round(xpx, 3), "return_pct": round(ret, 3)})
    return rows, {**base, "reason": "qualifying_breakout", "qualifies": True, "breakout_date": tdays[f],
                  "days_listing_to_breakout": (tdays[f] - listing_date).days, "entry_price": round(entry_px, 3)}


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
    print(f"\n=== PASS-RATE DIAGNOSTIC ===\nuniverse: {len(U)} | skipped (no/insufficient data): {n_data_issue} | "
          f"evaluable: {n_evaluable}\n  qualifying breakout within 3 months: {n_qualified} ({n_qualified/n_evaluable*100:.1f}% of evaluable)\n"
          f"  NEVER broke the 5-day high within 3 months: {n_no_breakout} ({n_no_breakout/n_evaluable*100:.1f}% of evaluable)", flush=True)

    def agg(g):
        return pd.Series({"n_trades": len(g), "avg_return_pct": round(g["return_pct"].mean(), 3), "median_return_pct": round(g["return_pct"].median(), 3),
                           "win_rate_pct": round((g["return_pct"] > 0).mean() * 100, 1), "std_pct": round(g["return_pct"].std(), 3)})
    G = R.groupby("holding_period").apply(agg, include_groups=False).reset_index()
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 25); pd.set_option("display.max_rows", 30)
    print("\n=== SWEEP TABLE (holding period T+1..T+20) ===\n" + G.to_string(index=False), flush=True)

    # stable region: best 5-consecutive-holding-period window (per the brief's own T+8..T+12 example)
    WIN = 5; best_win, best_mean = None, -1e18
    hp = G.set_index("holding_period")["avg_return_pct"]
    for i in range(1, 21 - WIN + 1):
        w = list(range(i, i + WIN))
        m = hp.loc[w].mean()
        if m > best_mean:
            best_mean, best_win = m, w
    print(f"\n=== BEST STABLE {WIN}-DAY HOLDING REGION ===\n  T+{best_win[0]}..T+{best_win[-1]} -> mean avg return {best_mean:.2f}% "
          f"(individual: {[round(hp.loc[w],2) for w in best_win]}, n_trades each: {G.set_index('holding_period').loc[best_win,'n_trades'].tolist()})", flush=True)
    single_best = int(hp.idxmax())
    print(f"  single best holding period (reference only, overfit risk): T+{single_best} -> {hp.loc[single_best]:.2f}%", flush=True)

    centre_N = best_win[len(best_win) // 2]
    detail = R[R["holding_period"] == centre_N][["symbol", "listing_date", "ref_window_start", "ref_window_end", "ref_high", "ref_low",
                                                   "breakout_date", "days_listing_to_breakout", "gap_fill", "entry_price",
                                                   "holding_period", "exit_date", "exit_price", "return_pct"]].sort_values("return_pct", ascending=False)
    print(f"\nper-trade detail at region-centre T+{centre_N} (n={len(detail)}):\n" + detail.to_string(index=False), flush=True)

    fn = OUT / "ipo_1week_breakout.xlsx"
    with pd.ExcelWriter(fn, engine="openpyxl") as w:
        pd.DataFrame([{"note": "Touch-based breakout confirmation (daily high as the intraday-touch proxy), as specified. Entry = the locked "
                                "5-day high, or the breakout day's open if it gapped past that level. Only the FIRST breakout within 3 months "
                                "of listing counts; exit can fall beyond the 3-month mark since only the breakout deadline is 3 months."},
                       {"note": f"SAMPLE SIZE: of {len(U)} universe stocks, only {n_qualified} ({n_qualified/n_evaluable*100:.1f}% of evaluable) "
                                f"ever got a qualifying breakout within 3 months. {n_no_breakout} never did. This is a narrow, selective setup -- "
                                "treat any 'best' holding period with real caution."}]).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        G.to_excel(w, sheet_name="Sweep_table", index=False)
        pd.DataFrame({"holding_period": best_win, "avg_return_pct": [round(hp.loc[x], 3) for x in best_win]}).to_excel(w, sheet_name="Best_stable_region", index=False)
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
