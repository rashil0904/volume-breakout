# -*- coding: utf-8 -*-
"""ipo_consolidation_breakout_backtest.py — "newly listed stock enters a tight consolidation within its first 3
months, trade the breakout" strategy. 3-WAY GRID confirmed with the user: window length (5-20 trading days) x
range-tightness threshold (5-10%) x post-breakout holding period (5-20 trading days) = 16 x 6 x 16 = 1,536 cells,
each window length tested INDEPENDENTLY (not just the longest valid window per stock).

UNIVERSE: the existing 801-stock newly-listed universe (results/newly_listed_universe_combined.csv), listing date
proxied by each stock's own first available 1-min candle (first_date), all now in master_data/ after the folder merge.

DETECTION (per stock, per window length L, per tightness threshold X%):
  Scan trading days within [first_date, first_date + 3 calendar months) for the EARLIEST L-consecutive-trading-day
  window (start and end both inside the 3-month span) where (window_high - window_low) / window_low * 100 <= X.
  window_high/low = max(daily high) / min(daily low) over the L days. Only the first qualifying window per (stock,L,X)
  is used -- this is a DIFFERENT, independent scan for every L (a stock can have a different, unrelated window at L=5
  than at L=12), per the confirmed design.

BREAKOUT (touch-based, using daily high/low as the intraday touch proxy -- a daily high/low IS the exact highest/
lowest traded price that day, so it equals a "touch" at 1-min resolution): scanning forward day-by-day after the
window ends (no time limit), the first day where daily high > window_high (upside) or daily low < window_low
(downside). If both trigger on the same day, flagged as an ambiguous same-day dual breakout and NOT traded.
LONG-ONLY on upside breakouts (default, per instruction); downside breakouts are counted/reported as a diagnostic,
not traded.

ENTRY PRICE: the breakout level (window_high) if the breakout day's open <= window_high (a stop order would fill
exactly at the trigger); if the day GAPPED open above window_high, entry = that day's open (the trigger was already
passed, so a stop order fills at the open, not at the stale lower level).

EXIT: close of the Nth trading day after the breakout day, for every N in 5..20 independently (no re-scan needed --
the breakout is fixed per (stock,L,X); only the exit date/price varies with N). Trades right-censored (not enough
days of data yet) are dropped for that N, not counted.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUT = rb.RESULTS / "ipo_consolidation_breakout"; OUT.mkdir(parents=True, exist_ok=True)
IST = "Asia/Kolkata"
TIGHTNESS = list(range(5, 11))          # 5,6,7,8,9,10 %
WINDOW_LENS = list(range(5, 21))        # 5..20 trading days
HOLD_PERIODS = list(range(5, 21))       # 5..20 trading days
MONTHS = 3


def daily_ohlc(sym):
    fn = rb.MASTER_DIR / f"{sym}.parquet"
    if not fn.exists():
        return None
    df = pd.read_parquet(fn, columns=["timestamp", "open", "high", "low", "close"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    df = df.assign(date=ts.dt.date).sort_values("timestamp")
    g = df.groupby("date")
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    return d.sort_index()


def scan_stock(sym, listing_date):
    D = daily_ohlc(sym)
    if D is None or len(D) < 5:
        return [], {"symbol": sym, "reason": "no_or_insufficient_price_data"}
    tdays = list(D.index)
    hi = D["high"].values; lo = D["low"].values; op = D["open"].values; cl = D["close"].values
    end3 = (pd.Timestamp(listing_date) + pd.DateOffset(months=MONTHS)).date()
    in3 = [i for i, d in enumerate(tdays) if listing_date <= d < end3]
    if len(in3) < 5:
        return [], {"symbol": sym, "reason": "fewer_than_5_trading_days_in_first_3_months"}
    lo3, hi3 = in3[0], in3[-1]     # contiguous by construction (date-range mask on a sorted trading-day list)

    rows, no_window = [], []
    for L in WINDOW_LENS:
        if hi3 - lo3 + 1 < L:
            no_window.append((L, "window_would_extend_past_3_months")); continue
        ends = np.arange(lo3 + L - 1, hi3 + 1)
        starts = ends - L + 1
        w_hi = np.array([hi[s:e + 1].max() for s, e in zip(starts, ends)])
        w_lo = np.array([lo[s:e + 1].min() for s, e in zip(starts, ends)])
        tight = (w_hi - w_lo) / w_lo * 100.0

        for X in TIGHTNESS:
            ok = np.where(tight <= X)[0]
            if len(ok) == 0:
                continue
            k = ok[0]                          # earliest qualifying end position (index into ends[])
            e, s = int(ends[k]), int(starts[k])
            wh, wl, tt = float(w_hi[k]), float(w_lo[k]), float(tight[k])

            # breakout scan forward from the day after the window ends
            f = None; btype = None
            for j in range(e + 1, len(tdays)):
                up = hi[j] > wh + 1e-9; dn = lo[j] < wl - 1e-9
                if up and dn:
                    f, btype = j, "AMBIGUOUS_same_day_both"; break
                if up:
                    f, btype = j, "upside"; break
                if dn:
                    f, btype = j, "downside"; break
            base = {"symbol": sym, "listing_date": listing_date, "window_len": L, "tightness_threshold": X,
                    "window_start": tdays[s], "window_end": tdays[e], "window_high": round(wh, 3), "window_low": round(wl, 3),
                    "tightness_actual_pct": round(tt, 3)}
            if f is None:
                rows.append({**base, "breakout_date": None, "breakout_type": "none_found_yet", "traded": False})
                continue
            base.update({"breakout_date": tdays[f], "breakout_type": btype})
            if btype != "upside":
                rows.append({**base, "traded": False}); continue
            entry_px = wh if op[f] <= wh + 1e-9 else float(op[f])
            for N in HOLD_PERIODS:
                xi = f + N
                if xi >= len(tdays):
                    continue
                xpx = float(cl[xi]); ret = (xpx - entry_px) / entry_px * 100.0
                rows.append({**base, "traded": True, "entry_price": round(entry_px, 3), "gap_fill": op[f] > wh + 1e-9,
                             "holding_period": N, "exit_date": tdays[xi], "exit_price": round(xpx, 3), "return_pct": round(ret, 3)})
    return rows, None


def main():
    U = pd.read_csv(rb.RESULTS / "newly_listed_universe_combined.csv", parse_dates=["first_date"])
    U["first_date"] = U["first_date"].dt.date
    print(f"universe: {len(U)} stocks", flush=True)

    all_rows, issues = [], []
    for i, r in enumerate(U.itertuples(), 1):
        rows, issue = scan_stock(r.symbol, r.first_date)
        all_rows.extend(rows)
        if issue: issues.append(issue)
        if i % 150 == 0: print(f"  ...{i}/{len(U)}", flush=True)
    R = pd.DataFrame(all_rows)
    print(f"\nstocks skipped (no/insufficient data): {len(issues)}", flush=True)
    print(f"total (symbol,L,threshold) window-defs found: {R.drop_duplicates(['symbol','window_len','tightness_threshold']).shape[0] if len(R) else 0}", flush=True)

    windows = R.drop_duplicates(["symbol", "window_len", "tightness_threshold"])
    print(f"  breakout types (per window-def): {windows['breakout_type'].value_counts(dropna=False).to_dict()}", flush=True)
    traded = R[R["traded"] == True].copy()
    print(f"\ntradeable (stock,L,X) upside breakouts: {traded.drop_duplicates(['symbol','window_len','tightness_threshold']).shape[0]}", flush=True)
    print(f"trade-holding rows (each = one exercisable N-day exit): {len(traded)}", flush=True)

    def agg(g):
        return pd.Series({"n_trades": len(g), "avg_return_pct": round(g["return_pct"].mean(), 3), "median_return_pct": round(g["return_pct"].median(), 3),
                           "win_rate_pct": round((g["return_pct"] > 0).mean() * 100, 1), "std_pct": round(g["return_pct"].std(), 3)})

    G_tx = traded.groupby(["tightness_threshold", "holding_period"]).apply(agg, include_groups=False).reset_index()
    grid_ret = G_tx.pivot(index="tightness_threshold", columns="holding_period", values="avg_return_pct")
    grid_win = G_tx.pivot(index="tightness_threshold", columns="holding_period", values="win_rate_pct")
    grid_n = G_tx.pivot(index="tightness_threshold", columns="holding_period", values="n_trades")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 25)
    print("\n=== GRID (tightness % rows x holding-period days cols): avg return % ===\n" + grid_ret.round(2).to_string())
    print("\n=== GRID: n_trades ===\n" + grid_n.fillna(0).astype(int).to_string())
    print("\n=== GRID: win rate % ===\n" + grid_win.round(1).to_string())

    G_lx = traded.groupby(["window_len", "tightness_threshold"]).apply(agg, include_groups=False).reset_index()
    grid_lx_ret = G_lx.pivot(index="window_len", columns="tightness_threshold", values="avg_return_pct")
    grid_lx_n = G_lx.pivot(index="window_len", columns="tightness_threshold", values="n_trades")
    print("\n=== GRID (window_len rows x tightness % cols): avg return % ===\n" + grid_lx_ret.round(2).to_string())
    print("\n=== GRID (window_len x tightness): n_trades ===\n" + grid_lx_n.fillna(0).astype(int).to_string())

    # ---- stable-region search on the tightness x holding-period grid (3x3 neighbourhood, min n) ----
    MIN_N = 15
    idxs, cols = list(grid_ret.index), list(grid_ret.columns)
    best = None
    for xi, X in enumerate(idxs):
        for ni, N in enumerate(cols):
            neigh = [(idxs[a], cols[b]) for a in range(max(0, xi - 1), min(len(idxs), xi + 2))
                     for b in range(max(0, ni - 1), min(len(cols), ni + 2))]
            vals = [grid_ret.loc[a, b] for a, b in neigh if pd.notna(grid_ret.loc[a, b])]
            ns = [grid_n.loc[a, b] for a, b in neigh if pd.notna(grid_n.loc[a, b])]
            if len(vals) < 5 or sum(ns) < MIN_N:
                continue
            rec = {"tightness": X, "holding_period": N, "cell_return": grid_ret.loc[X, N], "cell_n": int(grid_n.loc[X, N]),
                   "neigh_mean_return": np.mean(vals), "neigh_min_return": np.min(vals), "neigh_n_cells": len(vals), "neigh_total_trades": int(sum(ns))}
            if best is None or rec["neigh_mean_return"] > best["neigh_mean_return"]:
                best = rec
    print(f"\n=== BEST STABLE 3x3 NEIGHBOURHOOD (min {MIN_N} trades in neighbourhood) ===\n{best}", flush=True)

    single_best = G_tx.loc[G_tx["avg_return_pct"].idxmax()] if len(G_tx) else None
    print(f"\nsingle best CELL (for reference only, overfit risk): {dict(single_best) if single_best is not None else 'n/a'}", flush=True)

    # ---- per-trade detail for the stable region's centre cell ----
    detail = pd.DataFrame()
    if best is not None:
        detail = traded[(traded["tightness_threshold"] == best["tightness"]) & (traded["holding_period"] == best["holding_period"])]
        detail = detail[["symbol", "listing_date", "window_len", "tightness_threshold", "window_start", "window_end", "tightness_actual_pct",
                          "breakout_date", "gap_fill", "entry_price", "holding_period", "exit_date", "exit_price", "return_pct"]].sort_values("return_pct", ascending=False)
        print(f"\nper-trade detail for centre cell (tightness={best['tightness']}%, hold={best['holding_period']}d, n={len(detail)}):\n" + detail.to_string(index=False), flush=True)

    downside = windows[windows["breakout_type"] == "downside"]
    ambiguous = windows[windows["breakout_type"] == "AMBIGUOUS_same_day_both"]
    none_yet = windows[windows["breakout_type"] == "none_found_yet"]
    print(f"\ndiagnostics: downside breakouts (not traded) = {len(downside)} | ambiguous same-day both = {len(ambiguous)} | "
          f"no breakout yet (still inside consolidation-adjacent range) = {len(none_yet)}", flush=True)

    fn = OUT / "ipo_consolidation_breakout.xlsx"
    with pd.ExcelWriter(fn, engine="openpyxl") as w:
        pd.DataFrame([{"note": "3-way grid confirmed with user: window length (5-20d) x tightness (5-10%) x holding period (5-20d) tested "
                                "independently per stock -- each window length is its own scan, not nested/dependent on other lengths. "
                                "Entry = breakout level (window_high) or the breakout day's open if it gapped past that level. Exit = close "
                                "N trading days after breakout. Downside breakouts and ambiguous same-day dual breakouts are reported but NOT traded (long-only)."},
                       {"note": f"SAMPLE SIZE: total tradeable upside-breakout (stock,L,X) combos = {traded.drop_duplicates(['symbol','window_len','tightness_threshold']).shape[0]} "
                                f"out of {len(U)} stocks in the universe. Grid cells (tightness x holding period) each average far fewer trades than that "
                                "since they're further split by window length. Treat the 'best' cell/region with real caution."}]).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        grid_ret.to_excel(w, sheet_name="Grid_avg_return_pct")
        grid_win.to_excel(w, sheet_name="Grid_win_rate_pct")
        grid_n.to_excel(w, sheet_name="Grid_n_trades")
        grid_lx_ret.to_excel(w, sheet_name="Grid_by_windowlen_tightness")
        pd.DataFrame([best]).to_excel(w, sheet_name="Best_stable_region", index=False)
        detail.to_excel(w, sheet_name="Trades_stable_region_detail", index=False)
        traded.to_excel(w, sheet_name="All_tradeable_rows", index=False)
        windows.to_excel(w, sheet_name="All_windows_incl_no_breakout", index=False)
        pd.DataFrame(issues).to_excel(w, sheet_name="Stocks_skipped", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 34)
    print(f"\nSaved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
