# -*- coding: utf-8 -*-
"""newly_listed_first_results_rally_filter.py — adds a pre-entry filter to the SIMPLE-RULE version of the newly-listed
first-quarterly-result strategy (newly_listed_first_results_simple_rule.py):
  during_market -> buy D open,  sell D close
  post_market   -> buy D close, sell next trading day open
  non-trading-day filing -> treated like post_market (prev close -> next open)
  pre_market    -> prev close -> D open   [not user-specified; kept from the simple-rule script]

FILTER: skip (do not take) any trade where the stock already rallied >= 7% over the 3 trading days immediately
before the entry price is set, i.e. using only information known at entry time:
  during_market (entry = D open, before the result): rally measured close[D-4] -> close[D-1] (3-day return
      ending the trading day BEFORE the filing day -- D's own candle is not yet known at the open).
  post_market / non-trading-day (entry = D close or prior close, after D's full session): rally measured
      close[D-3] -> close[D] (3-day return ending at the entry reference day's own close).
  pre_market (entry = prior close, before D opens): rally measured close[D-1-3] -> close[D-1] (ending the day
      before the reference/pre-market day, same idea as during_market).
Trades with fewer than 3 prior trading days of history are kept but flagged (rally_pct = NaN, cannot be evaluated).
New file; reuses newly_listed_first_results_simple_rule.py's daily()/rule logic unmodified where possible.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import newly_listed_first_results_backtest as N
import newly_listed_first_results_simple_rule as SR

OUT = rb.RESULTS / "newly_listed_first_results_rally_filter"; OUT.mkdir(parents=True, exist_ok=True)
SRC = rb.RESULTS / "newly_listed_first_results"
RALLY_THRESH = 7.0


def rally_pct(D, tdays, end_date):
    """3-trading-day return ending at end_date (inclusive): (close[end] - close[end-3]) / close[end-3] * 100."""
    if end_date not in D.index:
        return np.nan
    i = tdays.index(end_date)
    if i < 3:
        return np.nan
    c0, c1 = D.loc[tdays[i - 3], "close"], D.loc[end_date, "close"]
    if c0 <= 0:
        return np.nan
    return (c1 - c0) / c0 * 100


def main():
    FR = pd.read_csv(SRC / "first_results_raw.csv", parse_dates=["announcement_datetime"])
    rows, issues = [], []
    for _, r in FR.iterrows():
        sym, dt = r["symbol"], r["announcement_datetime"]
        D = SR.daily(sym, r["source"])
        if D is None:
            issues.append({"symbol": sym, "reason": "no_price_data"}); continue
        tdays = list(D.index); d0 = dt.date(); is_td = d0 in D.index
        sess = N.classify_session(dt)
        prior = [d for d in tdays if d < d0]; later = [d for d in tdays if d > d0]
        if is_td and sess == "during_market":
            rule, ed, ep, xd, xp = "during: D open -> D close", d0, D.loc[d0, "open"], d0, D.loc[d0, "close"]
            rally_end = prior[-1] if prior else None
        elif is_td and sess == "post_market":
            if not later: issues.append({"symbol": sym, "reason": "no_next_trading_day"}); continue
            rule, ed, ep, xd, xp = "post: D close -> next open", d0, D.loc[d0, "close"], later[0], D.loc[later[0], "open"]
            rally_end = d0
        elif is_td and sess == "pre_market":
            if not prior: issues.append({"symbol": sym, "reason": "no_prior_trading_day"}); continue
            rule, ed, ep, xd, xp = "pre: prev close -> D open", prior[-1], D.loc[prior[-1], "close"], d0, D.loc[d0, "open"]
            rally_end = prior[-2] if len(prior) >= 2 else None
        else:
            if not prior or not later: issues.append({"symbol": sym, "reason": "no_prior_or_next_trading_day"}); continue
            rule, ed, ep, xd, xp = "non-trading day: prev close -> next open", prior[-1], D.loc[prior[-1], "close"], later[0], D.loc[later[0], "open"]
            rally_end = prior[-1]
        if not ep or ep <= 0:
            issues.append({"symbol": sym, "reason": "invalid_entry_price"}); continue
        rp = rally_pct(D, tdays, rally_end) if rally_end is not None else np.nan
        rows.append({"symbol": sym, "listing_date": r["listing_date"], "first_result_date": d0, "announce_time": dt.strftime("%H:%M"),
                     "session": sess, "filed_on_trading_day": is_td, "rule": rule, "entry_date": ed, "entry_price": float(ep),
                     "exit_date": xd, "exit_price": float(xp), "return_pct": round((xp - ep) / ep * 100, 3),
                     "rally_3d_pct": round(rp, 2) if rp == rp else np.nan,
                     "rally_evaluable": rp == rp, "rally_ge_7pct": bool(rp >= RALLY_THRESH) if rp == rp else False,
                     "headline": r["headline"]})
    T = pd.DataFrame(rows); DI = pd.DataFrame(issues)
    kept = T[~T["rally_ge_7pct"]].copy(); dropped = T[T["rally_ge_7pct"]].copy()
    print(f"trades: {len(T)} | issues: {len(DI)} | not evaluable (kept, <3 prior trading days): {int((~T['rally_evaluable']).sum())} | "
          f"dropped (rallied >={RALLY_THRESH}% in prior 3 trading days): {len(dropped)}", flush=True)

    def stats(x, label):
        if x.empty: return {"group": label, "n": 0}
        return {"group": label, "n": len(x), "win_pct": round((x > 0).mean() * 100, 1), "avg_pct": round(x.mean(), 3),
                "median_pct": round(x.median(), 3), "sum_pct": round(x.sum(), 1),
                "avg_win_pct": round(x[x > 0].mean(), 3) if (x > 0).any() else np.nan,
                "avg_loss_pct": round(x[x <= 0].mean(), 3) if (x <= 0).any() else np.nan,
                "worst_pct": round(x.min(), 2), "best_pct": round(x.max(), 2)}

    R = [stats(T["return_pct"], "ALL (no filter)"), stats(kept["return_pct"], f"FILTERED (drop rally>={RALLY_THRESH}%)"),
         stats(dropped["return_pct"], "the DROPPED trades themselves")]
    for rule in sorted(T["rule"].unique()):
        R.append(stats(T[T["rule"] == rule]["return_pct"], f"[ALL] {rule}"))
        R.append(stats(kept[kept["rule"] == rule]["return_pct"], f"[FILTERED] {rule}"))
    S = pd.DataFrame(R)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S.to_string(index=False), flush=True)

    print(f"\ndropped trades: {len(dropped)} of {len(T)} ({len(dropped)/len(T)*100:.1f}%)", flush=True)
    print(dropped.groupby("rule").size().to_string(), flush=True)
    print("\ndropped trades detail:\n" + dropped[["symbol", "entry_date", "rule", "rally_3d_pct", "return_pct"]].sort_values("rally_3d_pct", ascending=False).to_string(index=False), flush=True)

    fn = OUT / "newly_listed_first_results_rally_filter.xlsx"
    with pd.ExcelWriter(fn, engine="openpyxl") as w:
        pd.DataFrame([{"note": f"Filter: skip entry if the stock rallied >={RALLY_THRESH}% over the 3 trading days immediately before entry (during_market: close[D-4]->close[D-1], ending the day before the filing day since D's own candle is not yet known at the open; post_market/non-trading-day: close[D-3]->close[D], ending the reference day's own close since it is known by the close). Trades with fewer than 3 prior trading days of history are KEPT (filter not evaluable), flagged in rally_evaluable."}]).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        S.to_excel(w, sheet_name="Summary", index=False)
        T.to_excel(w, sheet_name="All_trades_flagged", index=False)
        kept.to_excel(w, sheet_name="Trades_FILTERED_kept", index=False)
        dropped.to_excel(w, sheet_name="Trades_DROPPED_by_filter", index=False)
        if len(DI): DI.to_excel(w, sheet_name="Data_Issues", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 34)
    print(f"\nSaved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
