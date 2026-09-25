# -*- coding: utf-8 -*-
"""newly_listed_first_results_simple_rule.py — first post-listing quarterly result, SIMPLE session rule (user spec):
  during_market (9:15-15:30 filing on a trading day D):  BUY at D OPEN,  SELL at D CLOSE
  post_market   (>15:30 on a trading day D):             BUY at D CLOSE, SELL at next trading day OPEN
  filing on a NON-trading day (weekend/holiday, any time): no session to trade -> treated like post_market:
                                                        BUY at prior trading day CLOSE, SELL at next trading day OPEN
  pre_market    (<9:15 on a trading day D):              BUY at prior trading day CLOSE, SELL at D OPEN   [not specified by user; same
                                                        'enter before, exit at the first open after the news' logic]
Reads results/newly_listed_first_results/first_results_raw.csv (no BSE re-fetch). New folder; nothing else touched.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import newly_listed_first_results_backtest as N

OUT = rb.RESULTS / "newly_listed_first_results_simple_rule"; OUT.mkdir(parents=True, exist_ok=True)
SRC = rb.RESULTS / "newly_listed_first_results"


def daily(sym, source):
    sd = N.load_stock_days(sym, source)
    if sd is None or sd.empty:
        return None
    g = sd.groupby("date")
    return pd.DataFrame({"open": g["open"].first(), "close": g["close"].last()})   # sd is sorted by ts


def stats(T, label):
    if T.empty:
        return {"group": label, "n": 0}
    x = T["return_pct"]
    return {"group": label, "n": len(T), "win_pct": round((x > 0).mean() * 100, 1), "avg_pct": round(x.mean(), 3),
            "median_pct": round(x.median(), 3), "sum_pct": round(x.sum(), 1),
            "avg_win_pct": round(x[x > 0].mean(), 3) if (x > 0).any() else np.nan,
            "avg_loss_pct": round(x[x <= 0].mean(), 3) if (x <= 0).any() else np.nan,
            "worst_pct": round(x.min(), 2), "best_pct": round(x.max(), 2)}


def main():
    FR = pd.read_csv(SRC / "first_results_raw.csv", parse_dates=["announcement_datetime"])
    rows, issues = [], []
    for _, r in FR.iterrows():
        sym, dt = r["symbol"], r["announcement_datetime"]
        D = daily(sym, r["source"])
        if D is None:
            issues.append({"symbol": sym, "reason": "no_price_data"}); continue
        tdays = list(D.index); d0 = dt.date(); is_td = d0 in D.index
        sess = N.classify_session(dt)
        prior = [d for d in tdays if d < d0]; later = [d for d in tdays if d > d0]
        if is_td and sess == "during_market":
            rule, ed, ep, xd, xp = "during: D open -> D close", d0, D.loc[d0, "open"], d0, D.loc[d0, "close"]
        elif is_td and sess == "post_market":
            if not later: issues.append({"symbol": sym, "reason": "no_next_trading_day"}); continue
            rule, ed, ep, xd, xp = "post: D close -> next open", d0, D.loc[d0, "close"], later[0], D.loc[later[0], "open"]
        elif is_td and sess == "pre_market":
            if not prior: issues.append({"symbol": sym, "reason": "no_prior_trading_day"}); continue
            rule, ed, ep, xd, xp = "pre: prev close -> D open", prior[-1], D.loc[prior[-1], "close"], d0, D.loc[d0, "open"]
        else:   # non-trading-day filing
            if not prior or not later: issues.append({"symbol": sym, "reason": "no_prior_or_next_trading_day"}); continue
            rule, ed, ep, xd, xp = "non-trading day: prev close -> next open", prior[-1], D.loc[prior[-1], "close"], later[0], D.loc[later[0], "open"]
        if not ep or ep <= 0:
            issues.append({"symbol": sym, "reason": "invalid_entry_price"}); continue
        rows.append({"symbol": sym, "listing_date": r["listing_date"], "first_result_date": d0, "announce_time": dt.strftime("%H:%M"),
                     "session": sess, "filed_on_trading_day": is_td, "rule": rule, "entry_date": ed, "entry_price": float(ep),
                     "exit_date": xd, "exit_price": float(xp), "return_pct": round((xp - ep) / ep * 100, 3), "headline": r["headline"]})
    T = pd.DataFrame(rows); DI = pd.DataFrame(issues)
    print(f"trades: {len(T)} | issues: {len(DI)}", flush=True)

    S = pd.DataFrame([stats(T, "ALL")] + [stats(g, k) for k, g in T.groupby("rule")])
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S.to_string(index=False), flush=True)

    def bucket(t):
        h, m = map(int, t.split(":")); v = h * 60 + m
        return "09:15-11:00" if v < 660 else "11:00-13:00" if v < 780 else "13:00-14:30" if v < 870 else "14:30-15:30"
    m = T[T["rule"].str.startswith("during")].copy(); m["bucket"] = m["announce_time"].map(bucket)
    Bk = pd.DataFrame([stats(g, b) for b, g in m.groupby("bucket")])
    print("\nduring-market by filing time:\n" + Bk.to_string(index=False), flush=True)
    T["year"] = pd.to_datetime(T["first_result_date"]).dt.year
    Y = pd.DataFrame([stats(g, str(y)) for y, g in T.groupby("year")])
    print("\nby year:\n" + Y.to_string(index=False), flush=True)

    fn = OUT / "newly_listed_first_results_simple_rule.xlsx"
    with pd.ExcelWriter(fn, engine="openpyxl") as w:
        pd.DataFrame([{"note": "during_market: buy D open, sell D close. post_market: buy D close, sell next trading day open. Non-trading-day filing: prev trading close -> next trading open. pre_market (unspecified by user): prev close -> D open."}]).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        S.to_excel(w, sheet_name="Summary", index=False)
        Bk.to_excel(w, sheet_name="During_by_filing_time", index=False)
        Y.to_excel(w, sheet_name="By_year", index=False)
        T.to_excel(w, sheet_name="Trades", index=False)
        if len(DI): DI.to_excel(w, sheet_name="Data_Issues", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 34)
    print(f"\nSaved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
