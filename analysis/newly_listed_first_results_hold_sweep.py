# -*- coding: utf-8 -*-
"""newly_listed_first_results_hold_sweep.py — extends the "newly listed stocks - first quarterly result" backtest
with a THIRD exit variant: instead of exiting on the reaction day itself (open or close), hold further and exit at
the close of trading day T+N AFTER the reaction day, N swept 1..20. Entry (T-1 close), universe, session
classification and reaction-day mapping are UNCHANGED from newly_listed_first_results_backtest.py (same first-
results source, same PEAD-style convention: pre_market->same day; during_market/post_market->next trading day) --
reused via that module's own functions, not reimplemented differently. Also recomputes variants 1 (morning open)
and 2 (closing) alongside, for the three-way comparison. New file; original untouched.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import newly_listed_first_results_backtest as N

OUT = rb.RESULTS / "newly_listed_first_results_hold_sweep"; OUT.mkdir(parents=True, exist_ok=True)
SRC = rb.RESULTS / "newly_listed_first_results"
HOLD_PERIODS = list(range(1, 21))


def build(FR):
    v1, v2, v3, issues = [], [], [], []
    cache = {}
    for _, r in FR.iterrows():
        sym, source, dt = r["symbol"], r["source"], r["announcement_datetime"]
        if sym not in cache:
            cache[sym] = N.load_stock_days(sym, source)
        sd = cache[sym]
        if sd is None or sd.empty:
            issues.append({"symbol": sym, "reason": "no_price_data"}); continue
        tdays = sorted(sd["date"].unique())
        d0 = dt.date(); sess = N.classify_session(dt); is_td = d0 in set(tdays)

        if sess == "pre_market" and is_td:
            reaction_day = d0
        else:
            later = [d for d in tdays if d > d0]
            reaction_day = later[0] if later else None
        if reaction_day is None:
            issues.append({"symbol": sym, "reason": "no_trading_day_after_announcement"}); continue
        earlier = [d for d in tdays if d < reaction_day]
        if not earlier:
            issues.append({"symbol": sym, "reason": "no_trading_day_before_reaction_day"}); continue
        entry_date = earlier[-1]
        entry_rows = sd[sd["date"] == entry_date]
        reaction_rows = sd[sd["date"] == reaction_day]
        if entry_rows.empty or reaction_rows.empty:
            issues.append({"symbol": sym, "reason": "missing_entry_or_reaction_day_candles"}); continue
        entry_price = float(entry_rows.sort_values("ts")["close"].iloc[-1])
        if entry_price <= 0:
            issues.append({"symbol": sym, "reason": "invalid_entry_price"}); continue
        exit_open = float(reaction_rows.sort_values("ts")["open"].iloc[0])
        exit_close = float(reaction_rows.sort_values("ts")["close"].iloc[-1])
        base = {"symbol": sym, "listing_date": r["listing_date"], "first_result_date": d0, "session": sess,
                "reaction_day": reaction_day, "entry_date": entry_date, "entry_price": entry_price, "headline": r["headline"]}
        v1.append({**base, "exit_price": exit_open, "return_pct": round((exit_open - entry_price) / entry_price * 100, 3)})
        v2.append({**base, "exit_price": exit_close, "return_pct": round((exit_close - entry_price) / entry_price * 100, 3)})

        r_idx = tdays.index(reaction_day)
        for N_ in HOLD_PERIODS:
            xi = r_idx + N_
            if xi >= len(tdays):
                continue
            xdate = tdays[xi]
            xpx = float(sd[sd["date"] == xdate].sort_values("ts")["close"].iloc[-1])
            v3.append({**base, "holding_period": N_, "exit_date": xdate, "exit_price": xpx,
                       "days_reaction_to_exit": (xdate - reaction_day).days,
                       "return_pct": round((xpx - entry_price) / entry_price * 100, 3)})
    return pd.DataFrame(v1), pd.DataFrame(v2), pd.DataFrame(v3), pd.DataFrame(issues)


def stats(x):
    if len(x) == 0:
        return {"n": 0}
    return {"n": len(x), "avg_pct": round(x.mean(), 3), "median_pct": round(x.median(), 3), "win_pct": round((x > 0).mean() * 100, 1),
            "std_pct": round(x.std(), 3), "p10": round(x.quantile(.1), 2), "p90": round(x.quantile(.9), 2)}


def main():
    FR = pd.read_csv(SRC / "first_results_raw.csv", parse_dates=["announcement_datetime"])
    print(f"first results: {len(FR)}", flush=True)
    V1, V2, V3, DI = build(FR)
    print(f"variant1 (morning open): {len(V1)} | variant2 (closing): {len(V2)} | variant3 (T+1..T+20) rows: {len(V3)} | issues: {len(DI)}", flush=True)

    G = V3.groupby("holding_period")["return_pct"].apply(lambda x: pd.Series(stats(x))).unstack().reset_index()
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 30)
    print("\n=== SWEEP TABLE: holding period T+1..T+20 past the reaction day ===\n" + G.to_string(index=False), flush=True)

    WIN = 5; hp = G.set_index("holding_period")["avg_pct"]; hp_med = G.set_index("holding_period")["median_pct"]
    best_win_avg, best_mean = None, -1e18
    for i in range(1, 21 - WIN + 1):
        w = list(range(i, i + WIN))
        m = hp.loc[w].mean()
        if m > best_mean:
            best_mean, best_win_avg = m, w
    best_win_med, best_mean_med = None, -1e18
    for i in range(1, 21 - WIN + 1):
        w = list(range(i, i + WIN))
        m = hp_med.loc[w].mean()
        if m > best_mean_med:
            best_mean_med, best_win_med = m, w
    print(f"\n=== BEST STABLE {WIN}-DAY REGION BY AVG ===\n  T+{best_win_avg[0]}..T+{best_win_avg[-1]} -> mean avg {best_mean:.3f}% "
          f"(individual: {[round(hp.loc[w],2) for w in best_win_avg]})", flush=True)
    print(f"=== BEST STABLE {WIN}-DAY REGION BY MEDIAN ===\n  T+{best_win_med[0]}..T+{best_win_med[-1]} -> mean median {best_mean_med:.3f}% "
          f"(individual: {[round(hp_med.loc[w],2) for w in best_win_med]})", flush=True)

    # three-way comparison
    centre_avg = best_win_avg[len(best_win_avg)//2]; centre_med = best_win_med[len(best_win_med)//2]
    s1 = stats(V1["return_pct"]); s2 = stats(V2["return_pct"])
    s3_avgbest = stats(V3[V3["holding_period"] == centre_avg]["return_pct"])
    s3_medbest = stats(V3[V3["holding_period"] == centre_med]["return_pct"])
    CMP = pd.DataFrame([{"variant": "1_MORNING_OPEN (reaction day open)", **s1},
                        {"variant": "2_CLOSING (reaction day close)", **s2},
                        {"variant": f"3_HOLD_T+{centre_avg} (best-by-avg centre)", **s3_avgbest},
                        {"variant": f"3_HOLD_T+{centre_med} (best-by-median centre)", **s3_medbest}])
    print("\n=== THREE-WAY COMPARISON ===\n" + CMP.to_string(index=False), flush=True)

    # outlier check on the avg-best cell
    x = V3[V3["holding_period"] == centre_avg]["return_pct"].sort_values()
    print(f"\noutlier check, T+{centre_avg}: avg incl all = {x.mean():.2f}% | avg excl top-5 winners = {x.iloc[:-5].mean():.2f}% | "
          f"top-5: {x.iloc[-5:].round(1).tolist()}", flush=True)

    fn = OUT / "newly_listed_first_results_hold_sweep.xlsx"
    with pd.ExcelWriter(fn, engine="openpyxl") as w:
        pd.DataFrame([{"note": "Variant 3 = hold PAST the reaction day, exit at close of trading day T+N after the reaction day "
                                "(N=1..20). Entry, universe, session classification and reaction-day mapping are unchanged from "
                                "the original two-variant backtest. Sample: only the first post-listing quarterly result per stock "
                                "qualifies -- narrow, exploratory. Both an avg-based and a median-based stable region are reported "
                                "since prior IPO-strategy work showed averages here can be tail-outlier driven."}]).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        G.to_excel(w, sheet_name="Sweep_table_T1_T20", index=False)
        CMP.to_excel(w, sheet_name="Three_way_comparison", index=False)
        V1.to_excel(w, sheet_name="Trades_MorningOpen", index=False)
        V2.to_excel(w, sheet_name="Trades_Closing", index=False)
        V3.to_excel(w, sheet_name="Trades_HoldSweep_all_N", index=False)
        if len(DI): DI.to_excel(w, sheet_name="Data_Issues", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 34)
    print(f"\nSaved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
