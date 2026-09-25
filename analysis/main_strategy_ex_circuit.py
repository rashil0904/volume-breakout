# -*- coding: utf-8 -*-
"""
main_strategy_ex_circuit.py
===========================
Filtered variant of the main strategy (mcap ₹1,500–5,000 Cr, lookback 36, volume 6x,
3:15pm entry, +5% daily return, 09:45/12:00 conditional split + 14% target, ₹5L pool /
₹1L per trade) that EXCLUDES trades whose entry-day return was >= 19.95% (at/near the
20% upper circuit, where fills and next-day behaviour are unreliable).

Reuses the canonical trade list (final_performance_report.build_trades) — entries/exits
are NOT recomputed, only filtered. The exclusion field is the diagnostic table's
return_pct_vs_prev_close — the EXACT field the +5% entry condition screens on:
  return_pct_vs_prev_close = (15:00-candle open − prev-day VWAP close) / prev-day VWAP close × 100
(prev close is VWAP-based, per the existing +5% filter — no new close definition introduced).

All fpr formulas (Sharpe rf=7.5%/√252, daily win/loss streaks, period extremes) are
replicated exactly, so the unfiltered column matches the existing final report.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import final_performance_report as fpr

OUTDIR = rb.RESULTS / "ex_circuit"
BASE_POOL, EXPENSE, RF_ANNUAL = fpr.BASE_POOL, fpr.EXPENSE, fpr.RF_ANNUAL
THRESHOLD = 19.95            # inclusive: exclude return_at_entry_pct >= 19.95
EXIT_CATS = ["early_target_pre_t1", "positive_at_t1",
             "early_target_between_t1_t2", "exit_at_t2_no_target"]


def period_tables(T):
    def tab(col):
        return pd.DataFrame([{col if col != "exit_date" else "date": k, **fpr.pmetrics(g),
                              "gross_is_winning": bool(g["gross_pnl"].sum() > 0),
                              "net_is_winning": bool(g["net_pnl"].sum() > 0)}
                             for k, g in T.groupby(col)]).sort_values(
            col if col != "exit_date" else "date")
    return (tab("month"), tab("quarter"), tab("half_year"), tab("year"), tab("exit_date"))


def _streaks(ex_dates, pnls):
    bw_len = bl_len = 0
    bw = bl = (None, None, 0.0)
    ct = None; cl = 0; cs = ce = None; cp = 0.0
    for d, p in zip(ex_dates, pnls):
        typ = "win" if p > 0 else "los"
        if typ == ct:
            cl += 1; cp += p; ce = d
        else:
            ct, cl, cp, cs, ce = typ, 1, float(p), d, d
        if ct == "win" and cl > bw_len:
            bw_len, bw = cl, (cs, ce, cp)
        if ct == "los" and cl > bl_len:
            bl_len, bl = cl, (cs, ce, cp)
    return bw_len, bw, bl_len, bl


def metrics_rows(T):
    """List of (metric, gross, net) replicating fpr's summary exactly, for any T."""
    n = len(T)
    gw, gl = T[T["gross_pnl"] > 0], T[T["gross_pnl"] <= 0]
    nw, nl = T[T["net_pnl"] > 0], T[T["net_pnl"] <= 0]
    monthly, quarterly, halfy, yearly, daily_perf = period_tables(T)

    daily = T.groupby("entry_date").agg(g=("gross_pnl", "sum"), nt=("net_pnl", "sum"),
                                        c=("capital_deployed", "sum")).reset_index()
    daily["g_ret"] = daily["g"] / daily["c"] * 100
    daily["n_ret"] = daily["nt"] / daily["c"] * 100
    rf_daily = RF_ANNUAL / 252 * 100

    def sharpe(s):
        return round((s.mean() - rf_daily) / s.std(ddof=1) * np.sqrt(252), 4)

    de = (T.groupby("exit_date").agg(g=("gross_pnl", "sum"), nt=("net_pnl", "sum"))
          .reset_index().sort_values("exit_date"))
    ex_dates = de["exit_date"].tolist()
    g_wl, g_w, g_ll, g_l = _streaks(ex_dates, de["g"].values)
    n_wl, n_w, n_ll, n_l = _streaks(ex_dates, de["nt"].values)

    def per(t):
        return f"{t[0]} -> {t[1]}" if t[0] is not None else "—"

    def S(m, g, nv): return {"metric": m, "gross": g, "net": nv}
    rows = [
        S("n_trades", n, n),
        S("total_return_fixedbase_pct", round(T["gross_pnl"].sum()/BASE_POOL*100, 4), round(T["net_pnl"].sum()/BASE_POOL*100, 4)),
        S("total_pnl_inr", round(T["gross_pnl"].sum(), 0), round(T["net_pnl"].sum(), 0)),
        S("win_rate_pct", round(len(gw)/n*100, 2), round(len(nw)/n*100, 2)),
        S("avg_return_per_trade_pct", round(T["gross_ret"].mean(), 4), round(T["net_ret"].mean(), 4)),
        S("median_return_per_trade_pct", round(T["gross_ret"].median(), 4), round(T["net_ret"].median(), 4)),
        S("avg_return_winning_trades_pct", round(gw["gross_ret"].mean(), 4), round(nw["net_ret"].mean(), 4)),
        S("avg_return_losing_trades_pct", round(gl["gross_ret"].mean(), 4), round(nl["net_ret"].mean(), 4)),
        S("median_return_winning_trades_pct", round(gw["gross_ret"].median(), 4), round(nw["net_ret"].median(), 4)),
        S("median_return_losing_trades_pct", round(gl["gross_ret"].median(), 4), round(nl["net_ret"].median(), 4)),
        S("n_winning_trades", len(gw), len(nw)),
        S("n_losing_trades", len(gl), len(nl)),
        S("n_winning_days", int((daily["g"] > 0).sum()), int((daily["nt"] > 0).sum())),
        S("n_losing_days", int((daily["g"] <= 0).sum()), int((daily["nt"] <= 0).sum())),
        S("max_consecutive_winning_days", g_wl, n_wl),
        S("max_consecutive_winning_days_pnl_inr", round(g_w[2], 0), round(n_w[2], 0)),
        S("max_consecutive_winning_days_period", per(g_w), per(n_w)),
        S("max_consecutive_losing_days", g_ll, n_ll),
        S("max_consecutive_losing_days_pnl_inr", round(g_l[2], 0), round(n_l[2], 0)),
        S("max_consecutive_losing_days_period", per(g_l), per(n_l)),
        S("avg_return_per_month_pct", round(monthly["gross_total_return_fixedbase_pct"].mean(), 4), round(monthly["net_total_return_fixedbase_pct"].mean(), 4)),
        S("avg_return_per_quarter_pct", round(quarterly["gross_total_return_fixedbase_pct"].mean(), 4), round(quarterly["net_total_return_fixedbase_pct"].mean(), 4)),
        S("avg_return_per_half_year_pct", round(halfy["gross_total_return_fixedbase_pct"].mean(), 4), round(halfy["net_total_return_fixedbase_pct"].mean(), 4)),
        S("avg_return_per_year_pct", round(yearly["gross_total_return_fixedbase_pct"].mean(), 4), round(yearly["net_total_return_fixedbase_pct"].mean(), 4)),
        S("sharpe_ratio", sharpe(daily["g_ret"]), sharpe(daily["n_ret"])),
        S("avg_capital_deployed_per_trade", round(T["capital_deployed"].mean(), 0), round(T["capital_deployed"].mean(), 0)),
    ]

    def _lbl(v):
        return str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v)

    def add_extremes(df, name, label_col, plbl, llbl):
        gmax = df.loc[df["gross_total_pnl_inr"].idxmax()]; nmax = df.loc[df["net_total_pnl_inr"].idxmax()]
        gmin = df.loc[df["gross_total_pnl_inr"].idxmin()]; nmin = df.loc[df["net_total_pnl_inr"].idxmin()]
        rows.extend([
            S(f"highest_profit_{name}_pnl_inr", round(float(gmax["gross_total_pnl_inr"]), 0), round(float(nmax["net_total_pnl_inr"]), 0)),
            S(f"highest_profit_{name}_return_pct", round(float(gmax["gross_total_return_fixedbase_pct"]), 4), round(float(nmax["net_total_return_fixedbase_pct"]), 4)),
            S(plbl, _lbl(gmax[label_col]), _lbl(nmax[label_col])),
            S(f"highest_loss_{name}_pnl_inr", round(float(gmin["gross_total_pnl_inr"]), 0), round(float(nmin["net_total_pnl_inr"]), 0)),
            S(f"highest_loss_{name}_return_pct", round(float(gmin["gross_total_return_fixedbase_pct"]), 4), round(float(nmin["net_total_return_fixedbase_pct"]), 4)),
            S(llbl, _lbl(gmin[label_col]), _lbl(nmin[label_col])),
        ])
    add_extremes(daily_perf, "day", "date", "highest_profit_day", "highest_loss_day")
    add_extremes(monthly, "month", "month", "highest_profit_month", "highest_loss_month")
    add_extremes(quarterly, "quarter", "quarter", "highest_profit_quarter", "highest_loss_quarter")
    add_extremes(yearly, "year", "year", "highest_profit_year", "highest_loss_year")
    return rows


def exit_breakdown(T, label):
    n = len(T)
    out = []
    for cat in EXIT_CATS:
        s = T[T["exit_type"] == cat]
        out.append({
            "set": label, "exit_type": cat, "n": len(s),
            "pct_of_trades": round(len(s) / n * 100, 2) if n else 0,
            "gross_avg_return_pct": round(s["gross_ret"].mean(), 4) if len(s) else np.nan,
            "gross_median_return_pct": round(s["gross_ret"].median(), 4) if len(s) else np.nan,
            "net_avg_return_pct": round(s["net_ret"].mean(), 4) if len(s) else np.nan,
            "net_median_return_pct": round(s["net_ret"].median(), 4) if len(s) else np.nan,
        })
    return pd.DataFrame(out)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── canonical trades + entry-day return from the SAME field the +5% filter uses ──
    T = fpr.build_trades()
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "return_pct_vs_prev_close"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    ret_map = diag.set_index(["symbol", "date"])["return_pct_vs_prev_close"].to_dict()
    T["return_at_entry_pct"] = [ret_map.get((s, d), np.nan) for s, d in zip(T["symbol"], T["entry_date"])]
    miss = int(T["return_at_entry_pct"].isna().sum())
    if miss:
        print(f"  WARNING: {miss} trades missing entry-day return — kept (not excluded)")

    excl = T[T["return_at_entry_pct"] >= THRESHOLD].copy()
    filt = T[~(T["return_at_entry_pct"] >= THRESHOLD)].copy()   # keeps NaN too

    print(f"Total trades: {len(T):,}")
    print(f"Excluded (return_at_entry >= {THRESHOLD}%): {len(excl):,} "
          f"({len(excl)/len(T)*100:.2f}% of total)")
    print(f"Surviving (filtered): {len(filt):,}")
    if len(excl):
        eg = excl["gross_pnl"]
        print(f"\n  EXCLUDED-GROUP stats (near-circuit trades):")
        print(f"    n={len(excl)} | win_rate={round((eg>0).mean()*100,2)}% | "
              f"avg_ret={round(excl['gross_ret'].mean(),4)}% | "
              f"median_ret={round(excl['gross_ret'].median(),4)}% | "
              f"total_pnl=₹{eg.sum():,.0f} (gross) | ₹{excl['net_pnl'].sum():,.0f} (net)")
        print(f"    entry-return range: {excl['return_at_entry_pct'].min():.2f}% – "
              f"{excl['return_at_entry_pct'].max():.2f}%")

    # ── comparison table ──
    allm = pd.DataFrame(metrics_rows(T)).rename(columns={"gross": "all_gross", "net": "all_net"})
    fim = pd.DataFrame(metrics_rows(filt)).rename(columns={"gross": "filt_gross", "net": "filt_net"})
    comp = allm.merge(fim, on="metric")
    comp = comp[["metric", "all_gross", "all_net", "filt_gross", "filt_net"]]

    exit_all = exit_breakdown(T, "all_trades")
    exit_filt = exit_breakdown(filt, "filtered_ex_circuit")
    exit_tbl = pd.concat([exit_all, exit_filt], ignore_index=True)

    fcols = ["entry_date", "symbol", "return_at_entry_pct", "entry_price", "shares",
             "capital_deployed", "exit_date", "exit_price", "exit_type", "exit_time",
             "gross_ret", "gross_pnl", "net_ret", "net_pnl"]
    # ex-circuit period breakdowns (same helper/format as final_performance_report.xlsx)
    ex_sheets = fpr.ex_circuit_period_sheets(T)

    with pd.ExcelWriter(OUTDIR / "main_strategy_ex_circuit.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="comparison", index=False)
        exit_tbl.to_excel(w, sheet_name="exit_type_breakdown", index=False)
        filt[fcols].to_excel(w, sheet_name="filtered_trades", index=False)
        excl[fcols].to_excel(w, sheet_name="excluded_trades", index=False)
        for name, df in ex_sheets.items():
            df.to_excel(w, sheet_name=name, index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 110)
    print("COMPARISON — All trades  vs  Excluding entry return >= 19.95%")
    print("=" * 110)
    print(comp.to_string(index=False))
    print("\n--- EXIT-TYPE BREAKDOWN ---")
    print(exit_tbl.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
