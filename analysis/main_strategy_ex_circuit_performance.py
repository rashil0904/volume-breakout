# -*- coding: utf-8 -*-
"""
main_strategy_ex_circuit_performance.py
=======================================
Full performance report (mirrors final_performance_report.xlsx sheet-for-sheet) for the
CIRCUIT-EXCLUDED main-strategy trade set: mcap ₹1,500–5,000 Cr, lookback 36, volume 6x,
3:15pm entry, +5% daily return, 09:45/12:00 conditional split + 14% target, ₹5L pool /
₹1L per trade, EXCLUDING trades with return_at_entry_pct >= 19.95 (at/near 20% circuit).

Reuses the canonical trade list (fpr.build_trades) + fpr helpers (pmetrics, scaled) and
fpr's CURRENT full-compounding rule (scale up on a positive prior period, scale DOWN /
bake in the loss on a negative one) — identical to final_performance_report.xlsx, so the
two files are directly comparable. Exclusion field = diagnostic return_pct_vs_prev_close
(the exact field the +5% filter screens on).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import final_performance_report as fpr

OUTDIR = rb.RESULTS / "final_report"
XLSX = OUTDIR / "main_strategy_ex_circuit_performance.xlsx"
BP, BA, EXPENSE, RF = fpr.BASE_POOL, fpr.BASE_ALLOC, fpr.EXPENSE, fpr.RF_ANNUAL
THRESHOLD = 19.95


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── filter (reuse existing trades; exclusion on the +5%-filter's own field) ──
    T_all = fpr.build_trades()
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "return_pct_vs_prev_close"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    rmap = diag.set_index(["symbol", "date"])["return_pct_vs_prev_close"].to_dict()
    T_all["return_at_entry_pct"] = [rmap.get((s, d), np.nan)
                                    for s, d in zip(T_all["symbol"], T_all["entry_date"])]
    n_excl = int((T_all["return_at_entry_pct"] >= THRESHOLD).sum())
    T = T_all[~(T_all["return_at_entry_pct"] >= THRESHOLD)].copy()   # keeps NaN
    print(f"Excluded {n_excl} circuit trades ({n_excl/len(T_all)*100:.2f}%); surviving {len(T):,}")

    # ── period tables (fixed-base) ──
    def ptab(col, key=None):
        key = key or col
        return pd.DataFrame([{key: k, **fpr.pmetrics(g)} for k, g in T.groupby(col)]).sort_values(key)
    monthly = ptab("month")
    quarterly = ptab("quarter")
    halfy = ptab("half_year")
    yearly_base = {int(y): fpr.pmetrics(g) for y, g in T.groupby("year")}

    # ── quarterly compounding (CURRENT full-compounding rule) ──
    q_rows = []; pool, alloc = BP, BA
    for q, g in sorted(T.groupby("quarter"), key=lambda x: x[0]):
        cg, cn = fpr.scaled(g, alloc)
        q_rows.append({"quarter": q, **fpr.pmetrics(g),
                       "quarterly_allocation_used": round(alloc, 0),
                       "compounded_pool_value": round(pool, 0),
                       "gross_compounded_return_pct": round(cg / pool * 100, 4),
                       "net_compounded_return_pct": round(cn / pool * 100, 4),
                       "gross_compounded_pnl_inr": round(cg, 0),
                       "net_compounded_pnl_inr": round(cn, 0)})
        f = 1 + cg / pool * 100 / 100; pool *= f; alloc *= f     # full compounding (up or down)
    quarterly_c = pd.DataFrame(q_rows)

    # ── yearly compounding (avg prior-year half-return driver; full compounding) ──
    half_gross = {k: fpr.pmetrics(g)["gross_total_return_fixedbase_pct"] for k, g in T.groupby("half_year")}
    years = sorted(int(y) for y in T["year"].unique())
    y_rows = []; pool, alloc = BP, BA
    for idx, y in enumerate(years):
        if idx > 0:
            py = years[idx - 1]
            hs = [half_gross.get(f"{py}H1"), half_gross.get(f"{py}H2")]
            hs = [h for h in hs if h is not None]
            avg_half = float(np.mean(hs)) if hs else 0.0
            f = 1 + avg_half / 100; pool *= f; alloc *= f        # full compounding (up or down)
        g = T[T["year"] == y]
        cg, cn = fpr.scaled(g, alloc)
        y_rows.append({"year": y, **yearly_base[y],
                       "yearly_allocation_used": round(alloc, 0),
                       "compounded_pool_value": round(pool, 0),
                       "gross_compounded_return_pct": round(cg / pool * 100, 4),
                       "net_compounded_return_pct": round(cn / pool * 100, 4),
                       "gross_compounded_pnl_inr": round(cg, 0),
                       "net_compounded_pnl_inr": round(cn, 0)})
    yearly_c = pd.DataFrame(y_rows)

    # ── daily (sharpe by entry day; winning/losing days) ──
    daily = T.groupby("entry_date").agg(g=("gross_pnl", "sum"), n=("net_pnl", "sum"),
                                        c=("capital_deployed", "sum")).reset_index()
    daily["g_ret"] = daily["g"] / daily["c"] * 100
    daily["n_ret"] = daily["n"] / daily["c"] * 100
    rf_daily = RF / 252 * 100
    def sharpe(s): return round((s.mean() - rf_daily) / s.std(ddof=1) * np.sqrt(252), 4)

    # ── daily_performance sheet ──
    dcols = ["date", "n_trades", "gross_total_pnl_inr", "gross_total_return_fixedbase_pct",
             "net_total_pnl_inr", "net_total_return_fixedbase_pct", "gross_win_rate_pct",
             "net_win_rate_pct", "gross_avg_return_per_trade_pct", "net_avg_return_per_trade_pct",
             "gross_median_return_per_trade_pct", "net_median_return_per_trade_pct",
             "gross_is_winning_day", "net_is_winning_day"]
    daily_perf = pd.DataFrame([
        {"date": k, **fpr.pmetrics(g),
         "gross_is_winning_day": bool(g["gross_pnl"].sum() > 0),
         "net_is_winning_day": bool(g["net_pnl"].sum() > 0)}
        for k, g in T.groupby("exit_date")]).sort_values("date")[dcols]

    # ── streaks (by exit day) ──
    de = (T.groupby("exit_date").agg(g=("gross_pnl", "sum"), n=("net_pnl", "sum"))
          .reset_index().sort_values("exit_date"))
    ex_dates = de["exit_date"].tolist()
    def streaks(pnls):
        bwl = bll = 0; bw = bl = (None, None, 0.0)
        ct = None; cl = 0; cs = ce = None; cp = 0.0
        for d, p in zip(ex_dates, pnls):
            typ = "win" if p > 0 else "los"
            if typ == ct: cl += 1; cp += p; ce = d
            else: ct, cl, cp, cs, ce = typ, 1, float(p), d, d
            if ct == "win" and cl > bwl: bwl, bw = cl, (cs, ce, cp)
            if ct == "los" and cl > bll: bll, bl = cl, (cs, ce, cp)
        return bwl, bw, bll, bl
    g_wl, g_w, g_ll, g_l = streaks(de["g"].values)
    n_wl, n_w, n_ll, n_l = streaks(de["n"].values)
    def per(t): return f"{t[0]} -> {t[1]}" if t[0] is not None else "—"

    # ── summary (mirrors fpr order; n_trades prepended, circuit rows appended) ──
    n = len(T)
    gw, gl = T[T["gross_pnl"] > 0], T[T["gross_pnl"] <= 0]
    nw, nl = T[T["net_pnl"] > 0], T[T["net_pnl"] <= 0]
    def S(m, g, nv): return {"metric": m, "gross": g, "net": nv}
    rows = [
        S("n_trades", n, n),
        S("total_return_fixedbase_pct", round(T["gross_pnl"].sum()/BP*100, 4), round(T["net_pnl"].sum()/BP*100, 4)),
        S("total_pnl_inr", round(T["gross_pnl"].sum(), 0), round(T["net_pnl"].sum(), 0)),
        S("win_rate_pct", round(len(gw)/n*100, 2), round(len(nw)/n*100, 2)),
        S("n_winning_trades", len(gw), len(nw)),
        S("n_losing_trades", len(gl), len(nl)),
        S("n_winning_days", int((daily["g"] > 0).sum()), int((daily["n"] > 0).sum())),
        S("n_losing_days", int((daily["g"] <= 0).sum()), int((daily["n"] <= 0).sum())),
        S("avg_return_per_trade_pct", round(T["gross_ret"].mean(), 4), round(T["net_ret"].mean(), 4)),
        S("avg_pnl_per_trade_inr", round(T["gross_pnl"].mean(), 2), round(T["net_pnl"].mean(), 2)),
        S("avg_return_winning_trades_pct", round(gw["gross_ret"].mean(), 4), round(nw["net_ret"].mean(), 4)),
        S("avg_return_losing_trades_pct", round(gl["gross_ret"].mean(), 4), round(nl["net_ret"].mean(), 4)),
        S("median_return_per_trade_pct", round(T["gross_ret"].median(), 4), round(T["net_ret"].median(), 4)),
        S("median_return_winning_trades_pct", round(gw["gross_ret"].median(), 4), round(nw["net_ret"].median(), 4)),
        S("median_return_losing_trades_pct", round(gl["gross_ret"].median(), 4), round(nl["net_ret"].median(), 4)),
        S("avg_return_per_month_pct", round(monthly["gross_total_return_fixedbase_pct"].mean(), 4), round(monthly["net_total_return_fixedbase_pct"].mean(), 4)),
        S("avg_return_per_quarter_pct", round(quarterly["gross_total_return_fixedbase_pct"].mean(), 4), round(quarterly["net_total_return_fixedbase_pct"].mean(), 4)),
        S("avg_return_per_half_year_pct", round(halfy["gross_total_return_fixedbase_pct"].mean(), 4), round(halfy["net_total_return_fixedbase_pct"].mean(), 4)),
        S("avg_return_per_year_pct", round(pd.DataFrame(yearly_base).T["gross_total_return_fixedbase_pct"].mean(), 4), round(pd.DataFrame(yearly_base).T["net_total_return_fixedbase_pct"].mean(), 4)),
        S("sharpe_ratio", sharpe(daily["g_ret"]), sharpe(daily["n_ret"])),
        S("max_consecutive_winning_days", g_wl, n_wl),
        S("max_consecutive_winning_days_pnl_inr", round(g_w[2], 0), round(n_w[2], 0)),
        S("max_consecutive_winning_days_period", per(g_w), per(n_w)),
        S("max_consecutive_losing_days", g_ll, n_ll),
        S("max_consecutive_losing_days_pnl_inr", round(g_l[2], 0), round(n_l[2], 0)),
        S("max_consecutive_losing_days_period", per(g_l), per(n_l)),
    ]
    def _lbl(v): return str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v)
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
    yearly_ext = pd.DataFrame([{"year": y, **m} for y, m in yearly_base.items()])
    add_extremes(daily_perf, "day", "date", "highest_profit_day", "highest_loss_day")
    add_extremes(monthly, "month", "month", "highest_profit_month", "highest_loss_month")
    add_extremes(quarterly, "quarter", "quarter", "highest_profit_quarter", "highest_loss_quarter")
    add_extremes(yearly_ext, "year", "year", "highest_profit_year", "highest_loss_year")
    rows += [
        S("n_trades_excluded_as_circuit", n_excl, n_excl),
        S("pct_excluded_as_circuit", round(n_excl/len(T_all)*100, 2), round(n_excl/len(T_all)*100, 2)),
    ]
    summary = pd.DataFrame(rows)

    # ── all_trades (fpr columns + return_at_entry_pct) ──
    at = T[["entry_date", "symbol", "entry_price", "return_at_entry_pct", "exit_date", "exit_price",
            "exit_type", "exit_time", "shares", "capital_deployed", "gross_pnl", "gross_ret",
            "expense", "net_pnl", "net_ret", "month", "quarter", "half_year", "year"]].copy()
    at.columns = ["entry_date", "symbol", "entry_price", "return_at_entry_pct", "exit_date",
                  "exit_price", "exit_type", "exit_time", "shares", "capital_deployed",
                  "gross_trade_pnl_inr", "gross_trade_return_pct", "expense_inr",
                  "net_trade_pnl_inr", "net_trade_return_pct", "month", "quarter", "half_year", "year"]

    with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        at.to_excel(w, sheet_name="all_trades", index=False)
        daily_perf.to_excel(w, sheet_name="daily_performance", index=False)
        monthly.to_excel(w, sheet_name="monthly_performance", index=False)
        quarterly_c.to_excel(w, sheet_name="quarterly_performance", index=False)
        halfy.to_excel(w, sheet_name="half_yearly_performance", index=False)
        yearly_c.to_excel(w, sheet_name="yearly_performance", index=False)
        for sh in w.sheets.values():
            for col in sh.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                sh.column_dimensions[col[0].column_letter].width = min(width + 2, 30)

    pd.set_option("display.width", 200)
    print("\n=== SUMMARY (gross | net) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved -> {XLSX}")


if __name__ == "__main__":
    main()
