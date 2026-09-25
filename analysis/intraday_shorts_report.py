# -*- coding: utf-8 -*-
"""
intraday_shorts_report.py
=========================
Standalone performance report for the INTRADAY SHORT LEG ONLY of the full double-down.
NO long-leg P&L anywhere — every figure is the short book in isolation. A short is opened
at each main-strategy trade's long-exit price/time (target@entry*1.14, 9:45@9:45 open,
12:00@12:00 open) and covered at the 3:00pm candle open the SAME day. Reuses the existing
short-open data; not recomputed.

short_pnl = shares × (short_open − open_3pm); short_return_pct = (short_open − open_3pm)/
short_open × 100. Cost: 0.10% intraday round-trip on short notional (shares × short_open).
total_return_fixedbase_pct on the fixed ₹5,00,000 base. Compounding: carry-forward-on-loss
(scale by prior period gross short compounded return if >0, else carry prior STARTING level).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import double_down_short_variant as dd

OUTDIR = rb.RESULTS / "final_report"
XLSX = OUTDIR / "intraday_shorts_performance.xlsx"
BP, BA, SHORT_RATE, RF = 500_000, 100_000, 0.0010, 0.075


def pm(df):
    n = len(df); gp = df["short_pnl"].sum(); npl = df["net_short_pnl"].sum()
    return {"n_shorts": n,
            "gross_total_pnl_inr": round(gp, 0), "gross_return_fixedbase_pct": round(gp/BP*100, 4),
            "net_total_pnl_inr": round(npl, 0), "net_return_fixedbase_pct": round(npl/BP*100, 4),
            "gross_win_rate_pct": round((df["short_pnl"] > 0).mean()*100, 2) if n else 0,
            "net_win_rate_pct": round((df["net_short_pnl"] > 0).mean()*100, 2) if n else 0,
            "gross_avg_short_return_pct": round(df["short_ret"].mean(), 4) if n else 0,
            "net_avg_short_return_pct": round(df["net_short_ret"].mean(), 4) if n else 0,
            "gross_median_short_return_pct": round(df["short_ret"].median(), 4) if n else 0,
            "net_median_short_return_pct": round(df["net_short_ret"].median(), 4) if n else 0}


def short_scaled(g, alloc):
    sh = np.floor(alloc / g["entry_price"].values)
    gpnl = sh * (g["short_open"].values - g["open_3pm"].values)
    notl = sh * g["short_open"].values
    return float(gpnl.sum()), float((gpnl - SHORT_RATE * notl).sum())


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = dd.build_long_with_ohlc()
    S = pd.DataFrame({
        "symbol": T["symbol"], "entry_date": pd.to_datetime(T["entry_date"]),
        "cover_date": pd.to_datetime(T["exit_date"]), "short_open_time": T["exit_time"],
        "exit_type": T["exit_type"], "entry_price": T["entry_price"],
        "short_open": T["long_exit_price"], "open_3pm": T["price_3pm_open"], "shares": T["shares"],
    })
    S["short_pnl"] = S["shares"] * (S["short_open"] - S["open_3pm"])
    S["short_ret"] = (S["short_open"] - S["open_3pm"]) / S["short_open"] * 100
    S["short_notional"] = S["shares"] * S["short_open"]
    S["short_cost"] = S["short_notional"] * SHORT_RATE
    S["net_short_pnl"] = S["short_pnl"] - S["short_cost"]
    S["net_short_ret"] = S["short_ret"] - SHORT_RATE * 100
    ts = S["entry_date"]
    S["month"] = ts.dt.strftime("%Y-%m")
    S["quarter"] = ts.dt.year.astype(str) + "Q" + ts.dt.quarter.astype(str)
    S["half_year"] = ts.dt.year.astype(str) + "H" + np.where(ts.dt.month <= 6, "1", "2")
    S["year"] = ts.dt.year
    S = S.sort_values("entry_date").reset_index(drop=True)
    n = len(S)
    print(f"Short book: {n:,} shorts | gross {S['short_pnl'].sum()/BP*100:.2f}% | "
          f"net@0.1% {S['net_short_pnl'].sum()/BP*100:.2f}%")

    # ── period tables ──
    def ptab(col, key):
        return pd.DataFrame([{key: k, **pm(g)} for k, g in S.groupby(col)]).sort_values(key)
    monthly = ptab("month", "month"); halfy = ptab("half_year", "half_year")

    # daily by COVER day
    dcols = ["date", "n_shorts", "gross_total_pnl_inr", "gross_return_fixedbase_pct",
             "net_total_pnl_inr", "net_return_fixedbase_pct", "gross_win_rate_pct",
             "gross_avg_short_return_pct", "gross_median_short_return_pct", "net_is_winning_day"]
    daily = pd.DataFrame([{"date": k.date(), **pm(g),
                           "net_is_winning_day": bool(g["net_short_pnl"].sum() > 0)}
                          for k, g in S.groupby("cover_date")]).sort_values("date")[dcols]

    # compounding (carry-forward-flat, gross-driven)
    def compound(period_col, is_year=False, half_map=None, years=None):
        rows = []; pool, alloc = BP, BA
        keys = sorted(S[period_col].unique()) if not is_year else years
        prev_avg = None
        for i, k in enumerate(keys):
            if is_year and i > 0:
                py = keys[i-1]
                hs = [half_map.get(f"{py}H1"), half_map.get(f"{py}H2")]
                hs = [h for h in hs if h is not None]
                avg_half = float(np.mean(hs)) if hs else 0.0
                if avg_half > 0:
                    f = 1 + avg_half/100; pool *= f; alloc *= f     # carry-forward-flat: scale up only
            g = S[S[period_col] == k]
            cg, cn = short_scaled(g, alloc)
            base_m = pm(g)
            rows.append({period_col if not is_year else "year": (int(k) if is_year else k), **base_m,
                         "allocation_used": round(alloc, 0), "compounded_pool_value": round(pool, 0),
                         "gross_compounded_return_pct": round(cg/pool*100, 4),
                         "net_compounded_return_pct": round(cn/pool*100, 4),
                         "gross_compounded_pnl_inr": round(cg, 0), "net_compounded_pnl_inr": round(cn, 0)})
            if not is_year:
                cr = cg/pool*100
                if cr > 0:
                    f = 1 + cr/100; pool *= f; alloc *= f            # carry-forward-flat
        return pd.DataFrame(rows)
    quarterly = compound("quarter")
    half_gross = dict(zip(halfy["half_year"], halfy["gross_return_fixedbase_pct"]))
    yearly = compound("year", is_year=True, half_map=half_gross, years=sorted(S["year"].unique()))

    # ── sharpe (daily short return = daily short_pnl / daily short_notional) ──
    dser = S.groupby("cover_date").agg(g=("short_pnl", "sum"), nt=("net_short_pnl", "sum"),
                                       notl=("short_notional", "sum")).reset_index()
    dser["g_ret"] = dser["g"]/dser["notl"]*100; dser["n_ret"] = dser["nt"]/dser["notl"]*100
    rf_daily = RF/252*100
    def sharpe(x): return round((x.mean()-rf_daily)/x.std(ddof=1)*np.sqrt(252), 4)

    # ── streaks (by cover day, net) ──
    de = dser.sort_values("cover_date"); dts = de["cover_date"].tolist()
    def streaks(pnls):
        bwl=bll=0; bw=bl=(None,None,0.0); ct=None; cl=0; cs=ce=None; cp=0.0
        for d,p in zip(dts,pnls):
            typ = "w" if p>0 else "l"
            if typ==ct: cl+=1; cp+=p; ce=d
            else: ct,cl,cp,cs,ce = typ,1,float(p),d,d
            if ct=="w" and cl>bwl: bwl,bw=cl,(cs,ce,cp)
            if ct=="l" and cl>bll: bll,bl=cl,(cs,ce,cp)
        return bwl,bw,bll,bl
    g_wl,g_w,g_ll,g_l = streaks(de["g"].values)
    n_wl,n_w,n_ll,n_l = streaks(de["nt"].values)
    def per(t): return f"{pd.Timestamp(t[0]).date()} -> {pd.Timestamp(t[1]).date()}" if t[0] is not None else "—"

    # ── true peak-to-trough drawdown on net short equity ──
    eq = np.cumsum(de["nt"].values); peak = np.maximum.accumulate(eq)
    dd_curve = eq - peak; max_dd = float(dd_curve.min())
    tr_i = int(np.argmin(dd_curve)); pk_i = int(np.argmax(eq[:tr_i+1])) if tr_i > 0 else 0
    dd_period = f"{pd.Timestamp(dts[pk_i]).date()} -> {pd.Timestamp(dts[tr_i]).date()}"

    # ── summary ──
    gw,gl = S[S["short_pnl"]>0], S[S["short_pnl"]<=0]
    nw,nl = S[S["net_short_pnl"]>0], S[S["net_short_pnl"]<=0]
    def R(m,g,nv): return {"metric": m, "gross": g, "net@0.10%": nv}
    rows = [
        R("n_shorts", n, n),
        R("total_return_fixedbase_pct", round(S["short_pnl"].sum()/BP*100,4), round(S["net_short_pnl"].sum()/BP*100,4)),
        R("total_short_pnl_inr", round(S["short_pnl"].sum(),0), round(S["net_short_pnl"].sum(),0)),
        R("win_rate_pct", round(len(gw)/n*100,2), round(len(nw)/n*100,2)),
        R("n_winning_trades", len(gw), len(nw)),
        R("n_losing_trades", len(gl), len(nl)),
        R("avg_return_per_trade_pct", round(S["short_ret"].mean(),4), round(S["net_short_ret"].mean(),4)),
        R("avg_return_winning_trades_pct", round(gw["short_ret"].mean(),4), round(nw["net_short_ret"].mean(),4)),
        R("avg_return_losing_trades_pct", round(gl["short_ret"].mean(),4), round(nl["net_short_ret"].mean(),4)),
        R("median_return_per_trade_pct", round(S["short_ret"].median(),4), round(S["net_short_ret"].median(),4)),
        R("median_return_winning_trades_pct", round(gw["short_ret"].median(),4), round(nw["net_short_ret"].median(),4)),
        R("median_return_losing_trades_pct", round(gl["short_ret"].median(),4), round(nl["net_short_ret"].median(),4)),
        R("n_winning_days", int((dser["g"]>0).sum()), int((dser["nt"]>0).sum())),
        R("n_losing_days", int((dser["g"]<=0).sum()), int((dser["nt"]<=0).sum())),
        R("avg_return_per_month_pct", round(monthly["gross_return_fixedbase_pct"].mean(),4), round(monthly["net_return_fixedbase_pct"].mean(),4)),
        R("avg_return_per_quarter_pct", round(quarterly["gross_return_fixedbase_pct"].mean(),4), round(quarterly["net_return_fixedbase_pct"].mean(),4)),
        R("avg_return_per_half_year_pct", round(halfy["gross_return_fixedbase_pct"].mean(),4), round(halfy["net_return_fixedbase_pct"].mean(),4)),
        R("avg_return_per_year_pct", round(yearly["gross_return_fixedbase_pct"].mean(),4), round(yearly["net_return_fixedbase_pct"].mean(),4)),
        R("sharpe_ratio", sharpe(dser["g_ret"]), sharpe(dser["n_ret"])),
        R("avg_capital_deployed_per_short_inr", round(S["short_notional"].mean(),0), round(S["short_notional"].mean(),0)),
        R("max_consecutive_winning_days", g_wl, n_wl),
        R("max_consecutive_winning_days_pnl_inr", round(g_w[2],0), round(n_w[2],0)),
        R("max_consecutive_winning_days_period", per(g_w), per(n_w)),
        R("max_consecutive_losing_days", g_ll, n_ll),
        R("max_consecutive_losing_days_pnl_inr", round(g_l[2],0), round(n_l[2],0)),
        R("max_consecutive_losing_days_period", per(g_l), per(n_l)),
        R("max_peak_to_trough_drawdown_inr_net", round(max_dd,0), round(max_dd,0)),
        R("max_drawdown_period_net", dd_period, dd_period),
        R("--- CONTEXT ---", "SHORT LEG STANDALONE", "returns driven purely by exit->3pm fade"),
    ]
    def _lbl(v): return str(int(v)) if isinstance(v,float) and float(v).is_integer() else str(v)
    def extremes(df, name, lbl):
        gmax=df.loc[df["gross_total_pnl_inr"].idxmax()]; nmax=df.loc[df["net_total_pnl_inr"].idxmax()]
        gmin=df.loc[df["gross_total_pnl_inr"].idxmin()]; nmin=df.loc[df["net_total_pnl_inr"].idxmin()]
        c = "date" if name=="day" else lbl
        rows.extend([
            R(f"highest_profit_{name}_pnl_inr", round(float(gmax["gross_total_pnl_inr"]),0), round(float(nmax["net_total_pnl_inr"]),0)),
            R(f"highest_profit_{name}", _lbl(gmax[c]), _lbl(nmax[c])),
            R(f"highest_loss_{name}_pnl_inr", round(float(gmin["gross_total_pnl_inr"]),0), round(float(nmin["net_total_pnl_inr"]),0)),
            R(f"highest_loss_{name}", _lbl(gmin[c]), _lbl(nmin[c]))])
    extremes(daily.rename(columns={"date":"date"}).assign(), "day", "date") if False else None
    # daily has different pnl col names; build day extremes from dser
    dser2 = dser.rename(columns={"g":"gross_total_pnl_inr","nt":"net_total_pnl_inr"})
    dser2["date"] = dser2["cover_date"].dt.date
    extremes(dser2, "day", "date")
    extremes(monthly.rename(columns={"gross_total_pnl_inr":"gross_total_pnl_inr","net_total_pnl_inr":"net_total_pnl_inr"}), "month", "month")
    extremes(quarterly, "quarter", "quarter")
    extremes(yearly, "year", "year")
    # exit_type split context
    for et,g in S.groupby("exit_type"):
        rows.append(R(f"exit_type[{et}]_n / win% / avg_short%",
                      f"{len(g)} / {round((g['short_pnl']>0).mean()*100,2)}% / {round(g['short_ret'].mean(),4)}%",
                      f"{round((g['net_short_pnl']>0).mean()*100,2)}% net win"))
    summary = pd.DataFrame(rows)

    all_shorts = S[["entry_date","cover_date","short_open_time","exit_type","short_open","open_3pm",
                    "shares","short_pnl","short_ret","short_cost","net_short_pnl","net_short_ret",
                    "month","quarter","year"]].copy()
    all_shorts.columns = ["entry_date","short_open_cover_date","short_open_time","exit_type",
                          "short_open_price","open_at_3pm","shares","short_pnl_inr","short_return_pct",
                          "short_cost_inr","net_short_pnl_inr","net_short_return_pct","month","quarter","year"]

    with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        all_shorts.to_excel(w, sheet_name="all_shorts", index=False)
        daily.to_excel(w, sheet_name="daily_performance", index=False)
        monthly.to_excel(w, sheet_name="monthly_performance", index=False)
        quarterly.to_excel(w, sheet_name="quarterly_performance", index=False)
        halfy.to_excel(w, sheet_name="half_yearly_performance", index=False)
        yearly.to_excel(w, sheet_name="yearly_performance", index=False)

    pd.set_option("display.width", 200)
    print("\n=== SHORT-LEG SUMMARY (gross | net@0.10%) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved -> {XLSX}")


if __name__ == "__main__":
    main()
