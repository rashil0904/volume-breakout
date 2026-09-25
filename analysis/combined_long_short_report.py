# -*- coding: utf-8 -*-
"""
combined_long_short_report.py
=============================
Combined LONG + intraday SHORT performance report for the full double-down, as ONE
strategy — every trade's P&L is its combined long+short result. Mirrors
final_performance_report.xlsx. Reuses the canonical long trades + short-open data.

combined_pnl = long_pnl + short_pnl; combined_return = combined_pnl/capital_deployed×100.
Costs per leg: long delivery 0.23%/0.38% × long notional (capital_deployed); short
intraday 0.10% × short notional (shares×long_exit). net_A = long0.23%+short0.10%,
net_B = long0.38%+short0.10%. total_return_fixedbase on the fixed ₹5,00,000 base.
Compounding: carry-forward-on-loss (gross-driven pool; carry prior STARTING level on a
non-positive quarter). GROSS / net_A / net_B throughout.
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
XLSX = OUTDIR / "combined_long_short_performance.xlsx"
BP, BA, L023, L038, SRATE, RF = 500_000, 100_000, 0.0023, 0.0038, 0.0010, 0.075


def pm(df):
    n = len(df)
    g = df["combined_pnl"].sum(); a = df["netA_pnl"].sum(); b = df["netB_pnl"].sum()
    return {"n_trades": n,
            "gross_total_pnl_inr": round(g, 0), "gross_return_fixedbase_pct": round(g/BP*100, 4),
            "netA_total_pnl_inr": round(a, 0), "netA_return_fixedbase_pct": round(a/BP*100, 4),
            "netB_total_pnl_inr": round(b, 0), "netB_return_fixedbase_pct": round(b/BP*100, 4),
            "gross_win_rate_pct": round((df["combined_pnl"] > 0).mean()*100, 2) if n else 0,
            "netA_win_rate_pct": round((df["netA_pnl"] > 0).mean()*100, 2) if n else 0,
            "netB_win_rate_pct": round((df["netB_pnl"] > 0).mean()*100, 2) if n else 0,
            "gross_avg_return_pct": round(df["combined_ret"].mean(), 4) if n else 0,
            "netA_avg_return_pct": round(df["netA_ret"].mean(), 4) if n else 0,
            "netB_avg_return_pct": round(df["netB_ret"].mean(), 4) if n else 0,
            "gross_median_return_pct": round(df["combined_ret"].median(), 4) if n else 0,
            "netA_median_return_pct": round(df["netA_ret"].median(), 4) if n else 0,
            "netB_median_return_pct": round(df["netB_ret"].median(), 4) if n else 0}


def scaled_combined(g, alloc):
    sh = np.floor(alloc / g["entry_price"].values)
    lp = sh * (g["long_exit_price"].values - g["entry_price"].values)
    sp = sh * (g["long_exit_price"].values - g["price_3pm_open"].values)
    comb = lp + sp
    lcap = sh * g["entry_price"].values; snotl = sh * g["long_exit_price"].values
    return (float(comb.sum()),
            float((comb - L023*lcap - SRATE*snotl).sum()),
            float((comb - L038*lcap - SRATE*snotl).sum()))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = dd.build_long_with_ohlc()
    cap = T["capital_deployed"]
    T["long_pnl"] = T["shares"]*(T["long_exit_price"]-T["entry_price"])
    T["short_pnl"] = T["shares"]*(T["long_exit_price"]-T["price_3pm_open"])
    T["combined_pnl"] = T["long_pnl"] + T["short_pnl"]
    T["combined_ret"] = T["combined_pnl"]/cap*100
    T["long_cost"] = L023*cap; T["long_cost_038"] = L038*cap
    T["short_cost"] = SRATE*(T["shares"]*T["long_exit_price"])
    T["netA_pnl"] = T["combined_pnl"] - T["long_cost"] - T["short_cost"]
    T["netB_pnl"] = T["combined_pnl"] - T["long_cost_038"] - T["short_cost"]
    T["netA_ret"] = T["netA_pnl"]/cap*100; T["netB_ret"] = T["netB_pnl"]/cap*100
    T["entry_dt"] = pd.to_datetime(T["entry_date"]); T["exit_dt"] = pd.to_datetime(T["exit_date"])
    T["month"] = T["entry_dt"].dt.strftime("%Y-%m")
    T["quarter"] = T["entry_dt"].dt.year.astype(str)+"Q"+T["entry_dt"].dt.quarter.astype(str)
    T["half_year"] = T["entry_dt"].dt.year.astype(str)+"H"+np.where(T["entry_dt"].dt.month<=6,"1","2")
    T["year"] = T["entry_dt"].dt.year
    T = T.sort_values("entry_dt").reset_index(drop=True)
    n = len(T)
    print(f"Combined book: {n:,} trades | gross {T['combined_pnl'].sum()/BP*100:.2f}% | "
          f"net_A {T['netA_pnl'].sum()/BP*100:.2f}% | net_B {T['netB_pnl'].sum()/BP*100:.2f}%")

    monthly = pd.DataFrame([{"month": k, **pm(g)} for k, g in T.groupby("month")]).sort_values("month")
    halfy = pd.DataFrame([{"half_year": k, **pm(g)} for k, g in T.groupby("half_year")]).sort_values("half_year")

    dcols = ["date", "n_trades", "gross_total_pnl_inr", "gross_return_fixedbase_pct",
             "netA_total_pnl_inr", "netA_return_fixedbase_pct", "netB_total_pnl_inr",
             "netB_return_fixedbase_pct", "gross_win_rate_pct", "gross_avg_return_pct",
             "gross_median_return_pct", "netA_is_winning_day"]
    daily = pd.DataFrame([{"date": k.date(), **pm(g),
                           "netA_is_winning_day": bool(g["netA_pnl"].sum() > 0)}
                          for k, g in T.groupby("exit_dt")]).sort_values("date")[dcols]

    # ── compounding (gross-driven pool, carry-forward-flat) ──
    def compound(is_year=False, half_map=None, years=None):
        rows = []; pool, alloc = BP, BA
        keys = years if is_year else sorted(T["quarter"].unique())
        for i, k in enumerate(keys):
            if is_year and i > 0:
                py = keys[i-1]
                hs = [h for h in (half_map.get(f"{py}H1"), half_map.get(f"{py}H2")) if h is not None]
                avg_half = float(np.mean(hs)) if hs else 0.0
                if avg_half > 0:
                    f = 1+avg_half/100; pool *= f; alloc *= f          # carry-forward-flat
            g = T[T["year"] == k] if is_year else T[T["quarter"] == k]
            cg, cnA, cnB = scaled_combined(g, alloc)
            rows.append({("year" if is_year else "quarter"): (int(k) if is_year else k), **pm(g),
                         "allocation_used": round(alloc, 0), "compounded_pool_value": round(pool, 0),
                         "gross_compounded_return_pct": round(cg/pool*100, 4),
                         "netA_compounded_return_pct": round(cnA/pool*100, 4),
                         "netB_compounded_return_pct": round(cnB/pool*100, 4),
                         "gross_compounded_pnl_inr": round(cg, 0),
                         "netA_compounded_pnl_inr": round(cnA, 0), "netB_compounded_pnl_inr": round(cnB, 0)})
            if not is_year:
                cr = cg/pool*100
                if cr > 0:
                    f = 1+cr/100; pool *= f; alloc *= f
        return pd.DataFrame(rows)
    quarterly = compound()
    half_g = dict(zip(halfy["half_year"], halfy["gross_return_fixedbase_pct"]))
    yearly = compound(is_year=True, half_map=half_g, years=sorted(T["year"].unique()))

    # ── daily series for sharpe + winning days (by cover day) ──
    dser = T.groupby("exit_dt").agg(g=("combined_pnl","sum"), a=("netA_pnl","sum"),
                                    b=("netB_pnl","sum"), c=("capital_deployed","sum")).reset_index()
    dser["g_ret"]=dser["g"]/dser["c"]*100; dser["a_ret"]=dser["a"]/dser["c"]*100; dser["b_ret"]=dser["b"]/dser["c"]*100
    rf_daily = RF/252*100
    def sharpe(x): return round((x.mean()-rf_daily)/x.std(ddof=1)*np.sqrt(252), 4)
    dts = dser.sort_values("exit_dt")["exit_dt"].tolist()
    def streaks(pnls):
        bwl=bll=0; bw=bl=(None,None,0.0); ct=None; cl=0; cs=ce=None; cp=0.0
        for d,p in zip(dts,pnls):
            typ="w" if p>0 else "l"
            if typ==ct: cl+=1; cp+=p; ce=d
            else: ct,cl,cp,cs,ce=typ,1,float(p),d,d
            if ct=="w" and cl>bwl: bwl,bw=cl,(cs,ce,cp)
            if ct=="l" and cl>bll: bll,bl=cl,(cs,ce,cp)
        return bwl,bw,bll,bl
    de = dser.sort_values("exit_dt")
    g_wl,g_w,g_ll,g_l = streaks(de["g"].values); a_wl,a_w,a_ll,a_l = streaks(de["a"].values)
    def per(t): return f"{pd.Timestamp(t[0]).date()} -> {pd.Timestamp(t[1]).date()}" if t[0] is not None else "—"

    # ── summary (gross | net_A | net_B) ──
    gw,gl2 = T[T["combined_pnl"]>0], T[T["combined_pnl"]<=0]
    aw,al = T[T["netA_pnl"]>0], T[T["netA_pnl"]<=0]
    bw,bl = T[T["netB_pnl"]>0], T[T["netB_pnl"]<=0]
    def R(m,g,a,b): return {"metric": m, "gross": g, "net_A(long0.23+short0.1)": a, "net_B(long0.38+short0.1)": b}
    rows = [
        R("n_trades", n, n, n),
        R("total_return_fixedbase_pct", round(T["combined_pnl"].sum()/BP*100,4), round(T["netA_pnl"].sum()/BP*100,4), round(T["netB_pnl"].sum()/BP*100,4)),
        R("total_pnl_inr", round(T["combined_pnl"].sum(),0), round(T["netA_pnl"].sum(),0), round(T["netB_pnl"].sum(),0)),
        R("win_rate_pct", round(len(gw)/n*100,2), round(len(aw)/n*100,2), round(len(bw)/n*100,2)),
        R("n_winning_trades", len(gw), len(aw), len(bw)),
        R("n_losing_trades", len(gl2), len(al), len(bl)),
        R("avg_return_per_trade_pct", round(T["combined_ret"].mean(),4), round(T["netA_ret"].mean(),4), round(T["netB_ret"].mean(),4)),
        R("avg_return_winning_trades_pct", round(gw["combined_ret"].mean(),4), round(aw["netA_ret"].mean(),4), round(bw["netB_ret"].mean(),4)),
        R("avg_return_losing_trades_pct", round(gl2["combined_ret"].mean(),4), round(al["netA_ret"].mean(),4), round(bl["netB_ret"].mean(),4)),
        R("median_return_per_trade_pct", round(T["combined_ret"].median(),4), round(T["netA_ret"].median(),4), round(T["netB_ret"].median(),4)),
        R("median_return_winning_trades_pct", round(gw["combined_ret"].median(),4), round(aw["netA_ret"].median(),4), round(bw["netB_ret"].median(),4)),
        R("median_return_losing_trades_pct", round(gl2["combined_ret"].median(),4), round(al["netA_ret"].median(),4), round(bl["netB_ret"].median(),4)),
        R("n_winning_days", int((dser["g"]>0).sum()), int((dser["a"]>0).sum()), int((dser["b"]>0).sum())),
        R("n_losing_days", int((dser["g"]<=0).sum()), int((dser["a"]<=0).sum()), int((dser["b"]<=0).sum())),
        R("avg_return_per_month_pct", round(monthly["gross_return_fixedbase_pct"].mean(),4), round(monthly["netA_return_fixedbase_pct"].mean(),4), round(monthly["netB_return_fixedbase_pct"].mean(),4)),
        R("avg_return_per_quarter_pct", round(quarterly["gross_return_fixedbase_pct"].mean(),4), round(quarterly["netA_return_fixedbase_pct"].mean(),4), round(quarterly["netB_return_fixedbase_pct"].mean(),4)),
        R("avg_return_per_half_year_pct", round(halfy["gross_return_fixedbase_pct"].mean(),4), round(halfy["netA_return_fixedbase_pct"].mean(),4), round(halfy["netB_return_fixedbase_pct"].mean(),4)),
        R("avg_return_per_year_pct", round(yearly["gross_return_fixedbase_pct"].mean(),4), round(yearly["netA_return_fixedbase_pct"].mean(),4), round(yearly["netB_return_fixedbase_pct"].mean(),4)),
        R("sharpe_ratio", sharpe(dser["g_ret"]), sharpe(dser["a_ret"]), sharpe(dser["b_ret"])),
        R("avg_capital_deployed_per_trade_inr", round(cap.mean(),0), round(cap.mean(),0), round(cap.mean(),0)),
        R("max_consecutive_winning_days", g_wl, a_wl, a_wl),
        R("max_consecutive_winning_days_pnl_inr", round(g_w[2],0), round(a_w[2],0), round(a_w[2],0)),
        R("max_consecutive_winning_days_period", per(g_w), per(a_w), per(a_w)),
        R("max_consecutive_losing_days", g_ll, a_ll, a_ll),
        R("max_consecutive_losing_days_pnl_inr", round(g_l[2],0), round(a_l[2],0), round(a_l[2],0)),
        R("max_consecutive_losing_days_period", per(g_l), per(a_l), per(a_l)),
    ]
    # leg decomposition block
    tl_long = T["long_pnl"].sum(); tl_short = T["short_pnl"].sum(); tot = T["combined_pnl"].sum()
    rows += [
        R("=== LEG DECOMPOSITION ===", "", "", ""),
        R("total_long_pnl_inr", round(tl_long,0), "long return %", round(tl_long/BP*100,4)),
        R("total_short_pnl_inr", round(tl_short,0), "short return %", round(tl_short/BP*100,4)),
        R("long_share_of_combined_pct", round(tl_long/tot*100,2), "short_share_of_combined_pct", round(tl_short/tot*100,2)),
    ]
    # extremes
    def _lbl(v): return str(int(v)) if isinstance(v,float) and float(v).is_integer() else str(v)
    def extremes(df, name, key):
        gmax=df.loc[df["gross_total_pnl_inr"].idxmax()]; amax=df.loc[df["netA_total_pnl_inr"].idxmax()]
        gmin=df.loc[df["gross_total_pnl_inr"].idxmin()]; amin=df.loc[df["netA_total_pnl_inr"].idxmin()]
        rows.extend([
            R(f"highest_profit_{name}_pnl_inr", round(float(gmax["gross_total_pnl_inr"]),0), round(float(amax["netA_total_pnl_inr"]),0), round(float(df.loc[df['netB_total_pnl_inr'].idxmax(),'netB_total_pnl_inr']),0)),
            R(f"highest_profit_{name}", _lbl(gmax[key]), _lbl(amax[key]), _lbl(df.loc[df['netB_total_pnl_inr'].idxmax(),key])),
            R(f"highest_loss_{name}_pnl_inr", round(float(gmin["gross_total_pnl_inr"]),0), round(float(amin["netA_total_pnl_inr"]),0), round(float(df.loc[df['netB_total_pnl_inr'].idxmin(),'netB_total_pnl_inr']),0)),
            R(f"highest_loss_{name}", _lbl(gmin[key]), _lbl(amin[key]), _lbl(df.loc[df['netB_total_pnl_inr'].idxmin(),key]))])
    dser2 = dser.rename(columns={"g":"gross_total_pnl_inr","a":"netA_total_pnl_inr","b":"netB_total_pnl_inr"})
    dser2["date"] = dser2["exit_dt"].dt.date
    extremes(dser2, "day", "date")
    extremes(monthly, "month", "month"); extremes(quarterly, "quarter", "quarter"); extremes(yearly, "year", "year")
    summary = pd.DataFrame(rows)

    all_trades = T[["entry_date","entry_price","long_exit_price","exit_type","long_pnl","price_3pm_open",
                    "short_pnl","combined_pnl","combined_ret","long_cost","short_cost","netA_pnl","netB_pnl",
                    "shares","capital_deployed","month","quarter","year"]].copy()
    all_trades.columns = ["entry_date","entry_price","long_exit_price","long_exit_type","long_pnl_inr",
                          "open_at_3pm","short_pnl_inr","combined_pnl_inr","combined_return_pct",
                          "long_cost_0.23pct_inr","short_cost_0.10pct_inr","netA_combined_pnl_inr",
                          "netB_combined_pnl_inr","shares","capital_deployed","month","quarter","year"]

    with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        all_trades.to_excel(w, sheet_name="all_trades", index=False)
        daily.to_excel(w, sheet_name="daily_performance", index=False)
        monthly.to_excel(w, sheet_name="monthly_performance", index=False)
        quarterly.to_excel(w, sheet_name="quarterly_performance", index=False)
        halfy.to_excel(w, sheet_name="half_yearly_performance", index=False)
        yearly.to_excel(w, sheet_name="yearly_performance", index=False)

    pd.set_option("display.width", 210)
    print("\n=== COMBINED SUMMARY (gross | net_A | net_B) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved -> {XLSX}")


if __name__ == "__main__":
    main()
