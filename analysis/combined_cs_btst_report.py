# -*- coding: utf-8 -*-
"""combined_cs_btst_report.py — combined comparison report for the NIFTY Weekly Credit Spread and the NIFTY
Daily Close-Direction BTST strategies, from their existing trade-level outputs. GROSS OPTION PREMIUM POINTS
only (no capital / INR / sizing). Combined curve = pure point-additive stacking (unlimited-capacity assumption).
Sheets: Combined_Summary, Combined_TradeLog (interleaved + running cum/DD), Credit_Spread_Trades, BTST_Trades.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

CS = rb.RESULTS / "weekly_credit_spread" / "weekly_credit_spread_trades.csv"
BT = rb.RESULTS / "btst_close_direction" / "btst_close_direction_trades.csv"
OUT = rb.RESULTS / "combined_cs_btst" / "combined_credit_spread_btst_report.xlsx"; OUT.parent.mkdir(parents=True, exist_ok=True)


def drawdown_episodes(equity, times):
    peak = np.maximum.accumulate(equity); dd = equity - peak; eps = []; in_dd = False; s = t = 0; tv = 0.0
    for k in range(len(equity)):
        if not in_dd:
            if dd[k] < -1e-9: in_dd = True; s = k - 1; t = k; tv = dd[k]
        else:
            if dd[k] < tv: tv = dd[k]; t = k
            if dd[k] >= -1e-9: eps.append((s, t, k, tv)); in_dd = False
    if in_dd: eps.append((s, t, len(equity) - 1, tv))
    depths = [abs(e[3]) for e in eps]
    durs = [int((pd.Timestamp(times[e[2]]) - pd.Timestamp(times[e[0]])).days) for e in eps]
    return depths, durs


def core(pnl, dates):
    order = np.argsort(dates.values); p = pnl[order]; d = pd.to_datetime(dates.values[order])
    cum = np.concatenate([[0.0], p.cumsum()]); tms = np.concatenate([[d[0]], d.values])
    depths, durs = drawdown_episodes(cum, tms)
    win = p > 0; loss = p < 0
    m = {"Total return (points)": round(p.sum(), 1), "Total trades": len(p),
         "Winning trades": int(win.sum()), "Losing trades": int(loss.sum()), "Win rate %": round(win.mean() * 100, 1),
         "Avg return / trade (points)": round(p.mean(), 2), "Median return / trade (points)": round(float(np.median(p)), 2),
         "Avg WIN (points)": round(p[win].mean(), 2) if win.any() else 0, "Avg LOSS (points)": round(p[loss].mean(), 2) if loss.any() else 0,
         "Max profit / best trade (points)": round(p.max(), 1), "Max loss / worst trade (points)": round(p.min(), 1),
         "Number of DD episodes": len(depths),
         "Avg drawdown (points)": round(float(np.mean(depths)), 1) if depths else 0, "Max drawdown (points)": round(float(np.max(depths)), 1) if depths else 0,
         "Avg DD period (days, peak->recovery)": round(float(np.mean(durs)), 1) if durs else 0, "Max DD period (days)": int(np.max(durs)) if durs else 0,
         "Return / Max-DD": round(p.sum() / max(depths), 2) if depths and max(depths) else "-"}
    return m


def main():
    cs = pd.read_csv(CS); bt = pd.read_csv(BT)
    cs["_d"] = pd.to_datetime(cs["entry_date"]); bt["_d"] = pd.to_datetime(bt["entry_date"])
    m_cs = core(cs["pnl_points"].values, cs["_d"]); m_bt = core(bt["pnl_points"].values, bt["_d"])

    # combined = point-additive stacking (merge by date)
    merged = pd.concat([
        cs.assign(strategy="Credit Spread", detail=cs["type"] + " / " + cs["trigger"])[["_d", "strategy", "detail", "pnl_points"]],
        bt.assign(strategy="BTST", detail=bt["direction"] + " / " + bt["contract"])[["_d", "strategy", "detail", "pnl_points"]],
    ], ignore_index=True).sort_values(["_d", "strategy"]).reset_index(drop=True)
    m_comb = core(merged["pnl_points"].values, merged["_d"])
    merged["cum_points"] = merged["pnl_points"].cumsum().round(1)
    peak = merged["cum_points"].cummax(); merged["drawdown_points"] = (merged["cum_points"] - peak).round(1)
    merged.insert(0, "date", merged["_d"].dt.date); merged = merged.drop(columns=["_d"])

    # strategy-specific extras (per-strategy only)
    extra_cs = {"Avg net credit (points)": round(cs["net_credit"].mean(), 2),
                "% exits: 90% target": round((cs["exit_reason"] == "90% target").mean() * 100, 1),
                "% exits: DTE0 settlement": round((cs["exit_reason"] == "DTE0 settlement").mean() * 100, 1),
                "Avg DTE at entry": round(cs["DTE"].mean(), 2)}
    extra_bt = {"RED / GREEN trades": f"{int((bt.direction=='RED').sum())} / {int((bt.direction=='GREEN').sum())}",
                "Normal-day / expiry-day trades": f"{int(~bt.is_expiry_day.sum() if False else (~bt.is_expiry_day).sum())} / {int(bt.is_expiry_day.sum())}",
                "Entry / exit time": f"{bt.entry_time.iloc[0]} / {bt.exit_time.iloc[0]}"}

    order = list(m_cs.keys())
    rows = [{"metric": k, "Credit Spread": m_cs[k], "BTST": m_bt[k], "Combined": m_comb[k]} for k in order]
    rows.append({"metric": "", "Credit Spread": "", "BTST": "", "Combined": ""})
    rows.append({"metric": "--- strategy-specific ---", "Credit Spread": "", "BTST": "", "Combined": ""})
    for k, v in extra_cs.items(): rows.append({"metric": k, "Credit Spread": v, "BTST": "", "Combined": ""})
    for k, v in extra_bt.items(): rows.append({"metric": k, "Credit Spread": "", "BTST": v, "Combined": ""})
    rows.append({"metric": "", "Credit Spread": "", "BTST": "", "Combined": ""})
    rows.append({"metric": "P&L UNIT", "Credit Spread": "GROSS option-premium points (no capital/INR/sizing)", "BTST": "same", "Combined": "point-additive"})
    rows.append({"metric": "CAPACITY FLAG", "Credit Spread": "", "BTST": "",
                 "Combined": "Combined curve = pure point sum of both, assuming UNLIMITED CAPACITY to run both simultaneously; NO shared capital/margin modelled. Same-day overlaps need no conflict resolution (no shared pool)."})
    SUM = pd.DataFrame(rows)

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        SUM.to_excel(w, sheet_name="Combined_Summary", index=False)
        merged.to_excel(w, sheet_name="Combined_TradeLog", index=False)
        cs.drop(columns=["_d"]).to_excel(w, sheet_name="Credit_Spread_Trades", index=False)
        bt.drop(columns=["_d"]).to_excel(w, sheet_name="BTST_Trades", index=False)

    pd.set_option("display.width", 200)
    print("COMBINED CREDIT SPREAD + BTST REPORT (gross option-premium points)")
    print(pd.DataFrame(rows[:len(order)]).to_string(index=False))
    print(f"\ncombined trades {len(merged)} | final cum {merged.cum_points.iloc[-1]} | max DD {round(-merged.drawdown_points.min(),1)}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
