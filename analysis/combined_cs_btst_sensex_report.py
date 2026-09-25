# -*- coding: utf-8 -*-
"""combined_cs_btst_sensex_report.py — combined comparison report for the SENSEX Weekly Credit Spread (600-wide
hedge) and the SENSEX Daily Close-Direction BTST (09:17 exit / 800 offset / VIX 17-19 filter), from their
existing trade-level outputs. GROSS OPTION PREMIUM POINTS only (no capital/INR/sizing). Combined curve = pure
point-additive stacking (unlimited-capacity assumption). Mirrors combined_cs_btst_report.py (NIFTY).
Sheets: Combined_Summary, Combined_TradeLog (interleaved + running cum/DD), Credit_Spread_Trades, BTST_Trades.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

CS = rb.RESULTS / "sensex_weekly_credit_spread" / "sensex_weekly_credit_spread_trades.csv"
BT_XL = rb.RESULTS / "sensex_btst_close_direction" / "sensex_btst_close_direction_final_report.xlsx"
OUT = rb.RESULTS / "combined_cs_btst_sensex" / "combined_credit_spread_btst_sensex_report.xlsx"; OUT.parent.mkdir(parents=True, exist_ok=True)


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


def dd_table(equity, times):
    """episode-level drawdown detail on a cumulative-equity curve."""
    peak = np.maximum.accumulate(equity); dd = equity - peak; eps = []; in_dd = False; s = t = 0; tv = 0.0
    for k in range(len(equity)):
        if not in_dd:
            if dd[k] < -1e-9: in_dd = True; s = k - 1; t = k; tv = dd[k]
        else:
            if dd[k] < tv: tv = dd[k]; t = k
            if dd[k] >= -1e-9: eps.append((s, t, k, tv)); in_dd = False
    if in_dd: eps.append((s, t, len(equity) - 1, tv))
    rows = []
    for (s, t, r, v) in eps:
        rec = (r != len(equity) - 1) or (equity[r] >= peak[s] - 1e-9)
        rows.append({"peak_date": pd.Timestamp(times[s]).date(), "trough_date": pd.Timestamp(times[t]).date(),
                     "recovery_date": (pd.Timestamp(times[r]).date() if rec else "NOT RECOVERED"),
                     "drawdown_points": round(abs(v), 1),
                     "days_peak_to_trough": int((pd.Timestamp(times[t]) - pd.Timestamp(times[s])).days),
                     "days_peak_to_recovery": int((pd.Timestamp(times[r]) - pd.Timestamp(times[s])).days)})
    return pd.DataFrame(rows)


def core(pnl, dates):
    order = np.argsort(dates.values); p = pnl[order]; d = pd.to_datetime(dates.values[order])
    cum = np.concatenate([[0.0], p.cumsum()]); tms = np.concatenate([[d[0]], d.values])
    depths, durs = drawdown_episodes(cum, tms)
    win = p > 0; loss = p < 0
    return {"Total return (points)": round(p.sum(), 1), "Total trades": len(p),
            "Winning trades": int(win.sum()), "Losing trades": int(loss.sum()), "Win rate %": round(win.mean() * 100, 1),
            "Avg return / trade (points)": round(p.mean(), 2), "Median return / trade (points)": round(float(np.median(p)), 2),
            "Avg WIN (points)": round(p[win].mean(), 2) if win.any() else 0, "Avg LOSS (points)": round(p[loss].mean(), 2) if loss.any() else 0,
            "Max profit / best trade (points)": round(p.max(), 1), "Max loss / worst trade (points)": round(p.min(), 1),
            "Number of DD episodes": len(depths),
            "Avg drawdown (points)": round(float(np.mean(depths)), 1) if depths else 0, "Max drawdown (points)": round(float(np.max(depths)), 1) if depths else 0,
            "Avg DD period (days, peak->recovery)": round(float(np.mean(durs)), 1) if durs else 0, "Max DD period (days)": int(np.max(durs)) if durs else 0,
            "Return / Max-DD": round(p.sum() / max(depths), 2) if depths and max(depths) else "-"}


def main():
    cs = pd.read_csv(CS)
    bt = pd.read_excel(BT_XL, sheet_name="Trade_Log")
    bt = bt[~bt["excluded_by_vix_filter"].astype(bool)].reset_index(drop=True)   # FINAL filtered BTST set (VIX 17-19 excluded)
    if "entry_time" not in bt.columns: bt["entry_time"] = "15:15"
    cs["_d"] = pd.to_datetime(cs["entry_date"]); bt["_d"] = pd.to_datetime(bt["entry_date"])
    m_cs = core(cs["pnl_points"].values, cs["_d"]); m_bt = core(bt["pnl_points"].values, bt["_d"])

    merged = pd.concat([
        cs.assign(strategy="Credit Spread", detail=cs["type"] + " / " + cs["trigger"])[["_d", "strategy", "detail", "pnl_points"]],
        bt.assign(strategy="BTST", detail=bt["direction"] + " / " + bt["contract"])[["_d", "strategy", "detail", "pnl_points"]],
    ], ignore_index=True).sort_values(["_d", "strategy"]).reset_index(drop=True)
    m_comb = core(merged["pnl_points"].values, merged["_d"])
    merged["cum_points"] = merged["pnl_points"].cumsum().round(1)
    peak = merged["cum_points"].cummax(); merged["drawdown_points"] = (merged["cum_points"] - peak).round(1)
    # combined drawdown-episode detail (on the combined equity curve)
    m_ord = merged.sort_values("_d")
    eqc = np.concatenate([[0.0], m_ord["pnl_points"].cumsum().values]); tmc = np.concatenate([[m_ord["_d"].iloc[0]], m_ord["_d"].values])
    DDc = dd_table(eqc, tmc).sort_values("drawdown_points", ascending=False).reset_index(drop=True)
    DDc.insert(0, "rank", DDc.index + 1)
    merged.insert(0, "date", merged["_d"].dt.date); merged = merged.drop(columns=["_d"])

    extra_cs = {"Avg net credit (points)": round(cs["net_credit"].mean(), 2),
                "% exits: 90% target": round((cs["exit_reason"] == "90% target").mean() * 100, 1),
                "% exits: DTE0 settlement": round((cs["exit_reason"] == "DTE0 settlement").mean() * 100, 1),
                "Avg DTE at entry": round(cs["DTE"].mean(), 2),
                "Trigger % 3dH/3dL/fallback": " / ".join(str(round((cs["trigger"] == t).mean() * 100, 1)) for t in ["3d-high", "3d-low", "fallback-2:30"]),
                "Hedge width": 600}
    extra_bt = {"RED / GREEN trades": f"{int((bt.direction=='RED').sum())} / {int((bt.direction=='GREEN').sum())}",
                "Normal-day / expiry-day trades": f"{int((~bt.is_expiry_day.astype(bool)).sum())} / {int(bt.is_expiry_day.astype(bool).sum())}",
                "Entry / exit time": f"{bt.entry_time.iloc[0]} / {bt.exit_time.iloc[0]}",
                "Sell-leg offset": 800, "VIX 17-19 filter": "applied (excluded)"}

    order = list(m_cs.keys())
    rows = [{"metric": k, "Credit Spread": m_cs[k], "BTST": m_bt[k], "Combined": m_comb[k]} for k in order]
    rows.append({"metric": "", "Credit Spread": "", "BTST": "", "Combined": ""})
    rows.append({"metric": "--- strategy-specific ---", "Credit Spread": "", "BTST": "", "Combined": ""})
    for k, v in extra_cs.items(): rows.append({"metric": k, "Credit Spread": v, "BTST": "", "Combined": ""})
    for k, v in extra_bt.items(): rows.append({"metric": k, "Credit Spread": "", "BTST": v, "Combined": ""})
    rows.append({"metric": "", "Credit Spread": "", "BTST": "", "Combined": ""})
    rows.append({"metric": "P&L UNIT", "Credit Spread": "GROSS option-premium points (no capital/INR/sizing)", "BTST": "same", "Combined": "point-additive"})
    rows.append({"metric": "CAPACITY FLAG", "Credit Spread": "", "BTST": "",
                 "Combined": "Combined curve = pure point sum of both, assuming UNLIMITED CAPACITY to run both simultaneously; NO shared capital/margin modelled. Same-day overlaps need no conflict resolution (no shared pool). NOTE: the two legs also differ in point-scale (credit-spread net credits ~244 vs BTST ~40/trade) - additive points mix the two magnitudes."})
    SUM = pd.DataFrame(rows)

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        SUM.to_excel(w, sheet_name="Combined_Summary", index=False)
        # header row w/ max & avg on the combined DD-episode sheet
        hdr = pd.DataFrame([{"rank": "SUMMARY", "peak_date": "", "trough_date": "",
                             "recovery_date": f"episodes {len(DDc)}", "drawdown_points": f"MAX {DDc.drawdown_points.max()} / AVG {round(DDc.drawdown_points.mean(),1)}",
                             "days_peak_to_trough": "", "days_peak_to_recovery": f"MAX {int(DDc.days_peak_to_recovery.max())} / AVG {round(DDc.days_peak_to_recovery.mean(),1)}"}])
        pd.concat([hdr, DDc], ignore_index=True).to_excel(w, sheet_name="Combined_Drawdowns", index=False)
        merged.to_excel(w, sheet_name="Combined_TradeLog", index=False)
        cs.drop(columns=["_d"]).to_excel(w, sheet_name="Credit_Spread_Trades", index=False)
        bt.drop(columns=["_d"]).to_excel(w, sheet_name="BTST_Trades", index=False)

    pd.set_option("display.width", 210)
    print("COMBINED SENSEX CREDIT SPREAD + BTST REPORT (gross option-premium points)")
    print(pd.DataFrame([{"metric": k, "Credit Spread": m_cs[k], "BTST": m_bt[k], "Combined": m_comb[k]} for k in order]).to_string(index=False))
    print(f"\ncombined trades {len(merged)} | final cum {merged.cum_points.iloc[-1]} | max DD {round(-merged.drawdown_points.min(),1)}")
    print(f"\n--- COMBINED DRAWDOWN EPISODES (top 8 by depth) --- [total {len(DDc)} | MAX {DDc.drawdown_points.max()} avg {round(DDc.drawdown_points.mean(),1)} | max dur {int(DDc.days_peak_to_recovery.max())}d avg {round(DDc.days_peak_to_recovery.mean(),1)}d]")
    print(DDc.head(8).to_string(index=False))
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
