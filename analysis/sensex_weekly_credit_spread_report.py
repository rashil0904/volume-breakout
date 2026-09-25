# -*- coding: utf-8 -*-
"""sensex_weekly_credit_spread_report.py — Excel report for the SENSEX Weekly Credit Spread (hedge width 600).
Reuses the trade set from sensex_weekly_credit_spread.py (re-runs its main to get the fresh trade table), then
builds: Sheet1 Summary (full metric set incl. drawdown depth & duration) and Sheet2 Trade_Log (per-trade +
running cumulative P&L + running drawdown). Drawdown on the realized cumulative-equity curve ordered by exit.
Save: sensex_weekly_credit_spread_report.xlsx.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUTDIR = rb.RESULTS / "sensex_weekly_credit_spread"
CSV = OUTDIR / "sensex_weekly_credit_spread_trades.csv"


def drawdown_episodes(equity, times):
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
                     "recovery_date": (pd.Timestamp(times[r]).date() if rec else "NOT RECOVERED"), "drawdown_points": round(abs(v), 1),
                     "days_peak_to_recovery": int((pd.Timestamp(times[r]) - pd.Timestamp(times[s])).days)})
    return pd.DataFrame(rows)


def main():
    if not CSV.exists():
        import sensex_weekly_credit_spread as bt; bt.main()
    T = pd.read_csv(CSV)
    T = T.sort_values("exit_time").reset_index(drop=True)                 # chronological by realized exit
    T["cum_pnl"] = T["pnl_points"].cumsum().round(2)
    peak = T["cum_pnl"].cummax(); T["running_drawdown"] = (T["cum_pnl"] - peak).round(2)   # <=0 depth below peak

    equity = np.concatenate([[0.0], T["cum_pnl"].values])
    times = np.concatenate([[pd.to_datetime(T["entry_time"]).iloc[0]], pd.to_datetime(T["exit_time"]).values])
    DE = drawdown_episodes(equity, times)
    maxdd = DE.drawdown_points.max() if len(DE) else 0.0; avgdd = round(DE.drawdown_points.mean(), 1) if len(DE) else 0.0
    maxddp = int(DE.days_peak_to_recovery.max()) if len(DE) else 0; avgddp = round(DE.days_peak_to_recovery.mean(), 1) if len(DE) else 0

    pos = T[T.pnl_points > 0]; neg = T[T.pnl_points < 0]
    win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)
    by_trig = (T.trigger.value_counts(normalize=True) * 100).round(1).to_dict()

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "SENSEX Weekly Credit Spread — hedge WIDTH=600 (only change vs NIFTY 200); GROSS premium points, 1 spread"},
        {"metric": "Window", "value": f"{T.entry_date.min()} .. {T.exit_date.max()}"},
        {"metric": "Exit rule", "value": "90% of max profit (1-min) else DTE-0 spot-settlement intrinsic; no stop"},
        {"metric": "", "value": ""},
        {"metric": "Total trades", "value": len(T)},
        {"metric": "Winning trades", "value": int((T.pnl_points > 0).sum())},
        {"metric": "Losing trades", "value": int((T.pnl_points < 0).sum())},
        {"metric": "Win rate %", "value": win},
        {"metric": "Total P&L (points)", "value": tot},
        {"metric": "Average P&L / trade", "value": round(T.pnl_points.mean(), 2)},
        {"metric": "Median P&L / trade", "value": round(T.pnl_points.median(), 2)},
        {"metric": "Average positive P&L", "value": round(pos.pnl_points.mean(), 2) if len(pos) else 0},
        {"metric": "Average negative P&L", "value": round(neg.pnl_points.mean(), 2) if len(neg) else 0},
        {"metric": "Max profit", "value": round(T.pnl_points.max(), 1)},
        {"metric": "Max loss", "value": round(T.pnl_points.min(), 1)},
        {"metric": "Average holding period (days)", "value": round(T.days_held.mean(), 2)},
        {"metric": "Average DTE at entry", "value": round(T.DTE.mean(), 2)},
        {"metric": "% by trigger (3d-high / 3d-low / fallback-2:30)",
         "value": f"{by_trig.get('3d-high',0)} / {by_trig.get('3d-low',0)} / {by_trig.get('fallback-2:30',0)}"},
        {"metric": "% exit 90% target / DTE0 settlement", "value": f"{round((T.exit_reason=='90% target').mean()*100,1)} / {round((T.exit_reason=='DTE0 settlement').mean()*100,1)}"},
        {"metric": "", "value": ""},
        {"metric": "--- DRAWDOWN (realized cumulative equity, points) ---", "value": ""},
        {"metric": "Number of drawdown episodes", "value": len(DE)},
        {"metric": "Average drawdown (points)", "value": avgdd},
        {"metric": "Maximum drawdown (points)", "value": round(maxdd, 1)},
        {"metric": "Average drawdown period (days, peak->recovery)", "value": avgddp},
        {"metric": "Maximum drawdown period (days, peak->recovery)", "value": maxddp},
        {"metric": "Return / Max-DD", "value": round(tot / maxdd, 2) if maxdd else "-"},
    ])

    TL = T[["entry_date", "DTE", "type", "trigger", "net_credit", "exit_date", "exit_reason", "exit_debit",
            "pnl_points", "cum_pnl", "running_drawdown"]]

    OUTF = OUTDIR / "sensex_weekly_credit_spread_report.xlsx"
    with pd.ExcelWriter(OUTF, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        TL.to_excel(w, sheet_name="Trade_Log", index=False)
        DE.sort_values("drawdown_points", ascending=False).to_excel(w, sheet_name="Drawdowns", index=False)

    pd.set_option("display.width", 230)
    print("=" * 92 + "\nSENSEX WEEKLY CREDIT SPREAD — REPORT (hedge width 600)\n" + "=" * 92)
    print(f"trades {len(T)} | win {win}% | TOTAL {tot:,} | avg {round(T.pnl_points.mean(),2)} | median {round(T.pnl_points.median(),2)}")
    print(f"avg+ {round(pos.pnl_points.mean(),2)} | avg- {round(neg.pnl_points.mean(),2)} | maxP {round(T.pnl_points.max(),1)} | maxL {round(T.pnl_points.min(),1)}")
    print(f"avg hold {round(T.days_held.mean(),2)}d | avg DTE {round(T.DTE.mean(),2)} | trigger% 3dH/3dL/FB {by_trig.get('3d-high',0)}/{by_trig.get('3d-low',0)}/{by_trig.get('fallback-2:30',0)}")
    print(f"DD: episodes {len(DE)} | avg {avgdd} | MAX {round(maxdd,1)} | avg period {avgddp}d | max period {maxddp}d | ret/maxDD {round(tot/maxdd,2) if maxdd else '-'}")
    print(f"\nSaved -> {OUTF}")


if __name__ == "__main__":
    main()
