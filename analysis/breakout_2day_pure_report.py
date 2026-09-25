# -*- coding: utf-8 -*-
"""breakout_2day_pure_report.py — Excel report for the PURE daily 2-day breakout (0 buffer, no gap).
Sheets: Summary (performance + drawdown: # episodes, points, durations), Drawdown_Episodes (each DD period),
Trades. Drawdown computed on the realized cumulative-points equity curve (stepped at each trade exit)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUTDIR = rb.RESULTS / "breakout_2day_pure"
CSV = OUTDIR / "breakout_2day_pure_trades.csv"
DATA_1MIN = rb.BASE / "data" / "nifty_1min_ohlc.csv"


def add_excursions(T):
    """per-trade MFE (max profit reached) & MAE (max loss reached) from the 1-min path within each trade."""
    d1 = pd.read_csv(DATA_1MIN, usecols=["timestamp", "high", "low"])
    ts1 = pd.to_datetime(d1["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d1 = d1.assign(ts=ts1).sort_values("ts").reset_index(drop=True)
    tsv = d1["ts"].values; hv = d1["high"].values; lv = d1["low"].values
    mfe = []; mae = []
    for r in T.itertuples():
        e = np.datetime64(pd.Timestamp(r.entry_time)); x = np.datetime64(pd.Timestamp(r.exit_time))
        i0 = np.searchsorted(tsv, e, "left"); i1 = np.searchsorted(tsv, x, "right")
        if i1 <= i0:
            mfe.append(0.0); mae.append(0.0); continue
        hi = hv[i0:i1].max(); lo = lv[i0:i1].min()
        if r.direction == "Long":
            mfe.append(hi - r.entry_price); mae.append(lo - r.entry_price)          # best up / worst down while long
        else:
            mfe.append(r.entry_price - lo); mae.append(r.entry_price - hi)          # best down / worst up while short
    T["mfe_points"] = np.round(mfe, 1)      # max PROFIT reached intra-trade (>=0)
    T["mae_points"] = np.round(mae, 1)      # max LOSS reached intra-trade (<=0)
    return T


def drawdown_episodes(equity, times):
    """episodes where equity sits below a prior peak. returns list of (peak_i, trough_i, recover_i, depth)."""
    peak = np.maximum.accumulate(equity); dd = equity - peak
    eps = []; in_dd = False; s = t = 0; tv = 0.0
    for k in range(len(equity)):
        if not in_dd:
            if dd[k] < -1e-9: in_dd = True; s = k - 1; t = k; tv = dd[k]      # s = last peak index
        else:
            if dd[k] < tv: tv = dd[k]; t = k                                  # deeper trough
            if dd[k] >= -1e-9: eps.append((s, t, k, tv)); in_dd = False        # recovered to new high
    if in_dd: eps.append((s, t, len(equity) - 1, tv))                          # still under water at the end
    rows = []
    for (s, t, r, v) in eps:
        recovered = (r != len(equity) - 1) or (equity[r] >= peak[s] - 1e-9)
        rows.append({"peak_date": pd.Timestamp(times[s]).date(), "trough_date": pd.Timestamp(times[t]).date(),
                     "recovery_date": (pd.Timestamp(times[r]).date() if recovered else "NOT RECOVERED"),
                     "drawdown_points": round(abs(v), 1),
                     "days_peak_to_trough": int((pd.Timestamp(times[t]) - pd.Timestamp(times[s])).days),
                     "days_peak_to_recovery": int((pd.Timestamp(times[r]) - pd.Timestamp(times[s])).days)})
    return pd.DataFrame(rows)


def main():
    T = pd.read_csv(CSV); T = add_excursions(T)
    ent = pd.to_datetime(T["entry_time"]); ext = pd.to_datetime(T["exit_time"])
    equity = np.concatenate([[0.0], T["cum_points"].values]); times = np.concatenate([[ent.iloc[0]], ext.values])
    DE = drawdown_episodes(equity, times)

    tot = round(T.points_pnl.sum(), 1); win = round((T.points_pnl > 0).mean() * 100, 1)
    span_days = (ext.iloc[-1] - ent.iloc[0]).days
    maxdd = DE.drawdown_points.max() if len(DE) else 0
    maxdd_row = DE.loc[DE.drawdown_points.idxmax()] if len(DE) else None
    longest = DE.loc[DE.days_peak_to_recovery.idxmax()] if len(DE) else None

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY daily 2-day high/low breakout — PURE (0 buffer, no gap), always-in flip"},
        {"metric": "Period", "value": f"{ent.iloc[0].date()} .. {ext.iloc[-1].date()}"},
        {"metric": "P&L basis", "value": "GROSS index points; equity = realized cum points stepped at each trade exit"},
        {"metric": "", "value": ""},
        {"metric": "Total trades", "value": len(T)},
        {"metric": "Win rate %", "value": win},
        {"metric": "Total P&L (points)", "value": tot},
        {"metric": "Avg points / trade", "value": round(T.points_pnl.mean(), 2)},
        {"metric": "Best / worst REALIZED trade (pts)", "value": f"{round(T.points_pnl.max(),1)} / {round(T.points_pnl.min(),1)}"},
        {"metric": "Max PROFIT reached in a trade (MFE, pts)", "value": round(T.mfe_points.max(), 1)},
        {"metric": "Max LOSS reached in a trade (MAE, pts)", "value": round(T.mae_points.min(), 1)},
        {"metric": "Avg max-profit / trade (MFE)", "value": round(T.mfe_points.mean(), 1)},
        {"metric": "Avg max-loss / trade (MAE)", "value": round(T.mae_points.mean(), 1)},
        {"metric": "  -> note", "value": "MFE/MAE = best/worst UNREALIZED excursion within each trade (from 1-min path); realized P&L is between them"},
        {"metric": "Avg holding (calendar days)", "value": round(T.hold_cal_days.mean(), 2)},
        {"metric": "Long / Short trades", "value": f"{int((T.direction=='Long').sum())} / {int((T.direction=='Short').sum())}"},
        {"metric": "Signals / week", "value": round(len(T) / (span_days / 7), 2)},
        {"metric": "Signals / month", "value": round(len(T) / (span_days / 30.44), 2)},
        {"metric": "", "value": ""},
        {"metric": "--- DRAWDOWN (on realized equity curve) ---", "value": ""},
        {"metric": "Number of drawdown episodes", "value": len(DE)},
        {"metric": "Max drawdown (points)", "value": round(maxdd, 1)},
        {"metric": "Avg drawdown (points)", "value": round(DE.drawdown_points.mean(), 1) if len(DE) else 0},
        {"metric": "Max drawdown duration (days, peak->recovery)", "value": int(DE.days_peak_to_recovery.max()) if len(DE) else 0},
        {"metric": "Avg drawdown duration (days)", "value": round(DE.days_peak_to_recovery.mean(), 1) if len(DE) else 0},
        {"metric": "Return / Max-DD", "value": round(tot / maxdd, 2) if maxdd else "-"},
        {"metric": "Max-DD episode", "value": f"{maxdd_row.peak_date} -> {maxdd_row.trough_date} -> {maxdd_row.recovery_date} ({maxdd_row.drawdown_points} pts)" if maxdd_row is not None else "-"},
        {"metric": "Longest-DD episode", "value": f"{longest.peak_date} -> {longest.recovery_date} ({longest.days_peak_to_recovery} days, {longest.drawdown_points} pts)" if longest is not None else "-"},
    ])

    out = OUTDIR / "breakout_2day_pure.xlsx"
    try:
        wr = pd.ExcelWriter(out, engine="openpyxl")
    except PermissionError:
        out = OUTDIR / "breakout_2day_pure_new.xlsx"; wr = pd.ExcelWriter(out, engine="openpyxl")
    with wr as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        DE.sort_values("drawdown_points", ascending=False).to_excel(w, sheet_name="Drawdown_Episodes", index=False)
        T.to_excel(w, sheet_name="Trades", index=False)

    pd.set_option("display.width", 200)
    print(f"PURE 2-day breakout report | {ent.iloc[0].date()}..{ext.iloc[-1].date()}")
    print(f"trades {len(T)} | win {win}% | total {tot:,} pts | avg hold {round(T.hold_cal_days.mean(),2)}d")
    print(f"drawdown: {len(DE)} episodes | max {round(maxdd,1)} pts | avg {round(DE.drawdown_points.mean(),1)} pts | "
          f"max duration {int(DE.days_peak_to_recovery.max())} d | avg duration {round(DE.days_peak_to_recovery.mean(),1)} d")
    print("\ntop 5 drawdowns by points:")
    print(DE.sort_values("drawdown_points", ascending=False).head(5).to_string(index=False))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
