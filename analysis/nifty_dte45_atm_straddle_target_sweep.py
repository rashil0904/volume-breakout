# -*- coding: utf-8 -*-
"""nifty_dte45_atm_straddle_target_sweep.py — sweeps the profit-TARGET fraction (30%-50%, 5% steps) on
the "Nifty Monthly DTE 45 ATM Short Straddle" strategy, keeping everything else IDENTICAL to the locked
nifty_dte45_atm_straddle_backtest.py: same 22 monthly cycles, same futures-derived ATM strikes, same
15:15 OPEN entry/exit convention, same 200%-of-credit stoploss, same DTE=21 hard ceiling on the touch-
search window (the bug fixed in that script), same open-then-close per-minute touch checking. Reuses the
already-cached CE/PE leg data (data/dte45_straddle_legs/NIFTY) -- no new API calls needed.
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import opt_pull_nifty_full as op
import nifty_dte45_atm_straddle_backtest as base

FUT_MANIFEST = rb.BASE / "data" / "futures_intraday_full" / "manifest_nifty_fut.csv"
FUT_DIR = rb.BASE / "data" / "futures_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "nifty_dte45_atm_straddle"; OUTDIR.mkdir(parents=True, exist_ok=True)
STOPLOSS_FRAC = 2.00
TARGET_FRACS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def run_one_cycle(row, target_frac):
    exp = row["expiry"]; exp_str = exp.strftime("%Y-%m-%d")
    fut_fn = FUT_DIR / exp.strftime("%Y%m%d") / (row["symbol"].replace(" ", "_").replace("/", "-") + ".parquet")
    fut = pd.read_parquet(fut_fn)
    fut["timestamp"] = pd.to_datetime(fut["timestamp"]); fut["date"] = fut["timestamp"].dt.date
    trading_days = sorted(fut["date"].unique())

    entry_target = exp - pd.Timedelta(days=base.ENTRY_DTE)
    exit_target = exp - pd.Timedelta(days=base.EXIT_DTE)
    entry_date = base.nearest_trading_day(entry_target, trading_days)
    exit_date_fallback = base.nearest_trading_day(exit_target, trading_days)

    f1515 = fut[(fut["date"] == entry_date) & (fut["timestamp"].dt.strftime("%H:%M") == "15:15")]
    if f1515.empty:
        return None
    fut_px = float(f1515["open"].iloc[0])
    atm = round(fut_px / base.STRIKE_STEP) * base.STRIKE_STEP

    cons, sc = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(base.NIFTY)}&expiry_date={exp_str}")
    if not cons:
        return None
    ce, err_ce = base.fetch_leg(cons, exp_str, atm, "CE", entry_date, exp)
    pe, err_pe = base.fetch_leg(cons, exp_str, atm, "PE", entry_date, exp)
    if ce is None or pe is None:
        return None
    if ce["timestamp"].min().date() > entry_date or pe["timestamp"].min().date() > entry_date:
        return None

    ces = ce.set_index("timestamp")["close"]; pes = pe.set_index("timestamp")["close"]
    ceo = ce.set_index("timestamp")["open"]; peo = pe.set_index("timestamp")["open"]
    common_ts = ces.index.intersection(pes.index).sort_values()
    common_ts = common_ts[common_ts.normalize() >= pd.Timestamp(entry_date)]
    if common_ts.empty:
        return None
    entry_ts_c = common_ts[(common_ts.normalize() == pd.Timestamp(entry_date)) & (common_ts.strftime("%H:%M") == "15:15")]
    if entry_ts_c.empty:
        return None
    entry_ts = entry_ts_c[0]
    entry_ce = float(ceo.loc[entry_ts]); entry_pe = float(peo.loc[entry_ts]); entry_credit = entry_ce + entry_pe
    # target_frac = TARGET PROFIT fraction (e.g. 0.20 = exit once 20% of the credit is captured as profit),
    # i.e. combined value must decay to (1-target_frac) of entry credit -- NOT to target_frac itself. These
    # coincide only at 50% (the spec's own worked example), which is why the bug was invisible there.
    target_level = entry_credit * (1 - target_frac)
    stop_level = entry_credit * STOPLOSS_FRAC

    boundary_c = common_ts[common_ts.normalize() <= pd.Timestamp(exit_date_fallback)]
    boundary_ts = boundary_c.max() if len(boundary_c) else entry_ts
    path_ts = common_ts[(common_ts > entry_ts) & (common_ts <= boundary_ts)]
    combined_close = (ces.loc[path_ts] + pes.loc[path_ts])
    combined_open = (ceo.loc[path_ts] + peo.loc[path_ts])

    exit_ts, exit_reason, exit_val = None, None, None
    for t in path_ts:
        o = combined_open.loc[t]; c = combined_close.loc[t]
        if o <= target_level:
            exit_ts, exit_reason, exit_val = t, "target", float(o); break
        if o >= stop_level:
            exit_ts, exit_reason, exit_val = t, "stoploss", float(o); break
        if c <= target_level:
            exit_ts, exit_reason, exit_val = t, "target", float(c); break
        if c >= stop_level:
            exit_ts, exit_reason, exit_val = t, "stoploss", float(c); break
    if exit_ts is None:
        te = path_ts[(path_ts.normalize() == pd.Timestamp(exit_date_fallback)) & (path_ts.strftime("%H:%M") == "15:15")]
        if len(te):
            exit_ts = te[0]; exit_reason = "time_exit"; exit_val = float(combined_open.loc[exit_ts])
        elif len(path_ts):
            exit_ts = path_ts[-1]; exit_reason = "time_exit_no_1515"; exit_val = float(combined_open.loc[exit_ts])
        else:
            exit_ts = entry_ts; exit_reason = "no_data"; exit_val = entry_credit

    pnl = entry_credit - exit_val
    return {"expiry": exp_str, "entry_date": entry_date, "exit_date": exit_ts.date(), "exit_reason": exit_reason,
            "entry_credit": round(entry_credit, 2), "exit_value": round(exit_val, 2), "pnl_points": round(pnl, 2),
            "days_held": (exit_ts.date() - entry_date).days}


def main():
    man = pd.read_csv(FUT_MANIFEST)
    man["expiry"] = pd.to_datetime(man["expiry"]).dt.date
    man = man.sort_values("expiry").reset_index(drop=True)

    all_trades = {}
    summary_rows = []
    for tf in TARGET_FRACS:
        rows = [run_one_cycle(r, tf) for _, r in man.iterrows()]
        rows = [r for r in rows if r is not None]
        T = pd.DataFrame(rows)
        all_trades[tf] = T
        win = T["pnl_points"] > 0
        summary_rows.append({
            "target_pct": int(tf * 100), "n_trades": len(T), "win_rate_pct": round(win.mean() * 100, 2),
            "total_pnl": round(T["pnl_points"].sum(), 2), "avg_pnl": round(T["pnl_points"].mean(), 2),
            "avg_days_held": round(T["days_held"].mean(), 2), "max_profit": round(T["pnl_points"].max(), 2),
            "max_loss": round(T["pnl_points"].min(), 2),
            "pct_target": round((T["exit_reason"] == "target").mean() * 100, 2),
            "pct_stoploss": round((T["exit_reason"] == "stoploss").mean() * 100, 2),
            "pct_time_exit": round((T["exit_reason"].isin(["time_exit", "time_exit_no_1515"])).mean() * 100, 2),
        })
        print(f"target={int(tf*100)}%: n={len(T)} win%={win.mean()*100:.1f} total={T['pnl_points'].sum():.1f} "
              f"avg={T['pnl_points'].mean():.1f} pct_target={summary_rows[-1]['pct_target']:.1f} "
              f"pct_stop={summary_rows[-1]['pct_stoploss']:.1f} pct_time={summary_rows[-1]['pct_time_exit']:.1f}", flush=True)

    SWEEP = pd.DataFrame(summary_rows)
    pd.set_option("display.width", 200)
    print("\n=== TARGET SWEEP 30%-50% (5% steps), stoploss fixed at 200% of credit, DTE21 hard ceiling ===")
    print(SWEEP.to_string(index=False))

    out_fn = OUTDIR / "target_sweep_30to50.xlsx"
    try:
        out_fn.touch(exist_ok=True)
    except PermissionError:
        out_fn = OUTDIR / "target_sweep_30to50_v2.xlsx"
    with pd.ExcelWriter(out_fn, engine="openpyxl") as w:
        SWEEP.to_excel(w, sheet_name="Sweep_Summary", index=False)
        for tf, T in all_trades.items():
            T.to_excel(w, sheet_name=f"Trades_{int(tf*100)}pct", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 26)
    print(f"\nSaved -> {out_fn}")


if __name__ == "__main__":
    main()
