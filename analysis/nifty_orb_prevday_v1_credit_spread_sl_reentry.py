# -*- coding: utf-8 -*-
"""VARIANT 1 (credit spread) + SL + RE-ENTRY. Direction-aligned: bullish+OR-high -> Put Credit Spread (SELL ATM
PE / BUY ATM-200 PE); bearish+OR-low -> Call Credit Spread (SELL ATM CE / BUY ATM+200 CE). SL = opposite OR
extreme (spot touch), closes BOTH legs; re-enter fresh spread on each subsequent desired break (new ATM); final
3:15 exit. Reversal-entry trigger for entry 1. GROSS premium pts, 1 spread. Uses orb_reentry_engine + shared report.
"""
import sys, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import orb_reentry_engine as eng
from nifty_orb_prevday_v2_buy_sl_reentry import report

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = eng.OPTDIR; WIDTH = 200


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    daily = sp.groupby("date").agg(o=("open", "first"), c=("close", "last")); tdays = list(daily.index); spot_end = sp["date"].max()
    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")); exp_dates = [e.normalize() for e in expiries]

    rows = []; skipped = 0
    for i in range(1, len(tdays)):
        D = tdays[i]; P = tdays[i - 1]
        if pd.Timestamp(D) < eng.FLOOR or D > spot_end: continue
        bias = "bullish" if daily.loc[P, "c"] > daily.loc[P, "o"] else ("bearish" if daily.loc[P, "c"] < daily.loc[P, "o"] else "neutral")
        if bias == "neutral": continue
        ed = sp[sp["date"] == D]; segs, ORH, ORL = eng.day_segments(ed, bias, D)
        if not segs: continue
        j = bisect.bisect_left(exp_dates, pd.Timestamp(D))
        if j >= len(exp_dates): continue
        folder = expiries[j].strftime("%Y%m%d")
        for s in segs:
            atm = s["atm"]
            if bias == "bullish": ot, sK, lK, dirn = "PE", atm, atm - WIDTH, "LONG-dir (PCS)"
            else: ot, sK, lK, dirn = "CE", atm, atm + WIDTH, "SHORT-dir (CCS)"
            s_en = eng.opt_px(folder, sK, ot, D, s["entry_ts"]); l_en = eng.opt_px(folder, lK, ot, D, s["entry_ts"])
            s_ex = eng.opt_px(folder, sK, ot, D, s["exit_ts"]); l_ex = eng.opt_px(folder, lK, ot, D, s["exit_ts"])
            if any(np.isnan(v) for v in (s_en, l_en, s_ex, l_ex)): skipped += 1; continue
            credit = s_en - l_en; exit_debit = s_ex - l_ex
            rows.append({"date": D.date(), "prev_bias": bias, "OR_high": round(ORH, 1), "OR_low": round(ORL, 1),
                         "entry_no": s["entry_no"], "entry_kind": s["entry_kind"], "direction": dirn,
                         "entry_time": s["entry_ts"].strftime("%H:%M"), "short_K": sK, "long_K": lK, "option_type": ot,
                         "SL_level": round(ORL if bias == "bullish" else ORH, 1), "net_credit": round(credit, 2),
                         "exit_time": s["exit_ts"].strftime("%H:%M"), "exit_reason": s["exit_reason"], "exit_debit": round(exit_debit, 2),
                         "pnl_points": round(credit - exit_debit, 2)})
    report(pd.DataFrame(rows), "V1 CREDIT SPREAD (aligned) + SL + RE-ENTRY", "v1_credit_spread_sl_reentry", skipped,
           "V1 CS aligned no-SL: +1032.6, 61.7% win; V1 CS aligned +SL (no reentry): +760.2, 52.3% win")


if __name__ == "__main__":
    main()
