# -*- coding: utf-8 -*-
"""VARIANT 2 (buy ATM) + SL + RE-ENTRY. Buy ATM CE (bullish/long) or ATM PE (bearish/short) on each desired-dir
OR break; SL = opposite OR extreme (spot touch); re-enter on each fresh desired break after an SL, new ATM
each time; final 3:15 exit. Reversal-entry trigger for entry 1. GROSS premium pts, 1 lot. Uses orb_reentry_engine.
"""
import sys, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import orb_reentry_engine as eng

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = eng.OPTDIR
OUTDIR = rb.RESULTS / "nifty_orb_prevday_variants"; OUTDIR.mkdir(parents=True, exist_ok=True)


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
        folder = expiries[j].strftime("%Y%m%d"); ot = "CE" if bias == "bullish" else "PE"
        for s in segs:
            en = eng.opt_px(folder, s["atm"], ot, D, s["entry_ts"]); ex = eng.opt_px(folder, s["atm"], ot, D, s["exit_ts"])
            if np.isnan(en) or np.isnan(ex): skipped += 1; continue
            rows.append({"date": D.date(), "prev_bias": bias, "OR_high": round(ORH, 1), "OR_low": round(ORL, 1),
                         "entry_no": s["entry_no"], "entry_kind": s["entry_kind"], "direction": "LONG" if ot == "CE" else "SHORT-dir",
                         "entry_time": s["entry_ts"].strftime("%H:%M"), "ATM": s["atm"], "option_type": ot,
                         "SL_level": round(ORL if bias == "bullish" else ORH, 1), "entry_premium": round(en, 2),
                         "exit_time": s["exit_ts"].strftime("%H:%M"), "exit_reason": s["exit_reason"], "exit_premium": round(ex, 2),
                         "pnl_points": round(ex - en, 2)})
    report(pd.DataFrame(rows), "V2 BUY ATM + SL + RE-ENTRY", "v2_buy_sl_reentry", skipped,
           "base buy no-SL/no-reentry: 300 trades, +356.8; V2 buy+SL (no reentry): 300 trades, +909.0, 39.7% win")


def report(T, title, fname, skipped, cmp_note):
    win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)
    slh = T[T.exit_reason == "SL-hit"]; tex = T[T.exit_reason == "3:15pm"]
    e1 = T[T.entry_no == 1]; ere = T[T.entry_no > 1]
    day = T.groupby("date").agg(entries=("entry_no", "max"), day_pnl=("pnl_points", "sum"),
                                last_reason=("exit_reason", "last")).reset_index()
    day["ended"] = day["last_reason"].map(lambda r: "SL on last entry" if r == "SL-hit" else "3:15 holding")

    def blk(df, lbl):
        return {"group": lbl, "trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_points.sum(), 1), "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0}
    byn = [blk(T[T.entry_no == k], f"entry #{k}") for k in sorted(T.entry_no.unique()) if k <= 5]
    BY = pd.DataFrame([blk(T, "ALL"), blk(e1, "entry #1"), blk(ere, "re-entries (#2+)"),
                       blk(slh, "SL-hit exits"), blk(tex, "3:15pm exits")] + byn[5:])
    ENUM = pd.DataFrame(byn)
    summary = pd.DataFrame([
        {"metric": "Variant", "value": title + " | reversal-entry ORB+bias | SL=opp OR extreme (spot touch) | re-enter on fresh desired break | ATM recomputed each entry | GROSS pts"},
        {"metric": "Window", "value": f"{T.date.min()} .. {T.date.max()}"},
        {"metric": "Total trades (incl re-entries)", "value": len(T)},
        {"metric": "Trading days", "value": T.date.nunique()},
        {"metric": "Avg entries / day", "value": round(day.entries.mean(), 2)},
        {"metric": "Max entries in a day", "value": int(day.entries.max())},
        {"metric": "Days with re-entry (>1 entry)", "value": int((day.entries > 1).sum())},
        {"metric": "Win rate %", "value": win}, {"metric": "Total P&L (points)", "value": tot},
        {"metric": "Avg P&L / trade", "value": round(T.pnl_points.mean(), 2)}, {"metric": "Median P&L / trade", "value": round(T.pnl_points.median(), 2)},
        {"metric": "Avg day P&L", "value": round(day.day_pnl.mean(), 2)},
        {"metric": "entry #1 trades / total / avg", "value": f"{len(e1)} / {round(e1.pnl_points.sum(),1)} / {round(e1.pnl_points.mean(),2)}"},
        {"metric": "re-entries (#2+) / total / avg", "value": f"{len(ere)} / {round(ere.pnl_points.sum(),1)} / {round(ere.pnl_points.mean(),2) if len(ere) else 0}"},
        {"metric": "SL-hit / 3:15 exits", "value": f"{len(slh)} ({round(len(slh)/len(T)*100,1)}%) / {len(tex)} ({round(len(tex)/len(T)*100,1)}%)"},
        {"metric": "Avg P&L SL-hit / 3:15", "value": f"{round(slh.pnl_points.mean(),2) if len(slh) else 0} / {round(tex.pnl_points.mean(),2) if len(tex) else 0}"},
        {"metric": "Days ended SL-on-last / 3:15-holding", "value": f"{int((day.ended=='SL on last entry').sum())} / {int((day.ended=='3:15 holding').sum())}"},
        {"metric": "Skipped (missing price)", "value": skipped},
        {"metric": "Comparison", "value": cmp_note},
    ])
    with pd.ExcelWriter(OUTDIR / f"{fname}.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False); BY.to_excel(w, sheet_name="By_Group", index=False)
        ENUM.to_excel(w, sheet_name="By_Entry_Number", index=False); day.to_excel(w, sheet_name="Daily_Summary", index=False)
        T.to_excel(w, sheet_name="Trades", index=False)
    T.to_csv(OUTDIR / f"{fname}_trades.csv", index=False)
    pd.set_option("display.width", 230)
    print("=" * 96 + f"\n{title}\n" + "=" * 96)
    print(f"trades {len(T)} | days {T.date.nunique()} | avg entries/day {round(day.entries.mean(),2)} (max {int(day.entries.max())}) | reentry days {int((day.entries>1).sum())}")
    print(f"win {win}% | TOTAL {tot:,} | avg {round(T.pnl_points.mean(),2)} | median {round(T.pnl_points.median(),2)}")
    print(f"SL-hit {len(slh)} ({round(len(slh)/len(T)*100,1)}%) avg {round(slh.pnl_points.mean(),2) if len(slh) else 0} | 3:15 {len(tex)} avg {round(tex.pnl_points.mean(),2) if len(tex) else 0}")
    print(f"entry#1 {len(e1)} avg {round(e1.pnl_points.mean(),2)} | re-entries {len(ere)} avg {round(ere.pnl_points.mean(),2) if len(ere) else 0}")
    print("\n--- BY ENTRY NUMBER ---"); print(ENUM.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}/{fname}.xlsx")


if __name__ == "__main__":
    main()
