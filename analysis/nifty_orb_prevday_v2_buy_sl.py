# -*- coding: utf-8 -*-
"""nifty_orb_prevday_v2_buy_sl.py — VARIANT 2 of NIFTY ORB + prev-day bias: BUY ATM (as base) + SPOT stop-loss.
Same reversal-entry trigger as base (imported entry_signal). LONG buy ATM CE (bullish) / SHORT-dir buy ATM PE
(bearish). SL = opposite end of the 9:15-9:29 opening range: long SL = OR low, short SL = OR high. SL is on
NIFTY SPOT, TOUCH-based (intraday low/high touching level). On SL touch, exit the option at that minute's price;
else exit 15:15. Entry = break-minute CLOSE; fills at the trigger minute's CLOSE. GROSS premium pts, 1 lot.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
from nifty_orb_prevday_v1_credit_spread import entry_signal, opt_px, OR_START, OR_END, SCAN_START, EXIT_MOD, FLOOR

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "nifty_orb_prevday_variants"; OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    daily = sp.groupby("date").agg(o=("open", "first"), c=("close", "last")); tdays = list(daily.index); spot_end = sp["date"].max()
    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")); exp_dates = [e.normalize() for e in expiries]

    trades = []; skipped = 0
    for i in range(1, len(tdays)):
        D = tdays[i]; P = tdays[i - 1]
        if pd.Timestamp(D) < FLOOR or D > spot_end: continue
        popen = daily.loc[P, "o"]; pclose = daily.loc[P, "c"]
        bias = "bullish" if pclose > popen else ("bearish" if pclose < popen else "neutral")
        if bias == "neutral": continue
        ed = sp[sp["date"] == D]
        br, kind, first_side, ORH, ORL = entry_signal(ed, bias)
        if br is None: continue
        bspot = float(br["close"]); atm = int(round(bspot / 50) * 50); ot = "CE" if bias == "bullish" else "PE"
        t_en = pd.Timestamp(br["ts"]); en_mod = int(br["mod"]); t_time = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=15)
        sl_level = ORL if bias == "bullish" else ORH
        # SL scan on spot after entry (touch-based)
        post = ed[(ed["mod"] > en_mod) & (ed["mod"] <= EXIT_MOD)].sort_values("mod")
        if bias == "bullish": hit = post[post["low"] <= sl_level]
        else: hit = post[post["high"] >= sl_level]
        if len(hit): t_ex = pd.Timestamp(hit.iloc[0]["ts"]); reason = "SL-hit"
        else: t_ex = t_time; reason = "3:15pm"
        j = bisect.bisect_left(exp_dates, pd.Timestamp(D))
        if j >= len(exp_dates): continue
        folder = expiries[j].strftime("%Y%m%d")
        en = opt_px(folder, atm, ot, D, t_en); ex = opt_px(folder, atm, ot, D, t_ex)
        if np.isnan(en) or np.isnan(ex): skipped += 1; continue
        trades.append({"date": D.date(), "prev_bias": bias, "OR_high": round(ORH, 1), "OR_low": round(ORL, 1),
                       "breakout_type": kind, "first_side_broken": first_side, "direction": "LONG" if ot == "CE" else "SHORT-dir",
                       "entry_time": t_en.strftime("%H:%M"), "ATM": atm, "option_type": ot, "SL_level": round(sl_level, 1),
                       "entry_premium": round(en, 2), "exit_time": pd.Timestamp(t_ex).strftime("%H:%M"), "exit_reason": reason,
                       "exit_premium": round(ex, 2), "pnl_points": round(ex - en, 2)})

    T = pd.DataFrame(trades); win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)
    lg = T[T.option_type == "CE"]; sh = T[T.option_type == "PE"]
    slh = T[T.exit_reason == "SL-hit"]; tex = T[T.exit_reason == "3:15pm"]
    fb = T[T.breakout_type.str.startswith("first")]; rv = T[T.breakout_type.str.startswith("reversal")]

    def blk(df, lbl):
        return {"group": lbl, "trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_points.sum(), 1), "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0,
                "max": round(df.pnl_points.max(), 1) if len(df) else 0, "min": round(df.pnl_points.min(), 1) if len(df) else 0}
    BY = pd.DataFrame([blk(T, "ALL"), blk(lg, "LONG (CE)"), blk(sh, "SHORT-dir (PE)"), blk(fb, "first-breakout"), blk(rv, "reversal"),
                       blk(slh, "SL-hit exits"), blk(tex, "3:15pm exits")])
    summary = pd.DataFrame([
        {"metric": "Variant", "value": "V2 BUY ATM + SPOT SL (SL = opposite OR extreme, touch-based). Reversal-entry ORB+bias. Exit SL-hit or 15:15. GROSS pts, 1 lot."},
        {"metric": "Window", "value": f"{T.date.min()} .. {spot_end.date()}"}, {"metric": "Total trades", "value": len(T)},
        {"metric": "Win rate %", "value": win}, {"metric": "Total P&L (points)", "value": tot},
        {"metric": "Avg P&L / trade", "value": round(T.pnl_points.mean(), 2)}, {"metric": "Median P&L / trade", "value": round(T.pnl_points.median(), 2)},
        {"metric": "Max / min trade", "value": f"{round(T.pnl_points.max(),1)} / {round(T.pnl_points.min(),1)}"},
        {"metric": "LONG / SHORT-dir", "value": f"{len(lg)} ({round(lg.pnl_points.sum(),1)}) / {len(sh)} ({round(sh.pnl_points.sum(),1)})"},
        {"metric": "SL-hit trades / % of total", "value": f"{len(slh)} / {round(len(slh)/len(T)*100,1)}%"},
        {"metric": "3:15pm-exit trades / % of total", "value": f"{len(tex)} / {round(len(tex)/len(T)*100,1)}%"},
        {"metric": "Avg P&L on SL-hit trades", "value": round(slh.pnl_points.mean(), 2) if len(slh) else 0},
        {"metric": "Avg P&L on 3:15pm-exit trades", "value": round(tex.pnl_points.mean(), 2) if len(tex) else 0},
        {"metric": "first-breakout / reversal", "value": f"{len(fb)} ({round(fb.pnl_points.sum(),1)}) / {len(rv)} ({round(rv.pnl_points.sum(),1)})"},
        {"metric": "Skipped (missing option)", "value": skipped},
        {"metric": "vs BASE (buy-ATM, NO SL, reversal)", "value": "base: 300 trades, +356.8 pts, 44.3% win, avg 1.19, median -10.45"},
    ])
    with pd.ExcelWriter(OUTDIR / "v2_buy_sl.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False); BY.to_excel(w, sheet_name="By_Group", index=False); T.to_excel(w, sheet_name="Trades", index=False)
    T.to_csv(OUTDIR / "v2_buy_sl_trades.csv", index=False)
    pd.set_option("display.width", 230)
    print("=" * 96 + "\nORB+BIAS VARIANT 2 — BUY ATM + SPOT SL (opposite OR extreme, touch)\n" + "=" * 96)
    print(f"trades {len(T)} | win {win}% | TOTAL {tot:,} | avg {round(T.pnl_points.mean(),2)} | median {round(T.pnl_points.median(),2)} | skipped {skipped}")
    print(f"SL-hit {len(slh)} ({round(len(slh)/len(T)*100,1)}%) avg {round(slh.pnl_points.mean(),2) if len(slh) else 0} | 3:15 exits {len(tex)} ({round(len(tex)/len(T)*100,1)}%) avg {round(tex.pnl_points.mean(),2) if len(tex) else 0}")
    print("\n--- BY GROUP ---"); print(BY.to_string(index=False)); print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
