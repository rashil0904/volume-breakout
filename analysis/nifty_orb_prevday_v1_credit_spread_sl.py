# -*- coding: utf-8 -*-
"""nifty_orb_prevday_v1_credit_spread_sl.py — VARIANT 1 (credit spread) + STOP-LOSS. Direction-aligned:
bullish+OR-high -> Put Credit Spread (SELL ATM PE / BUY ATM-200 PE); bearish+OR-low -> Call Credit Spread
(SELL ATM CE / BUY ATM+200 CE). Reversal-entry ORB+bias trigger (imported). SL = opposite OR extreme (long
SL=OR low, short SL=OR high) on NIFTY SPOT, TOUCH-based; on SL hit close BOTH legs at that minute's prices.
Exit = SL-hit OR 15:15, whichever first. Entry/fills = trigger-minute CLOSE. GROSS premium pts, 1 spread.
Keeps the no-SL v1 module intact.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
from nifty_orb_prevday_v1_credit_spread import entry_signal, opt_px, EXIT_MOD, FLOOR

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "nifty_orb_prevday_variants"; OUTDIR.mkdir(parents=True, exist_ok=True); WIDTH = 200


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
        bspot = float(br["close"]); atm = int(round(bspot / 50) * 50); en_mod = int(br["mod"]); t_en = pd.Timestamp(br["ts"]); t_time = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=15)
        if bias == "bullish": ot, sK, lK, dirn, sl_level = "PE", atm, atm - WIDTH, "LONG-dir (Put Credit Spread)", ORL
        else: ot, sK, lK, dirn, sl_level = "CE", atm, atm + WIDTH, "SHORT-dir (Call Credit Spread)", ORH
        post = ed[(ed["mod"] > en_mod) & (ed["mod"] <= EXIT_MOD)].sort_values("mod")
        hit = post[post["low"] <= sl_level] if bias == "bullish" else post[post["high"] >= sl_level]
        if len(hit): t_ex = pd.Timestamp(hit.iloc[0]["ts"]); reason = "SL-hit"
        else: t_ex = t_time; reason = "3:15pm"
        j = bisect.bisect_left(exp_dates, pd.Timestamp(D))
        if j >= len(exp_dates): continue
        folder = expiries[j].strftime("%Y%m%d")
        s_en = opt_px(folder, sK, ot, D, t_en); l_en = opt_px(folder, lK, ot, D, t_en)
        s_ex = opt_px(folder, sK, ot, D, t_ex); l_ex = opt_px(folder, lK, ot, D, t_ex)
        if any(np.isnan(v) for v in (s_en, l_en, s_ex, l_ex)): skipped += 1; continue
        credit = s_en - l_en; exit_debit = s_ex - l_ex; pnl = credit - exit_debit
        trades.append({"date": D.date(), "prev_bias": bias, "OR_high": round(ORH, 1), "OR_low": round(ORL, 1),
                       "breakout_type": kind, "first_side_broken": first_side, "direction": dirn, "entry_time": t_en.strftime("%H:%M"),
                       "short_K": sK, "long_K": lK, "option_type": ot, "SL_level": round(sl_level, 1), "net_credit": round(credit, 2),
                       "exit_time": pd.Timestamp(t_ex).strftime("%H:%M"), "exit_reason": reason, "exit_debit": round(exit_debit, 2), "pnl_points": round(pnl, 2)})

    T = pd.DataFrame(trades); win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)
    lg = T[T.prev_bias == "bullish"]; sh = T[T.prev_bias == "bearish"]
    fb = T[T.breakout_type.str.startswith("first")]; rv = T[T.breakout_type.str.startswith("reversal")]
    slh = T[T.exit_reason == "SL-hit"]; tex = T[T.exit_reason == "3:15pm"]

    def blk(df, lbl):
        return {"group": lbl, "trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_points.sum(), 1), "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0,
                "max": round(df.pnl_points.max(), 1) if len(df) else 0, "min": round(df.pnl_points.min(), 1) if len(df) else 0}
    BY = pd.DataFrame([blk(T, "ALL"), blk(lg, "LONG-dir (PCS)"), blk(sh, "SHORT-dir (CCS)"), blk(fb, "first-breakout"), blk(rv, "reversal"),
                       blk(slh, "SL-hit exits"), blk(tex, "3:15pm exits")])
    summary = pd.DataFrame([
        {"metric": "Variant", "value": "V1 CREDIT SPREAD (direction-aligned) + SPOT SL (opposite OR extreme, touch). Reversal-entry ORB+bias. Exit SL-hit or 15:15. GROSS pts, 1 spread."},
        {"metric": "Window", "value": f"{T.date.min()} .. {spot_end.date()}"}, {"metric": "Total trades", "value": len(T)},
        {"metric": "Win rate %", "value": win}, {"metric": "Total P&L (points)", "value": tot},
        {"metric": "Avg P&L / trade", "value": round(T.pnl_points.mean(), 2)}, {"metric": "Median P&L / trade", "value": round(T.pnl_points.median(), 2)},
        {"metric": "Avg net credit", "value": round(T.net_credit.mean(), 2)}, {"metric": "Max / min trade", "value": f"{round(T.pnl_points.max(),1)} / {round(T.pnl_points.min(),1)}"},
        {"metric": "LONG-dir PCS / SHORT-dir CCS", "value": f"{len(lg)} ({round(lg.pnl_points.sum(),1)}) / {len(sh)} ({round(sh.pnl_points.sum(),1)})"},
        {"metric": "SL-hit trades / % of total", "value": f"{len(slh)} / {round(len(slh)/len(T)*100,1)}%"},
        {"metric": "3:15pm-exit trades / % of total", "value": f"{len(tex)} / {round(len(tex)/len(T)*100,1)}%"},
        {"metric": "Avg P&L on SL-hit trades", "value": round(slh.pnl_points.mean(), 2) if len(slh) else 0},
        {"metric": "Avg P&L on 3:15pm-exit trades", "value": round(tex.pnl_points.mean(), 2) if len(tex) else 0},
        {"metric": "first-breakout / reversal", "value": f"{len(fb)} ({round(fb.pnl_points.sum(),1)}) / {len(rv)} ({round(rv.pnl_points.sum(),1)})"},
        {"metric": "Skipped (missing leg)", "value": skipped},
        {"metric": "vs V1 no-SL", "value": "no-SL: 300 trades, +1032.6 pts, 61.7% win, avg 3.44, median 8.62, worst -168.4"},
    ])
    with pd.ExcelWriter(OUTDIR / "v1_credit_spread_sl.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False); BY.to_excel(w, sheet_name="By_Group", index=False); T.to_excel(w, sheet_name="Trades", index=False)
    T.to_csv(OUTDIR / "v1_credit_spread_sl_trades.csv", index=False)
    pd.set_option("display.width", 230)
    print("=" * 96 + "\nORB+BIAS VARIANT 1 — CREDIT SPREAD (aligned) + SPOT SL\n" + "=" * 96)
    print(f"trades {len(T)} | win {win}% | TOTAL {tot:,} | avg {round(T.pnl_points.mean(),2)} | median {round(T.pnl_points.median(),2)} | avg credit {round(T.net_credit.mean(),2)} | skipped {skipped}")
    print(f"SL-hit {len(slh)} ({round(len(slh)/len(T)*100,1)}%) avg {round(slh.pnl_points.mean(),2) if len(slh) else 0} | 3:15 exits {len(tex)} ({round(len(tex)/len(T)*100,1)}%) avg {round(tex.pnl_points.mean(),2) if len(tex) else 0}")
    print("\n--- BY GROUP ---"); print(BY.to_string(index=False)); print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
