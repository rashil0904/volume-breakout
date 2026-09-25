# -*- coding: utf-8 -*-
"""nifty_orb_prevday_v1_credit_spread.py — VARIANT 1 of NIFTY ORB + prev-day bias: CREDIT SPREAD instead of
buying. Same reversal-entry trigger as the base (desired-dir OR break taken whenever it occurs, incl. after
undesired-first). LONG dir (bullish+OR-high): SELL ATM CE + BUY (ATM+200) CE. SHORT dir (bearish+OR-low): SELL
ATM PE + BUY (ATM-200) PE. Entry at break-minute CLOSE; EXIT 15:15 CLOSE (fixed time, NO profit target/SL, as
specified). GROSS premium points (1 spread). Does not touch the base module.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "nifty_orb_prevday_variants"; OUTDIR.mkdir(parents=True, exist_ok=True)
OR_START, OR_END = 9 * 60 + 15, 9 * 60 + 29; SCAN_START, EXIT_MOD = 9 * 60 + 30, 15 * 60 + 15
FLOOR = pd.Timestamp("2024-10-01"); WIDTH = 200


def opt_px(folder, strike, ot, day, ts):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{ot}_*.parquet"))
    if not fs: return np.nan
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[o.ts.dt.normalize() == pd.Timestamp(day).normalize()].set_index("ts")["close"].sort_index()
    if not len(o): return np.nan
    v = o.asof(ts); return float(v) if pd.notna(v) else np.nan


def entry_signal(ed, bias):
    """returns (d_idx_row, entry_kind, first_side) or None. reversal-entry logic."""
    orb = ed[(ed["mod"] >= OR_START) & (ed["mod"] <= OR_END)]
    if len(orb) < 10: return None, None, None, None, None
    ORH = orb["high"].max(); ORL = orb["low"].min()
    scan = ed[(ed["mod"] >= SCAN_START) & (ed["mod"] <= EXIT_MOD)].sort_values("mod").reset_index(drop=True)
    up = scan["high"].values >= ORH; dn = scan["low"].values <= ORL
    up_idx = int(np.argmax(up)) if up.any() else None; dn_idx = int(np.argmax(dn)) if dn.any() else None
    desired_up = (bias == "bullish"); d_idx = up_idx if desired_up else dn_idx; u_idx = dn_idx if desired_up else up_idx
    if d_idx is None: return None, None, None, ORH, ORL
    undesired_first = (u_idx is not None and u_idx < d_idx)
    kind = "reversal (undesired-first)" if undesired_first else "first-breakout (desired)"
    return scan.iloc[d_idx], kind, ("undesired" if undesired_first else "desired"), ORH, ORL


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
        bspot = float(br["close"]); atm = int(round(bspot / 50) * 50); t_en = pd.Timestamp(br["ts"]); t_ex = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=15)
        if bias == "bullish": ot, sK, lK, dirn = "PE", atm, atm - WIDTH, "LONG-dir (Put Credit Spread)"     # bullish -> PCS (sell ATM PE / buy ATM-200 PE)
        else: ot, sK, lK, dirn = "CE", atm, atm + WIDTH, "SHORT-dir (Call Credit Spread)"                   # bearish -> CCS (sell ATM CE / buy ATM+200 CE)
        j = bisect.bisect_left(exp_dates, pd.Timestamp(D))
        if j >= len(exp_dates): continue
        folder = expiries[j].strftime("%Y%m%d")
        s_en = opt_px(folder, sK, ot, D, t_en); l_en = opt_px(folder, lK, ot, D, t_en)
        s_ex = opt_px(folder, sK, ot, D, t_ex); l_ex = opt_px(folder, lK, ot, D, t_ex)
        if any(np.isnan(v) for v in (s_en, l_en, s_ex, l_ex)): skipped += 1; continue
        credit = s_en - l_en; exit_debit = s_ex - l_ex; pnl = credit - exit_debit
        trades.append({"date": D.date(), "prev_bias": bias, "OR_high": round(ORH, 1), "OR_low": round(ORL, 1),
                       "breakout_type": kind, "first_side_broken": first_side, "direction": dirn, "entry_time": t_en.strftime("%H:%M"),
                       "short_K": sK, "long_K": lK, "option_type": ot, "short_entry": round(s_en, 2), "long_entry": round(l_en, 2),
                       "net_credit": round(credit, 2), "exit_time": "15:15", "short_exit": round(s_ex, 2), "long_exit": round(l_ex, 2),
                       "exit_debit": round(exit_debit, 2), "pnl_points": round(pnl, 2)})

    T = pd.DataFrame(trades); win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)
    lg = T[T.prev_bias == "bullish"]; sh = T[T.prev_bias == "bearish"]; fb = T[T.breakout_type.str.startswith("first")]; rv = T[T.breakout_type.str.startswith("reversal")]

    def blk(df, lbl):
        return {"group": lbl, "trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_points.sum(), 1), "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0,
                "avg_credit": round(df.net_credit.mean(), 2) if len(df) else 0, "max": round(df.pnl_points.max(), 1) if len(df) else 0, "min": round(df.pnl_points.min(), 1) if len(df) else 0}
    BY = pd.DataFrame([blk(T, "ALL"), blk(lg, "LONG-dir (Put Credit Spread)"), blk(sh, "SHORT-dir (Call Credit Spread)"), blk(fb, "first-breakout"), blk(rv, "reversal")])
    summary = pd.DataFrame([
        {"metric": "Variant", "value": "V1 CREDIT SPREAD, direction-ALIGNED: bullish->Put Credit Spread (sell ATM PE/buy ATM-200 PE); bearish->Call Credit Spread (sell ATM CE/buy ATM+200 CE). Reversal-entry ORB+bias. Exit 15:15 fixed (NO target/SL). GROSS pts, 1 spread."},
        {"metric": "Window", "value": f"{T.date.min()} .. {spot_end.date()}"}, {"metric": "Total trades", "value": len(T)},
        {"metric": "Win rate %", "value": win}, {"metric": "Total P&L (points)", "value": tot},
        {"metric": "Avg P&L / trade", "value": round(T.pnl_points.mean(), 2)}, {"metric": "Median P&L / trade", "value": round(T.pnl_points.median(), 2)},
        {"metric": "Avg net credit", "value": round(T.net_credit.mean(), 2)}, {"metric": "Max / min trade", "value": f"{round(T.pnl_points.max(),1)} / {round(T.pnl_points.min(),1)}"},
        {"metric": "LONG-dir PCS / SHORT-dir CCS", "value": f"{len(lg)} ({round(lg.pnl_points.sum(),1)}) / {len(sh)} ({round(sh.pnl_points.sum(),1)})"},
        {"metric": "first-breakout / reversal", "value": f"{len(fb)} ({round(fb.pnl_points.sum(),1)}, avg {round(fb.pnl_points.mean(),2) if len(fb) else 0}) / {len(rv)} ({round(rv.pnl_points.sum(),1)}, avg {round(rv.pnl_points.mean(),2) if len(rv) else 0})"},
        {"metric": "Skipped (missing leg)", "value": skipped},
        {"metric": "vs BASE (buy-ATM reversal)", "value": "base: 300 trades, +356.8 pts, 44.3% win, avg 1.19"},
    ])
    with pd.ExcelWriter(OUTDIR / "v1_credit_spread.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False); BY.to_excel(w, sheet_name="By_Group", index=False); T.to_excel(w, sheet_name="Trades", index=False)
    T.to_csv(OUTDIR / "v1_credit_spread_trades.csv", index=False)
    pd.set_option("display.width", 230)
    print("=" * 96 + "\nORB+BIAS VARIANT 1 — CREDIT SPREAD (exit 15:15 fixed)\n" + "=" * 96)
    print(f"trades {len(T)} | win {win}% | TOTAL {tot:,} | avg {round(T.pnl_points.mean(),2)} | median {round(T.pnl_points.median(),2)} | avg credit {round(T.net_credit.mean(),2)} | skipped {skipped}")
    print("\n--- BY GROUP ---"); print(BY.to_string(index=False)); print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
