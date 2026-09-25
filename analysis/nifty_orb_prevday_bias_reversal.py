# -*- coding: utf-8 -*-
"""nifty_orb_prevday_bias_reversal.py — NIFTY ORB + prev-day bias, REVERSAL-ENTRY variant. Unlike the skip-if-
undesired-first version, here the DESIRED-direction breakout always triggers entry whenever it first occurs,
even if the undesired side broke earlier the same day (reversal entry). Skip only if the desired level never
breaks. Prev-day bias = prev day CLOSE vs OPEN. OR = 09:15-09:29 hi/lo. TOUCH-basis (confirmed). bullish ->
enter LONG (ATM CE) at first OR-HIGH break ; bearish -> ATM PE at first OR-LOW break. ATM=round(spot/50)*50 at
that break. Entry = break-minute option CLOSE ; EXIT = 15:15 CLOSE (time-only, no SL/target, confirmed). Front
expiry >= day. GROSS premium pts, 1 lot. Classifies each trade first-breakout-desired vs undesired-first-reversal.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "nifty_orb_prevday_bias_reversal"; OUTDIR.mkdir(parents=True, exist_ok=True)
OR_START, OR_END = 9 * 60 + 15, 9 * 60 + 29; SCAN_START, EXIT_MOD = 9 * 60 + 30, 15 * 60 + 15
FLOOR = pd.Timestamp("2024-10-01")


def opt_px(folder, strike, ot, day, ts):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{ot}_*.parquet"))
    if not fs: return np.nan
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[o.ts.dt.normalize() == pd.Timestamp(day).normalize()].set_index("ts")["close"].sort_index()
    if not len(o): return np.nan
    v = o.asof(ts); return float(v) if pd.notna(v) else np.nan


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    daily = sp.groupby("date").agg(o=("open", "first"), c=("close", "last"))
    tdays = list(daily.index); spot_end = sp["date"].max()
    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")); exp_dates = [e.normalize() for e in expiries]

    trades = []; notrades = []; flags = []
    for i in range(1, len(tdays)):
        D = tdays[i]; P = tdays[i - 1]
        if pd.Timestamp(D) < FLOOR or D > spot_end: continue
        popen = daily.loc[P, "o"]; pclose = daily.loc[P, "c"]
        bias = "bullish" if pclose > popen else ("bearish" if pclose < popen else "neutral")
        ed = sp[sp["date"] == D]; orb = ed[(ed["mod"] >= OR_START) & (ed["mod"] <= OR_END)]
        if bias == "neutral" or len(orb) < 10:
            notrades.append({"date": D.date(), "prev_bias": bias, "reason": "neutral bias" if bias == "neutral" else "incomplete OR"}); continue
        ORH = orb["high"].max(); ORL = orb["low"].min()
        scan = ed[(ed["mod"] >= SCAN_START) & (ed["mod"] <= EXIT_MOD)].sort_values("mod").reset_index(drop=True)
        up = scan["high"].values >= ORH; dn = scan["low"].values <= ORL
        up_idx = int(np.argmax(up)) if up.any() else None; dn_idx = int(np.argmax(dn)) if dn.any() else None
        desired_up = (bias == "bullish")
        d_idx = up_idx if desired_up else dn_idx; u_idx = dn_idx if desired_up else up_idx
        if d_idx is None:
            broke = "undesired only" if u_idx is not None else "no breakout"
            notrades.append({"date": D.date(), "prev_bias": bias, "reason": f"desired never broke ({broke})"}); continue
        undesired_first = (u_idx is not None and u_idx < d_idx)
        first_side = "undesired" if undesired_first else "desired"
        entry_kind = "reversal (undesired-first)" if undesired_first else "first-breakout (desired)"
        br = scan.iloc[d_idx]
        if bool(up[d_idx]) and bool(dn[d_idx]) and u_idx == d_idx:
            flags.append({"date": D.date(), "flag": "outside_bar_on_desired_break", "detail": "both levels in the entry candle"})
        bspot = float(br["close"]); atm = int(round(bspot / 50) * 50); ot = "CE" if desired_up else "PE"
        t_en = pd.Timestamp(br["ts"]); t_ex = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=15)
        j = bisect.bisect_left(exp_dates, pd.Timestamp(D))
        if j >= len(exp_dates): continue
        folder = expiries[j].strftime("%Y%m%d")
        en = opt_px(folder, atm, ot, D, t_en); ex = opt_px(folder, atm, ot, D, t_ex)
        if np.isnan(en) or np.isnan(ex):
            flags.append({"date": D.date(), "flag": "missing_option_price", "detail": f"{atm}{ot}"})
            notrades.append({"date": D.date(), "prev_bias": bias, "reason": "missing option price"}); continue
        trades.append({"date": D.date(), "prev_bias": bias, "OR_high": round(ORH, 1), "OR_low": round(ORL, 1),
                       "first_side_broken": first_side, "entry_kind": entry_kind,
                       "breakout_dir": "up" if desired_up else "down", "direction": "LONG" if ot == "CE" else "SHORT-dir",
                       "entry_time": t_en.strftime("%H:%M"), "ATM": atm, "option_type": ot,
                       "entry_premium": round(en, 2), "exit_time": "15:15", "exit_premium": round(ex, 2),
                       "pnl_points": round(ex - en, 2), "expiry_used": expiries[j].date()})

    T = pd.DataFrame(trades); NT = pd.DataFrame(notrades); total_days = len(T) + len(NT)
    win = round((T.pnl_points > 0).mean() * 100, 1) if len(T) else 0; tot = round(T.pnl_points.sum(), 1)

    def blk(df, lbl):
        return {"group": lbl, "trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_points.sum(), 1), "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0,
                "median_pnl": round(df.pnl_points.median(), 2) if len(df) else 0,
                "max": round(df.pnl_points.max(), 1) if len(df) else 0, "min": round(df.pnl_points.min(), 1) if len(df) else 0}
    lg = T[T.option_type == "CE"]; sh = T[T.option_type == "PE"]
    fb = T[T.entry_kind.str.startswith("first")]; rv = T[T.entry_kind.str.startswith("reversal")]
    BY = pd.DataFrame([blk(T, "ALL"), blk(lg, "LONG (CE)"), blk(sh, "SHORT-dir (PE)"),
                       blk(fb, "first-breakout-desired"), blk(rv, "undesired-first-then-reversal")])
    nt_reasons = NT.reason.value_counts().to_dict() if len(NT) else {}

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY ORB + prev-day bias, REVERSAL-ENTRY variant (take desired-dir break whenever it occurs, incl. after undesired-first). Touch-basis; naked long ATM; time-only 15:15 exit."},
        {"metric": "Diff vs skip-variant", "value": "undesired-first NO LONGER skips the day; desired break still enters (reversal). Skip only if desired level never breaks."},
        {"metric": "Window", "value": f"{T.date.min() if len(T) else '-'} .. {spot_end.date()}"},
        {"metric": "Total days evaluated", "value": total_days},
        {"metric": "Total trades", "value": len(T)}, {"metric": "No-trade days", "value": len(NT)},
        {"metric": "% days NO trade", "value": round(len(NT) / total_days * 100, 1) if total_days else 0},
        {"metric": "Win rate %", "value": win}, {"metric": "Total P&L (points)", "value": tot},
        {"metric": "Avg P&L / trade", "value": round(T.pnl_points.mean(), 2) if len(T) else 0},
        {"metric": "Median P&L / trade", "value": round(T.pnl_points.median(), 2) if len(T) else 0},
        {"metric": "LONG / SHORT-dir trades", "value": f"{len(lg)} / {len(sh)}"},
        {"metric": "LONG total / avg", "value": f"{round(lg.pnl_points.sum(),1)} / {round(lg.pnl_points.mean(),2) if len(lg) else 0}"},
        {"metric": "SHORT-dir total / avg", "value": f"{round(sh.pnl_points.sum(),1)} / {round(sh.pnl_points.mean(),2) if len(sh) else 0}"},
        {"metric": "first-breakout-desired trades", "value": f"{len(fb)} | win {round((fb.pnl_points>0).mean()*100,1) if len(fb) else 0}% | total {round(fb.pnl_points.sum(),1)} | avg {round(fb.pnl_points.mean(),2) if len(fb) else 0}"},
        {"metric": "undesired-first-then-reversal trades", "value": f"{len(rv)} | win {round((rv.pnl_points>0).mean()*100,1) if len(rv) else 0}% | total {round(rv.pnl_points.sum(),1)} | avg {round(rv.pnl_points.mean(),2) if len(rv) else 0}"},
        {"metric": "No-trade reasons", "value": str(nt_reasons)}, {"metric": "Flag rows", "value": len(flags)},
    ])
    with pd.ExcelWriter(OUTDIR / "nifty_orb_prevday_bias_reversal.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False); BY.to_excel(w, sheet_name="By_Group", index=False)
        (T if len(T) else pd.DataFrame([{"note": "no trades"}])).to_excel(w, sheet_name="Trades", index=False)
        (NT if len(NT) else pd.DataFrame([{"note": "none"}])).to_excel(w, sheet_name="No_Trade_Days", index=False)
        (pd.DataFrame(flags) if flags else pd.DataFrame([{"flag": "none"}])).to_excel(w, sheet_name="Flags", index=False)
    if len(T): T.to_csv(OUTDIR / "nifty_orb_prevday_bias_reversal_trades.csv", index=False)

    pd.set_option("display.width", 230)
    print("=" * 96 + "\nNIFTY ORB + PREV-DAY BIAS — REVERSAL-ENTRY variant (touch-basis, time-only exit)\n" + "=" * 96)
    print(f"days {total_days} | trades {len(T)} | no-trade {len(NT)} ({round(len(NT)/total_days*100,1)}%)")
    print(f"win {win}% | TOTAL {tot:,} pts | avg {round(T.pnl_points.mean(),2) if len(T) else 0} | median {round(T.pnl_points.median(),2) if len(T) else 0}")
    print("\n--- BY GROUP ---"); print(BY.to_string(index=False))
    print(f"\nno-trade reasons: {nt_reasons} | flags {len(flags)}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
