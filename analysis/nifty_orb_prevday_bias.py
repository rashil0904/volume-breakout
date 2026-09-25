# -*- coding: utf-8 -*-
"""nifty_orb_prevday_bias.py — NIFTY Opening-Range Breakout with previous-day bias.
Prev-day bias: prev trading day CLOSE vs OPEN (bullish close>open / bearish close<open). OR = 9:15-9:29 hi/lo.
From 9:30, TOUCH-basis breakout (confirmed): bullish+up-break -> LONG buy ATM CE ; bearish+down-break -> buy
ATM PE. Disallowed direction breaking FIRST -> NO TRADE that day (even if it later reverses). Max 1 trade/day.
ATM = round(breakout-spot/50)*50. Entry = breakout-minute option CLOSE; EXIT = 15:15 CLOSE (time-only, no
SL/target, confirmed). Front expiry >= day (intraday, DTE-0 ok). GROSS option premium points, 1 lot.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "nifty_orb_prevday_bias"; OUTDIR.mkdir(parents=True, exist_ok=True)
OR_START, OR_END = 9 * 60 + 15, 9 * 60 + 29        # opening range 09:15-09:29 (first 15 one-min candles)
SCAN_START, EXIT_MOD = 9 * 60 + 30, 15 * 60 + 15   # breakout watched from 09:30; exit 15:15
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
        ed = sp[sp["date"] == D]
        orb = ed[(ed["mod"] >= OR_START) & (ed["mod"] <= OR_END)]
        if bias == "neutral" or len(orb) < 10:
            notrades.append({"date": D.date(), "prev_bias": bias, "first_break": "n/a", "reason": "neutral bias" if bias == "neutral" else "incomplete OR"}); continue
        ORH = orb["high"].max(); ORL = orb["low"].min()
        scan = ed[(ed["mod"] >= SCAN_START) & (ed["mod"] <= EXIT_MOD)].sort_values("mod")
        first_dir = None; b_row = None
        for _, r in scan.iterrows():
            up = r["high"] >= ORH; dn = r["low"] <= ORL
            if up or dn:
                if up and dn:                              # outside bar -> proxy by candle direction, flag
                    first_dir = "up" if r["close"] >= r["open"] else "down"
                    flags.append({"date": D.date(), "flag": "outside_bar_first_break", "detail": f"resolved {first_dir} by candle O->C"})
                else:
                    first_dir = "up" if up else "down"
                b_row = r; break
        if first_dir is None:
            notrades.append({"date": D.date(), "prev_bias": bias, "first_break": "none (stayed in range)", "reason": "no breakout"}); continue
        allowed = (bias == "bullish" and first_dir == "up") or (bias == "bearish" and first_dir == "down")
        if not allowed:
            notrades.append({"date": D.date(), "prev_bias": bias, "first_break": first_dir, "reason": "disallowed direction broke first"}); continue

        bspot = float(b_row["close"]); atm = int(round(bspot / 50) * 50); ot = "CE" if first_dir == "up" else "PE"
        t_en = pd.Timestamp(b_row["ts"]); t_ex = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=15)
        j = bisect.bisect_left(exp_dates, pd.Timestamp(D))
        if j >= len(exp_dates): continue
        folder = expiries[j].strftime("%Y%m%d")
        en = opt_px(folder, atm, ot, D, t_en); ex = opt_px(folder, atm, ot, D, t_ex)
        if np.isnan(en) or np.isnan(ex):
            flags.append({"date": D.date(), "flag": "missing_option_price", "detail": f"{atm}{ot}"});
            notrades.append({"date": D.date(), "prev_bias": bias, "first_break": first_dir, "reason": "missing option price"}); continue
        pnl = ex - en
        trades.append({"date": D.date(), "prev_bias": bias, "OR_high": round(ORH, 1), "OR_low": round(ORL, 1),
                       "breakout_dir": first_dir, "direction": "LONG" if ot == "CE" else "SHORT-dir",
                       "entry_time": t_en.strftime("%H:%M"), "ATM": atm, "option_type": ot,
                       "entry_premium": round(en, 2), "exit_time": "15:15", "exit_premium": round(ex, 2),
                       "pnl_points": round(pnl, 2), "expiry_used": expiries[j].date()})

    T = pd.DataFrame(trades); NT = pd.DataFrame(notrades)
    total_days = len(T) + len(NT)
    win = round((T.pnl_points > 0).mean() * 100, 1) if len(T) else 0; tot = round(T.pnl_points.sum(), 1)
    lg = T[T.option_type == "CE"]; sh = T[T.option_type == "PE"]

    def blk(df, lbl):
        return {"group": lbl, "trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_points.sum(), 1), "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0,
                "median_pnl": round(df.pnl_points.median(), 2) if len(df) else 0,
                "max": round(df.pnl_points.max(), 1) if len(df) else 0, "min": round(df.pnl_points.min(), 1) if len(df) else 0}
    BY = pd.DataFrame([blk(T, "ALL"), blk(lg, "LONG (CE)"), blk(sh, "SHORT-dir (PE)")])
    nt_reasons = NT.reason.value_counts().to_dict() if len(NT) else {}

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY ORB + prev-day bias. Touch-basis breakout; naked long ATM option; time-only 15:15 exit (no SL/target). GROSS premium pts, 1 lot."},
        {"metric": "Rules", "value": "prev-day bull(C>O)+up-break->CE ; bear(C<O)+down-break->PE ; disallowed dir first->skip ; OR 09:15-09:29 ; 1 trade/day"},
        {"metric": "Window", "value": f"{min([*(T.date if len(T) else []), *(NT.date if len(NT) else [])]) if total_days else '-'} .. {spot_end.date()}"},
        {"metric": "Total days evaluated", "value": total_days},
        {"metric": "Total trades", "value": len(T)},
        {"metric": "No-trade days", "value": len(NT)},
        {"metric": "% days NO trade", "value": round(len(NT) / total_days * 100, 1) if total_days else 0},
        {"metric": "Win rate %", "value": win},
        {"metric": "Total P&L (points)", "value": tot},
        {"metric": "Avg P&L / trade", "value": round(T.pnl_points.mean(), 2) if len(T) else 0},
        {"metric": "Median P&L / trade", "value": round(T.pnl_points.median(), 2) if len(T) else 0},
        {"metric": "LONG (CE) trades / total / avg", "value": f"{len(lg)} / {round(lg.pnl_points.sum(),1)} / {round(lg.pnl_points.mean(),2) if len(lg) else 0}"},
        {"metric": "SHORT-dir (PE) trades / total / avg", "value": f"{len(sh)} / {round(sh.pnl_points.sum(),1)} / {round(sh.pnl_points.mean(),2) if len(sh) else 0}"},
        {"metric": "No-trade reasons", "value": str(nt_reasons)},
        {"metric": "Flag rows", "value": len(flags)},
    ])
    with pd.ExcelWriter(OUTDIR / "nifty_orb_prevday_bias.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False); BY.to_excel(w, sheet_name="By_Direction", index=False)
        (T if len(T) else pd.DataFrame([{"note": "no trades"}])).to_excel(w, sheet_name="Trades", index=False)
        (NT if len(NT) else pd.DataFrame([{"note": "none"}])).to_excel(w, sheet_name="No_Trade_Days", index=False)
        (pd.DataFrame(flags) if flags else pd.DataFrame([{"flag": "none"}])).to_excel(w, sheet_name="Flags", index=False)
    if len(T): T.to_csv(OUTDIR / "nifty_orb_prevday_bias_trades.csv", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96 + "\nNIFTY ORB + PREVIOUS-DAY BIAS (touch-basis, time-only 15:15 exit)\n" + "=" * 96)
    print(f"days {total_days} | trades {len(T)} | no-trade {len(NT)} ({round(len(NT)/total_days*100,1)}%)")
    print(f"win {win}% | TOTAL {tot:,} pts | avg {round(T.pnl_points.mean(),2) if len(T) else 0} | median {round(T.pnl_points.median(),2) if len(T) else 0}")
    print("\n--- BY DIRECTION ---"); print(BY.to_string(index=False))
    print(f"\nno-trade reasons: {nt_reasons}")
    print(f"flags: {len(flags)}")
    print("\nfirst 8 trades:")
    if len(T): print(T.head(8)[["date", "prev_bias", "breakout_dir", "entry_time", "ATM", "option_type", "entry_premium", "exit_premium", "pnl_points"]].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
