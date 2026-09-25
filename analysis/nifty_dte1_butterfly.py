# -*- coding: utf-8 -*-
"""nifty_dte1_butterfly.py — DTE-1 SINGLE-ENTRY variant of the NIFTY Weekly 2-Day Breakout Butterfly.
NEW standalone variant (base full-week dual-fly version untouched). Own output folder.

Entry ONLY on DTE-1 (the trading day immediately before expiry). Prev-2-day high/low = the 2 completed
days before DTE-1. First breach that day wins ONE butterfly (low->PUT, high->CALL); opposite ignored.
2:30pm fallback if no breach: red->PUT, green->CALL. NARROW 100-pt wings: PUT = +1 ATM PE / -2 (ATM-100)
PE / +1 (ATM-200) PE ; CALL = +1 ATM CE / -2 (ATM+100) CE / +1 (ATM+200) CE. ATM=nearest 50 to entry spot
(breach candle close). Exit: 90% of max profit (=100-net_debit; 1-min) else 15:15 expiry square-off. No stop.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "dte1_single_butterfly_nifty"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIDTH = 100; TARGET_FRAC = 0.90; FB_MOD = 14 * 60 + 30                 # 100-pt wings ; 90% target ; 14:30 fallback


def leg_series(folder, strike, otype, t0, t1):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{otype}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts")["close"].sort_index()
    return o if len(o) else None


def process_fly(folder, atm, direction, entry_time, e_cur):
    ot = "PE" if direction == "PUT" else "CE"; sgn = -1 if direction == "PUT" else 1
    Kw1, Kb, Kw2 = atm, atm + sgn * WIDTH, atm + sgn * 2 * WIDTH; t1 = e_cur + pd.Timedelta(hours=15, minutes=16)
    w1 = leg_series(folder, Kw1, ot, entry_time, t1); bd = leg_series(folder, Kb, ot, entry_time, t1); w2 = leg_series(folder, Kw2, ot, entry_time, t1)
    if w1 is None or bd is None or w2 is None: return None, f"missing leg {Kw1}/{Kb}/{Kw2} {ot}"
    e1, eb, e2 = w1.asof(entry_time), bd.asof(entry_time), w2.asof(entry_time)
    if any(np.isnan(v) for v in (e1, eb, e2)): return None, "nan entry premium"
    net_debit = float(e1 + e2 - 2 * eb)
    if net_debit <= 0 or net_debit >= WIDTH: return None, f"bad net_debit {round(net_debit,1)}"
    max_profit = WIDTH - net_debit
    both = pd.concat([w1.rename("w1"), bd.rename("bd"), w2.rename("w2")], axis=1).ffill().dropna(); both = both[both.index > entry_time]
    if both.empty: return None, "no series"
    fv = (both["w1"] + both["w2"] - 2 * both["bd"]).values; bt = both.index.values
    hit = np.where(fv - net_debit >= TARGET_FRAC * max_profit)[0]
    if len(hit):
        j = hit[0]; exit_time = pd.Timestamp(bt[j]); exit_val = float(fv[j]); reason = "90% target"
    else:
        sq = both[both.index <= e_cur + pd.Timedelta(hours=15, minutes=15)]
        if sq.empty: return None, "no square-off"
        exit_time = sq.index[-1]; exit_val = float(sq["w1"].iloc[-1] + sq["w2"].iloc[-1] - 2 * sq["bd"].iloc[-1]); reason = "expiry 15:15"
    return {"Kw1": int(Kw1), "Kb": int(Kb), "Kw2": int(Kw2), "leg_type": ot, "entry_w1": round(float(e1), 2), "entry_body": round(float(eb), 2), "entry_w2": round(float(e2), 2),
            "net_debit": round(net_debit, 2), "max_profit": round(max_profit, 2), "exit_time": exit_time.strftime("%Y-%m-%d %H:%M"),
            "exit_value": round(exit_val, 2), "exit_reason": reason, "pnl_points": round(exit_val - net_debit, 2)}, None


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    daily = sp.groupby("date").agg(h=("high", "max"), l=("low", "min")); tdays = list(daily.index)
    expiries = pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")

    trades = []; skipped = 0; noentry = 0
    for e_cur in expiries:
        before = [d for d in tdays if d < e_cur]
        if len(before) < 3: continue
        dte1 = before[-1]; prev2 = before[-3:-1]                       # DTE-1 day + the 2 completed days before it
        if dte1 > spot_end or e_cur > spot_end: continue
        p2h = daily.loc[prev2, "h"].max(); p2l = daily.loc[prev2, "l"].min()
        ed = sp[sp["date"] == dte1]; hi = ed["high"].values; lo = ed["low"].values; cl = ed["close"].values; md = ed["mod"].values; ets = ed["ts"].values; d_open = ed["open"].iloc[0]
        entry = None
        for k in range(len(ed)):
            if md[k] >= FB_MOD:                                         # 14:30 reached with no breach -> fallback (red/green)
                entry = (pd.Timestamp(ets[k]), cl[k], "PUT" if cl[k] < d_open else "CALL", "fallback-2:30"); break
            bl = lo[k] <= p2l; bh = hi[k] >= p2h                        # first breach wins (one trade)
            if bl and bh:
                entry = (pd.Timestamp(ets[k]), cl[k], "PUT" if abs(d_open - p2l) <= abs(d_open - p2h) else "CALL", "2d-low breach" if abs(d_open - p2l) <= abs(d_open - p2h) else "2d-high breach"); break
            if bl: entry = (pd.Timestamp(ets[k]), cl[k], "PUT", "2d-low breach"); break
            if bh: entry = (pd.Timestamp(ets[k]), cl[k], "CALL", "2d-high breach"); break
        if entry is None:
            noentry += 1; continue
        et, espot, direction, trig = entry; atm = round(espot / 50) * 50
        fly, err = process_fly(e_cur.strftime("%Y%m%d"), atm, direction, et, e_cur)
        if fly is None:
            skipped += 1; continue
        trades.append({"expiry": e_cur.date(), "dte1_entry_date": dte1.date(), "direction": direction, "trigger": trig,
                       "entry_time": et.strftime("%Y-%m-%d %H:%M"), "entry_spot": round(espot, 1), "ATM": int(atm), **fly})

    T = pd.DataFrame(trades)
    T["hrs_held"] = (pd.to_datetime(T["exit_time"]) - pd.to_datetime(T["entry_time"])).dt.total_seconds() / 3600
    win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)
    tgt = T[T.exit_reason == "90% target"]; exp = T[T.exit_reason == "expiry 15:15"]
    trig_mix = T.trigger.value_counts(normalize=True).mul(100).round(1).to_dict()

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY DTE-1 SINGLE-ENTRY Butterfly (100-pt wings); NEW variant, own folder. GROSS points, 1 fly/wk max"},
        {"metric": "Window", "value": f"{T.dte1_entry_date.min()} .. {T.expiry.max()} (spot ends 2026-07-15)"},
        {"metric": "Max-profit", "value": "100 - net_debit (LTP = 1-min close). Target = 90% of that."},
        {"metric": "Total trades (<=1/week)", "value": len(T)}, {"metric": "Win rate %", "value": win},
        {"metric": "Total P&L (points)", "value": tot}, {"metric": "Avg P&L / trade", "value": round(T.pnl_points.mean(), 2)},
        {"metric": "PUT / CALL", "value": f"{int((T.direction=='PUT').sum())} / {int((T.direction=='CALL').sum())}"},
        {"metric": "Avg net debit / max profit", "value": f"{round(T.net_debit.mean(),1)} / {round(T.max_profit.mean(),1)}"},
        {"metric": "Trigger mix % (PE-low / CE-high / fallback)", "value": trig_mix},
        {"metric": "Exit: target / expiry-15:15 count", "value": f"{len(tgt)} / {len(exp)}"},
        {"metric": "Target flies: win% / avg P&L / avg hrs", "value": f"{round((tgt.pnl_points>0).mean()*100,1) if len(tgt) else 0} / {round(tgt.pnl_points.mean(),2) if len(tgt) else 0} / {round(tgt.hrs_held.mean(),1) if len(tgt) else 0}"},
        {"metric": "Expiry flies: win% / avg P&L", "value": f"{round((exp.pnl_points>0).mean()*100,1) if len(exp) else 0} / {round(exp.pnl_points.mean(),2) if len(exp) else 0}"},
        {"metric": "Weeks with no entry / skipped legs", "value": f"{noentry} / {skipped}"},
        {"metric": "", "value": ""},
        {"metric": "-- vs BASE (full-week, dual-fly, 200-wide) --", "value": "base: 152 flies, +464.2 pts, 31.6% win (2 flies/wk, all days)"},
        {"metric": "This variant vs base", "value": f"{len(T)} flies ({round(len(T)/93,2)}/wk), {tot} pts, {win}% win — DTE-1-only + single-entry + 100-wide"},
    ])
    with pd.ExcelWriter(OUTDIR / "dte1_single_butterfly.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False); T.to_excel(w, sheet_name="Trades", index=False)
    T.to_csv(OUTDIR / "dte1_single_butterfly_trades.csv", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96 + "\nNIFTY DTE-1 SINGLE-ENTRY BUTTERFLY (100-pt wings)\n" + "=" * 96)
    print(f"window {T.dte1_entry_date.min()} .. {T.expiry.max()} | trades {len(T)} | no-entry weeks {noentry} | skipped {skipped}")
    print(f"win {win}% | TOTAL {tot:,} pts | avg {round(T.pnl_points.mean(),2)} | PUT/CALL {int((T.direction=='PUT').sum())}/{int((T.direction=='CALL').sum())} | avg debit {round(T.net_debit.mean(),1)} maxprofit {round(T.max_profit.mean(),1)}")
    print(f"trigger mix: {trig_mix} | exit target/expiry: {len(tgt)}/{len(exp)}")
    print(f"target flies: win {round((tgt.pnl_points>0).mean()*100,1) if len(tgt) else 0}% avg {round(tgt.pnl_points.mean(),2) if len(tgt) else 0} | expiry flies: win {round((exp.pnl_points>0).mean()*100,1) if len(exp) else 0}% avg {round(exp.pnl_points.mean(),2) if len(exp) else 0}")
    print("\nfirst 8 trades:")
    print(T.head(8)[["dte1_entry_date", "direction", "trigger", "ATM", "net_debit", "max_profit", "exit_reason", "exit_value", "pnl_points"]].to_string(index=False))
    print(f"\nvs BASE (dual 200-wide): base 152 flies +464.2 (31.6% win) | this {len(T)} flies {tot} ({win}% win)")
    print(f"Saved -> {OUTDIR}")


if __name__ == "__main__":
    main()
