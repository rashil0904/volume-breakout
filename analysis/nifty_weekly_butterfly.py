# -*- coding: utf-8 -*-
"""nifty_weekly_butterfly.py — NIFTY Weekly 2-Day Breakout Butterfly. Weekly cycle tied to REAL expiries.

Window: first trading day after prev expiry -> current expiry; open flies squared off 15:15 on expiry day.
Rolling prev-2-day high/low recalculated EVERY day. PUT fly on breach BELOW prev-2-day low (any day):
+1 ATM PE, -2 (ATM-200) PE, +1 (ATM-400) PE. CALL fly on breach ABOVE prev-2-day high: +1 ATM CE, -2
(ATM+200) CE, +1 (ATM+400) CE. Max 1 PUT + 1 CALL / week (independent, can coexist). Day-1-only 2:30pm
fallback if no breach: red->PUT, green->CALL. Exit: 90% of own max profit (per fly, 1-min) else 15:15
expiry square-off. No stop. max_profit = 200 - net_debit (LTP=1-min close; no bid/ask in data). GROSS points.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "weekly_butterfly"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIDTH = 200; TARGET_FRAC = 0.90; FB_MOD = 14 * 60 + 30; SQUARE_MOD = 15 * 60 + 15   # 90% target; 14:30 fallback; 15:15 square-off


def drawdown_episodes(equity, times):
    peak = np.maximum.accumulate(equity); dd = equity - peak; eps = []; in_dd = False; s = t = 0; tv = 0.0
    for k in range(len(equity)):
        if not in_dd:
            if dd[k] < -1e-9: in_dd = True; s = k - 1; t = k; tv = dd[k]
        else:
            if dd[k] < tv: tv = dd[k]; t = k
            if dd[k] >= -1e-9: eps.append((s, t, k, tv)); in_dd = False
    if in_dd: eps.append((s, t, len(equity) - 1, tv))
    rows = []
    for (s, t, r, v) in eps:
        rec = (r != len(equity) - 1) or (equity[r] >= peak[s] - 1e-9)
        rows.append({"peak_date": pd.Timestamp(times[s]).date(), "trough_date": pd.Timestamp(times[t]).date(),
                     "recovery_date": (pd.Timestamp(times[r]).date() if rec else "NOT RECOVERED"), "drawdown_points": round(abs(v), 1),
                     "days_peak_to_trough": int((pd.Timestamp(times[t]) - pd.Timestamp(times[s])).days),
                     "days_peak_to_recovery": int((pd.Timestamp(times[r]) - pd.Timestamp(times[s])).days)})
    return pd.DataFrame(rows)


def leg_series(folder, strike, otype, t0, t1):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{otype}_*.parquet"))
    if not fs:
        return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts")["close"].sort_index()
    return o if len(o) else None


def process_fly(folder, atm, direction, entry_time, e_cur):
    ot = "PE" if direction == "PUT" else "CE"; sgn = -1 if direction == "PUT" else 1
    Kw1, Kb, Kw2 = atm, atm + sgn * WIDTH, atm + sgn * 2 * WIDTH          # wing1(ATM), body(+-200), wing2(+-400)
    t1 = e_cur + pd.Timedelta(hours=15, minutes=16)
    w1 = leg_series(folder, Kw1, ot, entry_time, t1); bd = leg_series(folder, Kb, ot, entry_time, t1); w2 = leg_series(folder, Kw2, ot, entry_time, t1)
    if w1 is None or bd is None or w2 is None:
        return None, f"missing leg {Kw1}/{Kb}/{Kw2} {ot}"
    try:
        e1, eb, e2 = w1.asof(entry_time), bd.asof(entry_time), w2.asof(entry_time)
    except Exception:
        return None, "no entry premium"
    if any(np.isnan(v) for v in (e1, eb, e2)):
        return None, "nan entry premium"
    net_debit = float(e1 + e2 - 2 * eb)
    if net_debit <= 0 or net_debit >= WIDTH:
        return None, f"bad net_debit {round(net_debit,1)}"
    max_profit = WIDTH - net_debit
    both = pd.concat([w1.rename("w1"), bd.rename("bd"), w2.rename("w2")], axis=1).ffill().dropna()
    both = both[both.index > entry_time]
    fly_val = (both["w1"] + both["w2"] - 2 * both["bd"]).values; bt = both.index.values
    live = fly_val - net_debit; thr = TARGET_FRAC * max_profit
    hit = np.where(live >= thr)[0]
    if len(hit):
        j = hit[0]; exit_time = pd.Timestamp(bt[j]); exit_val = float(fly_val[j]); reason = "90% target"
    else:
        sq = both[both.index <= e_cur + pd.Timedelta(hours=15, minutes=15)]
        if sq.empty:
            return None, "no square-off price"
        exit_time = sq.index[-1]; exit_val = float(sq["w1"].iloc[-1] + sq["w2"].iloc[-1] - 2 * sq["bd"].iloc[-1]); reason = "expiry 15:15"
    return {"Kw1": int(Kw1), "Kb": int(Kb), "Kw2": int(Kw2), "leg_type": ot, "entry_w1": round(float(e1), 2), "entry_body": round(float(eb), 2), "entry_w2": round(float(e2), 2),
            "net_debit": round(net_debit, 2), "max_profit": round(max_profit, 2), "exit_time": exit_time, "exit_value": round(exit_val, 2),
            "exit_reason": reason, "pnl_points": round(exit_val - net_debit, 2)}, None


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    daily = sp.groupby("date").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"))
    daily["p2h"] = daily["h"].shift(1).rolling(2).max(); daily["p2l"] = daily["l"].shift(1).rolling(2).min()
    tdays = list(daily.index)
    expiries = pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")

    trades = []; weeks = []; skipped = []
    for i in range(1, len(expiries)):
        e_prev, e_cur = expiries[i - 1], expiries[i]
        win_days = [d for d in tdays if e_prev < d <= e_cur]
        if not win_days or win_days[0] > spot_end or e_cur > spot_end:
            continue
        folder = e_cur.strftime("%Y%m%d")
        put_e = None; call_e = None; fb_done = False
        for di, D in enumerate(win_days):
            p2h = daily.loc[D, "p2h"]; p2l = daily.loc[D, "p2l"]
            if np.isnan(p2h):
                continue
            ed = sp[sp["date"] == D]; hi = ed["high"].values; lo = ed["low"].values; cl = ed["close"].values; md = ed["mod"].values; ets = ed["ts"].values
            d_open = ed["open"].iloc[0]
            for k in range(len(ed)):
                if di == 0 and md[k] >= FB_MOD and not fb_done and put_e is None and call_e is None:    # day-1 2:30 fallback
                    if cl[k] < d_open: put_e = (pd.Timestamp(ets[k]), cl[k], "fallback-2:30")
                    else: call_e = (pd.Timestamp(ets[k]), cl[k], "fallback-2:30")
                    fb_done = True
                if put_e is None and lo[k] <= p2l: put_e = (pd.Timestamp(ets[k]), cl[k], "2d-low breach")     # entry spot = breach candle CLOSE
                if call_e is None and hi[k] >= p2h: call_e = (pd.Timestamp(ets[k]), cl[k], "2d-high breach")
            if put_e is not None and call_e is not None:
                break
        nfly = 0
        for direction, ent in [("PUT", put_e), ("CALL", call_e)]:
            if ent is None:
                continue
            et, espot, trig = ent; atm = round(espot / 50) * 50
            fly, err = process_fly(folder, atm, direction, et, e_cur)
            if fly is None:
                skipped.append((e_cur.date(), direction, err)); continue
            nfly += 1
            trades.append({"entry_date": win_days[0].date(), "expiry": e_cur.date(), "direction": direction, "trigger": trig,
                           "entry_time": et.strftime("%Y-%m-%d %H:%M"), "entry_spot": round(espot, 1), "ATM": int(atm), **fly})
        weeks.append({"expiry": e_cur.date(), "n_butterflies": nfly, "put": int(put_e is not None), "call": int(call_e is not None)})

    T = pd.DataFrame(trades)
    for c in ("exit_time",):
        T[c] = pd.to_datetime(T[c]).dt.strftime("%Y-%m-%d %H:%M")
    T["hrs_to_exit"] = (pd.to_datetime(T["exit_time"]) - pd.to_datetime(T["entry_time"])).dt.total_seconds() / 3600
    W = pd.DataFrame(weeks)
    tgt = T[T.exit_reason == "90% target"]; exp = T[T.exit_reason == "expiry 15:15"]
    win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)
    T["month"] = pd.to_datetime(T["entry_date"]).dt.to_period("M").astype(str)
    MON = T.groupby("month").apply(lambda g: pd.Series({
        "trades": len(g), "wins": int((g.pnl_points > 0).sum()), "win_%": round((g.pnl_points > 0).mean() * 100, 1),
        "total_pnl": round(g.pnl_points.sum(), 1), "avg_pnl": round(g.pnl_points.mean(), 2),
        "PUT": int((g.direction == "PUT").sum()), "CALL": int((g.direction == "CALL").sum()),
        "target_exits": int((g.exit_reason == "90% target").sum()), "expiry_exits": int((g.exit_reason == "expiry 15:15").sum()),
        "best": round(g.pnl_points.max(), 1), "worst": round(g.pnl_points.min(), 1)}), include_groups=False).reset_index()
    MON["cum_pnl"] = MON["total_pnl"].cumsum().round(1)

    Tc = T.sort_values("exit_time").reset_index(drop=True); cum = Tc["pnl_points"].cumsum().values
    equity = np.concatenate([[0.0], cum]); dtimes = np.concatenate([[pd.to_datetime(Tc.entry_time.iloc[0])], pd.to_datetime(Tc.exit_time).values])
    DE = drawdown_episodes(equity, dtimes); maxdd = DE.drawdown_points.max() if len(DE) else 0.0; avgdd = round(DE.drawdown_points.mean(), 1) if len(DE) else 0.0

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY Weekly 2-Day Breakout Butterfly (real-expiry weekly cycle); GROSS points, 1 lot/fly"},
        {"metric": "Window", "value": f"{T.entry_date.min()} .. {T.expiry.max()} (spot ends 2026-07-15)"},
        {"metric": "Max-profit", "value": "200 - net_debit (LTP = 1-min close; no bid/ask in data). 90% target = 90% of that."},
        {"metric": "Exit", "value": "90% of own max profit (1-min) else 15:15 expiry-day square-off at market. No stop."},
        {"metric": "Total butterflies", "value": len(T)}, {"metric": "Win rate %", "value": win},
        {"metric": "Total P&L (points)", "value": tot}, {"metric": "Avg P&L / fly", "value": round(T.pnl_points.mean(), 2)},
        {"metric": "PUT / CALL flies", "value": f"{int((T.direction=='PUT').sum())} / {int((T.direction=='CALL').sum())}"},
        {"metric": "Avg net debit / max profit", "value": f"{round(T.net_debit.mean(),1)} / {round(T.max_profit.mean(),1)}"},
        {"metric": "", "value": ""},
        {"metric": "-- exit path comparison --", "value": ""},
        {"metric": "90%-target flies: count / win% / avg P&L", "value": f"{len(tgt)} / {round((tgt.pnl_points>0).mean()*100,1) if len(tgt) else 0} / {round(tgt.pnl_points.mean(),2) if len(tgt) else 0}"},
        {"metric": "90%-target flies: avg time-to-target (hrs / days)", "value": f"{round(tgt.hrs_to_exit.mean(),1) if len(tgt) else 0} / {round(tgt.hrs_to_exit.mean()/24,2) if len(tgt) else 0}"},
        {"metric": "Expiry-15:15 flies: count / win% / avg P&L", "value": f"{len(exp)} / {round((exp.pnl_points>0).mean()*100,1) if len(exp) else 0} / {round(exp.pnl_points.mean(),2) if len(exp) else 0}"},
        {"metric": "", "value": ""},
        {"metric": "Weeks 0 / 1 / 2 flies", "value": f"{int((W.n_butterflies==0).sum())} / {int((W.n_butterflies==1).sum())} / {int((W.n_butterflies==2).sum())}"},
        {"metric": "Trigger mix", "value": T.trigger.value_counts(normalize=True).mul(100).round(1).to_dict()},
        {"metric": "", "value": ""},
        {"metric": "--- DRAWDOWN (realized cumulative P&L, points) ---", "value": ""},
        {"metric": "Number of drawdown episodes", "value": len(DE)},
        {"metric": "Max drawdown (points)", "value": round(maxdd, 1)},
        {"metric": "Avg drawdown (points)", "value": avgdd},
        {"metric": "Max DD duration (days, peak->recovery)", "value": int(DE.days_peak_to_recovery.max()) if len(DE) else 0},
        {"metric": "Return / Max-DD", "value": round(tot / maxdd, 2) if maxdd else "-"},
        {"metric": "Skipped (missing/bad legs)", "value": len(skipped)},
    ])
    with pd.ExcelWriter(OUTDIR / "weekly_butterfly.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        MON.to_excel(w, sheet_name="Monthly_PnL", index=False)
        DE.sort_values("drawdown_points", ascending=False).to_excel(w, sheet_name="Drawdowns", index=False)
        T.to_excel(w, sheet_name="Butterflies", index=False)
        W.to_excel(w, sheet_name="Weekly", index=False)
        if skipped: pd.DataFrame(skipped, columns=["expiry", "direction", "reason"]).to_excel(w, sheet_name="Skipped", index=False)
    T.to_csv(OUTDIR / "weekly_butterfly_trades.csv", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96 + "\nNIFTY WEEKLY 2-DAY BREAKOUT BUTTERFLY\n" + "=" * 96)
    print(f"window {T.entry_date.min()} .. {T.expiry.max()} | butterflies {len(T)} | skipped {len(skipped)}")
    print(f"win {win}% | TOTAL {tot:,} pts | avg {round(T.pnl_points.mean(),2)} | PUT/CALL {int((T.direction=='PUT').sum())}/{int((T.direction=='CALL').sum())} | avg debit {round(T.net_debit.mean(),1)} maxprofit {round(T.max_profit.mean(),1)}")
    print(f"weeks 0/1/2 flies: {int((W.n_butterflies==0).sum())}/{int((W.n_butterflies==1).sum())}/{int((W.n_butterflies==2).sum())} | trigger {T.trigger.value_counts(normalize=True).mul(100).round(1).to_dict()}")
    print(f"\n90%-target: {len(tgt)} flies | win {round((tgt.pnl_points>0).mean()*100,1) if len(tgt) else 0}% | avg P&L {round(tgt.pnl_points.mean(),2) if len(tgt) else 0} | avg time {round(tgt.hrs_to_exit.mean()/24,2) if len(tgt) else 0} days")
    print(f"expiry-15:15: {len(exp)} flies | win {round((exp.pnl_points>0).mean()*100,1) if len(exp) else 0}% | avg P&L {round(exp.pnl_points.mean(),2) if len(exp) else 0}")
    print("\nfirst 6 butterflies:")
    print(T.head(6)[["entry_date", "direction", "trigger", "ATM", "net_debit", "max_profit", "exit_reason", "exit_value", "pnl_points"]].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
