# -*- coding: utf-8 -*-
"""sensex_weekly_credit_spread.py — SENSEX Weekly Credit Spread. Identical logic to the NIFTY version
(nifty_weekly_credit_spread.py) except the hedge-leg WIDTH = 600 (was 200) and SENSEX data/strike-step(100)/
actual expiry calendar. Single backtest — NO sweeps/SL/width variants.

ENTRY: first trading day after previous weekly expiry (expiry calendar = SENSEX option folders; weekday
derived, NOT assumed — currently Thursday, was Fri->Tue@2025-01-07->Thu@2025-09-04). DTE = calendar days
entry->current expiry. LEVELS: 3-day high/low of the 3 completed trading days before entry. BREAKOUT (entry
day, intraday TOUCH): low<=3dL -> Call Credit Spread (sell ATM CE, buy CE+600); high>=3dH -> Put Credit Spread
(sell ATM PE, buy PE-600). First touch wins; 1 trade/week. FALLBACK (neither by 14:30): red=CCS, green=PCS at
14:30. ATM=round(spot/100)*100. EXIT: (1) 90% target every 1-min when cost-to-close (short-long) <= 10% of
net credit; else (2) DTE-0 SETTLEMENT = spread intrinsic at spot settlement (avg last 30-min spot on expiry).
No stop. GROSS premium points (1 spread).
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "sensex_1min_ohlc.csv"
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "SENSEX"
OUTDIR = rb.RESULTS / "sensex_weekly_credit_spread"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIDTH = 600; TARGET_FRAC = 0.10; FB_MOD = 14 * 60 + 30; STEP = 100


def leg_series(folder, strike, otype, t0, t1):
    fs = glob.glob(str(OPTDIR / folder / f"SENSEX_{int(strike)}_{otype}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts")["close"].sort_index()
    return o if len(o) else None


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    daily = sp.groupby("date").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    settle = sp[(sp["mod"] >= 900) & (sp["mod"] <= 930)].groupby("date")["close"].mean()      # last-30-min settlement
    tdays = list(daily.index)
    expiries = pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")

    trades = []; skipped = []
    for i in range(1, len(expiries)):
        e_prev, e_cur = expiries[i - 1], expiries[i]
        after = [d for d in tdays if d > e_prev]
        if not after: continue
        entry_day = after[0]
        if entry_day > spot_end or e_cur > spot_end: skipped.append((e_cur.date(), "outside spot window")); continue
        dte = (e_cur - entry_day).days
        prev3 = [d for d in tdays if d < entry_day][-3:]
        if len(prev3) < 3: continue
        d3h = daily.loc[prev3, "h"].max(); d3l = daily.loc[prev3, "l"].min()

        ed = sp[sp["date"] == entry_day]
        hi = ed["high"].values; lo = ed["low"].values; cl = ed["close"].values; md = ed["mod"].values; ets = ed["ts"].values
        d_open = ed["open"].iloc[0]; trig_i = None; typ = None; label = None
        for k in range(len(ed)):
            if md[k] > FB_MOD: break
            bh = hi[k] >= d3h; bl = lo[k] <= d3l
            if bh or bl:
                if bh and bl: typ, label = ("PCS", "3d-high") if abs(d_open - d3h) <= abs(d_open - d3l) else ("CCS", "3d-low")
                elif bh: typ, label = "PCS", "3d-high"
                else: typ, label = "CCS", "3d-low"
                trig_i = k; break
        if trig_i is None:
            fb = ed[ed["mod"] >= FB_MOD]
            if fb.empty: continue
            trig_i = ed.index.get_loc(fb.index[0]); px = cl[trig_i]
            typ = "CCS" if px < d_open else "PCS"; label = "fallback-2:30"
        entry_time = pd.Timestamp(ets[trig_i]); entry_spot = cl[trig_i]; atm = round(entry_spot / STEP) * STEP

        folder = e_cur.strftime("%Y%m%d")
        if typ == "CCS": sK, lK, ot = atm, atm + WIDTH, "CE"
        else: sK, lK, ot = atm, atm - WIDTH, "PE"
        t_end = e_cur + pd.Timedelta(hours=15, minutes=30)
        sser = leg_series(folder, sK, ot, entry_time, t_end); lser = leg_series(folder, lK, ot, entry_time, t_end)
        if sser is None or lser is None:
            skipped.append((e_cur.date(), f"missing leg {sK}/{lK} {ot}")); continue
        try:
            s0 = sser.asof(entry_time); l0 = lser.asof(entry_time)
        except Exception:
            skipped.append((e_cur.date(), "no entry premium")); continue
        if np.isnan(s0) or np.isnan(l0):
            skipped.append((e_cur.date(), "nan entry premium")); continue
        net_credit = float(s0 - l0)
        if net_credit <= 0:
            skipped.append((e_cur.date(), f"non-positive credit {round(net_credit,1)}")); continue

        both = pd.concat([sser.rename("s"), lser.rename("l")], axis=1).ffill().dropna()
        both = both[both.index > entry_time]; sv = both["s"].values; lv = both["l"].values; spread_val = sv - lv; btimes = both.index.values
        thr = TARGET_FRAC * net_credit; hit = np.where(spread_val <= thr)[0]
        if len(hit):
            j = hit[0]; exit_time = pd.Timestamp(btimes[j]); short_x = float(sv[j]); long_x = float(lv[j]); reason = "90% target"
        else:
            S = settle.get(e_cur, np.nan)
            if np.isnan(S): S = daily.loc[e_cur, "c"] if e_cur in daily.index else np.nan
            if np.isnan(S):
                skipped.append((e_cur.date(), "no settlement spot")); continue
            if typ == "CCS": short_x = float(max(0.0, S - sK)); long_x = float(max(0.0, S - lK))
            else: short_x = float(max(0.0, sK - S)); long_x = float(max(0.0, lK - S))
            exit_time = e_cur + pd.Timedelta(hours=15, minutes=30); reason = "DTE0 settlement"
        exit_debit = short_x - long_x; pnl = net_credit - exit_debit
        trades.append({"entry_date": entry_day.date(), "DTE": dte, "type": ("Call Credit Spread" if typ == "CCS" else "Put Credit Spread"),
                       "trigger": label, "entry_time": entry_time.strftime("%Y-%m-%d %H:%M"), "ATM": int(atm),
                       "short_K": int(sK), "long_K": int(lK), "short_leg_entry": round(float(s0), 2), "long_leg_entry": round(float(l0), 2),
                       "net_credit": round(net_credit, 2), "exit_date": pd.Timestamp(exit_time).date(),
                       "exit_time": pd.Timestamp(exit_time).strftime("%Y-%m-%d %H:%M"), "exit_reason": reason,
                       "short_leg_exit": round(short_x, 2), "long_leg_exit": round(long_x, 2), "exit_debit": round(exit_debit, 2),
                       "pnl_points": round(pnl, 2), "days_held": (pd.Timestamp(exit_time).normalize() - entry_day).days})

    T = pd.DataFrame(trades)
    win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)
    by_trig = (T.trigger.value_counts(normalize=True) * 100).round(1).to_dict()
    wd = pd.to_datetime(T.entry_date).dt.day_name(); exp_wd = pd.to_datetime([pd.Timestamp(x) for x in expiries]).day_name().value_counts().to_dict()

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "SENSEX Weekly Credit Spread (real-expiry-derived); hedge WIDTH=600; GROSS premium points, 1 spread"},
        {"metric": "Change vs NIFTY spec", "value": "hedge leg width 200 -> 600; SENSEX data, strike step 100, ATM=round(spot/100)*100"},
        {"metric": "Expiry weekday (from data)", "value": f"{exp_wd} (NOT constant: Fri->Tue@2025-01-07->Thu@2025-09-04; actual calendar used)"},
        {"metric": "Window", "value": f"{T.entry_date.min()} .. {T.exit_date.max()} (spot ends {spot_end.date()}; {len(skipped)} cycles skipped)"},
        {"metric": "Settlement", "value": "spread intrinsic at SENSEX spot settlement (avg last 30-min spot on expiry); 90% target on option 1-min"},
        {"metric": "Total trades", "value": len(T)},
        {"metric": "Win rate %", "value": win},
        {"metric": "Total P&L (points)", "value": tot},
        {"metric": "Avg P&L / trade", "value": round(T.pnl_points.mean(), 2)},
        {"metric": "Median P&L / trade", "value": round(T.pnl_points.median(), 2)},
        {"metric": "Avg net credit", "value": round(T.net_credit.mean(), 2)},
        {"metric": "Best / worst trade", "value": f"{round(T.pnl_points.max(),1)} / {round(T.pnl_points.min(),1)}"},
        {"metric": "Avg DTE at entry", "value": round(T.DTE.mean(), 2)},
        {"metric": "DTE range (min..max)", "value": f"{int(T.DTE.min())}..{int(T.DTE.max())}"},
        {"metric": "Avg days held", "value": round(T.days_held.mean(), 2)},
        {"metric": "% by trigger", "value": str(by_trig)},
        {"metric": "% 90% target / DTE0 settlement", "value": f"{round((T.exit_reason=='90% target').mean()*100,1)} / {round((T.exit_reason=='DTE0 settlement').mean()*100,1)}"},
        {"metric": "Call / Put credit spreads", "value": f"{int((T.type=='Call Credit Spread').sum())} / {int((T.type=='Put Credit Spread').sum())}"},
    ])
    with pd.ExcelWriter(OUTDIR / "sensex_weekly_credit_spread.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        T.to_excel(w, sheet_name="Trades", index=False)
        if skipped: pd.DataFrame(skipped, columns=["expiry", "reason"]).to_excel(w, sheet_name="Skipped", index=False)
    T.to_csv(OUTDIR / "sensex_weekly_credit_spread_trades.csv", index=False)

    pd.set_option("display.width", 230)
    print("=" * 96 + "\nSENSEX WEEKLY CREDIT SPREAD (hedge width 600)\n" + "=" * 96)
    print(f"window {T.entry_date.min()} .. {T.exit_date.max()} | trades {len(T)} | skipped {len(skipped)}")
    print(f"win {win}% | TOTAL {tot:,} pts | avg {round(T.pnl_points.mean(),2)} | median {round(T.pnl_points.median(),2)} | avg credit {round(T.net_credit.mean(),2)}")
    print(f"avg DTE {round(T.DTE.mean(),2)} (range {int(T.DTE.min())}..{int(T.DTE.max())}) | avg held {round(T.days_held.mean(),2)}d")
    print(f"trigger mix %: {by_trig}")
    print(f"exit: 90%-target {round((T.exit_reason=='90% target').mean()*100,1)}% | settlement {round((T.exit_reason=='DTE0 settlement').mean()*100,1)}%")
    print(f"CCS/PCS: {int((T.type=='Call Credit Spread').sum())}/{int((T.type=='Put Credit Spread').sum())}")
    print("\nfirst 8 trades:")
    print(T.head(8)[["entry_date", "DTE", "type", "trigger", "net_credit", "exit_date", "exit_reason", "exit_debit", "pnl_points"]].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
