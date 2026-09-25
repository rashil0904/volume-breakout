# -*- coding: utf-8 -*-
"""nifty_weekly_credit_spread.py — NIFTY Weekly Credit Spread, weekly-cycle version tied to ACTUAL expiries.

ENTRY: first trading day after the previous weekly expiry (derived from the real expiry calendar = options
folders; no Tue/Thu hardcoding). DTE = calendar days entry->current expiry (varies with holidays).
LEVELS: 3-day high/low = max high / min low of the 3 completed trading days before entry.
BREAKOUT (entry day, intraday TOUCH): low<=3dL -> Bearish -> Call Credit Spread (sell ATM CE, buy CE+200);
high>=3dH -> Bullish -> Put Credit Spread (sell ATM PE, buy PE-200). First touch wins; one trade/week.
FALLBACK (if neither by 14:30): spot vs day-open -> red=CCS, green=PCS, enter at 14:30.
ATM = round(spot_at_entry/50)*50. EXIT: (1) 90% target - every 1-min, when cost-to-close (short-long) decays
to <=10% of entry net credit; else (2) DTE-0 SETTLEMENT = spread intrinsic at spot settlement (avg last 30
min spot on expiry). No stop loss. GROSS premium points (1 spread). Window 2024-10 -> 2026-07-15 (spot end).
"""
import sys, glob, os, re
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "weekly_credit_spread"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIDTH = 200; TARGET_FRAC = 0.10; FB_MOD = 14 * 60 + 30      # 200-pt spread; exit at 10% of credit; 14:30 fallback


def drawdown_episodes(equity, times):
    """DD episodes on the realized cumulative-P&L curve. returns DataFrame of episodes."""
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
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{strike}_{otype}_*.parquet"))
    if not fs:
        return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts")["close"].sort_index()
    return o if len(o) else None


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    daily = sp.groupby("date").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    settle = sp[(sp["mod"] >= 900) & (sp["mod"] <= 930)].groupby("date")["close"].mean()   # last-30-min settlement
    tdays = list(daily.index)
    expiries = pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")

    trades = []; skipped = []
    for i in range(1, len(expiries)):
        e_prev, e_cur = expiries[i - 1], expiries[i]
        after = [d for d in tdays if d > e_prev]
        if not after:
            continue
        entry_day = after[0]
        if entry_day > spot_end or e_cur > spot_end:
            skipped.append((e_cur.date(), "outside spot window")); continue
        dte = (e_cur - entry_day).days
        prev3 = [d for d in tdays if d < entry_day][-3:]
        if len(prev3) < 3:
            continue
        d3h = daily.loc[prev3, "h"].max(); d3l = daily.loc[prev3, "l"].min()

        # ---- entry-day breakout / 14:30 fallback ----
        ed = sp[sp["date"] == entry_day]
        hi = ed["high"].values; lo = ed["low"].values; cl = ed["close"].values; md = ed["mod"].values; ets = ed["ts"].values
        d_open = ed["open"].iloc[0]; trig_i = None; typ = None; label = None
        for k in range(len(ed)):
            if md[k] > FB_MOD:
                break
            bh = hi[k] >= d3h; bl = lo[k] <= d3l
            if bh or bl:
                if bh and bl: typ, label = ("PCS", "3d-high") if abs(d_open - d3h) <= abs(d_open - d3l) else ("CCS", "3d-low")
                elif bh: typ, label = "PCS", "3d-high"
                else: typ, label = "CCS", "3d-low"
                trig_i = k; break
        if trig_i is None:                                       # fallback at 14:30
            fb = ed[ed["mod"] >= FB_MOD]
            if fb.empty:
                continue
            trig_i = ed.index.get_loc(fb.index[0]); px = cl[trig_i]
            typ = "CCS" if px < d_open else "PCS"; label = "fallback-2:30"
        entry_time = pd.Timestamp(ets[trig_i]); entry_spot = cl[trig_i]; atm = round(entry_spot / 50) * 50

        # ---- legs & net credit ----
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

        # ---- exit: 90% target (1-min) else DTE-0 settlement ----
        both = pd.concat([sser.rename("s"), lser.rename("l")], axis=1).ffill().dropna()
        both = both[both.index > entry_time]; sv = both["s"].values; lv = both["l"].values; spread_val = sv - lv; btimes = both.index.values
        thr = TARGET_FRAC * net_credit; exit_time = None; exit_debit = None; reason = None
        hit = np.where(spread_val <= thr)[0]
        if len(hit):                                             # 90% target -> per-leg market prices at that minute
            j = hit[0]; exit_time = pd.Timestamp(btimes[j]); short_x = float(sv[j]); long_x = float(lv[j]); reason = "90% target"
        else:                                                    # settlement -> per-leg intrinsic at expiry spot settlement
            S = settle.get(e_cur, np.nan)
            if np.isnan(S): S = daily.loc[e_cur, "c"] if e_cur in daily.index else np.nan
            if np.isnan(S):
                skipped.append((e_cur.date(), "no settlement spot")); continue
            if typ == "CCS": short_x = float(max(0.0, S - sK)); long_x = float(max(0.0, S - lK))
            else: short_x = float(max(0.0, sK - S)); long_x = float(max(0.0, lK - S))
            exit_time = e_cur + pd.Timedelta(hours=15, minutes=30); reason = "DTE0 settlement"
        exit_debit = short_x - long_x; pnl = net_credit - exit_debit; leg = ot   # ot = CE (CCS) or PE (PCS)
        trades.append({"entry_date": entry_day.date(), "DTE": dte, "type": ("Call Credit Spread" if typ == "CCS" else "Put Credit Spread"),
                       "trigger": label, "entry_time": entry_time.strftime("%Y-%m-%d %H:%M"), "ATM": int(atm),
                       "leg_type": leg, "short_K": int(sK), "long_K": int(lK),
                       "short_leg_entry": round(float(s0), 2), "long_leg_entry": round(float(l0), 2), "net_credit": round(net_credit, 2),
                       "exit_date": pd.Timestamp(exit_time).date(), "exit_time": pd.Timestamp(exit_time).strftime("%Y-%m-%d %H:%M"), "exit_reason": reason,
                       "short_leg_exit": round(short_x, 2), "long_leg_exit": round(long_x, 2), "exit_debit": round(exit_debit, 2),
                       "pnl_points": round(pnl, 2), "days_held": (pd.Timestamp(exit_time).normalize() - entry_day).days})

    T = pd.DataFrame(trades)
    win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)
    by_trig = T.trigger.value_counts(normalize=True).mul(100).round(1)

    # ---- DRAWDOWN on realized cumulative P&L (chronological by exit) ----
    T = T.sort_values("exit_time").reset_index(drop=True); T["cum_pnl"] = T["pnl_points"].cumsum().round(2)
    xt = pd.to_datetime(T["exit_time"]); equity = np.concatenate([[0.0], T["cum_pnl"].values])
    times = np.concatenate([[pd.to_datetime(T["entry_time"]).iloc[0]], xt.values])
    DE = drawdown_episodes(equity, times)
    maxdd = DE.drawdown_points.max() if len(DE) else 0.0; avgdd = round(DE.drawdown_points.mean(), 1) if len(DE) else 0.0

    # ---- MONTHLY P&L (by entry month) ----
    T["month"] = pd.to_datetime(T["entry_date"]).dt.to_period("M").astype(str)
    MON = T.groupby("month").apply(lambda g: pd.Series({
        "trades": len(g), "wins": int((g.pnl_points > 0).sum()), "win_%": round((g.pnl_points > 0).mean() * 100, 1),
        "total_pnl": round(g.pnl_points.sum(), 1), "avg_pnl": round(g.pnl_points.mean(), 2), "avg_credit": round(g.net_credit.mean(), 1),
        "target_exits": int((g.exit_reason == "90% target").sum()), "settle_exits": int((g.exit_reason == "DTE0 settlement").sum()),
        "best": round(g.pnl_points.max(), 1), "worst": round(g.pnl_points.min(), 1)}), include_groups=False).reset_index()
    MON["cum_pnl"] = MON["total_pnl"].cumsum().round(1)
    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY Weekly Credit Spread (weekly-cycle, real-expiry-derived); GROSS premium points, 1 spread"},
        {"metric": "Window", "value": f"{T.entry_date.min()} .. {T.exit_date.max()} (spot ends 2026-07-15 -> {len(skipped)} cycles skipped)"},
        {"metric": "Settlement", "value": "spread intrinsic at spot settlement (avg last 30-min spot on expiry); 90% target uses option 1-min"},
        {"metric": "Total trades", "value": len(T)}, {"metric": "Win rate %", "value": win},
        {"metric": "Total P&L (points)", "value": tot}, {"metric": "Avg P&L / trade", "value": round(T.pnl_points.mean(), 2)},
        {"metric": "Avg net credit", "value": round(T.net_credit.mean(), 2)}, {"metric": "Best / worst", "value": f"{round(T.pnl_points.max(),1)} / {round(T.pnl_points.min(),1)}"},
        {"metric": "Avg DTE at entry", "value": round(T.DTE.mean(), 2)}, {"metric": "DTE spread (min..max)", "value": f"{int(T.DTE.min())}..{int(T.DTE.max())}"},
        {"metric": "Avg days held", "value": round(T.days_held.mean(), 2)},
        {"metric": "% 90% target / settlement", "value": f"{round((T.exit_reason=='90% target').mean()*100,1)} / {round((T.exit_reason=='DTE0 settlement').mean()*100,1)}"},
        {"metric": "", "value": ""},
        {"metric": "--- DRAWDOWN (realized cumulative P&L) ---", "value": ""},
        {"metric": "Number of drawdown episodes", "value": len(DE)},
        {"metric": "Max drawdown (points)", "value": round(maxdd, 1)},
        {"metric": "Avg drawdown (points)", "value": avgdd},
        {"metric": "Max DD duration (days, peak->recovery)", "value": int(DE.days_peak_to_recovery.max()) if len(DE) else 0},
        {"metric": "Avg DD duration (days)", "value": round(DE.days_peak_to_recovery.mean(), 1) if len(DE) else 0},
        {"metric": "Return / Max-DD", "value": round(tot / maxdd, 2) if maxdd else "-"},
        {"metric": "Trigger mix (%)", "value": by_trig.to_dict()},
        {"metric": "Call / Put credit spreads", "value": f"{int((T.type=='Call Credit Spread').sum())} / {int((T.type=='Put Credit Spread').sum())}"},
    ])
    with pd.ExcelWriter(OUTDIR / "weekly_credit_spread.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        MON.to_excel(w, sheet_name="Monthly_PnL", index=False)
        DE.sort_values("drawdown_points", ascending=False).to_excel(w, sheet_name="Drawdowns", index=False)
        T.to_excel(w, sheet_name="Trades", index=False)
        pd.DataFrame({"DTE": sorted(T.DTE.unique()), "count": [int((T.DTE == x).sum()) for x in sorted(T.DTE.unique())]}).to_excel(w, sheet_name="DTE_Distribution", index=False)
        if skipped: pd.DataFrame(skipped, columns=["expiry", "reason"]).to_excel(w, sheet_name="Skipped", index=False)
    T.to_csv(OUTDIR / "weekly_credit_spread_trades.csv", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96 + "\nNIFTY WEEKLY CREDIT SPREAD (weekly cycle, real expiries)\n" + "=" * 96)
    print(f"window {T.entry_date.min()} .. {T.exit_date.max()} | trades {len(T)} | skipped {len(skipped)}")
    print(f"win {win}% | TOTAL {tot:,} pts | avg {round(T.pnl_points.mean(),2)} | avg credit {round(T.net_credit.mean(),2)} | avg DTE {round(T.DTE.mean(),2)} (range {int(T.DTE.min())}..{int(T.DTE.max())}) | avg held {round(T.days_held.mean(),2)}d")
    print(f"exit: 90%-target {round((T.exit_reason=='90% target').mean()*100,1)}% | settlement {round((T.exit_reason=='DTE0 settlement').mean()*100,1)}%")
    print(f"drawdown: {len(DE)} episodes | MAX {round(maxdd,1)} pts | AVG {avgdd} pts | ret/maxDD {round(tot/maxdd,2) if maxdd else '-'}")
    print("\n--- MONTHLY P&L ---"); print(MON.to_string(index=False))
    print(f"trigger mix: {by_trig.to_dict()}")
    print(f"CCS/PCS: {int((T.type=='Call Credit Spread').sum())}/{int((T.type=='Put Credit Spread').sum())}")
    print("\nfirst 6 trades:")
    print(T.head(6)[["entry_date", "DTE", "type", "trigger", "net_credit", "exit_date", "exit_reason", "exit_debit", "pnl_points"]].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
