# -*- coding: utf-8 -*-
"""nifty_weekly_credit_spread_v2_235fallback.py — NIFTY Weekly Credit Spread, UPDATED: fallback check time
2:30pm -> 2:35pm (open price). Everything else identical to nifty_weekly_credit_spread.py (WIDTH=200, 90%
target, DTE-0 settlement, no SL, real-expiry-derived weekly cycle). Confirmed via the 14:30-14:40 sweep: ZERO
breach-driven trigger switches occur in this window, so the trigger mix is expected to match the 2:30pm
baseline exactly (same population) - only the fallback ENTRY PRICE (and hence net credit/P&L) differs.
BEFORE reference (confirmed): 92 trades, 70.7% win, 1971.8 total pts, trigger mix fallback-2:30 34.8% /
3d-high 33.7% / 3d-low 31.5%.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "weekly_credit_spread"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIDTH = 200; TARGET_FRAC = 0.10; FB_MOD = 14 * 60 + 35        # UPDATED: 200-pt spread; exit at 10% of credit; 14:35 fallback
SPOT_CUTOFF = pd.Timestamp("2026-07-15")   # locks the trade universe to the original 92-trade report's data window
                                             # (spot/options data has since been extended further; this cap keeps this
                                             # report's trade set unchanged for the entry_spot/exit_spot column addition)

BEFORE = {"trades": 92, "win_%": 70.7, "total_pnl": 1971.8, "fb_pct": 34.8, "hi_pct": 33.7, "lo_pct": 31.5}


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
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{strike}_{otype}_*.parquet"))
    if not fs:
        return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts")["close"].sort_index()
    return o if len(o) else None


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp = sp[sp["ts"] <= SPOT_CUTOFF + pd.Timedelta(days=1)].reset_index(drop=True)   # lock to original report's data window
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    daily = sp.groupby("date").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    settle = sp[(sp["mod"] >= 900) & (sp["mod"] <= 930)].groupby("date")["close"].mean()   # last-30-min settlement
    spot_close = sp.set_index("ts")["close"].sort_index()   # for entry/exit spot price lookups
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

        ed = sp[sp["date"] == entry_day]
        hi = ed["high"].values; lo = ed["low"].values; cl = ed["close"].values; op = ed["open"].values; md = ed["mod"].values; ets = ed["ts"].values
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
        if trig_i is None:                                       # fallback at 14:35 (OPEN price)
            fb = ed[ed["mod"] >= FB_MOD]
            if fb.empty:
                continue
            trig_i = ed.index.get_loc(fb.index[0]); px = op[trig_i]
            typ = "CCS" if px < d_open else "PCS"; label = "fallback-2:35"
        entry_time = pd.Timestamp(ets[trig_i])
        entry_spot = op[trig_i] if label == "fallback-2:35" else cl[trig_i]
        atm = round(entry_spot / 50) * 50

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
        if len(hit):
            j = hit[0]; exit_time = pd.Timestamp(btimes[j]); short_x = float(sv[j]); long_x = float(lv[j]); reason = "90% target"
            exit_spot = spot_close.asof(exit_time)
        else:
            S = settle.get(e_cur, np.nan)
            if np.isnan(S): S = daily.loc[e_cur, "c"] if e_cur in daily.index else np.nan
            if np.isnan(S):
                skipped.append((e_cur.date(), "no settlement spot")); continue
            if typ == "CCS": short_x = float(max(0.0, S - sK)); long_x = float(max(0.0, S - lK))
            else: short_x = float(max(0.0, sK - S)); long_x = float(max(0.0, lK - S))
            exit_time = e_cur + pd.Timedelta(hours=15, minutes=30); reason = "DTE0 settlement"
            exit_spot = S
        exit_debit = short_x - long_x; pnl = net_credit - exit_debit; leg = ot
        trades.append({"entry_date": entry_day.date(), "DTE": dte, "type": ("Call Credit Spread" if typ == "CCS" else "Put Credit Spread"),
                       "trigger": label, "entry_time": entry_time.strftime("%Y-%m-%d %H:%M"), "ATM": int(atm),
                       "leg_type": leg, "short_K": int(sK), "long_K": int(lK),
                       "entry_spot": round(float(entry_spot), 2),
                       "short_leg_entry": round(float(s0), 2), "long_leg_entry": round(float(l0), 2), "net_credit": round(net_credit, 2),
                       "exit_date": pd.Timestamp(exit_time).date(), "exit_time": pd.Timestamp(exit_time).strftime("%Y-%m-%d %H:%M"), "exit_reason": reason,
                       "exit_spot": round(float(exit_spot), 2) if pd.notna(exit_spot) else np.nan,
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

    # ---- BEFORE (2:30pm) vs AFTER (2:35pm) comparison ----
    fb_pct_after = round((T.trigger == "fallback-2:35").mean() * 100, 1)
    hi_pct_after = round((T.trigger == "3d-high").mean() * 100, 1)
    lo_pct_after = round((T.trigger == "3d-low").mean() * 100, 1)
    CMP = pd.DataFrame([
        {"metric": "Total trades", "BEFORE (2:30pm fallback)": BEFORE["trades"], "AFTER (2:35pm fallback)": len(T)},
        {"metric": "Win rate %", "BEFORE (2:30pm fallback)": BEFORE["win_%"], "AFTER (2:35pm fallback)": win},
        {"metric": "Total P&L (points)", "BEFORE (2:30pm fallback)": BEFORE["total_pnl"], "AFTER (2:35pm fallback)": tot},
        {"metric": "Delta total P&L", "BEFORE (2:30pm fallback)": "-", "AFTER (2:35pm fallback)": round(tot - BEFORE["total_pnl"], 1)},
        {"metric": "Fallback % of trades", "BEFORE (2:30pm fallback)": BEFORE["fb_pct"], "AFTER (2:35pm fallback)": fb_pct_after},
        {"metric": "3d-high % of trades", "BEFORE (2:30pm fallback)": BEFORE["hi_pct"], "AFTER (2:35pm fallback)": hi_pct_after},
        {"metric": "3d-low % of trades", "BEFORE (2:30pm fallback)": BEFORE["lo_pct"], "AFTER (2:35pm fallback)": lo_pct_after},
        {"metric": "Trigger-mix unchanged? (confirms 0 switches, per prior sweep)", "BEFORE (2:30pm fallback)": "-",
         "AFTER (2:35pm fallback)": "YES" if abs(fb_pct_after - BEFORE["fb_pct"]) < 0.2 else "NO - mix shifted"},
    ])

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY Weekly Credit Spread v2 — FALLBACK CHECK TIME UPDATED 2:30pm -> 2:35pm (open price). GROSS premium points, 1 spread."},
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
    def blk(df, lbl):
        return {"group": lbl, "trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_points.sum(), 1) if len(df) else 0, "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0}
    ccs = T[T.type == "Call Credit Spread"]; pcs = T[T.type == "Put Credit Spread"]
    CCS_PCS = pd.DataFrame([blk(T, "ALL"), blk(ccs, "CCS (bearish)"), blk(pcs, "PCS (bullish)")])

    with pd.ExcelWriter(OUTDIR / "weekly_credit_spread_v2_235fallback.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        r0 = len(summary) + 2
        pd.DataFrame([{"section": "--- BEFORE (2:30pm) vs AFTER (2:35pm) ---"}]).to_excel(w, sheet_name="Summary", index=False, startrow=r0, header=False)
        CMP.to_excel(w, sheet_name="Summary", index=False, startrow=r0 + 1)
        r1 = r0 + 1 + len(CMP) + 2
        pd.DataFrame([{"section": "--- CCS vs PCS ---"}]).to_excel(w, sheet_name="Summary", index=False, startrow=r1, header=False)
        CCS_PCS.to_excel(w, sheet_name="Summary", index=False, startrow=r1 + 1)
        MON.to_excel(w, sheet_name="Monthly_PnL", index=False)
        DE.sort_values("drawdown_points", ascending=False).to_excel(w, sheet_name="Drawdowns", index=False)
        T.to_excel(w, sheet_name="Trades", index=False)
        pd.DataFrame({"DTE": sorted(T.DTE.unique()), "count": [int((T.DTE == x).sum()) for x in sorted(T.DTE.unique())]}).to_excel(w, sheet_name="DTE_Distribution", index=False)
        if skipped: pd.DataFrame(skipped, columns=["expiry", "reason"]).to_excel(w, sheet_name="Skipped", index=False)
        ccs.to_excel(w, sheet_name="CCS_Trades", index=False)
        pcs.to_excel(w, sheet_name="PCS_Trades", index=False)
    T.to_csv(OUTDIR / "weekly_credit_spread_v2_235fallback_trades.csv", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96 + "\nNIFTY WEEKLY CREDIT SPREAD v2 — 2:35pm FALLBACK\n" + "=" * 96)
    print(f"window {T.entry_date.min()} .. {T.exit_date.max()} | trades {len(T)} | skipped {len(skipped)}")
    print(f"win {win}% | TOTAL {tot:,} pts | avg {round(T.pnl_points.mean(),2)} | avg credit {round(T.net_credit.mean(),2)} | avg DTE {round(T.DTE.mean(),2)} (range {int(T.DTE.min())}..{int(T.DTE.max())}) | avg held {round(T.days_held.mean(),2)}d")
    print(f"exit: 90%-target {round((T.exit_reason=='90% target').mean()*100,1)}% | settlement {round((T.exit_reason=='DTE0 settlement').mean()*100,1)}%")
    print(f"drawdown: {len(DE)} episodes | MAX {round(maxdd,1)} pts | AVG {avgdd} pts | ret/maxDD {round(tot/maxdd,2) if maxdd else '-'}")
    print(f"trigger mix: {by_trig.to_dict()}")
    print(f"CCS/PCS: {int((T.type=='Call Credit Spread').sum())}/{int((T.type=='Put Credit Spread').sum())}")
    print("\n--- BEFORE vs AFTER ---"); print(CMP.to_string(index=False))
    print("\n--- CCS vs PCS ---"); print(CCS_PCS.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
