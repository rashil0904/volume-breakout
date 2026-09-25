# -*- coding: utf-8 -*-
"""nifty_weekly_credit_spread_FINAL_v3_capped_july2026.py — NIFTY Weekly Credit Spread FINAL v3 (EOD-close
SL on 3-day-high-triggered PCS trades only), date-capped so the LAST trade entered is 2026-07-22 (the
weekly cycle after the 2026-07-21 expiry) — excludes the Aug-2026 CAS-transition period entirely. Full
underlying spot/options data is used (NOT truncated) so trades entered near the cutoff still resolve
correctly against their own forward settlement/target/SL data; only which weekly CYCLES are included is
capped, via ENTRY_CUTOFF on entry_day. Everything else identical to
nifty_weekly_credit_spread_v3_eod_close_sl.py (entry logic, 200pt width, 90% target, DTE-0 settlement,
2:35pm fallback, EOD-close SL mechanics, entry_spot/exit_spot columns).
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "weekly_credit_spread"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIDTH = 200; TARGET_FRAC = 0.10; FB_MOD = 14 * 60 + 35
ENTRY_CUTOFF = pd.Timestamp("2026-07-22")   # last entry day included (2026-07-21 expiry's weekly cycle)


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
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    daily = sp.groupby("date").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    settle = sp[(sp["mod"] >= 900) & (sp["mod"] <= 930)].groupby("date")["close"].mean()
    spot_close = sp.set_index("ts")["close"].sort_index()
    eod_close = sp[sp["mod"] >= 929].groupby("date")["close"].last()
    tdays = list(daily.index)
    expiries = pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")

    trades = []; skipped = []
    for i in range(1, len(expiries)):
        e_prev, e_cur = expiries[i - 1], expiries[i]
        after = [d for d in tdays if d > e_prev]
        if not after:
            continue
        entry_day = after[0]
        if entry_day > ENTRY_CUTOFF:
            continue   # NEW: date window cap (whole cycle excluded, not just filtered post-hoc)
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
        if trig_i is None:
            fb = ed[ed["mod"] >= FB_MOD]
            if fb.empty:
                continue
            trig_i = ed.index.get_loc(fb.index[0]); px = op[trig_i]
            typ = "CCS" if px < d_open else "PCS"; label = "fallback-2:35"
        entry_time = pd.Timestamp(ets[trig_i])
        entry_spot = op[trig_i] if label == "fallback-2:35" else cl[trig_i]
        atm = round(entry_spot / 50) * 50

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
        thr = TARGET_FRAC * net_credit
        hit = np.where(spread_val <= thr)[0]
        target_time = pd.Timestamp(btimes[hit[0]]) if len(hit) else None

        is_3dhigh_pcs = (label == "3d-high" and typ == "PCS")
        sl_level = round(float(d3h), 2) if is_3dhigh_pcs else np.nan
        sl_fired = False

        if target_time is not None:
            j = hit[0]; b_short, b_long, b_reason = float(sv[j]), float(lv[j]), "90% target"
        else:
            S = settle.get(e_cur, np.nan)
            if np.isnan(S): S = daily.loc[e_cur, "c"] if e_cur in daily.index else np.nan
            if np.isnan(S):
                skipped.append((e_cur.date(), "no settlement spot")); continue
            if typ == "CCS": b_short, b_long = float(max(0.0, S - sK)), float(max(0.0, S - lK))
            else: b_short, b_long = float(max(0.0, sK - S)), float(max(0.0, lK - S))
            b_reason = "DTE0 settlement"
        baseline_pnl = round(net_credit - (b_short - b_long), 2)

        sl_day = None
        if is_3dhigh_pcs:
            days_between = [d for d in tdays if entry_day <= d <= e_cur]
            for d in days_between:
                c = eod_close.get(d, np.nan)
                if pd.notna(c) and c < d3h:
                    sl_day = d; break

        if is_3dhigh_pcs and sl_day is not None and (target_time is None or target_time.normalize() > sl_day):
            eod_ts = sl_day + pd.Timedelta(hours=15, minutes=29)
            idx = both.index.searchsorted(eod_ts, side="left")
            if idx >= len(both): idx = len(both) - 1
            short_x, long_x, exit_time, reason = float(both["s"].iloc[idx]), float(both["l"].iloc[idx]), both.index[idx], "eod-close-sl"
            exit_spot = eod_close.get(sl_day, np.nan)
            sl_fired = True
        elif target_time is not None:
            j = hit[0]; short_x, long_x, exit_time, reason = float(sv[j]), float(lv[j]), target_time, "90% target"
            exit_spot = spot_close.asof(exit_time)
        else:
            S = settle.get(e_cur, np.nan)
            if np.isnan(S): S = daily.loc[e_cur, "c"] if e_cur in daily.index else np.nan
            if typ == "CCS": short_x, long_x = float(max(0.0, S - sK)), float(max(0.0, S - lK))
            else: short_x, long_x = float(max(0.0, sK - S)), float(max(0.0, lK - S))
            exit_time, reason = e_cur + pd.Timedelta(hours=15, minutes=30), "DTE0 settlement"
            exit_spot = S

        exit_debit = short_x - long_x; pnl = net_credit - exit_debit; leg = ot
        trades.append({"entry_date": entry_day.date(), "DTE": dte, "type": ("Call Credit Spread" if typ == "CCS" else "Put Credit Spread"),
                       "trigger": label, "entry_time": entry_time.strftime("%Y-%m-%d %H:%M"), "ATM": int(atm),
                       "leg_type": leg, "short_K": int(sK), "long_K": int(lK),
                       "entry_spot": round(float(entry_spot), 2),
                       "short_leg_entry": round(float(s0), 2), "long_leg_entry": round(float(l0), 2), "net_credit": round(net_credit, 2),
                       "exit_date": pd.Timestamp(exit_time).date(), "exit_time": pd.Timestamp(exit_time).strftime("%Y-%m-%d %H:%M"), "exit_reason": reason,
                       "exit_spot": round(float(exit_spot), 2) if pd.notna(exit_spot) else np.nan,
                       "sl_level": sl_level,
                       "short_leg_exit": round(short_x, 2), "long_leg_exit": round(long_x, 2), "exit_debit": round(exit_debit, 2),
                       "pnl_points": round(pnl, 2), "days_held": (pd.Timestamp(exit_time).normalize() - entry_day).days,
                       "baseline_pnl_no_sl": baseline_pnl, "sl_fired": sl_fired})

    T = pd.DataFrame(trades)
    win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)
    by_trig = T.trigger.value_counts(normalize=True).mul(100).round(1)

    T = T.sort_values("exit_time").reset_index(drop=True); T["cum_pnl"] = T["pnl_points"].cumsum().round(2)
    xt = pd.to_datetime(T["exit_time"]); equity = np.concatenate([[0.0], T["cum_pnl"].values])
    times = np.concatenate([[pd.to_datetime(T["entry_time"]).iloc[0]], xt.values])
    DE = drawdown_episodes(equity, times)
    maxdd = DE.drawdown_points.max() if len(DE) else 0.0; avgdd = round(DE.drawdown_points.mean(), 1) if len(DE) else 0.0

    T["month"] = pd.to_datetime(T["entry_date"]).dt.to_period("M").astype(str)
    MON = T.groupby("month").apply(lambda g: pd.Series({
        "trades": len(g), "wins": int((g.pnl_points > 0).sum()), "win_%": round((g.pnl_points > 0).mean() * 100, 1),
        "total_pnl": round(g.pnl_points.sum(), 1), "avg_pnl": round(g.pnl_points.mean(), 2), "avg_credit": round(g.net_credit.mean(), 1),
        "target_exits": int((g.exit_reason == "90% target").sum()), "settle_exits": int((g.exit_reason == "DTE0 settlement").sum()),
        "sl_exits": int((g.exit_reason == "eod-close-sl").sum()),
        "best": round(g.pnl_points.max(), 1), "worst": round(g.pnl_points.min(), 1)}), include_groups=False).reset_index()
    MON["cum_pnl"] = MON["total_pnl"].cumsum().round(1)

    baseline_tot = round(T["baseline_pnl_no_sl"].sum(), 1)
    affected = T[T["sl_level"].notna()]
    sl_hit = affected[affected["sl_fired"]]
    CMP = pd.DataFrame([
        {"metric": "Total trades", "BEFORE (no SL)": len(T), "AFTER (EOD-close SL on 3d-high PCS)": len(T)},
        {"metric": "Total P&L (points)", "BEFORE (no SL)": baseline_tot, "AFTER (EOD-close SL on 3d-high PCS)": tot},
        {"metric": "Delta total P&L", "BEFORE (no SL)": "-", "AFTER (EOD-close SL on 3d-high PCS)": round(tot - baseline_tot, 1)},
        {"metric": "", "BEFORE (no SL)": "", "AFTER (EOD-close SL on 3d-high PCS)": ""},
        {"metric": "3d-high PCS trades (subset)", "BEFORE (no SL)": len(affected), "AFTER (EOD-close SL on 3d-high PCS)": len(affected)},
        {"metric": "  of which SL fired", "BEFORE (no SL)": "-", "AFTER (EOD-close SL on 3d-high PCS)": len(sl_hit)},
        {"metric": "  subset total P&L", "BEFORE (no SL)": round(affected["baseline_pnl_no_sl"].sum(), 1), "AFTER (EOD-close SL on 3d-high PCS)": round(affected["pnl_points"].sum(), 1)},
        {"metric": "  subset avg P&L", "BEFORE (no SL)": round(affected["baseline_pnl_no_sl"].mean(), 2) if len(affected) else "-", "AFTER (EOD-close SL on 3d-high PCS)": round(affected["pnl_points"].mean(), 2) if len(affected) else "-"},
        {"metric": "  subset win %", "BEFORE (no SL)": round((affected["baseline_pnl_no_sl"] > 0).mean() * 100, 1) if len(affected) else "-", "AFTER (EOD-close SL on 3d-high PCS)": round((affected["pnl_points"] > 0).mean() * 100, 1) if len(affected) else "-"},
        {"metric": "  subset delta P&L", "BEFORE (no SL)": "-", "AFTER (EOD-close SL on 3d-high PCS)": round(affected["pnl_points"].sum() - affected["baseline_pnl_no_sl"].sum(), 1)},
    ])

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY Weekly Credit Spread FINAL v3 — EOD-close SL on 3-day-high-triggered PCS trades only. Date-capped: last entry 2026-07-22 (pre-CAS). GROSS premium points, 1 spread."},
        {"metric": "Window", "value": f"{T.entry_date.min()} .. {T.exit_date.max()} ({len(skipped)} cycles skipped)"},
        {"metric": "Last trade entry date", "value": str(T.entry_date.max())},
        {"metric": "Settlement", "value": "spread intrinsic at spot settlement (avg last 30-min spot on expiry); 90% target uses option 1-min; EOD-close SL uses spot ~15:29 close"},
        {"metric": "Total trades", "value": len(T)}, {"metric": "Win rate %", "value": win},
        {"metric": "Total P&L (points)", "value": tot}, {"metric": "Avg P&L / trade", "value": round(T.pnl_points.mean(), 2)},
        {"metric": "Median P&L / trade", "value": round(T.pnl_points.median(), 2)},
        {"metric": "Avg net credit", "value": round(T.net_credit.mean(), 2)}, {"metric": "Best / worst", "value": f"{round(T.pnl_points.max(),1)} / {round(T.pnl_points.min(),1)}"},
        {"metric": "Avg DTE at entry", "value": round(T.DTE.mean(), 2)}, {"metric": "DTE spread (min..max)", "value": f"{int(T.DTE.min())}..{int(T.DTE.max())}"},
        {"metric": "Avg days held", "value": round(T.days_held.mean(), 2)},
        {"metric": "% 90% target / settlement / eod-close-sl", "value": f"{round((T.exit_reason=='90% target').mean()*100,1)} / {round((T.exit_reason=='DTE0 settlement').mean()*100,1)} / {round((T.exit_reason=='eod-close-sl').mean()*100,1)}"},
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

    OUTPATH = OUTDIR / "weekly_credit_spread_FINAL_v3_capped_july2026.xlsx"
    with pd.ExcelWriter(OUTPATH, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        r0 = len(summary) + 2
        pd.DataFrame([{"section": "--- BEFORE (no SL) vs AFTER (EOD-close SL on 3d-high PCS) ---"}]).to_excel(w, sheet_name="Summary", index=False, startrow=r0, header=False)
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
    T.to_csv(OUTDIR / "weekly_credit_spread_FINAL_v3_capped_july2026_trades.csv", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96 + "\nNIFTY WEEKLY CREDIT SPREAD FINAL v3 — capped, last entry 2026-07-22\n" + "=" * 96)
    print(f"window {T.entry_date.min()} .. {T.exit_date.max()} | trades {len(T)} | skipped {len(skipped)}")
    print(f"last trade entry date: {T.entry_date.max()}  (expect 2026-07-22)")
    print(f"win {win}% | TOTAL {tot:,} pts | avg {round(T.pnl_points.mean(),2)} | avg credit {round(T.net_credit.mean(),2)}")
    print(f"exit mix: 90%-target {round((T.exit_reason=='90% target').mean()*100,1)}% | settlement {round((T.exit_reason=='DTE0 settlement').mean()*100,1)}% | eod-close-sl {round((T.exit_reason=='eod-close-sl').mean()*100,1)}%")
    print(f"drawdown: {len(DE)} episodes | MAX {round(maxdd,1)} pts | AVG {avgdd} pts | ret/maxDD {round(tot/maxdd,2) if maxdd else '-'}")
    print(f"CCS/PCS: {int((T.type=='Call Credit Spread').sum())}/{int((T.type=='Put Credit Spread').sum())}")
    print("\n--- BEFORE vs AFTER ---"); print(CMP.to_string(index=False))
    print("\n--- CCS vs PCS ---"); print(CCS_PCS.to_string(index=False))
    print(f"\nSaved -> {OUTPATH}")


if __name__ == "__main__":
    main()
