# -*- coding: utf-8 -*-
"""sensex_btst_close_direction_final.py — FINAL locked SENSEX Daily Close-Direction BTST report.
LOCKED: entry 15:15 (spot vs day-open), exit 09:17 next trading day (that minute's OPEN), sell-leg OFFSET=800,
ATM step 100, never DTE-0 (nearest expiry STRICTLY AFTER entry day; expiry-day -> next expiry). RED(spot<open):
+2 ATM PE / -1 (ATM-800) PE ; GREEN: +2 ATM CE / -1 (ATM+800) CE. P&L = OPTION PREMIUM points (2xlong-1xshort),
GROSS. India VIX (NIFTY vol index, proxy) at entry for segregation only. 3 sheets: Summary, VIX_Segregation,
Trade_Log. Output: sensex_btst_close_direction_final_report.xlsx.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "sensex_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "SENSEX"
VIXF = rb.BASE / "data" / "india_vix_1min.csv"
OUTDIR = rb.RESULTS / "sensex_btst_close_direction"; OUTDIR.mkdir(parents=True, exist_ok=True)
ENTRY_MOD = 15 * 60 + 15; EXIT_MOD = 9 * 60 + 17; STEP = 100; OFFSET = 800; LOT = 20
VIX_LO, VIX_HI = 17.0, 19.0          # EXCLUDE trades with entry-minute India VIX in [17,19] (matches NIFTY)
VIX_BUCKETS = ["<9"] + [f"{i}-{i+1}" for i in range(9, 20)] + [">20"]


def vbucket(v):
    if pd.isna(v): return "n/a"
    return "<9" if v < 9 else (">20" if v >= 20 else f"{int(v)}-{int(v)+1}")


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


def leg_series(folder, strike, ot, t0, t1):
    fs = glob.glob(str(OPTDIR / folder / f"SENSEX_{int(strike)}_{ot}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "open", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts").sort_index()
    return o if len(o) else None


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    day_open = sp.groupby("date")["open"].first(); spot_entry = sp[sp["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vx = vx.assign(mod=vts.dt.hour * 60 + vts.dt.minute, date=vts.dt.normalize())
    vix_ent = vx[vx["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    tdays = sorted(day_open.index)
    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")); expset = set(expiries); first_exp = expiries[0]

    trades = []; skipped = 0
    for i in range(len(tdays) - 1):
        D = tdays[i]; Dn = tdays[i + 1]
        if D < first_exp or Dn > spot_end or D not in spot_entry.index: continue
        j = bisect.bisect_right(expiries, D)
        if j >= len(expiries): continue
        E = expiries[j]; is_exp = D in expset; folder = E.strftime("%Y%m%d")
        espot = float(spot_entry.loc[D]); dopen = float(day_open.loc[D]); atm = round(espot / STEP) * STEP
        direction = "RED" if espot < dopen else "GREEN"; ot = "PE" if direction == "RED" else "CE"
        long_K = atm; short_K = atm - OFFSET if direction == "RED" else atm + OFFSET
        t_en = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=15); t_ex = pd.Timestamp(Dn) + pd.Timedelta(hours=9, minutes=17)
        t0 = t_en - pd.Timedelta(minutes=10); t1 = t_ex + pd.Timedelta(minutes=10)
        Lo = leg_series(folder, long_K, ot, t0, t1); So = leg_series(folder, short_K, ot, t0, t1)
        if Lo is None or So is None: skipped += 1; continue
        Le = Lo["close"].asof(t_en); Se = So["close"].asof(t_en)
        Lx = Lo["open"].reindex([t_ex], method="ffill").iloc[0]; Sx = So["open"].reindex([t_ex], method="ffill").iloc[0]
        if any(pd.isna(v) for v in (Le, Se, Lx, Sx)): skipped += 1; continue
        entry_cost = 2 * Le - Se; exit_val = 2 * Lx - Sx; pnl = exit_val - entry_cost
        hold_hrs = round((t_ex - t_en).total_seconds() / 3600, 2)
        trades.append({"entry_date": D.date(), "direction": direction,
                       "contract": "NEXT-expiry (expiry-day switch)" if is_exp else "current-expiry", "is_expiry_day": is_exp,
                       "expiry_used": E.date(), "ATM": int(atm), "long_leg": f"2x {int(long_K)} {ot}", "short_leg": f"1x {int(short_K)} {ot}",
                       "long_entry": round(float(Le), 2), "short_entry": round(float(Se), 2), "entry_cost_debit": round(float(entry_cost), 2),
                       "exit_date": Dn.date(), "exit_time": "09:17", "long_exit": round(float(Lx), 2), "short_exit": round(float(Sx), 2),
                       "exit_value": round(float(exit_val), 2), "pnl_points": round(float(pnl), 2),
                       "entry_vix": round(float(vix_ent.get(D, np.nan)), 2), "holding_hours": hold_hrs})

    T_full = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    # VIX 17-19 exclusion (NaN VIX kept — not in [17,19]); flag on full set, then filter
    T_full["excluded_by_vix_filter"] = (T_full.entry_vix >= VIX_LO) & (T_full.entry_vix <= VIX_HI)
    n_removed = int(T_full["excluded_by_vix_filter"].sum())
    T = T_full[~T_full["excluded_by_vix_filter"]].reset_index(drop=True)      # FINAL filtered set (Summary computed on this)
    pos = T[T.pnl_points > 0]; neg = T[T.pnl_points < 0]

    def dd_of(df):
        if not len(df): return None, 0.0, 0.0, 0, 0
        cum = df["pnl_points"].cumsum().values; eq = np.concatenate([[0.0], cum])
        dt = np.concatenate([[pd.Timestamp(df.entry_date.iloc[0])], pd.to_datetime(df.entry_date).values])
        de = drawdown_episodes(eq, dt)
        return (de, de.drawdown_points.max() if len(de) else 0.0, round(de.drawdown_points.mean(), 1) if len(de) else 0.0,
                int(de.days_peak_to_recovery.max()) if len(de) else 0, round(de.days_peak_to_recovery.mean(), 1) if len(de) else 0)
    DE, maxdd, avgdd, maxddp, avgddp = dd_of(T)
    _, maxdd_u, avgdd_u, _, _ = dd_of(T_full)             # unfiltered DD for comparison block

    def blk(df, lbl):
        return {"segment": lbl, "trades": len(df), "wins": int((df.pnl_points > 0).sum()), "losses": int((df.pnl_points < 0).sum()),
                "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_points.sum(), 1), "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0,
                "median_pnl": round(df.pnl_points.median(), 2) if len(df) else 0,
                "avg_pos": round(df[df.pnl_points > 0].pnl_points.mean(), 2) if (df.pnl_points > 0).any() else 0,
                "avg_neg": round(df[df.pnl_points < 0].pnl_points.mean(), 2) if (df.pnl_points < 0).any() else 0,
                "max_profit": round(df.pnl_points.max(), 1) if len(df) else 0, "max_loss": round(df.pnl_points.min(), 1) if len(df) else 0}
    BY = pd.DataFrame([blk(T, "ALL"), blk(T[T.direction == "RED"], "RED"), blk(T[T.direction == "GREEN"], "GREEN"),
                       blk(T[~T.is_expiry_day], "normal-day (current-expiry)"), blk(T[T.is_expiry_day], "expiry-day (next-expiry)")])

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "SENSEX Daily Close-Direction BTST — FINAL (entry 15:15, exit 09:17 next-day OPEN, offset 800, never DTE-0)"},
        {"metric": "Structure", "value": "RED(spot<open): +2 ATM PE / -1 (ATM-800) PE ; GREEN: +2 ATM CE / -1 (ATM+800) CE ; 1 trade/day"},
        {"metric": "VIX FILTER (applied)", "value": f"EXCLUDE trades with India VIX (at 15:15 entry) in [{int(VIX_LO)},{int(VIX_HI)}] -> {n_removed} trades removed. (India VIX = NIFTY vol index, used as proxy.)"},
        {"metric": "P&L unit", "value": "OPTION PREMIUM points (2xlong - 1xshort), GROSS (no costs). Lot 20 (ref; points unaffected)."},
        {"metric": "ATM / strike step / exit price", "value": "round(spot/100)*100 ; step 100 ; exit at next-day 09:17 OPEN"},
        {"metric": "Window", "value": f"{T.entry_date.min()} .. {T.exit_date.max()}"},
        {"metric": "", "value": ""},
        {"metric": "Total trades", "value": len(T)},
        {"metric": "Winning trades", "value": int((T.pnl_points > 0).sum())},
        {"metric": "Losing trades", "value": int((T.pnl_points < 0).sum())},
        {"metric": "Win rate %", "value": round((T.pnl_points > 0).mean() * 100, 1)},
        {"metric": "Total P&L (points)", "value": round(T.pnl_points.sum(), 1)},
        {"metric": "Average P&L / trade", "value": round(T.pnl_points.mean(), 2)},
        {"metric": "Median P&L / trade", "value": round(T.pnl_points.median(), 2)},
        {"metric": "Average positive P&L", "value": round(pos.pnl_points.mean(), 2) if len(pos) else 0},
        {"metric": "Average negative P&L", "value": round(neg.pnl_points.mean(), 2) if len(neg) else 0},
        {"metric": "Max profit", "value": round(T.pnl_points.max(), 1)},
        {"metric": "Max loss", "value": round(T.pnl_points.min(), 1)},
        {"metric": "Average holding period (hours)", "value": round(T.holding_hours.mean(), 2)},
        {"metric": "", "value": ""},
        {"metric": "--- DRAWDOWN (cumulative equity, premium points) ---", "value": ""},
        {"metric": "Number of drawdown episodes", "value": len(DE)},
        {"metric": "Average drawdown (points)", "value": avgdd},
        {"metric": "Maximum drawdown (points)", "value": round(maxdd, 1)},
        {"metric": "Average drawdown period (days, peak->recovery)", "value": avgddp},
        {"metric": "Maximum drawdown period (days, peak->recovery)", "value": maxddp},
        {"metric": "Return / Max-DD", "value": round(T.pnl_points.sum() / maxdd, 2) if maxdd else "-"},
        {"metric": "Skipped (missing leg data)", "value": skipped},
    ])

    # comparison block: filtered (VIX 17-19 excluded) vs unfiltered (original)
    def q(df): return dict(trades=len(df), win=round((df.pnl_points > 0).mean() * 100, 1), tot=round(df.pnl_points.sum(), 1), avg=round(df.pnl_points.mean(), 2))
    u = q(T_full); f = q(T)
    CMP = pd.DataFrame([
        {"metric": "Trades", "unfiltered": u["trades"], "filtered (VIX 17-19 excl)": f["trades"], "delta": f["trades"] - u["trades"]},
        {"metric": "Win rate %", "unfiltered": u["win"], "filtered (VIX 17-19 excl)": f["win"], "delta": round(f["win"] - u["win"], 1)},
        {"metric": "Total P&L (points)", "unfiltered": u["tot"], "filtered (VIX 17-19 excl)": f["tot"], "delta": round(f["tot"] - u["tot"], 1)},
        {"metric": "Avg P&L / trade", "unfiltered": u["avg"], "filtered (VIX 17-19 excl)": f["avg"], "delta": round(f["avg"] - u["avg"], 2)},
        {"metric": "Max drawdown (points)", "unfiltered": round(maxdd_u, 1), "filtered (VIX 17-19 excl)": round(maxdd, 1), "delta": round(maxdd - maxdd_u, 1)},
        {"metric": "Avg drawdown (points)", "unfiltered": avgdd_u, "filtered (VIX 17-19 excl)": avgdd, "delta": round(avgdd - avgdd_u, 1)},
    ])

    # VIX segregation: FULL unfiltered set (informational; 17-19 NOT excluded here)
    T_full["_vb"] = T_full.entry_vix.map(vbucket)
    present = [b for b in VIX_BUCKETS if (T_full._vb == b).any()]
    VIX = pd.DataFrame([{"vix_bucket": b, "trades": int((T_full._vb == b).sum()),
                         "win_%": round((T_full[T_full._vb == b].pnl_points > 0).mean() * 100, 1),
                         "total_pnl": round(T_full[T_full._vb == b].pnl_points.sum(), 1),
                         "avg_pnl": round(T_full[T_full._vb == b].pnl_points.mean(), 2),
                         "in_filter_zone(17-19,EXCLUDED)": b in ("17-18", "18-19")} for b in present])

    TL = T_full[["entry_date", "direction", "contract", "is_expiry_day", "expiry_used", "ATM", "long_leg", "short_leg",
                 "long_entry", "short_entry", "entry_cost_debit", "exit_date", "exit_time", "long_exit", "short_exit",
                 "exit_value", "pnl_points", "entry_vix", "holding_hours", "excluded_by_vix_filter"]]

    OUTF = OUTDIR / "sensex_btst_close_direction_final_report.xlsx"
    with pd.ExcelWriter(OUTF, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        pd.DataFrame([{"metric": "--- FILTERED (VIX 17-19 excluded) vs UNFILTERED ---", "value": ""}]).to_excel(w, sheet_name="Summary", index=False, startrow=len(summary) + 2, header=False)
        CMP.to_excel(w, sheet_name="Summary", index=False, startrow=len(summary) + 4)
        pd.DataFrame([{"segment": "--- SEGMENT SPLIT (filtered set) ---"}]).to_excel(w, sheet_name="Summary", index=False, startrow=len(summary) + 4 + len(CMP) + 2, header=False)
        BY.to_excel(w, sheet_name="Summary", index=False, startrow=len(summary) + 4 + len(CMP) + 3)
        VIX.to_excel(w, sheet_name="VIX_Segregation", index=False)
        TL.to_excel(w, sheet_name="Trade_Log", index=False)

    pd.set_option("display.width", 240)
    print("=" * 96 + "\nSENSEX DAILY CLOSE-DIRECTION BTST — FINAL REPORT (exit 09:17, offset 800, VIX 17-19 EXCLUDED)\n" + "=" * 96)
    print(f"VIX filter: removed {n_removed} trades in [17,19]")
    print("--- BEFORE vs AFTER ---"); print(CMP.to_string(index=False))
    print(f"\nwindow {T.entry_date.min()} .. {T.exit_date.max()} | trades {len(T)} | skipped {skipped}")
    print(f"win {round((T.pnl_points>0).mean()*100,1)}% | TOTAL {round(T.pnl_points.sum(),1):,} pts | avg {round(T.pnl_points.mean(),2)} | median {round(T.pnl_points.median(),2)}")
    print(f"avg+ {round(pos.pnl_points.mean(),2)} | avg- {round(neg.pnl_points.mean(),2)} | maxP {round(T.pnl_points.max(),1)} | maxL {round(T.pnl_points.min(),1)} | hold {round(T.holding_hours.mean(),2)}h")
    print(f"DD: episodes {len(DE)} | avg {avgdd} | MAX {round(maxdd,1)} | avg period {avgddp}d | max period {maxddp}d | ret/maxDD {round(T.pnl_points.sum()/maxdd,2) if maxdd else '-'}")
    print("\n--- BY SEGMENT ---"); print(BY.to_string(index=False))
    print("\n--- VIX SEGREGATION ---"); print(VIX.to_string(index=False))
    print(f"\nSaved -> {OUTF}")


if __name__ == "__main__":
    main()
