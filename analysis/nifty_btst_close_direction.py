# -*- coding: utf-8 -*-
"""nifty_btst_close_direction.py — NIFTY Daily Close-Direction BTST (FINAL: 15:20 in / 09:17 next-day out,
VIX 17-19 filter). Enter 15:20 daily (RED=spot<open -> 2x ATM PE / -1 (ATM-OFFSET) PE ; GREEN=spot>open ->
2x ATM CE / -1 (ATM+OFFSET) CE ; OFFSET=300). Exit 09:17 next day. Exclude trades with India VIX 17-19.
MOD: never use a DTE-0 contract -> contract = nearest expiry STRICTLY AFTER the entry day (current week on
normal days; next week on expiry day). GROSS index points. Spot for 15:15 direction/ATM; options for legs.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "btst_close_direction"; OUTDIR.mkdir(parents=True, exist_ok=True)
OFFSET = 300; ENTRY_MOD = 15 * 60 + 20; EXIT_MOD = 9 * 60 + 17       # FINAL: 15:20 entry ; 09:17 exit (sell leg +-300)
VIXF = rb.BASE / "data" / "india_vix_1min.csv"; VIX_LO, VIX_HI = 17.0, 19.0   # exclude trades with entry-min VIX in [17,19]
LOT = 75                                                             # NIFTY lot; P&L(INR) = premium_points x LOT


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


def leg_at(folder, strike, otype, t_entry, t_exit):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{otype}_*.parquet"))
    if not fs: return None, None
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    s = o[(o.ts >= t_entry - pd.Timedelta(minutes=5)) & (o.ts <= t_exit + pd.Timedelta(minutes=5))].set_index("ts")["close"].sort_index()
    if s.empty: return None, None
    return s.asof(t_entry), s.asof(t_exit)


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    day_open = sp.groupby("date")["open"].first()
    spot1515 = sp[sp["mod"] == ENTRY_MOD].groupby("date")["close"].last()          # spot at entry minute (15:20)
    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vx = vx.assign(mod=vts.dt.hour * 60 + vts.dt.minute, date=vts.dt.normalize())
    vix_ent = vx[vx["mod"] == ENTRY_MOD].groupby("date")["close"].last()            # India VIX at entry minute
    tdays = sorted(day_open.index)
    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d"))
    expset = set(expiries); first_exp = expiries[0]

    trades = []; skipped = 0
    for i in range(len(tdays) - 1):
        D = tdays[i]; Dn = tdays[i + 1]
        if D < first_exp or Dn > spot_end or D not in spot1515.index:
            continue
        j = bisect.bisect_right(expiries, D)                                        # nearest expiry STRICTLY AFTER D
        if j >= len(expiries):
            continue
        E = expiries[j]; is_exp = D in expset
        espot = float(spot1515.loc[D]); dopen = float(day_open.loc[D]); atm = round(espot / 50) * 50
        direction = "RED" if espot < dopen else "GREEN"; ot = "PE" if direction == "RED" else "CE"
        long_K = atm; short_K = atm - OFFSET if direction == "RED" else atm + OFFSET
        t_en = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=20); t_ex = pd.Timestamp(Dn) + pd.Timedelta(hours=9, minutes=17)
        folder = E.strftime("%Y%m%d")
        Le, Lx = leg_at(folder, long_K, ot, t_en, t_ex); Se, Sx = leg_at(folder, short_K, ot, t_en, t_ex)
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (Le, Lx, Se, Sx)):
            skipped += 1; continue
        entry_cost = 2 * Le - Se; exit_val = 2 * Lx - Sx; pnl = exit_val - entry_cost
        trades.append({"entry_date": D.date(), "entry_time": "15:20", "entry_vix": round(float(vix_ent.get(D, np.nan)), 2), "direction": direction,
                       "contract": "NEXT-WEEK (expiry-day switch)" if is_exp else "current-week", "expiry_used": E.date(), "is_expiry_day": is_exp,
                       "ATM": int(atm), "long_leg": f"2x {int(long_K)} {ot}", "short_leg": f"1x {int(short_K)} {ot}",
                       "long_entry": round(float(Le), 2), "short_entry": round(float(Se), 2), "entry_cost": round(float(entry_cost), 2),
                       "exit_date": Dn.date(), "exit_time": "09:17", "long_exit": round(float(Lx), 2), "short_exit": round(float(Sx), 2),
                       "exit_value": round(float(exit_val), 2), "pnl_premium_points": round(float(pnl), 2)})

    T_all = pd.DataFrame(trades).rename(columns={"pnl_premium_points": "pnl_points"})     # ALL trades (pre-filter)
    pre_n = len(T_all); pre_tot = round(T_all.pnl_points.sum(), 1); pre_win = round((T_all.pnl_points > 0).mean() * 100, 1)
    fmask = (T_all.entry_vix >= VIX_LO) & (T_all.entry_vix <= VIX_HI)                       # VIX 17-19 -> excluded
    n_removed = int(fmask.sum()); T = T_all[~fmask].reset_index(drop=True)                  # FINAL filtered set
    win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)
    red = T[T.direction == "RED"]; grn = T[T.direction == "GREEN"]; nrm = T[~T.is_expiry_day]; exp = T[T.is_expiry_day]

    def blk(df, lbl):
        return {"group": lbl, "trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_points.sum(), 1), "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0}
    BY = pd.DataFrame([blk(T, "ALL"), blk(red, "RED"), blk(grn, "GREEN"), blk(nrm, "normal-day (current-week)"), blk(exp, "expiry-day (next-week)")])

    T["month"] = pd.to_datetime(T["entry_date"]).dt.to_period("M").astype(str)
    MON = T.groupby("month").apply(lambda g: pd.Series({
        "trades": len(g), "wins": int((g.pnl_points > 0).sum()), "win_%": round((g.pnl_points > 0).mean() * 100, 1),
        "total_pnl": round(g.pnl_points.sum(), 1), "avg_pnl": round(g.pnl_points.mean(), 2),
        "RED": int((g.direction == "RED").sum()), "GREEN": int((g.direction == "GREEN").sum()),
        "best": round(g.pnl_points.max(), 1), "worst": round(g.pnl_points.min(), 1)}), include_groups=False).reset_index()
    MON["cum_pnl"] = MON["total_pnl"].cumsum().round(1)

    Tc = T.sort_values("entry_date").reset_index(drop=True); cum = Tc["pnl_points"].cumsum().values
    equity = np.concatenate([[0.0], cum]); dtimes = np.concatenate([[pd.Timestamp(Tc.entry_date.iloc[0])], pd.to_datetime(Tc.entry_date).values])
    DE = drawdown_episodes(equity, dtimes); maxdd = DE.drawdown_points.max() if len(DE) else 0.0; avgdd = round(DE.drawdown_points.mean(), 1) if len(DE) else 0.0

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY Close-Direction BTST — FINAL (15:20 in, 09:17-next-day out, VIX 17-19 filter); never DTE-0 (expiry-day uses next week)"},
        {"metric": "VIX filter", "value": f"EXCLUDE trades with India VIX (at 15:20 entry) in [{int(VIX_LO)},{int(VIX_HI)}] -> {n_removed} trades removed"},
        {"metric": "-- filter impact (before -> after) --", "value": f"trades {pre_n} -> {len(T)} | P&L {pre_tot} -> {tot} | win {pre_win}% -> {win}%"},
        {"metric": "Structure", "value": f"RED: +2 ATM PE / -1 (ATM-{OFFSET}) PE ; GREEN: +2 ATM CE / -1 (ATM+{OFFSET}) CE ; 1 trade/day"},
        {"metric": "Sell-leg offset", "value": f"{OFFSET} points from ATM"},
        {"metric": "P&L unit", "value": "OPTION PREMIUM points (structure = 2xlong - 1xshort). GROSS (no cost)."},
        {"metric": "Window", "value": f"{T.entry_date.min()} .. {T.exit_date.max()} (spot ends 2026-07-15)"},
        {"metric": "Total trades", "value": len(T)}, {"metric": "Win rate %", "value": win},
        {"metric": "Total P&L (option premium points)", "value": tot}, {"metric": "Avg P&L / trade (points)", "value": round(T.pnl_points.mean(), 2)},
        {"metric": "Median P&L / trade (points)", "value": round(T.pnl_points.median(), 2)},
        {"metric": "RED / GREEN trades", "value": f"{len(red)} / {len(grn)}"},
        {"metric": "RED avg / GREEN avg P&L", "value": f"{round(red.pnl_points.mean(),2)} / {round(grn.pnl_points.mean(),2)}"},
        {"metric": "Normal-day trades / avg P&L", "value": f"{len(nrm)} / {round(nrm.pnl_points.mean(),2)}"},
        {"metric": "Expiry-day (next-week) trades / avg P&L", "value": f"{len(exp)} / {round(exp.pnl_points.mean(),2) if len(exp) else 0}"},
        {"metric": "", "value": ""},
        {"metric": "--- DRAWDOWN (realized cumulative P&L, premium points) ---", "value": ""},
        {"metric": "Number of drawdown episodes", "value": len(DE)},
        {"metric": "Max drawdown (points)", "value": round(maxdd, 1)},
        {"metric": "Avg drawdown (points)", "value": avgdd},
        {"metric": "Max DD duration (days, peak->recovery)", "value": int(DE.days_peak_to_recovery.max()) if len(DE) else 0},
        {"metric": "Return / Max-DD", "value": round(tot / maxdd, 2) if maxdd else "-"},
        {"metric": "Skipped (missing leg data)", "value": skipped},
    ])
    with pd.ExcelWriter(OUTDIR / "btst_close_direction.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False); BY.to_excel(w, sheet_name="By_Group", index=False)
        MON.to_excel(w, sheet_name="Monthly_PnL", index=False)
        DE.sort_values("drawdown_points", ascending=False).to_excel(w, sheet_name="Drawdowns", index=False)
        T.to_excel(w, sheet_name="Trades", index=False)
    T.to_csv(OUTDIR / "btst_close_direction_trades.csv", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96 + "\nNIFTY DAILY CLOSE-DIRECTION BTST (option premium points; never DTE-0; expiry-day uses next-week contract)\n" + "=" * 96)
    print(f"window {T.entry_date.min()} .. {T.exit_date.max()} | trades {len(T)} | skipped {skipped}")
    print(f"win {win}% | TOTAL {tot:,} option-premium pts | avg {round(T.pnl_points.mean(),2)} pts")
    print("\n--- BY GROUP ---"); print(BY.to_string(index=False))
    print("\nfirst 8 trades:")
    print(T.head(8)[["entry_date", "direction", "contract", "ATM", "entry_cost", "exit_date", "exit_value", "pnl_points"]].to_string(index=False))
    print(f"\nexpiry-day (next-week-contract) trades: {len(exp)} | avg P&L {round(exp.pnl_points.mean(),2) if len(exp) else 0} | normal {len(nrm)} avg {round(nrm.pnl_points.mean(),2)}")
    print(f"Saved -> {OUTDIR}")


if __name__ == "__main__":
    main()
