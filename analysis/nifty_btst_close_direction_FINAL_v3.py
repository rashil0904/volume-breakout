# -*- coding: utf-8 -*-
"""nifty_btst_close_direction_FINAL_v3.py — NIFTY Close-Direction BTST FINAL v3, on top of the confirmed
FINAL v2 baseline (entry CLOSE@15:20, exit CLOSE@09:17, VIX[17,19] excluded, DTE-0 next-week rule).

TWO CHANGES vs v2:
1. DTE-1 REMOVED ENTIRELY: any entry day where the current week's contract is at DTE==1 is SKIPPED (no
   trade at all), instead of v2's next-week-contract shift. Derived from the actual expiry calendar, both
   regimes (old Thu-expiry: DTE1=Wed; new Tue-expiry: DTE1=Mon).
2. DATE WINDOW CAPPED at 2026-07-30 entry / 2026-07-31 exit (last trade) — excludes the Aug-2026 CAS-
   transition period entirely. Start boundary unchanged.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "btst_close_direction_FINAL"; OUTDIR.mkdir(parents=True, exist_ok=True)
OFFSET = 300; ENTRY_MOD = 15 * 60 + 20; EXIT_MOD = 9 * 60 + 17
VIXF = rb.BASE / "data" / "india_vix_1min.csv"; VIX_LO, VIX_HI = 17.0, 19.0
DATE_CUTOFF_ENTRY = pd.Timestamp("2026-07-30")   # last entry day; exit lands 2026-07-31 -- excludes Aug-2026 CAS period
VIX_BUCKETS = ["<9"] + [f"{i}-{i+1}" for i in range(9, 20)] + [">20"]

# v2's confirmed numbers, NOW DATE-MATCHED to v3's window (v2 also capped at 2026-07-30 entry / 07-31 exit)
BEFORE_V2 = {"trades": 397, "win_%": 53.1, "total_pnl": 9464.5, "max_dd": 558.2, "avg_dd": 128.6, "max_dd_days": 35}


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


def vbucket(v):
    if pd.isna(v): return "n/a"
    return "<9" if v < 9 else (">20" if v >= 20 else f"{int(v)}-{int(v)+1}")


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    day_open = sp.groupby("date")["open"].first(); spot1520 = sp[sp["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vx = vx.assign(mod=vts.dt.hour * 60 + vts.dt.minute, date=vts.dt.normalize())
    vix_ent = vx[vx["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    tdays = sorted(day_open.index)
    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")); expset = set(expiries); first_exp = expiries[0]

    wd_all = pd.Series([e.day_name() for e in expiries], index=expiries)
    tue_thu = wd_all[wd_all.isin(["Tuesday", "Thursday"])]
    regime_start_new = None
    for i in range(1, len(tue_thu)):
        if tue_thu.iloc[i] != tue_thu.iloc[i - 1]:
            regime_start_new = tue_thu.index[i]; break

    trades = []; skipped = 0; dte1_skipped = 0
    for i in range(len(tdays) - 1):
        D = tdays[i]; Dn = tdays[i + 1]
        if D < first_exp or Dn > spot_end or D not in spot1520.index: continue
        if D > DATE_CUTOFF_ENTRY: continue     # NEW: date window cap
        j = bisect.bisect_right(expiries, D)
        if j >= len(expiries): continue
        E0 = expiries[j]; is_exp_day = D in expset; dte0 = (E0.normalize() - pd.Timestamp(D).normalize()).days

        if dte0 == 1:                          # NEW: skip DTE-1 entirely, no next-week shift
            dte1_skipped += 1; continue
        E = E0
        regime = "NEW (Tue-expiry)" if E0 >= regime_start_new else "OLD (Thu-expiry)"

        espot = float(spot1520.loc[D]); dopen = float(day_open.loc[D]); atm = round(espot / 50) * 50
        direction = "RED" if espot < dopen else "GREEN"; ot = "PE" if direction == "RED" else "CE"
        long_K = atm; short_K = atm - OFFSET if direction == "RED" else atm + OFFSET
        t_en = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=20); t_ex = pd.Timestamp(Dn) + pd.Timedelta(hours=9, minutes=17)
        folder = E.strftime("%Y%m%d")
        Le, Lx = leg_at(folder, long_K, ot, t_en, t_ex); Se, Sx = leg_at(folder, short_K, ot, t_en, t_ex)
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (Le, Lx, Se, Sx)):
            skipped += 1; continue
        entry_cost = 2 * Le - Se; exit_val = 2 * Lx - Sx; pnl = exit_val - entry_cost

        group = "expiry-day (next-week)" if is_exp_day else "normal (current-week)"
        trades.append({"entry_date": D.date(), "entry_time": "15:20", "entry_vix": round(float(vix_ent.get(D, np.nan)), 2), "direction": direction,
                       "contract": "NEXT-WEEK (expiry-day switch)" if is_exp_day else "current-week",
                       "expiry_used": E.date(), "is_expiry_day": is_exp_day, "dte1_shift_applied": False, "group": group,
                       "DTE_original": dte0, "regime": regime,
                       "ATM": int(atm), "long_leg": f"2x {int(long_K)} {ot}", "short_leg": f"1x {int(short_K)} {ot}",
                       "long_entry": round(float(Le), 2), "short_entry": round(float(Se), 2), "entry_cost": round(float(entry_cost), 2),
                       "exit_date": Dn.date(), "exit_time": "09:17", "long_exit": round(float(Lx), 2), "short_exit": round(float(Sx), 2),
                       "exit_value": round(float(exit_val), 2), "pnl_points": round(float(pnl), 2)})

    T_all = pd.DataFrame(trades)
    pre_n = len(T_all); pre_tot = round(T_all.pnl_points.sum(), 1); pre_win = round((T_all.pnl_points > 0).mean() * 100, 1)
    fmask = (T_all.entry_vix >= VIX_LO) & (T_all.entry_vix <= VIX_HI)
    n_removed = int(fmask.sum()); T = T_all[~fmask].reset_index(drop=True)
    T["month"] = pd.to_datetime(T["entry_date"]).dt.to_period("M").astype(str)
    T["vix_bucket"] = T["entry_vix"].map(vbucket)

    t_en_all = pd.to_datetime(T["entry_date"]) + pd.Timedelta(hours=15, minutes=20)
    t_ex_all = pd.to_datetime(T["exit_date"]) + pd.Timedelta(hours=9, minutes=17)
    T["holding_hours"] = (t_ex_all - t_en_all).dt.total_seconds() / 3600

    win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)
    red = T[T.direction == "RED"]; grn = T[T.direction == "GREEN"]
    nrm = T[T.group == "normal (current-week)"]; exp = T[T.group == "expiry-day (next-week)"]

    def blk(df, lbl=None, extra=None):
        d = {"trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
             "total_pnl": round(df.pnl_points.sum(), 1) if len(df) else 0, "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0}
        if lbl is not None: d = {"group": lbl, **d}
        if extra: d.update(extra)
        return d

    BY = pd.DataFrame([blk(T, "ALL"), blk(red, "RED"), blk(grn, "GREEN"),
                       blk(nrm, "normal-day (current-week)"), blk(exp, "expiry-day (next-week)")])

    Tc = T.sort_values("entry_date").reset_index(drop=True); cum = Tc["pnl_points"].cumsum().values
    equity = np.concatenate([[0.0], cum]); dtimes = np.concatenate([[pd.Timestamp(Tc.entry_date.iloc[0])], pd.to_datetime(Tc.entry_date).values])
    DE = drawdown_episodes(equity, dtimes); maxdd = DE.drawdown_points.max() if len(DE) else 0.0; avgdd = round(DE.drawdown_points.mean(), 1) if len(DE) else 0.0
    maxddp = int(DE.days_peak_to_recovery.max()) if len(DE) else 0

    MON = T.groupby("month").apply(lambda g: pd.Series({
        "trades": len(g), "wins": int((g.pnl_points > 0).sum()), "win_%": round((g.pnl_points > 0).mean() * 100, 1),
        "total_pnl": round(g.pnl_points.sum(), 1), "avg_pnl": round(g.pnl_points.mean(), 2),
        "best": round(g.pnl_points.max(), 1), "worst": round(g.pnl_points.min(), 1)}), include_groups=False).reset_index()
    MON["cum_pnl"] = MON["total_pnl"].cumsum().round(1)

    AFTER = {"trades": len(T), "win_%": win, "total_pnl": tot, "max_dd": round(maxdd, 1), "avg_dd": avgdd, "max_dd_days": maxddp}
    CMP = pd.DataFrame([
        {"metric": "Total trades", "BEFORE (v2, DTE-1 next-week shift)": BEFORE_V2["trades"], "AFTER (v3, DTE-1 removed)": AFTER["trades"]},
        {"metric": "Win rate %", "BEFORE (v2, DTE-1 next-week shift)": BEFORE_V2["win_%"], "AFTER (v3, DTE-1 removed)": AFTER["win_%"]},
        {"metric": "Total P&L (points)", "BEFORE (v2, DTE-1 next-week shift)": BEFORE_V2["total_pnl"], "AFTER (v3, DTE-1 removed)": AFTER["total_pnl"]},
        {"metric": "Max drawdown (points)", "BEFORE (v2, DTE-1 next-week shift)": BEFORE_V2["max_dd"], "AFTER (v3, DTE-1 removed)": AFTER["max_dd"]},
        {"metric": "Avg drawdown (points)", "BEFORE (v2, DTE-1 next-week shift)": BEFORE_V2["avg_dd"], "AFTER (v3, DTE-1 removed)": AFTER["avg_dd"]},
        {"metric": "Max DD duration (days)", "BEFORE (v2, DTE-1 next-week shift)": BEFORE_V2["max_dd_days"], "AFTER (v3, DTE-1 removed)": AFTER["max_dd_days"]},
        {"metric": "Delta total P&L (DTE-1 removal effect, isolated)", "BEFORE (v2, DTE-1 next-week shift)": "-", "AFTER (v3, DTE-1 removed)": round(AFTER["total_pnl"] - BEFORE_V2["total_pnl"], 1)},
        {"metric": "NOTE", "BEFORE (v2, DTE-1 next-week shift)": "both date-matched: entries capped 2026-07-30, last exit 2026-07-31 -- delta isolates the DTE-1 rule change only", "AFTER (v3, DTE-1 removed)": ""},
    ])

    # ---- VIX_Summary / DTE_Summary sheets ----
    VIX = pd.DataFrame([{"vix_bucket": b, **blk(T[T.vix_bucket == b], lbl=None, extra={"avg_holding_hours": round(T[T.vix_bucket == b].holding_hours.mean(), 2) if len(T[T.vix_bucket == b]) else 0})} for b in VIX_BUCKETS])

    def dte_summary(regime_label):
        sub = T[T.regime == regime_label]
        rows = []
        for d in range(1, 7):
            g = sub[sub.DTE_original == d]
            contract_counts = g["contract"].value_counts().to_dict() if len(g) else {}
            rows.append({"DTE": d, **blk(g), "contract_used": str(contract_counts)})
        return pd.DataFrame(rows)
    DTE_OLD = dte_summary("OLD (Thu-expiry)"); DTE_NEW = dte_summary("NEW (Tue-expiry)")

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY Close-Direction BTST — FINAL v3: DTE-1 trades REMOVED ENTIRELY (no next-week shift), date window capped at 2026-07-30 entry / 2026-07-31 exit (pre-CAS)"},
        {"metric": "Base logic", "value": "entry CLOSE@15:20, exit CLOSE@09:17, VIX[17,19] excluded, DTE-0 next-week rule unchanged"},
        {"metric": "VIX filter", "value": f"EXCLUDE trades with India VIX (at 15:20 entry) in [{int(VIX_LO)},{int(VIX_HI)}] -> {n_removed} trades removed from {pre_n} unfiltered ({pre_tot} pts, {pre_win}% win)"},
        {"metric": "Structure", "value": f"RED: +2 ATM PE / -1 (ATM-{OFFSET}) PE ; GREEN: +2 ATM CE / -1 (ATM+{OFFSET}) CE ; 1 trade/day"},
        {"metric": "DTE-0 rule", "value": "unchanged — expiry-day entries use next week"},
        {"metric": "DTE-1 rule (NEW)", "value": f"DTE==1 entries SKIPPED ENTIRELY (no trade) — {dte1_skipped} days skipped for this reason"},
        {"metric": "Date window cap (NEW)", "value": f"entries capped at {DATE_CUTOFF_ENTRY.date()} (last trade: entry 2026-07-30, exit 2026-07-31) — excludes Aug-2026 CAS period"},
        {"metric": "Regime boundary", "value": f"OLD (Thu-expiry) through last Thu expiry; NEW (Tue-expiry) from {regime_start_new.date()} onward"},
        {"metric": "P&L unit", "value": "OPTION PREMIUM points (2xlong-1xshort). GROSS (no cost)."},
        {"metric": "Window", "value": f"{T.entry_date.min()} .. {T.exit_date.max()}"},
        {"metric": "Total trades", "value": len(T)}, {"metric": "Win rate %", "value": win},
        {"metric": "Total P&L (points)", "value": tot}, {"metric": "Avg P&L / trade", "value": round(T.pnl_points.mean(), 2)},
        {"metric": "Median P&L / trade", "value": round(T.pnl_points.median(), 2)},
        {"metric": "Avg positive P&L", "value": round(T[T.pnl_points > 0].pnl_points.mean(), 2) if (T.pnl_points > 0).any() else 0},
        {"metric": "Avg negative P&L", "value": round(T[T.pnl_points <= 0].pnl_points.mean(), 2) if (T.pnl_points <= 0).any() else 0},
        {"metric": "Max profit / Max loss", "value": f"{round(T.pnl_points.max(),1)} / {round(T.pnl_points.min(),1)}"},
        {"metric": "Avg holding period (hours)", "value": round(T.holding_hours.mean(), 2)},
        {"metric": "RED / GREEN trades", "value": f"{len(red)} / {len(grn)}"},
        {"metric": "RED avg / GREEN avg P&L", "value": f"{round(red.pnl_points.mean(),2)} / {round(grn.pnl_points.mean(),2)}"},
        {"metric": "Normal-day trades / avg P&L", "value": f"{len(nrm)} / {round(nrm.pnl_points.mean(),2) if len(nrm) else 0}"},
        {"metric": "Expiry-day (next-week) trades / avg P&L", "value": f"{len(exp)} / {round(exp.pnl_points.mean(),2) if len(exp) else 0}"},
        {"metric": "DTE breakdown (2-6 only, DTE-1 excluded)", "value": str({d: int((T.DTE_original == d).sum()) for d in range(2, 7)})},
        {"metric": "", "value": ""},
        {"metric": "--- DRAWDOWN (realized cumulative P&L, premium points) ---", "value": ""},
        {"metric": "Number of drawdown episodes", "value": len(DE)}, {"metric": "Max drawdown (points)", "value": round(maxdd, 1)},
        {"metric": "Avg drawdown (points)", "value": avgdd}, {"metric": "Max DD duration (days, peak->recovery)", "value": maxddp},
        {"metric": "Avg DD duration (days)", "value": round(DE.days_peak_to_recovery.mean(), 1) if len(DE) else 0},
        {"metric": "Return / Max-DD", "value": round(tot / maxdd, 2) if maxdd else "-"},
        {"metric": "Skipped (missing leg data)", "value": skipped},
        {"metric": "", "value": ""},
        {"metric": "--- BEFORE (v2) vs AFTER (v3) ---", "value": ""},
    ])

    OUTF = OUTDIR / "btst_close_direction_FINAL_v3.xlsx"
    with pd.ExcelWriter(OUTF, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        r0 = len(summary) + 1
        CMP.to_excel(w, sheet_name="Summary", index=False, startrow=r0)

        TL = T[["entry_date", "entry_time", "entry_vix", "direction", "contract", "expiry_used", "is_expiry_day", "dte1_shift_applied",
                "ATM", "long_leg", "short_leg", "long_entry", "short_entry", "entry_cost", "exit_date", "exit_time",
                "long_exit", "short_exit", "exit_value", "pnl_points", "month"]]
        TL.to_excel(w, sheet_name="Trades", index=False)
        VIX.to_excel(w, sheet_name="VIX_Summary", index=False)
        DTE_OLD.to_excel(w, sheet_name="DTE_Summary_OldRegime", index=False)
        DTE_NEW.to_excel(w, sheet_name="DTE_Summary_NewRegime", index=False)
        MON.to_excel(w, sheet_name="Monthly_PnL", index=False)
        DE.sort_values("drawdown_points", ascending=False).to_excel(w, sheet_name="Drawdowns", index=False)
        BY.to_excel(w, sheet_name="By_Group", index=False)
    T.to_csv(OUTDIR / "btst_close_direction_FINAL_v3_trades.csv", index=False)

    pd.set_option("display.width", 230)
    print("=" * 96 + "\nNIFTY BTST FINAL v3 (DTE-1 removed, capped Jul-2026)\n" + "=" * 96)
    print(f"unfiltered: {pre_n} trades, {pre_tot} pts, {pre_win}% win | VIX[17,19] removed {n_removed} | DTE-1 skipped {dte1_skipped}")
    print(f"window {T.entry_date.min()} .. {T.exit_date.max()} | trades {len(T)} | skipped {skipped}")
    print(f"last trade: entry {T.entry_date.max()} -> exit {T.exit_date.max()}")
    print(f"win {win}% | TOTAL {tot:,} pts | avg {round(T.pnl_points.mean(),2)} | median {round(T.pnl_points.median(),2)}")
    print(f"DD: episodes {len(DE)} | avg {avgdd} | MAX {round(maxdd,1)} | max period {maxddp}d")
    print("\n--- BY GROUP ---"); print(BY.to_string(index=False))
    print("\n--- BEFORE (v2) vs AFTER (v3) ---"); print(CMP.to_string(index=False))
    print(f"\nSaved -> {OUTF}")


if __name__ == "__main__":
    main()
