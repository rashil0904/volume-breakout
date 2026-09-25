# -*- coding: utf-8 -*-
"""nifty_btst_dte1_included_variant.py — DIAGNOSTIC variant, NOT a change to any locked FINAL file.
Reuses FINAL v3's exact entry/exit/structure/VIX-filter logic (entry CLOSE@15:20, exit CLOSE@09:17,
2x-ATM/1x-offset-300 structure, DTE-0 next-week rule, VIX[17,19] exclusion) but with NO DTE-1 special
handling at all: DTE-1 entries use the CURRENT-WEEK contract, exactly like DTE 2-6 -- neither v2's
next-week shift nor v3's skip-entirely rule. Requested explicitly for a month-wise comparison against
the Mar2022-Sep2024 spot-only proxy (also run with VIX filter only, DTE-1 included). Output goes to its
own new folder; nothing under results/btst_close_direction_FINAL/ is touched.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "btst_dte1_included_variant"; OUTDIR.mkdir(parents=True, exist_ok=True)
OFFSET = 300; ENTRY_MOD = 15 * 60 + 20; EXIT_MOD = 9 * 60 + 17
VIXF = rb.BASE / "data" / "india_vix_1min.csv"; VIX_LO, VIX_HI = 17.0, 19.0
DATE_CUTOFF_ENTRY = pd.Timestamp("2026-07-30")   # match FINAL v2/v3's pre-CAS cap for a clean comparison window


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
    day_open = sp.groupby("date")["open"].first(); spot1520 = sp[sp["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vx = vx.assign(mod=vts.dt.hour * 60 + vts.dt.minute, date=vts.dt.normalize())
    vix_ent = vx[vx["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    tdays = sorted(day_open.index)
    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")); expset = set(expiries); first_exp = expiries[0]

    trades = []; skipped = 0
    for i in range(len(tdays) - 1):
        D = tdays[i]; Dn = tdays[i + 1]
        if D < first_exp or Dn > spot_end or D not in spot1520.index: continue
        if D > DATE_CUTOFF_ENTRY: continue
        j = bisect.bisect_right(expiries, D)
        if j >= len(expiries): continue
        E = expiries[j]; is_exp_day = D in expset            # DTE-0 rule unchanged: strictly-after-D expiry
        dte0 = (E.normalize() - pd.Timestamp(D).normalize()).days
        # NO DTE-1 special-casing here -- DTE-1 uses the SAME current-week contract E as any other DTE

        espot = float(spot1520.loc[D]); dopen = float(day_open.loc[D]); atm = round(espot / 50) * 50
        direction = "RED" if espot < dopen else "GREEN"; ot = "PE" if direction == "RED" else "CE"
        long_K = atm; short_K = atm - OFFSET if direction == "RED" else atm + OFFSET
        t_en = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=20); t_ex = pd.Timestamp(Dn) + pd.Timedelta(hours=9, minutes=17)
        folder = E.strftime("%Y%m%d")
        Le, Lx = leg_at(folder, long_K, ot, t_en, t_ex); Se, Sx = leg_at(folder, short_K, ot, t_en, t_ex)
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (Le, Lx, Se, Sx)):
            skipped += 1; continue
        entry_cost = 2 * Le - Se; exit_val = 2 * Lx - Sx; pnl = exit_val - entry_cost

        trades.append({"entry_date": D.date(), "entry_vix": round(float(vix_ent.get(D, np.nan)), 2), "direction": direction,
                       "expiry_used": E.date(), "is_expiry_day": is_exp_day, "DTE": dte0,
                       "contract": "next-week (expiry-day)" if is_exp_day else "current-week (incl. DTE-1)",
                       "exit_date": Dn.date(), "pnl_points": round(float(pnl), 2)})

    T_all = pd.DataFrame(trades)
    T_all.to_csv(OUTDIR / "nifty_btst_dte1_included_UNFILTERED_trades.csv", index=False)
    fmask = (T_all.entry_vix >= VIX_LO) & (T_all.entry_vix <= VIX_HI)
    n_removed = int(fmask.sum()); T = T_all[~fmask].reset_index(drop=True)

    win = round((T.pnl_points > 0).mean() * 100, 2); tot = round(T.pnl_points.sum(), 1)
    n_dte1 = int((T.DTE == 1).sum())
    print(f"total (unfiltered): {len(T_all)} | VIX-removed: {n_removed} | FINAL (VIX-filtered, DTE-1 INCLUDED as current-week): {len(T)}")
    print(f"DTE-1 trades included (current-week contract): {n_dte1}")
    print(f"win rate: {win}% | total pnl: {tot} pts | skipped(missing leg data): {skipped}")

    T.to_csv(OUTDIR / "nifty_btst_dte1_included_variant_trades.csv", index=False)
    print(f"\nSaved -> {OUTDIR}/nifty_btst_dte1_included_variant_trades.csv")


if __name__ == "__main__":
    main()
