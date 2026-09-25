# -*- coding: utf-8 -*-
"""nifty_btst_close_direction_FINAL.py — FINAL LOCKED NIFTY Daily Close-Direction BTST. Baseline params
(previously finalized): entry 15:20 daily (RED=spot<open -> 2x ATM PE / -1 (ATM-300) PE ; GREEN=spot>open ->
2x ATM CE / -1 (ATM+300) CE). Exit 09:17 next trading day, fill = that minute's OPTION OPEN. ATM = nearest
50-pt strike from spot at 15:20. GROSS option premium points.

DTE-0 RULE (unchanged): never trade a contract expiring same day -> nearest expiry STRICTLY after entry day
(bisect_right on the real expiry calendar).
NEW DTE-1 RULE (this version): for ANY entry where the selected (DTE-0-compliant) contract is at DTE==1,
shift to the NEXT week's contract instead (one further bisect step). Applied by actual DTE value (not
weekday), so it covers both regimes without hardcoding: OLD (Thu-expiry) DTE=1 -> Wednesday ; NEW (Tue-expiry)
DTE=1 -> Monday. Regime boundary derived from the actual expiry calendar's weekday change point.
DTE 2-6 entries continue using the current-week contract as before (unchanged).
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "btst_close_direction_FINAL"; OUTDIR.mkdir(parents=True, exist_ok=True)
OFFSET = 300; ENTRY_MOD = 15 * 60 + 20; EXIT_MOD = 9 * 60 + 17


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
                     "days_peak_to_recovery": int((pd.Timestamp(times[r]) - pd.Timestamp(times[s])).days)})
    return pd.DataFrame(rows)


def leg_open_close(folder, strike, otype, t_entry, t_exit):
    """entry via CLOSE (asof), exit via OPEN (asof) — matches the established finalized convention."""
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{otype}_*.parquet"))
    if not fs: return None, None
    o = pd.read_parquet(fs[0], columns=["timestamp", "open", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[(o.ts >= t_entry - pd.Timedelta(minutes=5)) & (o.ts <= t_exit + pd.Timedelta(minutes=5))].set_index("ts").sort_index()
    if o.empty: return None, None
    en = o["close"].asof(t_entry); ex = o["open"].asof(t_exit)
    return en, ex


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    day_open = sp.groupby("date")["open"].first(); spot1520 = sp[sp["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    tdays = sorted(day_open.index)
    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")); expset = set(expiries); first_exp = expiries[0]

    # regime boundary derived from the actual calendar (weekday change point among Tue/Thu expiries)
    wd = pd.Series([e.day_name() for e in expiries], index=expiries)
    regime_start_new = None
    for i in range(1, len(expiries)):
        if wd.iloc[i] != wd.iloc[i - 1] and wd.iloc[i] in ("Tuesday", "Thursday"):
            regime_start_new = expiries[i]

    trades = []; skipped = 0
    for i in range(len(tdays) - 1):
        D = tdays[i]; Dn = tdays[i + 1]
        if D < first_exp or Dn > spot_end or D not in spot1520.index: continue
        j = bisect.bisect_right(expiries, D)
        if j >= len(expiries): continue
        E0 = expiries[j]; is_exp_day = D in expset; dte0 = (E0.normalize() - pd.Timestamp(D).normalize()).days

        dte1_shift = False; E = E0
        if dte0 == 1:
            if j + 1 < len(expiries): E = expiries[j + 1]; dte1_shift = True
            else: skipped += 1; continue
        dte_used = (E.normalize() - pd.Timestamp(D).normalize()).days
        reason = "DTE-1 rule (shifted to next week)" if dte1_shift else ("DTE-0 rule (expiry-day, next-week)" if is_exp_day else "normal (current week)")
        regime = "NEW (Tue-expiry)" if E0.day_name() == "Tuesday" else "OLD (Thu-expiry)"

        espot = float(spot1520.loc[D]); dopen = float(day_open.loc[D]); atm = round(espot / 50) * 50
        direction = "RED" if espot < dopen else "GREEN"; ot = "PE" if direction == "RED" else "CE"
        long_K = atm; short_K = atm - OFFSET if direction == "RED" else atm + OFFSET
        t_en = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=20); t_ex = pd.Timestamp(Dn) + pd.Timedelta(hours=9, minutes=17)
        folder = E.strftime("%Y%m%d")
        Le, Lx = leg_open_close(folder, long_K, ot, t_en, t_ex); Se, Sx = leg_open_close(folder, short_K, ot, t_en, t_ex)
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (Le, Lx, Se, Sx)):
            skipped += 1; continue
        entry_cost = 2 * Le - Se; exit_val = 2 * Lx - Sx; pnl = exit_val - entry_cost

        pnl_before_fix = pnl
        if dte1_shift:
            folder0 = E0.strftime("%Y%m%d")
            Le0, Lx0 = leg_open_close(folder0, long_K, ot, t_en, t_ex); Se0, Sx0 = leg_open_close(folder0, short_K, ot, t_en, t_ex)
            if not any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (Le0, Lx0, Se0, Sx0)):
                pnl_before_fix = (2 * Lx0 - Sx0) - (2 * Le0 - Se0)

        trades.append({"entry_date": D, "DTE": int(dte_used), "DTE_original": int(dte0), "regime": regime, "contract_used": "next-week" if (dte1_shift or is_exp_day) else "current-week",
                       "reason": reason, "direction": direction, "ATM": int(atm), "long_leg": f"2x {int(long_K)} {ot}", "short_leg": f"1x {int(short_K)} {ot}",
                       "long_entry": round(float(Le), 2), "short_entry": round(float(Se), 2), "entry_cost": round(float(entry_cost), 2),
                       "exit_date": Dn, "exit_time": "09:17", "long_exit": round(float(Lx), 2), "short_exit": round(float(Sx), 2),
                       "exit_value": round(float(exit_val), 2), "pnl_points": round(float(pnl), 2),
                       "pnl_before_dte1_fix": round(float(pnl_before_fix), 2), "holding_hours": round((t_ex - t_en).total_seconds() / 3600, 2)})

    T = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    T["cum_pnl"] = T["pnl_points"].cumsum().round(2); peak = T["cum_pnl"].cummax(); T["running_drawdown"] = (T["cum_pnl"] - peak).round(2)
    pos = T[T.pnl_points > 0]; neg = T[T.pnl_points < 0]
    win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)

    equity = np.concatenate([[0.0], T["cum_pnl"].values]); dtimes = np.concatenate([[pd.Timestamp(T.entry_date.iloc[0])], pd.to_datetime(T.entry_date).values])
    DE = drawdown_episodes(equity, dtimes); maxdd = DE.drawdown_points.max() if len(DE) else 0.0; avgdd = round(DE.drawdown_points.mean(), 1) if len(DE) else 0.0
    maxddp = int(DE.days_peak_to_recovery.max()) if len(DE) else 0; avgddp = round(DE.days_peak_to_recovery.mean(), 1) if len(DE) else 0

    def blk(df, lbl):
        return {"group": lbl, "trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_points.sum(), 1), "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0,
                "median_pnl": round(df.pnl_points.median(), 2) if len(df) else 0}
    red = T[T.direction == "RED"]; grn = T[T.direction == "GREEN"]
    BY_DIR = pd.DataFrame([blk(T, "ALL"), blk(red, "RED"), blk(grn, "GREEN")])
    BY_DTE = pd.DataFrame([blk(T[T.DTE_original == d], f"DTE {d}" + (" (shifted to next-week)" if d == 1 else "")) for d in range(1, 7) if (T.DTE_original == d).any()])
    BY_REGIME = pd.DataFrame([blk(T[T.regime == r], r) for r in sorted(T.regime.unique())])

    dte1_rows = T[T.reason.str.startswith("DTE-1")]
    before_total = round((T.pnl_points.sum() - dte1_rows.pnl_points.sum() + dte1_rows.pnl_before_dte1_fix.sum()), 1)
    after_total = tot; fix_delta = round(after_total - before_total, 1)
    CMP = pd.DataFrame([
        {"metric": "Total trades", "before(current-week DTE1)": len(T), "after(next-week DTE1, FINAL)": len(T)},
        {"metric": "Total P&L", "before(current-week DTE1)": before_total, "after(next-week DTE1, FINAL)": after_total},
        {"metric": "DTE-1 trades affected", "before(current-week DTE1)": len(dte1_rows), "after(next-week DTE1, FINAL)": len(dte1_rows)},
        {"metric": "DTE-1 subset P&L", "before(current-week DTE1)": round(dte1_rows.pnl_before_dte1_fix.sum(), 1), "after(next-week DTE1, FINAL)": round(dte1_rows.pnl_points.sum(), 1)},
        {"metric": "Delta attributable to DTE-1 fix", "before(current-week DTE1)": "-", "after(next-week DTE1, FINAL)": fix_delta},
    ])

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY Close-Direction BTST — FINAL LOCKED (entry 15:20, exit 09:17 next day OPEN, offset 300) + DTE-1 next-week-contract fix"},
        {"metric": "Structure", "value": f"RED: +2 ATM PE / -1 (ATM-{OFFSET}) PE ; GREEN: +2 ATM CE / -1 (ATM+{OFFSET}) CE ; 1 trade/day"},
        {"metric": "Fills", "value": "entry = option CLOSE @15:20 ; exit = next-day 09:17 option OPEN"},
        {"metric": "DTE-0 rule", "value": "never trade a same-day-expiring contract; expiry-day entries use next week"},
        {"metric": "DTE-1 rule (NEW)", "value": "any DTE-0-compliant contract landing at DTE==1 is shifted ONE FURTHER week; applied by actual DTE value, covers both regimes without hardcoding weekday"},
        {"metric": "Regime boundary (from actual calendar)", "value": f"OLD (Thu-expiry) through the last Thursday expiry; NEW (Tue-expiry) from {regime_start_new.date()} onward" if regime_start_new is not None else "single regime only"},
        {"metric": "P&L unit", "value": "OPTION PREMIUM points (2xlong-1xshort). GROSS (no cost)."},
        {"metric": "Window", "value": f"{T.entry_date.min()} .. {T.exit_date.max()}"},
        {"metric": "Total trades", "value": len(T)}, {"metric": "Winning trades", "value": int((T.pnl_points > 0).sum())},
        {"metric": "Losing trades", "value": int((T.pnl_points < 0).sum())}, {"metric": "Win rate %", "value": win},
        {"metric": "Total P&L (points)", "value": tot}, {"metric": "Avg P&L / trade", "value": round(T.pnl_points.mean(), 2)},
        {"metric": "Median P&L / trade", "value": round(T.pnl_points.median(), 2)},
        {"metric": "Avg positive P&L", "value": round(pos.pnl_points.mean(), 2) if len(pos) else 0},
        {"metric": "Avg negative P&L", "value": round(neg.pnl_points.mean(), 2) if len(neg) else 0},
        {"metric": "Max profit", "value": round(T.pnl_points.max(), 1)}, {"metric": "Max loss", "value": round(T.pnl_points.min(), 1)},
        {"metric": "Average holding period (hours)", "value": round(T.holding_hours.mean(), 2)},
        {"metric": "", "value": ""},
        {"metric": "--- DRAWDOWN (cumulative equity, premium points) ---", "value": ""},
        {"metric": "Number of drawdown episodes", "value": len(DE)}, {"metric": "Average drawdown (points)", "value": avgdd},
        {"metric": "Maximum drawdown (points)", "value": round(maxdd, 1)}, {"metric": "Average drawdown period (days)", "value": avgddp},
        {"metric": "Maximum drawdown period (days)", "value": maxddp}, {"metric": "Return / Max-DD", "value": round(tot / maxdd, 2) if maxdd else "-"},
        {"metric": "Skipped (missing leg data / no next-week available)", "value": skipped},
    ])

    OUTF = OUTDIR / "nifty_btst_close_direction_FINAL.xlsx"
    with pd.ExcelWriter(OUTF, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        r0 = len(summary) + 2
        pd.DataFrame([{"section": "--- BY DIRECTION ---"}]).to_excel(w, sheet_name="Summary", index=False, startrow=r0, header=False)
        BY_DIR.to_excel(w, sheet_name="Summary", index=False, startrow=r0 + 1)
        r1 = r0 + 1 + len(BY_DIR) + 2
        pd.DataFrame([{"section": "--- BY DTE ---"}]).to_excel(w, sheet_name="Summary", index=False, startrow=r1, header=False)
        BY_DTE.to_excel(w, sheet_name="Summary", index=False, startrow=r1 + 1)
        r2 = r1 + 1 + len(BY_DTE) + 2
        pd.DataFrame([{"section": "--- BY REGIME ---"}]).to_excel(w, sheet_name="Summary", index=False, startrow=r2, header=False)
        BY_REGIME.to_excel(w, sheet_name="Summary", index=False, startrow=r2 + 1)
        r3 = r2 + 1 + len(BY_REGIME) + 2
        pd.DataFrame([{"section": "--- BEFORE (current-week DTE1) vs AFTER (next-week DTE1, FINAL) ---"}]).to_excel(w, sheet_name="Summary", index=False, startrow=r3, header=False)
        CMP.to_excel(w, sheet_name="Summary", index=False, startrow=r3 + 1)

        TL = T[["entry_date", "DTE", "DTE_original", "regime", "contract_used", "reason", "direction", "ATM", "long_leg", "short_leg",
                "long_entry", "short_entry", "entry_cost", "exit_date", "exit_time", "long_exit", "short_exit",
                "exit_value", "pnl_points", "cum_pnl", "running_drawdown"]]
        TL.to_excel(w, sheet_name="Trade_Log", index=False)
    T.to_csv(OUTDIR / "nifty_btst_close_direction_FINAL_trades.csv", index=False)

    pd.set_option("display.width", 230)
    print("=" * 96 + "\nNIFTY BTST FINAL (entry 15:20 / exit 09:17 next-day OPEN / offset 300) + DTE-1 next-week fix\n" + "=" * 96)
    print(f"window {T.entry_date.min()} .. {T.exit_date.max()} | trades {len(T)} | skipped {skipped}")
    print(f"win {win}% | TOTAL {tot:,} pts | avg {round(T.pnl_points.mean(),2)} | median {round(T.pnl_points.median(),2)}")
    print(f"DD: episodes {len(DE)} | avg {avgdd} | MAX {round(maxdd,1)} | avg period {avgddp}d | max period {maxddp}d")
    print("\n--- BY DIRECTION ---"); print(BY_DIR.to_string(index=False))
    print("\n--- BY DTE ---"); print(BY_DTE.to_string(index=False))
    print("\n--- BY REGIME ---"); print(BY_REGIME.to_string(index=False))
    print("\n--- BEFORE vs AFTER (DTE-1 fix) ---"); print(CMP.to_string(index=False))
    print(f"\nSaved -> {OUTF}")


if __name__ == "__main__":
    main()
