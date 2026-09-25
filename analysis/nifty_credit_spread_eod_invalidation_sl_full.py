# -*- coding: utf-8 -*-
"""nifty_credit_spread_eod_invalidation_sl_full.py — ANALYSIS-ONLY test of an EOD-close invalidation stop
applied across ALL 92 trades of the NIFTY Weekly Credit Spread v2 (2:35pm fallback) strategy, not just the
3d-high PCS subset. Each trade gets a "reference level" from its own entry thesis:
  - 3d-high breakout (PCS): reference = d3h (the 3-day high that was broken)
  - 3d-low breakout (CCS):  reference = d3l (the 3-day low that was broken)
  - fallback-2:35 (CCS or PCS): reference = day's open (the level that determined red/green direction)
From entry day onward, checked at each day's ~15:29 close (NOT intraday 1-min — avoids same-day noise):
  - CCS invalidated if EOD close > reference (price recovered back above the bearish thesis level)
  - PCS invalidated if EOD close < reference (price fell back below the bullish thesis level)
On invalidation, exit at that day's ~15:29 option spread value. Otherwise unchanged: 90% target still exits
first if it fires before invalidation; else DTE-0 settlement as before. Width/entry logic unchanged.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
BASE_TRADES_CSV = rb.RESULTS / "weekly_credit_spread" / "weekly_credit_spread_v2_235fallback_trades.csv"
OUTDIR = rb.RESULTS / "weekly_credit_spread_diagnostic"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIDTH = 200; TARGET_FRAC = 0.10; FB_MOD = 14 * 60 + 35
OOS_SPLIT = pd.Timestamp("2025-08-29")


def leg_full(folder, strike, otype, t0, t1):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{strike}_{otype}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    return o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts")["close"].sort_index()


def drawdown_episodes(equity, times):
    peak = np.maximum.accumulate(equity); dd = equity - peak; eps = []; in_dd = False; s = t = 0; tv = 0.0
    for k in range(len(equity)):
        if not in_dd:
            if dd[k] < -1e-9: in_dd = True; s = k - 1; t = k; tv = dd[k]
        else:
            if dd[k] < tv: tv = dd[k]; t = k
            if dd[k] >= -1e-9: eps.append((s, t, k, tv)); in_dd = False
    if in_dd: eps.append((s, t, len(equity) - 1, tv))
    return pd.DataFrame([{"drawdown_points": round(abs(v), 1)} for (_, _, _, v) in eps])


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    daily = sp.groupby("date").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    settle = sp[(sp["mod"] >= 900) & (sp["mod"] <= 930)].groupby("date")["close"].mean()
    tdays = list(daily.index)
    expiries = pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")
    eod_close = sp[sp["mod"] >= 929].groupby("date")["close"].last()

    trades = []; skipped = []
    for i in range(1, len(expiries)):
        e_prev, e_cur = expiries[i - 1], expiries[i]
        after = [d for d in tdays if d > e_prev]
        if not after: continue
        entry_day = after[0]
        if entry_day > spot_end or e_cur > spot_end:
            skipped.append((e_cur.date(), "outside spot window")); continue
        dte = (e_cur - entry_day).days
        prev3 = [d for d in tdays if d < entry_day][-3:]
        if len(prev3) < 3: continue
        d3h = daily.loc[prev3, "h"].max(); d3l = daily.loc[prev3, "l"].min()

        ed = sp[sp["date"] == entry_day]
        hi = ed["high"].values; lo = ed["low"].values; cl = ed["close"].values; op = ed["open"].values; md = ed["mod"].values; ets = ed["ts"].values
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
            trig_i = ed.index.get_loc(fb.index[0]); px = op[trig_i]
            typ = "CCS" if px < d_open else "PCS"; label = "fallback-2:35"
        entry_time = pd.Timestamp(ets[trig_i])
        entry_spot = op[trig_i] if label == "fallback-2:35" else cl[trig_i]
        atm = round(entry_spot / 50) * 50

        # ---- reference level for invalidation, per trigger type ----
        if label == "3d-high": ref_level = d3h
        elif label == "3d-low": ref_level = d3l
        else: ref_level = d_open   # fallback-2:35 (either CCS or PCS)

        folder = e_cur.strftime("%Y%m%d")
        if typ == "CCS": sK, lK, ot = atm, atm + WIDTH, "CE"
        else: sK, lK, ot = atm, atm - WIDTH, "PE"
        t_end = e_cur + pd.Timedelta(hours=15, minutes=30)
        sser = leg_full(folder, sK, ot, entry_time, t_end); lser = leg_full(folder, lK, ot, entry_time, t_end)
        if sser is None or lser is None or sser.empty or lser.empty:
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
        target_time = pd.Timestamp(btimes[hit[0]]) if len(hit) else None

        # ---- EOD-close invalidation scan, entry day onward ----
        days_between = [d for d in tdays if entry_day <= d <= e_cur]
        inval_day = None
        for d in days_between:
            c = eod_close.get(d, np.nan)
            if pd.isna(c): continue
            broke = (c > ref_level) if typ == "CCS" else (c < ref_level)
            if broke:
                inval_day = d; break

        if target_time is not None and (inval_day is None or target_time.normalize() <= inval_day):
            short_x, long_x, exit_time, reason = float(sv[hit[0]]), float(lv[hit[0]]), target_time, "90% target"
        elif inval_day is not None:
            eod_ts = inval_day + pd.Timedelta(hours=15, minutes=29)
            idx = both.index.searchsorted(eod_ts, side="left")
            if idx >= len(both): idx = len(both) - 1
            short_x, long_x, exit_time, reason = float(both["s"].iloc[idx]), float(both["l"].iloc[idx]), both.index[idx], "SL: EOD invalidated"
        else:
            S = settle.get(e_cur, np.nan)
            if np.isnan(S): S = daily.loc[e_cur, "c"] if e_cur in daily.index else np.nan
            if np.isnan(S):
                skipped.append((e_cur.date(), "no settlement spot")); continue
            if typ == "CCS": short_x, long_x = float(max(0.0, S - sK)), float(max(0.0, S - lK))
            else: short_x, long_x = float(max(0.0, sK - S)), float(max(0.0, lK - S))
            exit_time, reason = t_end, "DTE0 settlement"

        exit_debit = short_x - long_x; pnl = net_credit - exit_debit
        trades.append({"entry_date": entry_day.date(), "DTE": dte, "type": ("Call Credit Spread" if typ == "CCS" else "Put Credit Spread"),
                       "trigger": label, "entry_time": entry_time.strftime("%Y-%m-%d %H:%M"), "ATM": int(atm),
                       "net_credit": round(net_credit, 2), "exit_date": pd.Timestamp(exit_time).date(),
                       "exit_time": pd.Timestamp(exit_time).strftime("%Y-%m-%d %H:%M"), "exit_reason": reason,
                       "pnl_points": round(pnl, 2)})

    T = pd.DataFrame(trades)
    win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)

    Bfull = pd.read_csv(BASE_TRADES_CSV); Bfull["entry_date"] = pd.to_datetime(Bfull["entry_date"])
    base_win = round((Bfull.pnl_points > 0).mean() * 100, 1); base_tot = round(Bfull.pnl_points.sum(), 1)

    # drawdowns (chronological by exit)
    def dd_stats(df):
        d = df.sort_values("exit_time").reset_index(drop=True); eq = np.concatenate([[0.0], d["pnl_points"].cumsum().values])
        times = np.concatenate([[pd.to_datetime(d["entry_time"]).iloc[0]], pd.to_datetime(d["exit_time"]).values])
        DE = drawdown_episodes(eq, times)
        return (round(DE.drawdown_points.max(), 1) if len(DE) else 0.0), (round(DE.drawdown_points.mean(), 1) if len(DE) else 0.0), len(DE)

    T["oos"] = pd.to_datetime(T["entry_date"]) >= OOS_SPLIT
    Bfull["oos"] = Bfull["entry_date"] >= OOS_SPLIT
    maxdd_v, avgdd_v, nep_v = dd_stats(T)
    Bfull_ren = Bfull.rename(columns={"entry_time": "entry_time", "exit_time": "exit_time"})
    maxdd_b, avgdd_b, nep_b = dd_stats(Bfull_ren)

    def blk(df, lbl):
        return {"group": lbl, "trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_points.sum(), 1) if len(df) else 0, "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0}

    OVERALL = pd.DataFrame([
        {"metric": "Total trades", "BASELINE (no SL)": len(Bfull), "WITH EOD-close SL": len(T)},
        {"metric": "Win rate %", "BASELINE (no SL)": base_win, "WITH EOD-close SL": win},
        {"metric": "Total P&L (points)", "BASELINE (no SL)": base_tot, "WITH EOD-close SL": tot},
        {"metric": "Delta total P&L", "BASELINE (no SL)": "-", "WITH EOD-close SL": round(tot - base_tot, 1)},
        {"metric": "Avg P&L / trade", "BASELINE (no SL)": round(Bfull.pnl_points.mean(), 2), "WITH EOD-close SL": round(T.pnl_points.mean(), 2)},
        {"metric": "Max drawdown (pts)", "BASELINE (no SL)": maxdd_b, "WITH EOD-close SL": maxdd_v},
        {"metric": "Avg drawdown (pts)", "BASELINE (no SL)": avgdd_b, "WITH EOD-close SL": avgdd_v},
        {"metric": "# drawdown episodes", "BASELINE (no SL)": nep_b, "WITH EOD-close SL": nep_v},
    ])

    BY_TRIG = pd.DataFrame([blk(T, "ALL")] + [blk(T[T.trigger == g], g) for g in ["3d-high", "3d-low", "fallback-2:35"]])
    BY_TRIG_BASE = pd.DataFrame([blk(Bfull, "ALL")] + [blk(Bfull[Bfull.trigger == g], g) for g in ["3d-high", "3d-low", "fallback-2:35"]])

    ISOOS = pd.DataFrame([
        {"period": "IS", "baseline_trades": len(Bfull[~Bfull.oos]), "baseline_total": round(Bfull[~Bfull.oos].pnl_points.sum(), 1),
         "variant_trades": len(T[~T.oos]), "variant_total": round(T[~T.oos].pnl_points.sum(), 1)},
        {"period": "OOS", "baseline_trades": len(Bfull[Bfull.oos]), "baseline_total": round(Bfull[Bfull.oos].pnl_points.sum(), 1),
         "variant_trades": len(T[T.oos]), "variant_total": round(T[T.oos].pnl_points.sum(), 1)},
    ])

    reason_mix = T.exit_reason.value_counts().reset_index(); reason_mix.columns = ["exit_reason", "count"]

    with pd.ExcelWriter(OUTDIR / "eod_invalidation_sl_full.xlsx", engine="openpyxl") as w:
        OVERALL.to_excel(w, sheet_name="Overall", index=False)
        BY_TRIG.to_excel(w, sheet_name="By_Trigger_WithSL", index=False)
        BY_TRIG_BASE.to_excel(w, sheet_name="By_Trigger_Baseline", index=False)
        ISOOS.to_excel(w, sheet_name="IS_OOS", index=False)
        reason_mix.to_excel(w, sheet_name="Exit_Reason_Mix", index=False)
        T.to_excel(w, sheet_name="Trades", index=False)
        if skipped: pd.DataFrame(skipped, columns=["expiry", "reason"]).to_excel(w, sheet_name="Skipped", index=False)
    T.to_csv(OUTDIR / "eod_invalidation_sl_full_trades.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 100 + "\nEOD-CLOSE INVALIDATION SL -- ALL 92 TRADES\n" + "=" * 100)
    print("\n--- Overall ---"); print(OVERALL.to_string(index=False))
    print("\n--- By trigger (WITH SL) ---"); print(BY_TRIG.to_string(index=False))
    print("\n--- By trigger (BASELINE) ---"); print(BY_TRIG_BASE.to_string(index=False))
    print("\n--- IS/OOS ---"); print(ISOOS.to_string(index=False))
    print("\n--- Exit reason mix ---"); print(reason_mix.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}/eod_invalidation_sl_full.xlsx")


if __name__ == "__main__":
    main()
