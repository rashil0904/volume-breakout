# -*- coding: utf-8 -*-
"""nifty_credit_spread_3dhigh_pcs_deepdive.py — ANALYSIS-ONLY deep-dive diagnostic on the 31 underperforming
3-day-high-breakout Put Credit Spread trades from the NIFTY Weekly Credit Spread v2 (2:35pm fallback) strategy
(baseline: 58.1% win, -424.4 total pts). Reproduces those 31 trades with full enrichment (entry VIX, breach
strength/timing, MFE/MAE path, spot path entry->exit, week-of-month, DTE) to look for a data-justified filter
or exit-rule tweak. Read-only — does not change the live strategy.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
VIXF = rb.BASE / "data" / "india_vix_1min.csv"
OUTDIR = rb.RESULTS / "weekly_credit_spread_diagnostic"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIDTH = 200; TARGET_FRAC = 0.10; FB_MOD = 14 * 60 + 35
OOS_SPLIT = pd.Timestamp("2025-08-29")


def leg_full(folder, strike, otype, t0, t1):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{strike}_{otype}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    return o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts")["close"].sort_index()


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    daily = sp.groupby("date").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    settle = sp[(sp["mod"] >= 900) & (sp["mod"] <= 930)].groupby("date")["close"].mean()
    tdays = list(daily.index)
    expiries = pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")

    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vser = vx.assign(ts=vts)[["ts", "close"]].dropna().sort_values("ts").set_index("ts")["close"]

    trades = []
    for i in range(1, len(expiries)):
        e_prev, e_cur = expiries[i - 1], expiries[i]
        after = [d for d in tdays if d > e_prev]
        if not after: continue
        entry_day = after[0]
        if entry_day > spot_end or e_cur > spot_end: continue
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

        if not (typ == "PCS" and label == "3d-high"):
            continue   # only care about the 31-trade subset

        entry_time = pd.Timestamp(ets[trig_i]); entry_spot = cl[trig_i]; atm = round(entry_spot / 50) * 50
        breach_strength_pct = round((entry_spot - d3h) / d3h * 100, 3)
        breach_min_from_open = int(md[trig_i] - 555)   # minutes after 09:15

        folder = e_cur.strftime("%Y%m%d")
        sK, lK, ot = atm, atm - WIDTH, "PE"
        t_end = e_cur + pd.Timedelta(hours=15, minutes=30)
        sser = leg_full(folder, sK, ot, entry_time, t_end); lser = leg_full(folder, lK, ot, entry_time, t_end)
        if sser is None or lser is None or sser.empty or lser.empty: continue
        try: s0 = sser.asof(entry_time); l0 = lser.asof(entry_time)
        except Exception: continue
        if np.isnan(s0) or np.isnan(l0): continue
        net_credit = float(s0 - l0)
        if net_credit <= 0: continue

        both = pd.concat([sser.rename("s"), lser.rename("l")], axis=1).ffill().dropna()
        both = both[both.index > entry_time]; sv = both["s"].values; lv = both["l"].values; spread_val = sv - lv; btimes = both.index.values
        thr = TARGET_FRAC * net_credit; hit = np.where(spread_val <= thr)[0]
        if len(hit):
            j = hit[0]; exit_time = pd.Timestamp(btimes[j]); short_x = float(sv[j]); long_x = float(lv[j]); reason = "90% target"
            path_to_exit = spread_val[:j + 1]
        else:
            S = settle.get(e_cur, np.nan)
            if np.isnan(S): S = daily.loc[e_cur, "c"] if e_cur in daily.index else np.nan
            if np.isnan(S): continue
            short_x = float(max(0.0, sK - S)); long_x = float(max(0.0, lK - S))
            exit_time = t_end; reason = "DTE0 settlement"; path_to_exit = spread_val

        exit_debit = short_x - long_x; pnl = net_credit - exit_debit
        pnl_path = net_credit - path_to_exit
        mfe = float(pnl_path.max()) if len(pnl_path) else pnl
        mae = float(pnl_path.min()) if len(pnl_path) else pnl
        mfe_time_idx = int(np.argmax(pnl_path)) if len(pnl_path) else 0
        mfe_reached_before_end = mfe_time_idx < (len(pnl_path) - 1)

        sp_win = sp[(sp["ts"] > entry_time) & (sp["ts"] <= exit_time)]
        worst_adverse_pct = round(((entry_spot - sp_win["low"].min()) / entry_spot * 100), 3) if len(sp_win) else np.nan
        exit_spot = None
        if reason == "DTE0 settlement":
            exit_spot = settle.get(e_cur, np.nan)
        else:
            exit_spot = sp[sp["ts"] <= exit_time]["close"].iloc[-1] if len(sp[sp["ts"] <= exit_time]) else np.nan
        spot_move_pct = round((exit_spot - entry_spot) / entry_spot * 100, 3) if pd.notna(exit_spot) else np.nan

        entry_vix = vser.asof(entry_time)
        week_of_month = (entry_day.day - 1) // 7 + 1

        trades.append({"entry_date": entry_day.date(), "day_of_week": entry_day.day_name(), "DTE": dte, "week_of_month": week_of_month,
                       "entry_time": entry_time.strftime("%H:%M"), "breach_min_from_open": breach_min_from_open,
                       "entry_vix": round(float(entry_vix), 2) if pd.notna(entry_vix) else np.nan,
                       "d3h": round(float(d3h), 1), "entry_spot": round(float(entry_spot), 1), "breach_strength_pct": breach_strength_pct,
                       "ATM": int(atm), "short_K": int(sK), "long_K": int(lK), "net_credit": round(net_credit, 2),
                       "exit_date": pd.Timestamp(exit_time).date(), "exit_reason": reason, "days_held": (pd.Timestamp(exit_time).normalize() - entry_day).days,
                       "pnl_points": round(pnl, 2), "mfe": round(mfe, 2), "mae": round(mae, 2),
                       "mfe_reached_before_final_exit": mfe_reached_before_end,
                       "worst_adverse_spot_pct": worst_adverse_pct, "exit_spot": round(float(exit_spot), 1) if pd.notna(exit_spot) else np.nan,
                       "spot_move_pct_entry_to_exit": spot_move_pct})

    T = pd.DataFrame(trades)
    T["oos"] = pd.to_datetime(T["entry_date"]) >= OOS_SPLIT
    print(f"reproduced 3d-high PCS trades: {len(T)} (expect 31)", flush=True)

    def blk(df, lbl):
        return {"group": lbl, "trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_points.sum(), 1) if len(df) else 0, "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0}

    # ===== CUT 1: IS vs OOS =====
    C1 = pd.DataFrame([blk(T, "ALL"), blk(T[~T.oos], "IS (<2025-08-29)"), blk(T[T.oos], "OOS (>=2025-08-29)")])

    # ===== CUT 2: day of week =====
    C2 = pd.DataFrame([blk(T[T.day_of_week == d], d) for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"] if (T.day_of_week == d).any()])

    # ===== CUT 3: DTE bucket =====
    def dte_bucket(x): return "1-2" if x <= 2 else ("3-4" if x <= 4 else "5-7")
    T["dte_bucket"] = T.DTE.map(dte_bucket)
    C3 = pd.DataFrame([blk(T[T.dte_bucket == b], b) for b in ["1-2", "3-4", "5-7"] if (T.dte_bucket == b).any()])

    # ===== CUT 4: VIX bucket =====
    def vbucket(v):
        if pd.isna(v): return "n/a"
        return "<13" if v < 13 else ("13-15" if v < 15 else ("15-17" if v < 17 else ">=17"))
    T["vix_bucket"] = T.entry_vix.map(vbucket)
    C4 = pd.DataFrame([blk(T[T.vix_bucket == b], b) for b in ["<13", "13-15", "15-17", ">=17"] if (T.vix_bucket == b).any()])

    # ===== CUT 5: breach strength (how far above 3d-high) =====
    med_bs = T.breach_strength_pct.median()
    T["breach_strength_grp"] = np.where(T.breach_strength_pct <= med_bs, "weak (<=median)", "strong (>median)")
    C5 = pd.DataFrame([blk(T[T.breach_strength_grp == g], g) for g in ["weak (<=median)", "strong (>median)"]])

    # ===== CUT 6: breach timing (early vs late in the day) =====
    med_t = T.breach_min_from_open.median()
    T["breach_timing_grp"] = np.where(T.breach_min_from_open <= med_t, "early (<=median)", "late (>median)")
    C6 = pd.DataFrame([blk(T[T.breach_timing_grp == g], g) for g in ["early (<=median)", "late (>median)"]])

    # ===== CUT 7: exit reason =====
    C7 = pd.DataFrame([blk(T[T.exit_reason == r], r) for r in ["90% target", "DTE0 settlement"] if (T.exit_reason == r).any()])

    # ===== CUT 8: MFE-based — trades that were profitable at some point (MFE>0) but lost overall =====
    T["had_positive_mfe"] = T.mfe > 0
    C8 = pd.DataFrame([blk(T[T.had_positive_mfe], "had MFE>0 (profitable at some point)"),
                        blk(T[~T.had_positive_mfe], "never profitable (MFE<=0)")])
    gave_back = T[(T.had_positive_mfe) & (T.pnl_points <= 0)]

    # ===== CUT 9: spot reversal magnitude (entry->exit) — did NIFTY fall back after the breakout? =====
    T["reversed_down"] = T.spot_move_pct_entry_to_exit < 0
    C9 = pd.DataFrame([blk(T[T.reversed_down], "spot fell after breakout"), blk(T[~T.reversed_down], "spot held/rose after breakout")])

    # ===== CUT 10: week of month =====
    C10 = pd.DataFrame([blk(T[T.week_of_month == w], f"week {w}") for w in sorted(T.week_of_month.unique())])

    # ===== candidate rule tests =====
    def rule_impact(mask, name):
        kept = T[~mask]; removed = T[mask]
        return {"rule": name, "trades_removed": len(removed), "removed_pnl": round(removed.pnl_points.sum(), 1),
                "remaining_trades": len(kept), "remaining_pnl": round(kept.pnl_points.sum(), 1),
                "remaining_win_%": round((kept.pnl_points > 0).mean() * 100, 1) if len(kept) else 0,
                "net_impact_vs_baseline": round(kept.pnl_points.sum() - T.pnl_points.sum(), 1)}
    candidates = []
    # A: spot-based SL — exit if spot falls back below 3-day-high (breakout invalidated) at any point
    T["breach_invalidated"] = T.worst_adverse_spot_pct >= (T.entry_spot - T.d3h) / T.entry_spot * 100  # spot round-tripped back through d3h
    candidates.append(rule_impact(T.breach_invalidated & (T.pnl_points < 0), "Hypothetical: cut losers where spot round-tripped back below 3d-high (perfect-foresight upper bound)"))
    # B: skip weak breaches
    candidates.append(rule_impact(T.breach_strength_grp == "weak (<=median)", "Exclude weak breaches (<=median strength)"))
    # C: skip late-day breaches
    candidates.append(rule_impact(T.breach_timing_grp == "late (>median)", "Exclude late breaches (>median minute)"))
    # D: skip OOS-like VIX
    candidates.append(rule_impact(T.vix_bucket == "<13", "Exclude VIX <13"))
    # E: lock in MFE — hypothetical profit-take at MFE (upper bound only, for reference)
    mfe_upper_bound = round(T.mfe.sum(), 1)
    CAND = pd.DataFrame(candidates)

    with pd.ExcelWriter(OUTDIR / "3dhigh_pcs_deepdive.xlsx", engine="openpyxl") as w:
        pd.DataFrame([{"metric": "Baseline (all 31)", "trades": len(T), "win_%": round((T.pnl_points>0).mean()*100,1), "total_pnl": round(T.pnl_points.sum(),1), "avg_pnl": round(T.pnl_points.mean(),2)},
                      {"metric": "Sum of MFE (perfect profit-take upper bound, for reference only)", "trades": "-", "win_%": "-", "total_pnl": mfe_upper_bound, "avg_pnl": round(mfe_upper_bound/len(T),2)}]).to_excel(w, sheet_name="Overview", index=False)
        C1.to_excel(w, sheet_name="Cut1_IS_OOS", index=False)
        C2.to_excel(w, sheet_name="Cut2_DayOfWeek", index=False)
        C3.to_excel(w, sheet_name="Cut3_DTE", index=False)
        C4.to_excel(w, sheet_name="Cut4_VIX", index=False)
        C5.to_excel(w, sheet_name="Cut5_BreachStrength", index=False)
        C6.to_excel(w, sheet_name="Cut6_BreachTiming", index=False)
        C7.to_excel(w, sheet_name="Cut7_ExitReason", index=False)
        C8.to_excel(w, sheet_name="Cut8_MFE_GaveBack", index=False)
        gave_back.to_excel(w, sheet_name="Cut8_GaveBack_Trades", index=False)
        C9.to_excel(w, sheet_name="Cut9_SpotReversal", index=False)
        C10.to_excel(w, sheet_name="Cut10_WeekOfMonth", index=False)
        CAND.to_excel(w, sheet_name="Candidate_Rules", index=False)
        T.to_excel(w, sheet_name="Full_Trade_Data", index=False)
    T.to_csv(OUTDIR / "3dhigh_pcs_deepdive_trades.csv", index=False)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 100 + "\n3-DAY-HIGH PCS DEEP-DIVE (31 trades)\n" + "=" * 100)
    for name, df in [("Cut1 IS/OOS", C1), ("Cut2 DayOfWeek", C2), ("Cut3 DTE", C3), ("Cut4 VIX", C4),
                      ("Cut5 BreachStrength", C5), ("Cut6 BreachTiming", C6), ("Cut7 ExitReason", C7),
                      ("Cut8 MFE-GaveBack", C8), ("Cut9 SpotReversal", C9), ("Cut10 WeekOfMonth", C10)]:
        print(f"\n--- {name} ---"); print(df.to_string(index=False))
    print(f"\n--- Trades that had MFE>0 but ended a loser ({len(gave_back)}) ---")
    print(gave_back[["entry_date","net_credit","mfe","mae","pnl_points","exit_reason"]].to_string(index=False))
    print("\n--- Candidate rules ---"); print(CAND.to_string(index=False))
    print(f"\nSum of MFE (upper bound if perfectly profit-taken): {mfe_upper_bound}")
    print(f"\nSaved -> {OUTDIR}/3dhigh_pcs_deepdive.xlsx")


if __name__ == "__main__":
    main()
