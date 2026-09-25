# -*- coding: utf-8 -*-
"""nifty_credit_spread_diagnostic.py — ANALYSIS-ONLY diagnostic on the NIFTY Weekly Credit Spread strategy.
Reproduces the exact 92-trade base backtest (nifty_weekly_credit_spread.py logic: WIDTH=200, 90% target,
DTE-0 settlement, no SL) while capturing enrichment fields not in the saved CSV: entry VIX, full spread-value
path (for MFE/MAE + near-miss-to-target), spot path (for adverse-move validation), week-of-month. Cross-
references the already-built SL variant, width sweep, and fallback-time sweep outputs for cuts 5/6/10.
Does not change strategy logic — read-only diagnostic.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "weekly_credit_spread_diagnostic"; OUTDIR.mkdir(parents=True, exist_ok=True)
VIXF = rb.BASE / "data" / "india_vix_1min.csv"
WIDTH = 200; TARGET_FRAC = 0.10; FB_MOD = 14 * 60 + 30
VIX_BUCKETS = ["<9"] + [f"{i}-{i+1}" for i in range(9, 20)] + [">20"]
OOS_SPLIT = pd.Timestamp("2025-08-29")


def vbucket(v):
    if pd.isna(v): return "n/a"
    return "<9" if v < 9 else (">20" if v >= 20 else f"{int(v)}-{int(v)+1}")


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
        hi = ed["high"].values; lo = ed["low"].values; cl = ed["close"].values; md = ed["mod"].values; ets = ed["ts"].values
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
            trig_i = ed.index.get_loc(fb.index[0]); px = cl[trig_i]
            typ = "CCS" if px < d_open else "PCS"; label = "fallback-2:30"
        entry_time = pd.Timestamp(ets[trig_i]); entry_spot = cl[trig_i]; atm = round(entry_spot / 50) * 50

        folder = e_cur.strftime("%Y%m%d")
        if typ == "CCS": sK, lK, ot = atm, atm + WIDTH, "CE"
        else: sK, lK, ot = atm, atm - WIDTH, "PE"
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
            if typ == "CCS": short_x = float(max(0.0, S - sK)); long_x = float(max(0.0, S - lK))
            else: short_x = float(max(0.0, sK - S)); long_x = float(max(0.0, lK - S))
            exit_time = t_end; reason = "DTE0 settlement"; path_to_exit = spread_val

        exit_debit = short_x - long_x; pnl = net_credit - exit_debit
        # MFE/MAE on the SPREAD VALUE (lower spread_val = more profit for the credit seller: profit = credit - spread_val)
        pnl_path = net_credit - path_to_exit
        mfe = float(pnl_path.max()) if len(pnl_path) else pnl  # best (highest) profit point reached
        mae = float(pnl_path.min()) if len(pnl_path) else pnl  # worst (lowest/most negative) point reached
        closest_to_target_frac = float((path_to_exit.min() / net_credit)) if len(path_to_exit) and net_credit else np.nan  # lowest cost-to-close reached, as frac of credit

        # spot path from entry to exit (for adverse-move validation)
        sp_win = sp[(sp["ts"] > entry_time) & (sp["ts"] <= exit_time)]
        if typ == "CCS":  # bearish; adverse = spot UP
            worst_adverse_pct = round(((sp_win["high"].max() - entry_spot) / entry_spot * 100), 3) if len(sp_win) else np.nan
        else:
            worst_adverse_pct = round(((entry_spot - sp_win["low"].min()) / entry_spot * 100), 3) if len(sp_win) else np.nan

        entry_vix = vser.asof(entry_time)
        week_of_month = (entry_day.day - 1) // 7 + 1

        trades.append({"entry_date": entry_day.date(), "DTE": dte, "type": typ, "trigger": label,
                       "trig_kind": "fallback" if label == "fallback-2:30" else "breakout",
                       "entry_vix": round(float(entry_vix), 2) if pd.notna(entry_vix) else np.nan,
                       "net_credit": round(net_credit, 2), "exit_date": pd.Timestamp(exit_time).date(), "exit_reason": reason,
                       "pnl_points": round(pnl, 2), "days_held": (pd.Timestamp(exit_time).normalize() - entry_day).days,
                       "mfe": round(mfe, 2), "mae": round(mae, 2), "closest_to_target_pct_of_credit_remaining": round(closest_to_target_frac * 100, 2) if pd.notna(closest_to_target_frac) else np.nan,
                       "worst_adverse_spot_pct": worst_adverse_pct, "week_of_month": week_of_month})

    T = pd.DataFrame(trades)
    T["vix_bucket"] = T["entry_vix"].map(vbucket)
    T["oos"] = pd.to_datetime(T["entry_date"]) >= OOS_SPLIT
    T.to_csv(OUTDIR / "diagnostic_trades_full.csv", index=False)
    print(f"reproduced trades: {len(T)} (expect 92)", flush=True)

    def blk(df, lbl):
        return {"group": lbl, "trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_points.sum(), 1) if len(df) else 0, "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0,
                "avg_days_held": round(df.days_held.mean(), 2) if len(df) else 0}

    # ===== CUT 1: trigger type =====
    C1 = pd.DataFrame([blk(T, "ALL")] + [blk(T[T.trig_kind == k], k) for k in ["breakout", "fallback"]])

    # ===== CUT 2: DTE distribution =====
    C2 = pd.DataFrame([{"DTE": d, **blk(T[T.DTE == d], str(d))} for d in sorted(T.DTE.unique())]).drop(columns=["group"])

    # ===== CUT 3: exit reason =====
    C3 = pd.DataFrame([blk(T, "ALL")] + [blk(T[T.exit_reason == r], r) for r in ["90% target", "DTE0 settlement"]])

    # ===== CUT 4: MFE/MAE / near-miss =====
    tgt = T[T.exit_reason == "90% target"]; settle_ = T[T.exit_reason == "DTE0 settlement"]
    near_miss = settle_[settle_.closest_to_target_pct_of_credit_remaining <= 20]  # got within 20% of credit remaining (i.e. close to the 10% target) but never crossed
    C4 = pd.DataFrame([
        {"metric": "Avg MFE (all trades)", "value": round(T.mfe.mean(), 2)},
        {"metric": "Avg MAE (all trades)", "value": round(T.mae.mean(), 2)},
        {"metric": "Settlement-exit trades: avg closest-approach to target (%cost remaining, lower=closer)", "value": round(settle_.closest_to_target_pct_of_credit_remaining.mean(), 2) if len(settle_) else 0},
        {"metric": "Settlement trades within 20% of target threshold (near-miss)", "value": f"{len(near_miss)} / {len(settle_)}"},
        {"metric": "Settlement trades within 15% of target threshold", "value": int((settle_.closest_to_target_pct_of_credit_remaining <= 15).sum())},
        {"metric": "Settlement trades within 25% of target threshold", "value": int((settle_.closest_to_target_pct_of_credit_remaining <= 25).sum())},
    ])
    C4_detail = settle_[["entry_date", "net_credit", "closest_to_target_pct_of_credit_remaining", "pnl_points"]].sort_values("closest_to_target_pct_of_credit_remaining")

    # ===== CUT 5: spot move / SL validation =====
    L = T[T.pnl_points < 0]
    SLF = pd.read_csv(rb.RESULTS / "weekly_credit_spread" / "credit_spread_spot_sl_trades.csv")
    C5 = pd.DataFrame([
        {"metric": "Losing trades (no-SL base)", "value": len(L)},
        {"metric": "Avg worst adverse spot move % (losers)", "value": round(L.worst_adverse_spot_pct.mean(), 3) if len(L) else 0},
        {"metric": "Losers with adverse move >= 2.5%", "value": int((L.worst_adverse_spot_pct >= 2.5).sum())},
        {"metric": "Losers with adverse move < 2.5% (still lost anyway)", "value": int((L.worst_adverse_spot_pct < 2.5).sum())},
        {"metric": "Trades touching >=2.5% adverse but recovering to a WIN by settlement", "value": int(((T.worst_adverse_spot_pct >= 2.5) & (T.pnl_points > 0)).sum())},
        {"metric": "SL file (2.01% threshold) total P&L, base vs SL", "value": f"base={round(SLF.base_pnl.sum(),1)} / SL={round(SLF.sl_pnl.sum(),1)}"},
    ])
    C5_detail = T[T.worst_adverse_spot_pct >= 2.5][["entry_date", "type", "worst_adverse_spot_pct", "exit_reason", "pnl_points"]].sort_values("worst_adverse_spot_pct", ascending=False)

    # ===== CUT 6: width x DTE cross-check =====
    WS = pd.read_csv(rb.RESULTS / "weekly_credit_spread" / "credit_spread_width_sweep_trades.csv")
    WS["dte_bucket"] = pd.cut(WS.DTE, [-1, 2, 4, 10], labels=["low(0-2)", "mid(3-4)", "high(5+)"])
    C6 = WS.groupby(["width", "dte_bucket"], observed=True).agg(trades=("pnl_points", "size"), total_pnl=("pnl_points", "sum"), avg_pnl=("pnl_points", "mean")).round(2).reset_index()

    # ===== CUT 7: VIX buckets =====
    present = [b for b in VIX_BUCKETS if (T.vix_bucket == b).any()]
    C7 = pd.DataFrame([{"vix_bucket": b, **blk(T[T.vix_bucket == b], b)} for b in present]).drop(columns=["group"])

    # ===== CUT 8: CCS vs PCS =====
    C8 = pd.DataFrame([blk(T, "ALL"), blk(T[T.type == "CCS"], "CCS (bearish)"), blk(T[T.type == "PCS"], "PCS (bullish)")])

    # ===== CUT 9: week of month =====
    C9 = pd.DataFrame([{"week_of_month": w, **blk(T[T.week_of_month == w], str(w))} for w in sorted(T.week_of_month.unique())]).drop(columns=["group"])

    # ===== CUT 10: fallback-time sweep cross-check =====
    FB = pd.read_csv(rb.RESULTS.parent / "results" / "credit_spread_fallback_sweep" / "fallback_time_sweep.csv") if (rb.RESULTS.parent / "results" / "credit_spread_fallback_sweep" / "fallback_time_sweep.csv").exists() else pd.read_csv(rb.RESULTS / "credit_spread_fallback_sweep" / "fallback_time_sweep.csv")
    best_fb = FB.sort_values("total_pnl", ascending=False).head(5)

    # ===== improvement candidates with IS/OOS =====
    def rule_impact(mask_bad, label):
        keep = T[~mask_bad]; bad = T[mask_bad]
        isb = bad[~bad.oos]; oosb = bad[bad.oos]
        return {"rule": label, "trades_removed": len(bad), "IS_removed_pnl": round(isb.pnl_points.sum(), 1), "OOS_removed_pnl": round(oosb.pnl_points.sum(), 1),
                "new_total_if_applied": round(keep.pnl_points.sum(), 1), "delta_vs_baseline": round(keep.pnl_points.sum() - T.pnl_points.sum(), 1)}
    candidates = []
    if (T.vix_bucket == "16-17").any(): candidates.append(rule_impact(T.vix_bucket.isin(["16-17"]), "Exclude VIX 16-17"))
    candidates.append(rule_impact(T.trig_kind == "fallback", "Exclude fallback-triggered trades"))
    candidates.append(rule_impact(T.DTE <= 2, "Exclude DTE<=2 entries"))
    CAND = pd.DataFrame(candidates)

    with pd.ExcelWriter(OUTDIR / "nifty_credit_spread_diagnostic.xlsx", engine="openpyxl") as w:
        pd.DataFrame([{"metric": "Total trades (reproduced)", "value": len(T)}]).to_excel(w, sheet_name="Scope", index=False)
        C1.to_excel(w, sheet_name="Cut1_TriggerType", index=False)
        C2.to_excel(w, sheet_name="Cut2_DTE_Dist", index=False)
        C3.to_excel(w, sheet_name="Cut3_ExitReason", index=False)
        C4.to_excel(w, sheet_name="Cut4_MFE_MAE", index=False); C4_detail.to_excel(w, sheet_name="Cut4_MFE_MAE", index=False, startrow=len(C4) + 2)
        C5.to_excel(w, sheet_name="Cut5_SpotMove_SL", index=False); C5_detail.to_excel(w, sheet_name="Cut5_SpotMove_SL", index=False, startrow=len(C5) + 2)
        C6.to_excel(w, sheet_name="Cut6_Width_x_DTE", index=False)
        C7.to_excel(w, sheet_name="Cut7_VIX", index=False)
        C8.to_excel(w, sheet_name="Cut8_CCS_vs_PCS", index=False)
        C9.to_excel(w, sheet_name="Cut9_WeekOfMonth", index=False)
        best_fb.to_excel(w, sheet_name="Cut10_FallbackSweep", index=False)
        CAND.to_excel(w, sheet_name="Improvement_Candidates", index=False)
        T.to_excel(w, sheet_name="Full_Trade_Data", index=False)

    pd.set_option("display.width", 230)
    print("\n--- CUT 1: Trigger type ---"); print(C1.to_string(index=False))
    print("\n--- CUT 2: DTE distribution ---"); print(C2.to_string(index=False))
    print("\n--- CUT 3: Exit reason ---"); print(C3.to_string(index=False))
    print("\n--- CUT 4: MFE/MAE ---"); print(C4.to_string(index=False))
    print("\n--- CUT 5: Spot move / SL ---"); print(C5.to_string(index=False))
    print("\n--- CUT 6: Width x DTE (head) ---"); print(C6.head(15).to_string(index=False))
    print("\n--- CUT 7: VIX ---"); print(C7.to_string(index=False))
    print("\n--- CUT 8: CCS vs PCS ---"); print(C8.to_string(index=False))
    print("\n--- CUT 9: Week of month ---"); print(C9.to_string(index=False))
    print("\n--- CUT 10: Best fallback times ---"); print(best_fb.to_string(index=False))
    print("\n--- Improvement candidates ---"); print(CAND.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
