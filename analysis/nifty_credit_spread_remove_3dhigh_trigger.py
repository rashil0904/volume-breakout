# -*- coding: utf-8 -*-
"""nifty_credit_spread_remove_3dhigh_trigger.py — ANALYSIS-ONLY variant test on the NIFTY Weekly Credit
Spread v2 (2:35pm fallback) strategy: REMOVE the 3-day-high breakout trigger entirely. Days that would have
fired 3-day-high (touch-basis) instead fall through to the standard 2:35pm fallback (red/green vs day's open
-> CCS/PCS). 3-day-low breakout trigger, exit rules (90% target / DTE-0 settlement), spread width (200pts),
and the 2:35pm fallback timing itself are all UNCHANGED. If 3-day-low also breaches before 2:35pm on a day
that would have fired 3-day-high, the 3-day-low breakout still takes priority (this falls out naturally from
only checking the low-breach condition in the trigger loop).

BASELINE (confirmed, from weekly_credit_spread_v2_235fallback_trades.csv): 92 trades, 72.8% win, 2213.7 total
pts. Of the 42 PCS trades, 31 were 3d-high-triggered (58.1% win, -424.4 total -- the underperforming subset).
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
BASE_TRADES_CSV = rb.RESULTS / "weekly_credit_spread" / "weekly_credit_spread_v2_235fallback_trades.csv"
OUTDIR = rb.RESULTS / "weekly_credit_spread_diagnostic"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIDTH = 200; TARGET_FRAC = 0.10; FB_MOD = 14 * 60 + 35
IS_OOS_BOUNDARY = pd.Timestamp("2025-08-29")


def leg_series(folder, strike, otype, t0, t1):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{strike}_{otype}_*.parquet"))
    if not fs:
        return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts")["close"].sort_index()
    return o if len(o) else None


def stats(df, col="pnl_points"):
    p = df[col]
    return {"trades": len(p), "win_%": round((p > 0).mean() * 100, 1) if len(p) else 0,
            "total_pnl": round(p.sum(), 1) if len(p) else 0, "avg_pnl": round(p.mean(), 2) if len(p) else 0}


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    daily = sp.groupby("date").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    settle = sp[(sp["mod"] >= 900) & (sp["mod"] <= 930)].groupby("date")["close"].mean()
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
        d_open = ed["open"].iloc[0]

        # ---- ORIGINAL trigger (tag only, for comparison — does NOT drive the trade) ----
        orig_trig_i = None; orig_label = None
        for k in range(len(ed)):
            if md[k] > FB_MOD:
                break
            bh = hi[k] >= d3h; bl = lo[k] <= d3l
            if bh or bl:
                if bh and bl: orig_label = "3d-high" if abs(d_open - d3h) <= abs(d_open - d3l) else "3d-low"
                elif bh: orig_label = "3d-high"
                else: orig_label = "3d-low"
                orig_trig_i = k; break
        if orig_trig_i is None:
            orig_label = "fallback-2:35"

        # ---- VARIANT trigger: 3d-high REMOVED, only 3d-low checked; else fallback ----
        trig_i = None; typ = None; label = None
        for k in range(len(ed)):
            if md[k] > FB_MOD:
                break
            if lo[k] <= d3l:
                typ, label = "CCS", "3d-low"; trig_i = k; break
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
        thr = TARGET_FRAC * net_credit; exit_time = None; reason = None
        hit = np.where(spread_val <= thr)[0]
        if len(hit):
            j = hit[0]; exit_time = pd.Timestamp(btimes[j]); short_x = float(sv[j]); long_x = float(lv[j]); reason = "90% target"
        else:
            S = settle.get(e_cur, np.nan)
            if np.isnan(S): S = daily.loc[e_cur, "c"] if e_cur in daily.index else np.nan
            if np.isnan(S):
                skipped.append((e_cur.date(), "no settlement spot")); continue
            if typ == "CCS": short_x = float(max(0.0, S - sK)); long_x = float(max(0.0, S - lK))
            else: short_x = float(max(0.0, sK - S)); long_x = float(max(0.0, lK - S))
            exit_time = e_cur + pd.Timedelta(hours=15, minutes=30); reason = "DTE0 settlement"
        exit_debit = short_x - long_x; pnl = net_credit - exit_debit
        trades.append({"entry_date": entry_day.date(), "DTE": dte, "type": ("Call Credit Spread" if typ == "CCS" else "Put Credit Spread"),
                       "trigger": label, "original_trigger": orig_label, "entry_time": entry_time.strftime("%Y-%m-%d %H:%M"), "ATM": int(atm),
                       "leg_type": ot, "short_K": int(sK), "long_K": int(lK),
                       "short_leg_entry": round(float(s0), 2), "long_leg_entry": round(float(l0), 2), "net_credit": round(net_credit, 2),
                       "exit_date": pd.Timestamp(exit_time).date(), "exit_time": pd.Timestamp(exit_time).strftime("%Y-%m-%d %H:%M"), "exit_reason": reason,
                       "short_leg_exit": round(short_x, 2), "long_leg_exit": round(long_x, 2), "exit_debit": round(exit_debit, 2),
                       "pnl_points": round(pnl, 2)})

    V = pd.DataFrame(trades)
    win = round((V.pnl_points > 0).mean() * 100, 1); tot = round(V.pnl_points.sum(), 1)

    # ---- baseline reference ----
    Bfull = pd.read_csv(BASE_TRADES_CSV); Bfull["entry_date"] = pd.to_datetime(Bfull["entry_date"])
    base_win = round((Bfull.pnl_points > 0).mean() * 100, 1); base_tot = round(Bfull.pnl_points.sum(), 1)
    base_by_date = Bfull.set_index("entry_date")

    affected_dates = V[V.original_trigger == "3d-high"]["entry_date"].apply(pd.Timestamp)
    V_affected = V[V.original_trigger == "3d-high"].copy()
    B_affected = base_by_date.loc[base_by_date.index.isin(affected_dates)]

    print(f"total variant trades: {len(V)} | skipped {len(skipped)}", flush=True)
    print(f"affected (former 3d-high) days found in variant run: {len(V_affected)} | in baseline: {len(B_affected)}", flush=True)

    # ---- what did the affected days BECOME under the variant? ----
    became = V_affected.groupby(["trigger", "type"]).size().reset_index(name="count")

    # ---- overall comparison ----
    OVERALL = pd.DataFrame([
        {"metric": "Total trades", "BASELINE (3d-high active)": len(Bfull), "VARIANT (3d-high removed)": len(V)},
        {"metric": "Win rate %", "BASELINE (3d-high active)": base_win, "VARIANT (3d-high removed)": win},
        {"metric": "Total P&L (points)", "BASELINE (3d-high active)": base_tot, "VARIANT (3d-high removed)": tot},
        {"metric": "Delta total P&L", "BASELINE (3d-high active)": "-", "VARIANT (3d-high removed)": round(tot - base_tot, 1)},
        {"metric": "Avg P&L / trade", "BASELINE (3d-high active)": round(Bfull.pnl_points.mean(), 2), "VARIANT (3d-high removed)": round(V.pnl_points.mean(), 2)},
    ])

    # ---- affected-subset comparison (baseline PCS 3d-high vs variant fallback outcome) ----
    sB = stats(B_affected); sV = stats(V_affected)
    AFFECTED = pd.DataFrame([
        {"metric": "Trades", "BASELINE (3d-high PCS)": sB["trades"], "VARIANT (fallback instead)": sV["trades"]},
        {"metric": "Win rate %", "BASELINE (3d-high PCS)": sB["win_%"], "VARIANT (fallback instead)": sV["win_%"]},
        {"metric": "Total P&L", "BASELINE (3d-high PCS)": sB["total_pnl"], "VARIANT (fallback instead)": sV["total_pnl"]},
        {"metric": "Avg P&L", "BASELINE (3d-high PCS)": sB["avg_pnl"], "VARIANT (fallback instead)": sV["avg_pnl"]},
        {"metric": "Delta total P&L", "BASELINE (3d-high PCS)": "-", "VARIANT (fallback instead)": round(sV["total_pnl"] - sB["total_pnl"], 1)},
    ])

    # ---- IS/OOS on the affected subset (paired by entry_date) ----
    pair_dates = sorted(set(B_affected.index) & set(pd.to_datetime(V_affected.entry_date)))
    Bp = B_affected.loc[B_affected.index.isin(pair_dates)]
    Vp = V_affected[pd.to_datetime(V_affected.entry_date).isin(pair_dates)]
    Bp_is = Bp[Bp.index < IS_OOS_BOUNDARY]; Bp_oos = Bp[Bp.index >= IS_OOS_BOUNDARY]
    Vp_is = Vp[pd.to_datetime(Vp.entry_date) < IS_OOS_BOUNDARY]; Vp_oos = Vp[pd.to_datetime(Vp.entry_date) >= IS_OOS_BOUNDARY]
    ISOOS = pd.DataFrame([
        {"period": f"IS (< {IS_OOS_BOUNDARY.date()})", "baseline_trades": len(Bp_is), "baseline_total_pnl": round(Bp_is.pnl_points.sum(), 1), "baseline_avg": round(Bp_is.pnl_points.mean(), 2) if len(Bp_is) else None,
         "variant_trades": len(Vp_is), "variant_total_pnl": round(Vp_is.pnl_points.sum(), 1), "variant_avg": round(Vp_is.pnl_points.mean(), 2) if len(Vp_is) else None},
        {"period": f"OOS (>= {IS_OOS_BOUNDARY.date()})", "baseline_trades": len(Bp_oos), "baseline_total_pnl": round(Bp_oos.pnl_points.sum(), 1), "baseline_avg": round(Bp_oos.pnl_points.mean(), 2) if len(Bp_oos) else None,
         "variant_trades": len(Vp_oos), "variant_total_pnl": round(Vp_oos.pnl_points.sum(), 1), "variant_avg": round(Vp_oos.pnl_points.mean(), 2) if len(Vp_oos) else None},
    ])

    with pd.ExcelWriter(OUTDIR / "remove_3dhigh_trigger_variant.xlsx", engine="openpyxl") as w:
        OVERALL.to_excel(w, sheet_name="Overall_Comparison", index=False)
        AFFECTED.to_excel(w, sheet_name="Affected_Subset", index=False)
        became.to_excel(w, sheet_name="Affected_Became", index=False)
        ISOOS.to_excel(w, sheet_name="IS_OOS_Affected", index=False)
        V.to_excel(w, sheet_name="Variant_All_Trades", index=False)
        V_affected.to_excel(w, sheet_name="Variant_Affected_Trades", index=False)
        B_affected.reset_index().to_excel(w, sheet_name="Baseline_Affected_Trades", index=False)
        if skipped: pd.DataFrame(skipped, columns=["expiry", "reason"]).to_excel(w, sheet_name="Skipped", index=False)
    V.to_csv(OUTDIR / "remove_3dhigh_trigger_variant_trades.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 100 + "\nVARIANT: 3-DAY-HIGH TRIGGER REMOVED -> FALLBACK\n" + "=" * 100)
    print("\n--- Overall ---"); print(OVERALL.to_string(index=False))
    print("\n--- Affected (former 3d-high) subset ---"); print(AFFECTED.to_string(index=False))
    print("\n--- What the affected days became ---"); print(became.to_string(index=False))
    print("\n--- IS/OOS on affected subset ---"); print(ISOOS.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}/remove_3dhigh_trigger_variant.xlsx")


if __name__ == "__main__":
    main()
