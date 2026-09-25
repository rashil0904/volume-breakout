# -*- coding: utf-8 -*-
"""nifty_credit_spread_fallback_time_sweep_narrow.py — NIFTY Weekly Credit Spread: fallback CHECK-TIME sweep
14:30-14:40 (1-min steps, 11 values), explicitly tracking the breakout/fallback trigger-mix shift as T
increases. For each entry day, the FIRST 3-day-high/low breach minute (if any, scanned through the max T of
14:40) is found ONCE; for a given T, the day resolves to BREAKOUT if that breach minute <= T, else FALLBACK
at T's open price. This correctly captures days that "switch" from fallback (at T=14:30) to breakout (at a
later T) as the breach minute falls within the newly-extended window. Base strategy (WIDTH=200, 90% target,
DTE-0 settlement, no SL) otherwise unchanged. Baseline for reference: 92 trades, 70.7% win, 1971.8 pts @T=14:30.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "weekly_credit_spread_diagnostic"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIDTH = 200; TARGET_FRAC = 0.10
T_VALUES = list(range(14 * 60 + 30, 14 * 60 + 41))          # 14:30 .. 14:40 inclusive (11)
BASELINE_T = 14 * 60 + 30
OOS_SPLIT = pd.Timestamp("2025-08-29")


def leg_full(folder, strike, otype, t0, t1):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{strike}_{otype}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    return o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts")["close"].sort_index()


def price_and_exit(folder, e_cur, entry_time, atm, typ, daily, settle):
    ot = "CE" if typ == "CCS" else "PE"
    if typ == "CCS": sK, lK = atm, atm + WIDTH
    else: sK, lK = atm, atm - WIDTH
    t_end = e_cur + pd.Timedelta(hours=15, minutes=30)
    sser = leg_full(folder, sK, ot, entry_time, t_end); lser = leg_full(folder, lK, ot, entry_time, t_end)
    if sser is None or lser is None or sser.empty or lser.empty: return None
    try: s0 = sser.asof(entry_time); l0 = lser.asof(entry_time)
    except Exception: return None
    if np.isnan(s0) or np.isnan(l0): return None
    net_credit = float(s0 - l0)
    if net_credit <= 0: return None
    both = pd.concat([sser.rename("s"), lser.rename("l")], axis=1).ffill().dropna(); both = both[both.index > entry_time]
    sv = both["s"].values; lv = both["l"].values; spread_val = sv - lv; bt = both.index.values
    thr = TARGET_FRAC * net_credit; hit = np.where(spread_val <= thr)[0]
    if len(hit):
        j = hit[0]; short_x = float(sv[j]); long_x = float(lv[j]); reason = "90% target"
    else:
        S = settle.get(e_cur, np.nan)
        if np.isnan(S): S = daily.loc[e_cur, "c"] if e_cur in daily.index else np.nan
        if np.isnan(S): return None
        if typ == "CCS": short_x = float(max(0.0, S - sK)); long_x = float(max(0.0, S - lK))
        else: short_x = float(max(0.0, sK - S)); long_x = float(max(0.0, lK - S))
        reason = "DTE0 settlement"
    pnl = net_credit - (short_x - long_x)
    return {"net_credit": round(net_credit, 2), "exit_reason": reason, "pnl_points": round(pnl, 2)}


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    daily = sp.groupby("date").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    settle = sp[(sp["mod"] >= 900) & (sp["mod"] <= 930)].groupby("date")["close"].mean()
    tdays = list(daily.index)
    expiries = pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")

    weeks = []
    for i in range(1, len(expiries)):
        e_prev, e_cur = expiries[i - 1], expiries[i]
        after = [d for d in tdays if d > e_prev]
        if not after: continue
        entry_day = after[0]
        if entry_day > spot_end or e_cur > spot_end: continue
        prev3 = [d for d in tdays if d < entry_day][-3:]
        if len(prev3) < 3: continue
        d3h = daily.loc[prev3, "h"].max(); d3l = daily.loc[prev3, "l"].min()
        ed = sp[sp["date"] == entry_day]
        max_scan = ed[ed["mod"] <= T_VALUES[-1]]
        hi = max_scan["high"].values; lo = max_scan["low"].values; cl = max_scan["close"].values; op = max_scan["open"].values
        md = max_scan["mod"].values; ets = max_scan["ts"].values
        d_open = ed["open"].iloc[0] if len(ed) else None
        if d_open is None or len(max_scan) == 0: continue

        breach_min = None; breach_typ = None; breach_label = None; breach_i = None
        for k in range(len(max_scan)):
            bh = hi[k] >= d3h; bl = lo[k] <= d3l
            if bh or bl:
                if bh and bl: btyp, blab = ("PCS", "3d-high") if abs(d_open - d3h) <= abs(d_open - d3l) else ("CCS", "3d-low")
                elif bh: btyp, blab = "PCS", "3d-high"
                else: btyp, blab = "CCS", "3d-low"
                breach_min = int(md[k]); breach_typ = btyp; breach_label = blab; breach_i = k
                break
        weeks.append({"entry_day": entry_day, "e_cur": e_cur, "d3h": d3h, "d3l": d3l, "d_open": d_open,
                      "breach_min": breach_min, "breach_typ": breach_typ, "breach_label": breach_label,
                      "breach_ts": pd.Timestamp(ets[breach_i]) if breach_i is not None else None,
                      "breach_close": float(cl[breach_i]) if breach_i is not None else None,
                      "ed_full": max_scan})

    per_T_results = {}; per_T_trades = {}
    for T in T_VALUES:
        rows = []
        for w in weeks:
            entry_day = w["entry_day"]; e_cur = w["e_cur"]
            if w["breach_min"] is not None and w["breach_min"] <= T:
                typ = w["breach_typ"]; label = w["breach_label"]; entry_time = w["breach_ts"]; entry_spot = w["breach_close"]; trig_kind = "breakout"
            else:
                ed = w["ed_full"]; frow = ed[ed["mod"] == T]
                if frow.empty: continue
                px = float(frow["open"].iloc[0]); entry_spot = px; entry_time = pd.Timestamp(frow["ts"].iloc[0])
                typ = "CCS" if px < w["d_open"] else "PCS"; label = f"fallback-{T//60}:{T%60:02d}"; trig_kind = "fallback"
            atm = round(entry_spot / 50) * 50
            folder = e_cur.strftime("%Y%m%d")
            res = price_and_exit(folder, e_cur, entry_time, atm, typ, daily, settle)
            if res is None: continue
            rows.append({"entry_date": entry_day.date(), "T": T, "trig_kind": trig_kind, "trigger": label, "type": typ,
                        "entry_time": entry_time.strftime("%H:%M"), "ATM": int(atm), **res})
        Tdf = pd.DataFrame(rows)
        per_T_trades[T] = Tdf
        n_bo = int((Tdf.trig_kind == "breakout").sum()) if len(Tdf) else 0; n_fb = int((Tdf.trig_kind == "fallback").sum()) if len(Tdf) else 0
        per_T_results[T] = {"T": f"{T//60}:{T%60:02d}", "trades": len(Tdf), "win_%": round((Tdf.pnl_points > 0).mean() * 100, 1) if len(Tdf) else 0,
                            "total_pnl": round(Tdf.pnl_points.sum(), 1) if len(Tdf) else 0, "avg_pnl": round(Tdf.pnl_points.mean(), 2) if len(Tdf) else 0,
                            "breakout_n": n_bo, "fallback_n": n_fb, "breakout_%": round(n_bo / len(Tdf) * 100, 1) if len(Tdf) else 0,
                            "fallback_%": round(n_fb / len(Tdf) * 100, 1) if len(Tdf) else 0}

    SUM = pd.DataFrame([per_T_results[T] for T in T_VALUES])
    baseline = per_T_trades[BASELINE_T].set_index("entry_date")

    # switched-day detection: compare each T's trigger to the BASELINE (T=14:30) trigger for the same entry_date
    switch_rows = []
    for T in T_VALUES:
        if T == BASELINE_T: continue
        cur = per_T_trades[T].set_index("entry_date")
        common = baseline.index.intersection(cur.index)
        for d in common:
            if baseline.loc[d, "trig_kind"] == "fallback" and cur.loc[d, "trig_kind"] == "breakout":
                switch_rows.append({"entry_date": d, "switched_at_T": f"{T//60}:{T%60:02d}",
                                    "baseline_trigger": baseline.loc[d, "trigger"], "baseline_type": baseline.loc[d, "type"], "baseline_pnl": baseline.loc[d, "pnl_points"],
                                    "new_trigger": cur.loc[d, "trigger"], "new_type": cur.loc[d, "type"], "new_pnl": cur.loc[d, "pnl_points"],
                                    "pnl_delta": round(cur.loc[d, "pnl_points"] - baseline.loc[d, "pnl_points"], 2)})
    SWITCH = pd.DataFrame(switch_rows).drop_duplicates(subset=["entry_date"], keep="first").sort_values("entry_date") if switch_rows else pd.DataFrame(columns=["entry_date", "switched_at_T", "baseline_trigger", "baseline_type", "baseline_pnl", "new_trigger", "new_type", "new_pnl", "pnl_delta"])

    # best stable region (3-neighbour mean) on total_pnl
    v = SUM["total_pnl"].values; nb = np.array([np.nanmean(v[max(0, k - 1):k + 2]) for k in range(len(v))])
    bi = int(np.nanargmax(nb)); region = [SUM.T.iloc[x] if False else SUM["T"].iloc[x] for x in range(max(0, bi - 1), min(len(v), bi + 2))]

    # IS/OOS per T
    isoos_rows = []
    for T in T_VALUES:
        df = per_T_trades[T]
        if not len(df): continue
        df2 = df.copy(); df2["oos"] = pd.to_datetime(df2.entry_date) >= OOS_SPLIT
        isoos_rows.append({"T": f"{T//60}:{T%60:02d}", "IS_trades": int((~df2.oos).sum()), "IS_total_pnl": round(df2[~df2.oos].pnl_points.sum(), 1),
                           "OOS_trades": int(df2.oos.sum()), "OOS_total_pnl": round(df2[df2.oos].pnl_points.sum(), 1)})
    ISOOS = pd.DataFrame(isoos_rows)

    with pd.ExcelWriter(OUTDIR / "fallback_time_sweep_narrow.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"metric": "Sweep", "value": "fallback check-time 14:30-14:40 (1-min steps, 11 values)"},
            {"metric": "Baseline (T=14:30)", "value": "92 trades, 70.7% win, 1971.8 pts (confirmed reference)"},
            {"metric": "Breakout-detection rule", "value": "breach minute found ONCE per week (scanned through max T=14:40); day = breakout if breach_min<=T else fallback at T"},
            {"metric": "Best stable region (3-T nbhd)", "value": f"{region} (center {SUM['T'].iloc[bi]})"},
            {"metric": "NOTE", "value": "population is NOT constant across T - see Switched_Days sheet. Not a clean apples-to-apples comparison."},
        ]).to_excel(w, sheet_name="Summary", index=False)
        SUM.to_excel(w, sheet_name="Summary", index=False, startrow=7)
        SWITCH.to_excel(w, sheet_name="Switched_Days", index=False)
        ISOOS.to_excel(w, sheet_name="IS_OOS", index=False)
        for T in T_VALUES:
            per_T_trades[T].to_excel(w, sheet_name=f"Trades_{T//60}_{T%60:02d}", index=False)

    pd.set_option("display.width", 240)
    print("=" * 100 + "\nNIFTY CREDIT SPREAD — FALLBACK CHECK-TIME SWEEP 14:30-14:40\n" + "=" * 100)
    print(SUM.to_string(index=False))
    print(f"\nBest stable region: {region} (center {SUM['T'].iloc[bi]})")
    print(f"\n--- SWITCHED DAYS (fallback@14:30 -> breakout at a later T) ---")
    print(SWITCH.to_string(index=False) if len(SWITCH) else "  NONE")
    print("\n--- IS/OOS per T ---"); print(ISOOS.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}/fallback_time_sweep_narrow.xlsx")


if __name__ == "__main__":
    main()
