# -*- coding: utf-8 -*-
"""nifty_credit_spread_3dhigh_pcs_invalidation_sl.py — ANALYSIS-ONLY test of a CAUSAL spot-based stop-loss
on the 31 3-day-high-breakout PCS trades: if NIFTY spot closes back BELOW the 3-day-high level that triggered
entry (d3h — known at entry time, not future info) at any point after entry, exit immediately at that minute's
spread value instead of riding to DTE-0 settlement. Otherwise, exit rules are unchanged (90% target still takes
priority if it fires first). This directly targets the deep-dive finding: 100% of the 13 DTE0-settlement exits
in this subset were losers (-1546.0 total) because spot reversed back down after the breakout.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
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
    spot_close_ser = sp.set_index("ts")["close"].sort_index()

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
            continue

        entry_time = pd.Timestamp(ets[trig_i]); entry_spot = cl[trig_i]; atm = round(entry_spot / 50) * 50
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
        both = both[both.index > entry_time]; sv_full = both["s"].values; lv_full = both["l"].values
        spread_val_full = sv_full - lv_full; btimes_full = both.index.values
        thr = TARGET_FRAC * net_credit

        # ---- ORIGINAL exit (90% target else DTE0 settlement) ----
        hit = np.where(spread_val_full <= thr)[0]
        if len(hit):
            j0 = hit[0]; orig_exit_time = pd.Timestamp(btimes_full[j0]); orig_short = float(sv_full[j0]); orig_long = float(lv_full[j0]); orig_reason = "90% target"
        else:
            S = settle.get(e_cur, np.nan)
            if np.isnan(S): S = daily.loc[e_cur, "c"] if e_cur in daily.index else np.nan
            if np.isnan(S): continue
            orig_short = float(max(0.0, sK - S)); orig_long = float(max(0.0, lK - S))
            orig_exit_time = t_end; orig_reason = "DTE0 settlement"
        orig_pnl = net_credit - (orig_short - orig_long)

        # ---- WITH SL: exit if spot CLOSES back below d3h (breakout invalidated) at any 1-min bar after entry, or 90% target, whichever first ----
        spot_path = spot_close_ser[(spot_close_ser.index > entry_time) & (spot_close_ser.index <= orig_exit_time)]
        inval_times = spot_path[spot_path < d3h].index
        sl_exit_time = None; sl_reason = None; sl_short = None; sl_long = None
        if len(hit) and pd.Timestamp(btimes_full[hit[0]]) <= (inval_times[0] if len(inval_times) else pd.Timestamp.max):
            sl_exit_time = orig_exit_time; sl_short = orig_short; sl_long = orig_long; sl_reason = "90% target"
        elif len(inval_times):
            inv_t = inval_times[0]
            idx = both.index.searchsorted(inv_t, side="left")
            if idx >= len(both): idx = len(both) - 1
            row_t = both.index[idx]
            sl_exit_time = row_t; sl_short = float(both["s"].iloc[idx]); sl_long = float(both["l"].iloc[idx]); sl_reason = "SL: breakout invalidated"
        else:
            sl_exit_time = orig_exit_time; sl_short = orig_short; sl_long = orig_long; sl_reason = orig_reason
        sl_pnl = net_credit - (sl_short - sl_long)

        trades.append({"entry_date": entry_day.date(), "DTE": dte, "d3h": round(float(d3h), 1), "entry_spot": round(float(entry_spot), 1),
                       "net_credit": round(net_credit, 2),
                       "orig_exit_date": pd.Timestamp(orig_exit_time).date(), "orig_reason": orig_reason, "orig_pnl": round(orig_pnl, 2),
                       "sl_exit_date": pd.Timestamp(sl_exit_time).date(), "sl_reason": sl_reason, "sl_pnl": round(sl_pnl, 2),
                       "delta": round(sl_pnl - orig_pnl, 2)})

    T = pd.DataFrame(trades)
    T["oos"] = pd.to_datetime(T["entry_date"]) >= OOS_SPLIT
    print(f"reproduced: {len(T)} (expect 31)", flush=True)

    def blk(df, lbl, col):
        p = df[col]
        return {"group": lbl, "trades": len(p), "win_%": round((p > 0).mean() * 100, 1) if len(p) else 0,
                "total_pnl": round(p.sum(), 1) if len(p) else 0, "avg_pnl": round(p.mean(), 2) if len(p) else 0}

    OVERALL = pd.DataFrame([blk(T, "ORIGINAL (ride to settlement)", "orig_pnl"), blk(T, "WITH SL (exit on breakout invalidation)", "sl_pnl")])
    IS = T[~T.oos]; OOS = T[T.oos]
    ISOOS = pd.DataFrame([
        {"period": "IS", **{f"orig_{k}": v for k, v in blk(IS, "", "orig_pnl").items() if k != "group"}, **{f"sl_{k}": v for k, v in blk(IS, "", "sl_pnl").items() if k != "group"}},
        {"period": "OOS", **{f"orig_{k}": v for k, v in blk(OOS, "", "orig_pnl").items() if k != "group"}, **{f"sl_{k}": v for k, v in blk(OOS, "", "sl_pnl").items() if k != "group"}},
    ])

    reason_shift = T.sl_reason.value_counts().reset_index(); reason_shift.columns = ["sl_exit_reason", "count"]

    # strategy-level combined impact: replace the 31-trade baseline sub-total (-424.4) with the SL sub-total, within the 92-trade v2 total (2213.7)
    v2_total = 2213.7
    combined_with_sl = round(v2_total - T.orig_pnl.sum() + T.sl_pnl.sum(), 1)

    with pd.ExcelWriter(OUTDIR / "3dhigh_pcs_invalidation_sl.xlsx", engine="openpyxl") as w:
        OVERALL.to_excel(w, sheet_name="Overall", index=False)
        ISOOS.to_excel(w, sheet_name="IS_OOS", index=False)
        reason_shift.to_excel(w, sheet_name="SL_Exit_Reasons", index=False)
        T.to_excel(w, sheet_name="Trades", index=False)
    T.to_csv(OUTDIR / "3dhigh_pcs_invalidation_sl_trades.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 100 + "\n3-DAY-HIGH PCS: CAUSAL SPOT-INVALIDATION SL TEST\n" + "=" * 100)
    print(OVERALL.to_string(index=False))
    print("\n--- IS/OOS ---"); print(ISOOS.to_string(index=False))
    print("\n--- SL exit-reason mix ---"); print(reason_shift.to_string(index=False))
    print(f"\nCombined 92-trade strategy total: baseline {v2_total} -> with SL applied to this subset: {combined_with_sl} (delta {round(combined_with_sl - v2_total,1)})")
    print(f"\nSaved -> {OUTDIR}/3dhigh_pcs_invalidation_sl.xlsx")


if __name__ == "__main__":
    main()
