# -*- coding: utf-8 -*-
"""nifty_credit_spread_spot_sl.py — ADDITIVE spot-based 2.01% adverse stop-loss layered on the NIFTY Weekly
Credit Spread. Entry/breakout/ATM/width and the existing exits (90% target, DTE-0 settlement) are UNCHANGED;
this only adds a 3rd exit path and reports before/after.

SL RULE (touch-based on NIFTY SPOT, entry->expiry):
  reference = spot at the moment of entry (trigger-minute price = same basis used for ATM). *** FLAG: using the
  trade's ENTRY spot as the 2.01% base, not day-open. ***
  CCS/bearish (adverse=UP)   -> SL when spot HIGH >= entry_spot*1.0201
  PCS/bullish (adverse=DOWN)  -> SL when spot LOW  <= entry_spot*0.9799
FILL at the SL minute (deliberately worst-case for the short):
  SOLD ATM leg  -> bought back at THAT MINUTE'S HIGH (most expensive close-out)
  HEDGE leg     -> closed at THAT MINUTE'S CLOSE (normal reference)
PRIORITY: earliest of {SL touch, 90% target, DTE-0 settlement} wins. SL vs target are mutually exclusive states
(2.01% adverse vs spread-decayed-to-10%); any same-minute co-fire is flagged.
GROSS premium points, 1 spread. Baseline (no-SL) P&L is computed in the same pass for exact before/after.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "weekly_credit_spread"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIDTH = 200; TARGET_FRAC = 0.10; FB_MOD = 14 * 60 + 30; SL_ADVERSE = 0.0201


def leg_df(folder, strike, otype, t0, t1, cols):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{strike}_{otype}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp"] + cols); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts")[cols].sort_index()
    return o if len(o) else None


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    daily = sp.groupby("date").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    settle = sp[(sp["mod"] >= 900) & (sp["mod"] <= 930)].groupby("date")["close"].mean()
    tdays = list(daily.index)
    expiries = pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")

    trades = []; skipped = []; conflicts = 0
    for i in range(1, len(expiries)):
        e_prev, e_cur = expiries[i - 1], expiries[i]
        after = [d for d in tdays if d > e_prev]
        if not after: continue
        entry_day = after[0]
        if entry_day > spot_end or e_cur > spot_end: skipped.append((e_cur.date(), "outside spot window")); continue
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
        entry_time = pd.Timestamp(ets[trig_i]); entry_spot = float(cl[trig_i]); atm = round(entry_spot / 50) * 50
        trig_kind = "fallback" if label == "fallback-2:30" else "breakout"
        folder = e_cur.strftime("%Y%m%d"); ot = "CE" if typ == "CCS" else "PE"
        sK, lK = (atm, atm + WIDTH) if typ == "CCS" else (atm, atm - WIDTH)
        t_end = e_cur + pd.Timedelta(hours=15, minutes=30)
        sdf = leg_df(folder, sK, ot, entry_time, t_end, ["close", "high"])
        ldf = leg_df(folder, lK, ot, entry_time, t_end, ["close"])
        if sdf is None or ldf is None: skipped.append((e_cur.date(), f"missing leg {sK}/{lK}")); continue
        sser = sdf["close"]; sser_hi = sdf["high"]; lser = ldf["close"]
        try: s0 = float(sser.asof(entry_time)); l0 = float(lser.asof(entry_time))
        except Exception: skipped.append((e_cur.date(), "no entry premium")); continue
        if np.isnan(s0) or np.isnan(l0): skipped.append((e_cur.date(), "nan entry premium")); continue
        net_credit = s0 - l0
        if net_credit <= 0: skipped.append((e_cur.date(), f"non-pos credit {round(net_credit,1)}")); continue

        # existing target series (per-min spread value)
        both = pd.concat([sser.rename("s"), lser.rename("l")], axis=1).ffill().dropna(); both = both[both.index > entry_time]
        sv = both["s"].values; lv = both["l"].values; spread_val = sv - lv; bt = both.index.values
        thr = TARGET_FRAC * net_credit; hit = np.where(spread_val <= thr)[0]
        target_time = pd.Timestamp(bt[hit[0]]) if len(hit) else None
        tgt_short = float(sv[hit[0]]) if len(hit) else None; tgt_long = float(lv[hit[0]]) if len(hit) else None

        # DTE-0 settlement intrinsic (shared by both versions when neither target nor SL fires)
        S = settle.get(e_cur, np.nan)
        if np.isnan(S): S = daily.loc[e_cur, "c"] if e_cur in daily.index else np.nan
        def settle_pnl():
            if np.isnan(S): return None
            if typ == "CCS": sx, lx = max(0.0, S - sK), max(0.0, S - lK)
            else: sx, lx = max(0.0, sK - S), max(0.0, lK - S)
            return net_credit - (sx - lx)

        # ---- SPOT SL detection (touch, entry->expiry) ----
        sw = sp[(sp["ts"] > entry_time) & (sp["ts"] <= t_end)]
        if typ == "CCS":
            level = entry_spot * (1 + SL_ADVERSE); sl_hits = np.where(sw["high"].values >= level)[0]
        else:
            level = entry_spot * (1 - SL_ADVERSE); sl_hits = np.where(sw["low"].values <= level)[0]
        sl_time = pd.Timestamp(sw["ts"].values[sl_hits[0]]) if len(sl_hits) else None
        touched_25 = sl_time is not None

        # ---- BASELINE exit (no SL): earliest of target / settlement ----
        if target_time is not None:
            base_pnl = net_credit - (tgt_short - tgt_long); base_reason = "90% target"
        else:
            bp = settle_pnl()
            if bp is None: skipped.append((e_cur.date(), "no settlement")); continue
            base_pnl = bp; base_reason = "DTE0 settlement"

        # ---- SL-APPLIED exit: earliest of SL / target / settlement ----
        if sl_time is not None and target_time is not None and sl_time == target_time: conflicts += 1
        if sl_time is not None and (target_time is None or sl_time <= target_time):
            short_x = float(sser_hi.asof(sl_time)); long_x = float(lser.asof(sl_time))     # worst-case short = minute HIGH; hedge = minute close
            if np.isnan(short_x): short_x = float(sser.asof(sl_time))
            if np.isnan(long_x): long_x = l0
            sl_pnl = net_credit - (short_x - long_x); sl_reason = "SL-2.01%"; sl_exit_time = sl_time
        elif target_time is not None:
            sl_pnl = net_credit - (tgt_short - tgt_long); sl_reason = "90% target"; sl_exit_time = target_time
        else:
            sl_pnl = settle_pnl(); sl_reason = "DTE0 settlement"; sl_exit_time = t_end

        trades.append({"entry_date": entry_day.date(), "DTE": dte, "type": ("Call Credit Spread" if typ == "CCS" else "Put Credit Spread"),
                       "trigger": label, "trig_kind": trig_kind, "entry_time": entry_time.strftime("%Y-%m-%d %H:%M"),
                       "entry_spot": round(entry_spot, 1), "SL_level": round(level, 1), "ATM": int(atm), "short_K": int(sK), "long_K": int(lK),
                       "net_credit": round(net_credit, 2), "touched_2.01%": touched_25,
                       "sl_time": sl_time.strftime("%Y-%m-%d %H:%M") if sl_time is not None else "",
                       "base_reason": base_reason, "base_pnl": round(base_pnl, 2),
                       "sl_reason": sl_reason, "sl_exit_time": pd.Timestamp(sl_exit_time).strftime("%Y-%m-%d %H:%M"), "sl_pnl": round(sl_pnl, 2),
                       "oos": pd.Timestamp(entry_day) >= pd.Timestamp("2025-08-29")})

    T = pd.DataFrame(trades)
    nb, ns = T["base_pnl"], T["sl_pnl"]
    # premise validation: of trades that touched 2.01%, how many recovered to >=0 by DTE0 in the NO-SL baseline
    tt = T[T["touched_2.01%"]]
    recovered = int((tt["base_pnl"] >= 0).sum()); touched_n = len(tt)
    sl_hit = T[T["sl_reason"] == "SL-2.01%"]
    settle_noSL = T[(T["sl_reason"] == "DTE0 settlement")]
    base_settle_losers = T[(T["base_reason"] == "DTE0 settlement")]

    summary = pd.DataFrame([
        {"metric": "Rule", "value": "ADDITIVE spot 2.01% adverse SL (touch); short bought back at SL-minute HIGH (worst-case), hedge at SL-minute CLOSE"},
        {"metric": "SL reference", "value": "*** ENTRY spot (trigger-minute price, = ATM basis), NOT day-open. Flagged for confirmation. ***"},
        {"metric": "Priority", "value": f"earliest of SL / 90%-target / DTE0-settlement; same-minute SL&target co-fires: {conflicts} (expected 0, mutually exclusive)"},
        {"metric": "Total trades", "value": len(T)},
        {"metric": "", "value": ""},
        {"metric": "--- EXIT PATH COUNTS (with SL) ---", "value": ""},
        {"metric": "SL-2.01% hits", "value": int((T.sl_reason == "SL-2.01%").sum())},
        {"metric": "90% target", "value": int((T.sl_reason == "90% target").sum())},
        {"metric": "DTE0 settlement", "value": int((T.sl_reason == "DTE0 settlement").sum())},
        {"metric": "", "value": ""},
        {"metric": "--- BEFORE vs AFTER (total P&L) ---", "value": ""},
        {"metric": "Baseline total P&L (no SL)", "value": round(nb.sum(), 1)},
        {"metric": "With-SL total P&L", "value": round(ns.sum(), 1)},
        {"metric": "SL impact (after - before)", "value": round(ns.sum() - nb.sum(), 1)},
        {"metric": "Baseline win %", "value": round((nb > 0).mean() * 100, 1)},
        {"metric": "With-SL win %", "value": round((ns > 0).mean() * 100, 1)},
        {"metric": "", "value": ""},
        {"metric": "--- LOSS PROFILE ---", "value": ""},
        {"metric": "Avg P&L on SL-hit trades", "value": round(sl_hit.sl_pnl.mean(), 2) if len(sl_hit) else np.nan},
        {"metric": "  (same trades' baseline P&L, no SL)", "value": round(sl_hit.base_pnl.mean(), 2) if len(sl_hit) else np.nan},
        {"metric": "Avg P&L on DTE0-settlement trades (no SL fired)", "value": round(settle_noSL.sl_pnl.mean(), 2) if len(settle_noSL) else np.nan},
        {"metric": "", "value": ""},
        {"metric": "--- PREMISE CHECK: 'once 2.01% adverse touched, never returns to entry' ---", "value": ""},
        {"metric": "Trades that touched 2.01% adverse", "value": touched_n},
        {"metric": "  ...of those, recovered to >=0 by DTE0 WITHOUT SL", "value": f"{recovered} ({round(recovered/touched_n*100,1) if touched_n else 0}%)"},
        {"metric": "  ...ended negative by DTE0 without SL", "value": touched_n - recovered},
        {"metric": "SL saved vs cost on touched trades (sum sl_pnl - base_pnl)", "value": round((tt.sl_pnl - tt.base_pnl).sum(), 1)},
    ])

    with pd.ExcelWriter(OUTDIR / "credit_spread_spot_sl.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        cmp = pd.DataFrame([
            {"version": "Baseline (no SL)", "total_pnl": round(nb.sum(), 1), "win_%": round((nb > 0).mean() * 100, 1),
             "avg_pnl": round(nb.mean(), 2), "worst": round(nb.min(), 1), "best": round(nb.max(), 1),
             "target": int((T.base_reason == "90% target").sum()), "settlement": int((T.base_reason == "DTE0 settlement").sum()), "SL": 0},
            {"version": "With 2.01% spot SL", "total_pnl": round(ns.sum(), 1), "win_%": round((ns > 0).mean() * 100, 1),
             "avg_pnl": round(ns.mean(), 2), "worst": round(ns.min(), 1), "best": round(ns.max(), 1),
             "target": int((T.sl_reason == "90% target").sum()), "settlement": int((T.sl_reason == "DTE0 settlement").sum()), "SL": int((T.sl_reason == "SL-2.01%").sum())},
        ])
        cmp.to_excel(w, sheet_name="Before_After", index=False)
        # per touched trade detail
        tt[["entry_date", "type", "entry_spot", "SL_level", "net_credit", "sl_time", "base_reason", "base_pnl", "sl_pnl"]].to_excel(w, sheet_name="Touched_2.01pct", index=False)
        T.to_excel(w, sheet_name="All_Trades", index=False)
        if skipped: pd.DataFrame(skipped, columns=["expiry", "reason"]).to_excel(w, sheet_name="Skipped", index=False)
    T.to_csv(OUTDIR / "credit_spread_spot_sl_trades.csv", index=False)

    pd.set_option("display.width", 230)
    print("=" * 96 + "\nNIFTY WEEKLY CREDIT SPREAD + 2.01% SPOT STOP-LOSS (additive, worst-case short fill)\n" + "=" * 96)
    print(f"trades {len(T)} | SL reference = ENTRY spot (flagged) | same-minute SL&target co-fires {conflicts}")
    print(f"\nexit paths WITH SL:  SL-2.01% {int((T.sl_reason=='SL-2.01%').sum())} | 90%-target {int((T.sl_reason=='90% target').sum())} | DTE0-settle {int((T.sl_reason=='DTE0 settlement').sum())}")
    print(f"\nBEFORE (no SL): total {round(nb.sum(),1)} | win {round((nb>0).mean()*100,1)}% | worst {round(nb.min(),1)}")
    print(f"AFTER  (SL):    total {round(ns.sum(),1)} | win {round((ns>0).mean()*100,1)}% | worst {round(ns.min(),1)}")
    print(f"SL IMPACT: {round(ns.sum()-nb.sum(),1)} points")
    print(f"\nAvg SL-hit P&L {round(sl_hit.sl_pnl.mean(),2) if len(sl_hit) else 'n/a'} (same trades no-SL: {round(sl_hit.base_pnl.mean(),2) if len(sl_hit) else 'n/a'}) | avg DTE0-settle (no SL fired) {round(settle_noSL.sl_pnl.mean(),2) if len(settle_noSL) else 'n/a'}")
    print(f"\nPREMISE: {touched_n} trades touched 2.01% adverse | recovered to >=0 by DTE0 WITHOUT SL: {recovered} ({round(recovered/touched_n*100,1) if touched_n else 0}%) | net SL effect on them {round((tt.sl_pnl-tt.base_pnl).sum(),1)}")
    print("\n--- touched-2.01% trades (base vs SL P&L) ---")
    print(tt[["entry_date", "type", "entry_spot", "SL_level", "net_credit", "base_reason", "base_pnl", "sl_pnl"]].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}/credit_spread_spot_sl.xlsx")


if __name__ == "__main__":
    main()
