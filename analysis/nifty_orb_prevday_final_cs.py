# -*- coding: utf-8 -*-
"""nifty_orb_prevday_final_cs.py — FINALIZED ORB + prev-day bias CREDIT SPREAD.
Entry = FIRST breakout only and it must be the DESIRED direction (else no trade, no watching). Bullish +
OR-high-first -> Put Credit Spread (SELL ATM PE / BUY ATM-200 PE). Bearish + OR-low-first -> Call Credit Spread
(SELL ATM CE / BUY ATM+200 CE). ATM=round(spot/50)*50 at the break (break-minute close). NO SL, NO re-entry.
EXIT = 90% profit target (cost-to-close = short-long <= 10% of net credit, checked every 1-min) else 15:15
close. GROSS premium points, 1 spread. Front expiry >= day (intraday). Reuses no other ORB module.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "nifty_orb_prevday_final_cs"; OUTDIR.mkdir(parents=True, exist_ok=True)
OR_START, OR_END = 9 * 60 + 15, 9 * 60 + 29; SCAN_START, EXIT_MOD = 9 * 60 + 30, 15 * 60 + 15
FLOOR = pd.Timestamp("2024-10-01"); WIDTH = 200; TARGET_FRAC = 0.10


def leg_series(folder, strike, ot, day, t0, t1):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{ot}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts")["close"].sort_index()
    return o if len(o) else None


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    daily = sp.groupby("date").agg(o=("open", "first"), c=("close", "last")); tdays = list(daily.index); spot_end = sp["date"].max()
    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")); exp_dates = [e.normalize() for e in expiries]

    trades = []; notrades = []; flags = []
    for i in range(1, len(tdays)):
        D = tdays[i]; P = tdays[i - 1]
        if pd.Timestamp(D) < FLOOR or D > spot_end: continue
        bias = "bullish" if daily.loc[P, "c"] > daily.loc[P, "o"] else ("bearish" if daily.loc[P, "c"] < daily.loc[P, "o"] else "neutral")
        ed = sp[sp["date"] == D]; orb = ed[(ed["mod"] >= OR_START) & (ed["mod"] <= OR_END)]
        if bias == "neutral" or len(orb) < 10:
            notrades.append({"date": D.date(), "prev_bias": bias, "first_break": "n/a", "reason": "neutral bias" if bias == "neutral" else "incomplete OR"}); continue
        ORH = orb["high"].max(); ORL = orb["low"].min()
        scan = ed[(ed["mod"] >= SCAN_START) & (ed["mod"] <= EXIT_MOD)].sort_values("mod").reset_index(drop=True)
        up = scan["high"].values >= ORH; dn = scan["low"].values <= ORL
        up_i = int(np.argmax(up)) if up.any() else None; dn_i = int(np.argmax(dn)) if dn.any() else None
        # FIRST breakout of either level
        if up_i is None and dn_i is None:
            notrades.append({"date": D.date(), "prev_bias": bias, "first_break": "none", "reason": "no breakout"}); continue
        if up_i is None: first_dir = "down"
        elif dn_i is None: first_dir = "up"
        elif up_i < dn_i: first_dir = "up"
        elif dn_i < up_i: first_dir = "down"
        else:                                            # outside bar first candle -> proxy by O->C
            r0 = scan.iloc[up_i]; first_dir = "up" if r0["close"] >= r0["open"] else "down"
            flags.append({"date": D.date(), "flag": "outside_bar_first_break", "detail": f"resolved {first_dir}"})
        desired = "up" if bias == "bullish" else "down"
        if first_dir != desired:
            notrades.append({"date": D.date(), "prev_bias": bias, "first_break": first_dir, "reason": "undesired direction broke first"}); continue

        b_i = up_i if desired == "up" else dn_i; br = scan.iloc[b_i]
        bspot = float(br["close"]); atm = int(round(bspot / 50) * 50); t_en = pd.Timestamp(br["ts"]); t_end = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=15)
        if bias == "bullish": ot, sK, lK, typ = "PE", atm, atm - WIDTH, "Put Credit Spread"
        else: ot, sK, lK, typ = "CE", atm, atm + WIDTH, "Call Credit Spread"
        j = bisect.bisect_left(exp_dates, pd.Timestamp(D))
        if j >= len(exp_dates): continue
        folder = expiries[j].strftime("%Y%m%d")
        sser = leg_series(folder, sK, ot, D, t_en - pd.Timedelta(minutes=2), t_end); lser = leg_series(folder, lK, ot, D, t_en - pd.Timedelta(minutes=2), t_end)
        if sser is None or lser is None:
            notrades.append({"date": D.date(), "prev_bias": bias, "first_break": first_dir, "reason": "missing leg data"}); flags.append({"date": D.date(), "flag": "missing_leg", "detail": f"{sK}/{lK}{ot}"}); continue
        s0 = sser.asof(t_en); l0 = lser.asof(t_en)
        if np.isnan(s0) or np.isnan(l0):
            notrades.append({"date": D.date(), "prev_bias": bias, "first_break": first_dir, "reason": "nan entry premium"}); continue
        credit = float(s0 - l0)
        if credit <= 0:
            notrades.append({"date": D.date(), "prev_bias": bias, "first_break": first_dir, "reason": f"non-positive credit {round(credit,1)}"}); flags.append({"date": D.date(), "flag": "non_positive_credit", "detail": round(credit, 1)}); continue
        # 90% target scan (cost-to-close <= 10% of credit), else 15:15
        both = pd.concat([sser.rename("s"), lser.rename("l")], axis=1).ffill().dropna(); both = both[both.index > t_en]
        sv = both["s"].values; lv = both["l"].values; spread_val = sv - lv; bt = both.index.values
        thr = TARGET_FRAC * credit; hit = np.where(spread_val <= thr)[0]
        if len(hit):
            k = hit[0]; t_ex = pd.Timestamp(bt[k]); short_x = float(sv[k]); long_x = float(lv[k]); reason = "90% target"
        else:
            short_x = float(sser.asof(t_end)); long_x = float(lser.asof(t_end)); t_ex = t_end; reason = "3:15pm"
            if np.isnan(short_x) or np.isnan(long_x):
                notrades.append({"date": D.date(), "prev_bias": bias, "first_break": first_dir, "reason": "nan exit premium"}); continue
        exit_debit = short_x - long_x; pnl = credit - exit_debit
        dte = int((expiries[j].normalize() - pd.Timestamp(D).normalize()).days)
        trades.append({"date": D.date(), "prev_bias": bias, "OR_high": round(ORH, 1), "OR_low": round(ORL, 1), "first_break": first_dir,
                       "type": typ, "expiry_used": expiries[j].date(), "DTE": dte, "entry_time": t_en.strftime("%H:%M"), "short_K": sK, "long_K": lK, "option_type": ot,
                       "short_entry": round(float(s0), 2), "long_entry": round(float(l0), 2), "net_credit": round(credit, 2),
                       "exit_time": pd.Timestamp(t_ex).strftime("%H:%M"), "exit_reason": reason, "short_exit": round(short_x, 2), "long_exit": round(long_x, 2),
                       "exit_debit": round(exit_debit, 2), "pnl_points": round(pnl, 2)})

    T = pd.DataFrame(trades); NT = pd.DataFrame(notrades); total = len(T) + len(NT)
    win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)
    pcs = T[T.type == "Put Credit Spread"]; ccs = T[T.type == "Call Credit Spread"]
    tgt = T[T.exit_reason == "90% target"]; tim = T[T.exit_reason == "3:15pm"]

    def blk(df, lbl):
        return {"group": lbl, "trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_points.sum(), 1), "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0,
                "avg_credit": round(df.net_credit.mean(), 2) if len(df) else 0, "max": round(df.pnl_points.max(), 1) if len(df) else 0, "min": round(df.pnl_points.min(), 1) if len(df) else 0}
    BY = pd.DataFrame([blk(T, "ALL"), blk(pcs, "Put Credit Spread (bullish)"), blk(ccs, "Call Credit Spread (bearish)"),
                       blk(tgt, "90%-target exits"), blk(tim, "3:15pm exits")])
    DTET = pd.DataFrame([{"DTE": d, "trades": int((T.DTE == d).sum()),
                          "win_%": round((T[T.DTE == d].pnl_points > 0).mean() * 100, 1) if (T.DTE == d).any() else 0,
                          "total_pnl": round(T[T.DTE == d].pnl_points.sum(), 1), "avg_pnl": round(T[T.DTE == d].pnl_points.mean(), 2) if (T.DTE == d).any() else 0,
                          "avg_credit": round(T[T.DTE == d].net_credit.mean(), 2) if (T.DTE == d).any() else 0,
                          "target_hits": int((T[T.DTE == d].exit_reason == "90% target").sum()),
                          "PCS/CCS": f"{int((T[T.DTE == d].type == 'Put Credit Spread').sum())}/{int((T[T.DTE == d].type == 'Call Credit Spread').sum())}"}
                         for d in sorted(T.DTE.unique())])
    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "FINAL ORB+prev-day-bias CREDIT SPREAD. First-breakout-desired only; aligned spread; NO SL/re-entry; 90% target else 15:15. GROSS pts, 1 spread."},
        {"metric": "Structure", "value": "bullish+OR-high-first: PCS (sell ATM PE/buy ATM-200 PE) ; bearish+OR-low-first: CCS (sell ATM CE/buy ATM+200 CE)"},
        {"metric": "Window", "value": f"{T.date.min()} .. {spot_end.date()}"},
        {"metric": "Total days evaluated", "value": total}, {"metric": "Total trades", "value": len(T)},
        {"metric": "No-trade days", "value": len(NT)}, {"metric": "% days NO trade", "value": round(len(NT) / total * 100, 1) if total else 0},
        {"metric": "Win rate %", "value": win}, {"metric": "Total P&L (points)", "value": tot},
        {"metric": "Avg P&L / trade", "value": round(T.pnl_points.mean(), 2)}, {"metric": "Median P&L / trade", "value": round(T.pnl_points.median(), 2)},
        {"metric": "Avg net credit", "value": round(T.net_credit.mean(), 2)}, {"metric": "Max / min trade", "value": f"{round(T.pnl_points.max(),1)} / {round(T.pnl_points.min(),1)}"},
        {"metric": "90%-target exits / %", "value": f"{len(tgt)} / {round(len(tgt)/len(T)*100,1)}%"},
        {"metric": "3:15pm exits / %", "value": f"{len(tim)} / {round(len(tim)/len(T)*100,1)}%"},
        {"metric": "Avg P&L target / 3:15", "value": f"{round(tgt.pnl_points.mean(),2) if len(tgt) else 0} / {round(tim.pnl_points.mean(),2) if len(tim) else 0}"},
        {"metric": "PCS (bullish) trades / total", "value": f"{len(pcs)} / {round(pcs.pnl_points.sum(),1)}"},
        {"metric": "CCS (bearish) trades / total", "value": f"{len(ccs)} / {round(ccs.pnl_points.sum(),1)}"},
        {"metric": "No-trade reasons", "value": str(NT.reason.value_counts().to_dict() if len(NT) else {})},
        {"metric": "vs prior CS variants", "value": "reversal-entry aligned no-SL: +1032.6 (300 tr); first-breakout subset there: +1117.4 (196 tr) held-to-3:15 (no target)"},
    ])
    with pd.ExcelWriter(OUTDIR / "nifty_orb_prevday_final_cs.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False); BY.to_excel(w, sheet_name="By_Group", index=False)
        DTET.to_excel(w, sheet_name="PnL_by_DTE", index=False)
        T.to_excel(w, sheet_name="Trades", index=False); (NT if len(NT) else pd.DataFrame([{"note": "none"}])).to_excel(w, sheet_name="No_Trade_Days", index=False)
        (pd.DataFrame(flags) if flags else pd.DataFrame([{"flag": "none"}])).to_excel(w, sheet_name="Flags", index=False)
    T.to_csv(OUTDIR / "nifty_orb_prevday_final_cs_trades.csv", index=False)

    pd.set_option("display.width", 230)
    print("=" * 96 + "\nFINAL ORB+PREV-DAY-BIAS CREDIT SPREAD (first-breakout-desired, 90% target, no SL/re-entry)\n" + "=" * 96)
    print(f"days {total} | trades {len(T)} | no-trade {len(NT)} ({round(len(NT)/total*100,1)}%)")
    print(f"win {win}% | TOTAL {tot:,} | avg {round(T.pnl_points.mean(),2)} | median {round(T.pnl_points.median(),2)} | avg credit {round(T.net_credit.mean(),2)}")
    print(f"90%-target {len(tgt)} ({round(len(tgt)/len(T)*100,1)}%) avg {round(tgt.pnl_points.mean(),2) if len(tgt) else 0} | 3:15 {len(tim)} ({round(len(tim)/len(T)*100,1)}%) avg {round(tim.pnl_points.mean(),2) if len(tim) else 0}")
    print("\n--- BY GROUP ---"); print(BY.to_string(index=False))
    print("\n--- PnL BY DTE ---"); print(DTET.to_string(index=False))
    print(f"\nno-trade reasons: {NT.reason.value_counts().to_dict() if len(NT) else {}} | flags {len(flags)}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
