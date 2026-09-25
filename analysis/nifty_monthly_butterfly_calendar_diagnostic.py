# -*- coding: utf-8 -*-
"""nifty_monthly_butterfly_calendar_diagnostic.py — ANALYSIS-ONLY diagnostic on the existing NIFTY Monthly
Butterfly+Calendar backtest (15 trades). Rebuilds each trade's FULL combined P&L path (for MFE/MAE), the
standalone butterfly-only and calendar-only P&L at the actual combined exit (structure attribution), entry
VIX, and entry-to-expiry spot return (for directional/seasonality bucketing). Does not change strategy logic.
"""
import sys, glob, os, time, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
import opt_pull_nifty_full as op

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "nifty_monthly_butterfly_calendar"; OUTDIR.mkdir(parents=True, exist_ok=True)
VIXF = rb.BASE / "data" / "india_vix_1min.csv"
ENTRY_MOD = 15 * 60 + 16; STEP = 50; TARGET, SL = 3000.0, -3000.0
NIFTY_KEY = "NSE_INDEX|Nifty 50"
VIX_BUCKETS = ["<9"] + [f"{i}-{i+1}" for i in range(9, 20)] + [">20"]


def vbucket(v):
    if pd.isna(v): return "n/a"
    return "<9" if v < 9 else (">20" if v >= 20 else f"{int(v)}-{int(v)+1}")


def lot_size_for(exp_dt):
    cons, _ = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(NIFTY_KEY)}&expiry_date={exp_dt.strftime('%Y-%m-%d')}")
    return int(cons[0]["lot_size"]) if cons else None


def leg_series(folder, strike, otype):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{otype}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "open", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    return o.set_index("ts").sort_index()


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    tdays = sorted(sp["date"].unique()); tdset = set(tdays); spot_end = sp["date"].max()

    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vx = vx.assign(mod=vts.dt.hour * 60 + vts.dt.minute, date=vts.dt.normalize())
    vix_ent_series = vx[vx["mod"] == ENTRY_MOD].groupby("date")["close"].last()

    all_exp = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d"))
    dfm = pd.DataFrame({"date": all_exp}); dfm["month"] = dfm["date"].dt.to_period("M")
    monthly = sorted(dfm.groupby("month")["date"].max().tolist())

    lot_cache = {}
    def lots(e):
        if e not in lot_cache: lot_cache[e] = lot_size_for(e); time.sleep(0.1)
        return lot_cache[e]

    trades = []
    for i in range(len(monthly) - 2):
        E_prev, E_cur, E_next = monthly[i], monthly[i + 1], monthly[i + 2]
        if E_cur > spot_end or E_next > spot_end: continue
        d0 = pd.Timestamp(E_prev) + pd.Timedelta(days=1)
        days_to_fri = (4 - d0.weekday()) % 7
        target_fri = (d0 + pd.Timedelta(days=days_to_fri)).normalize()
        if target_fri in tdset: entry_day = target_fri
        else:
            thu = (target_fri - pd.Timedelta(days=1)).normalize()
            if thu in tdset: entry_day = thu
            else: continue

        ed = sp[sp["date"] == entry_day]; e1516 = ed[ed["mod"] == ENTRY_MOD]
        if e1516.empty: continue
        espot = float(e1516["open"].iloc[0]); atm = round(espot / STEP) * STEP
        t_en = pd.Timestamp(entry_day) + pd.Timedelta(hours=15, minutes=16); t_ex_cap = pd.Timestamp(E_cur) + pd.Timedelta(hours=15, minutes=15)
        lot_cur = lots(E_cur); lot_next = lots(E_next)
        if lot_cur is None or lot_next is None: continue

        legs_def = [
            ("bfly", "buy1_ATM_PE", atm, "PE", E_cur, 1, lot_cur),
            ("bfly", "sell2_ATMm400_PE", atm - 400, "PE", E_cur, -2, lot_cur),
            ("bfly", "buy1_ATMm800_PE", atm - 800, "PE", E_cur, 1, lot_cur),
            ("cal", "sell1_ATMp200_CE_cur", atm + 200, "CE", E_cur, -1, lot_cur),
            ("cal", "buy1_ATMp200_CE_next", atm + 200, "CE", E_next, 1, lot_next),
        ]
        series = {}; ok = True
        for grp, lbl, K, ot, E, q, lot in legs_def:
            s = leg_series(E.strftime("%Y%m%d"), K, ot)
            if s is None: ok = False; break
            series[lbl] = s
        if not ok: continue

        entry_px = {}
        for grp, lbl, K, ot, E, q, lot in legs_def:
            v = series[lbl]["open"].asof(t_en)
            if pd.isna(v): ok = False; break
            entry_px[lbl] = float(v)
        if not ok: continue
        entry_value = sum(q * entry_px[lbl] * lot for grp, lbl, K, ot, E, q, lot in legs_def)
        bfly_entry = sum(q * entry_px[lbl] * lot for grp, lbl, K, ot, E, q, lot in legs_def if grp == "bfly")
        cal_entry = sum(q * entry_px[lbl] * lot for grp, lbl, K, ot, E, q, lot in legs_def if grp == "cal")

        idx = None
        for grp, lbl, K, ot, E, q, lot in legs_def:
            s = series[lbl][(series[lbl].index > t_en) & (series[lbl].index <= t_ex_cap)]["close"]
            idx = s.index if idx is None else idx.union(s.index)
        if idx is None or len(idx) == 0: continue
        idx = idx.sort_values()
        combined = pd.Series(0.0, index=idx); bfly_val = pd.Series(0.0, index=idx); cal_val = pd.Series(0.0, index=idx)
        for grp, lbl, K, ot, E, q, lot in legs_def:
            px = series[lbl]["close"].reindex(idx).ffill()
            combined = combined + q * px * lot
            if grp == "bfly": bfly_val = bfly_val + q * px * lot
            else: cal_val = cal_val + q * px * lot
        pnl_path = combined - entry_value

        hit = pnl_path[(pnl_path >= TARGET) | (pnl_path <= SL)]
        if len(hit):
            exit_time = hit.index[0]; exit_reason = "target hit" if pnl_path.loc[exit_time] >= TARGET else "SL hit"
        else:
            exit_time = pnl_path.index[-1]; exit_reason = "3:15pm expiry square-off"
        pnl = float(combined.loc[exit_time] - entry_value)

        path_to_exit = pnl_path.loc[:exit_time]
        mfe = float(path_to_exit.max()); mae = float(path_to_exit.min())
        bfly_pnl_at_exit = float(bfly_val.loc[exit_time] - bfly_entry)
        cal_pnl_at_exit = float(cal_val.loc[exit_time] - cal_entry)

        # entry-to-expiry spot return (E_cur's expiry-day close, or exit day if earlier/exp itself missing)
        exp_day_spot = sp[(sp["date"] == pd.Timestamp(E_cur).normalize())]
        exp_close = float(exp_day_spot[exp_day_spot["mod"] <= 15 * 60 + 15]["close"].iloc[-1]) if len(exp_day_spot) else np.nan
        spot_ret_pct = round((exp_close - espot) / espot * 100, 3) if not np.isnan(exp_close) else np.nan

        dte_at_entry = (pd.Timestamp(E_cur).normalize() - entry_day).days
        days_to_exit = (pd.Timestamp(exit_time).normalize() - entry_day).days
        frac_of_dte = round(days_to_exit / dte_at_entry, 3) if dte_at_entry else np.nan

        trades.append({"entry_date": entry_day.date(), "entry_month": entry_day.strftime("%Y-%m"), "ATM": int(atm),
                       "E_cur": E_cur.date(), "E_next": E_next.date(), "entry_vix": round(float(vix_ent_series.get(entry_day, np.nan)), 2),
                       "dte_at_entry": dte_at_entry, "exit_date": pd.Timestamp(exit_time).date(), "exit_reason": exit_reason,
                       "days_held": days_to_exit, "frac_of_dte_at_exit": frac_of_dte, "pnl_rs": round(pnl, 2),
                       "mfe": round(mfe, 2), "mae": round(mae, 2), "bfly_pnl_at_exit": round(bfly_pnl_at_exit, 2), "cal_pnl_at_exit": round(cal_pnl_at_exit, 2),
                       "entry_spot": round(espot, 2), "expiry_close_spot": round(exp_close, 2) if not np.isnan(exp_close) else np.nan, "spot_return_pct": spot_ret_pct,
                       "lot_cur": lot_cur, "lot_next": lot_next})
        print(f"  {entry_day.date()} -> {exit_reason} pnl={round(pnl,2)} mfe={round(mfe,1)} mae={round(mae,1)} bfly={round(bfly_pnl_at_exit,1)} cal={round(cal_pnl_at_exit,1)} vix={round(float(vix_ent_series.get(entry_day, np.nan)),1)} spot_ret={spot_ret_pct}%", flush=True)

    T = pd.DataFrame(trades)
    T["vix_bucket"] = T["entry_vix"].map(vbucket)
    T["direction_bucket"] = pd.cut(T["spot_return_pct"], [-100, -2, 2, 100], labels=["DOWN (<-2%)", "SIDEWAYS (-2%..2%)", "UP (>2%)"])
    T.to_csv(OUTDIR / "diagnostic_trades_full.csv", index=False)

    # ===== CUT 1: exit reason breakdown =====
    def blk(df, lbl):
        return {"group": lbl, "trades": len(df), "win_%": round((df.pnl_rs > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_rs.sum(), 1) if len(df) else 0, "avg_pnl": round(df.pnl_rs.mean(), 2) if len(df) else 0,
                "avg_days_held": round(df.days_held.mean(), 2) if len(df) else 0}
    C1 = pd.DataFrame([blk(T, "ALL")] + [blk(T[T.exit_reason == r], r) for r in ["target hit", "SL hit", "3:15pm expiry square-off"]])

    # ===== CUT 2: held-to-expiry deeper look (likely empty) =====
    exp_trades = T[T.exit_reason == "3:15pm expiry square-off"]
    C2_note = f"{len(exp_trades)} trades held to expiry (out of {len(T)} total)."
    # time-based-exit-at-fraction-of-DTE hypothetical: for EVERY trade, what if we forced a check at 50%/75% DTE and exited if neither target/SL had hit yet?
    C2_hyp = []
    for frac_cut in [0.5, 0.75]:
        # approximate: trades whose ACTUAL exit came at/after this fraction AND via target/SL - would an earlier forced exit at frac_cut have preempted a winning trade, or avoided further loss on a losing one?
        would_be_early = T[T.frac_of_dte_at_exit >= frac_cut]
        C2_hyp.append({"frac_of_DTE_checkpoint": frac_cut, "trades_still_open_at_checkpoint": len(would_be_early),
                       "of_those_eventual_outcome": would_be_early.exit_reason.value_counts().to_dict()})

    # ===== CUT 3: SL-hit trades — what went wrong =====
    sl = T[T.exit_reason == "SL hit"]
    C3 = pd.DataFrame([
        {"metric": "SL-hit trades", "value": len(sl)},
        {"metric": "Avg days held (SL trades)", "value": round(sl.days_held.mean(), 2) if len(sl) else 0},
        {"metric": "Avg |spot return| to expiry on SL trades", "value": round(sl.spot_return_pct.abs().mean(), 3) if len(sl) else 0},
        {"metric": "Avg |spot return| to expiry on ALL trades (context)", "value": round(T.spot_return_pct.abs().mean(), 3)},
        {"metric": "SL trades where |spot move by exit-day close| > 2%", "value": int((sl.spot_return_pct.abs() > 2).sum()) if len(sl) else 0},
        {"metric": "Avg days_held / dte_at_entry (SL trades, fast-bleed check)", "value": round(sl.frac_of_dte_at_exit.mean(), 3) if len(sl) else 0},
    ])
    C3_detail = sl[["entry_date", "days_held", "dte_at_entry", "frac_of_dte_at_exit", "entry_spot", "expiry_close_spot", "spot_return_pct", "pnl_rs", "mae"]]

    # ===== CUT 4: structure attribution =====
    C4 = pd.DataFrame([
        {"metric": "Total combined P&L (realized)", "value": round(T.pnl_rs.sum(), 1)},
        {"metric": "Sum of BUTTERFLY-only P&L (at actual exit times)", "value": round(T.bfly_pnl_at_exit.sum(), 1)},
        {"metric": "Sum of CALENDAR-only P&L (at actual exit times)", "value": round(T.cal_pnl_at_exit.sum(), 1)},
        {"metric": "Check (bfly+cal should == combined total)", "value": round(T.bfly_pnl_at_exit.sum() + T.cal_pnl_at_exit.sum(), 1)},
        {"metric": "Butterfly win rate (per-trade bfly_pnl>0)", "value": round((T.bfly_pnl_at_exit > 0).mean() * 100, 1)},
        {"metric": "Calendar win rate (per-trade cal_pnl>0)", "value": round((T.cal_pnl_at_exit > 0).mean() * 100, 1)},
        {"metric": "Avg butterfly P&L / trade", "value": round(T.bfly_pnl_at_exit.mean(), 2)},
        {"metric": "Avg calendar P&L / trade", "value": round(T.cal_pnl_at_exit.mean(), 2)},
        {"metric": "Corr(bfly_pnl, cal_pnl) across trades", "value": round(T.bfly_pnl_at_exit.corr(T.cal_pnl_at_exit), 3)},
    ])
    C4_detail = T[["entry_date", "exit_reason", "pnl_rs", "bfly_pnl_at_exit", "cal_pnl_at_exit"]]

    # ===== CUT 5: MFE/MAE =====
    C5 = pd.DataFrame([
        {"metric": "Avg MFE (best point reached, all trades)", "value": round(T.mfe.mean(), 2)},
        {"metric": "Avg MAE (worst point reached, all trades)", "value": round(T.mae.mean(), 2)},
        {"metric": "Trades where MFE >= 2500 but exit_reason != target hit (near-miss)", "value": int(((T.mfe >= 2500) & (T.exit_reason != "target hit")).sum())},
        {"metric": "Trades where MAE <= -2500 but exit_reason != SL hit (near-miss)", "value": int(((T.mae <= -2500) & (T.exit_reason != "SL hit")).sum())},
        {"metric": "SL-hit trades: avg MFE reached before the SL (reversal check)", "value": round(sl.mfe.mean(), 2) if len(sl) else 0},
        {"metric": "Target-hit trades: avg MAE reached before the target (drawdown-before-win check)", "value": round(T[T.exit_reason=='target hit'].mae.mean(), 2)},
    ])
    C5_detail = T[["entry_date", "exit_reason", "pnl_rs", "mfe", "mae"]]

    # ===== CUT 6: VIX buckets =====
    present = [b for b in VIX_BUCKETS if (T.vix_bucket == b).any()]
    C6 = pd.DataFrame([{"vix_bucket": b, **blk(T[T.vix_bucket == b], b)} for b in present]).drop(columns=["group"])

    # ===== CUT 7: monthly seasonality / direction =====
    C7_month = T.groupby("entry_month").agg(trades=("pnl_rs", "size"), total_pnl=("pnl_rs", "sum"), avg_pnl=("pnl_rs", "mean")).reset_index()
    C7_dir = pd.DataFrame([{"direction": d, **blk(T[T.direction_bucket == d], str(d))} for d in ["DOWN (<-2%)", "SIDEWAYS (-2%..2%)", "UP (>2%)"] if (T.direction_bucket == d).any()]).drop(columns=["group"]) if len(T) else pd.DataFrame()
    corr_dir_pnl = round(T["spot_return_pct"].corr(T["pnl_rs"]), 3)

    # ===== improvement candidates =====
    candidates = []
    # VIX filter candidates (small sample - just report, IS/OOS not meaningful)
    hi_vix = T[T.entry_vix > 15]; lo_vix = T[T.entry_vix <= 15]
    candidates.append({"idea": "Skip entries where VIX>15", "trades_affected": len(hi_vix), "pnl_of_affected": round(hi_vix.pnl_rs.sum(),1), "note": f"n={len(hi_vix)} - too small to trust" if len(hi_vix)<8 else ""})
    candidates.append({"idea": "Only trade DOWN months (spot_return<-2%)", "trades_affected": len(T[T.direction_bucket!='DOWN (<-2%)']), "pnl_of_affected": round(T[T.direction_bucket!='DOWN (<-2%)'].pnl_rs.sum(),1), "note": "n too small"})
    CAND = pd.DataFrame(candidates)

    with pd.ExcelWriter(OUTDIR / "nifty_monthly_bfly_cal_diagnostic.xlsx", engine="openpyxl") as w:
        pd.DataFrame([{"metric": "Total trades", "value": len(T)}, {"metric": "NOTE", "value": "SAMPLE SIZE IS VERY SMALL (n=%d) - this is a monthly-frequency strategy; almost every cut below has sub-buckets of 0-3 trades. Treat all patterns as suggestive, not statistically validated." % len(T)}]).to_excel(w, sheet_name="Scope", index=False)
        C1.to_excel(w, sheet_name="Cut1_ExitReason", index=False)
        pd.DataFrame([{"note": C2_note}]).to_excel(w, sheet_name="Cut2_HeldToExpiry", index=False)
        pd.DataFrame(C2_hyp).to_excel(w, sheet_name="Cut2_HeldToExpiry", index=False, startrow=2)
        C3.to_excel(w, sheet_name="Cut3_SLAnalysis", index=False); C3_detail.to_excel(w, sheet_name="Cut3_SLAnalysis", index=False, startrow=len(C3)+2)
        C4.to_excel(w, sheet_name="Cut4_StructureAttrib", index=False); C4_detail.to_excel(w, sheet_name="Cut4_StructureAttrib", index=False, startrow=len(C4)+2)
        C5.to_excel(w, sheet_name="Cut5_MFE_MAE", index=False); C5_detail.to_excel(w, sheet_name="Cut5_MFE_MAE", index=False, startrow=len(C5)+2)
        C6.to_excel(w, sheet_name="Cut6_VIX", index=False)
        C7_month.to_excel(w, sheet_name="Cut7_Seasonality", index=False)
        C7_dir.to_excel(w, sheet_name="Cut7_Seasonality", index=False, startrow=len(C7_month)+3)
        CAND.to_excel(w, sheet_name="Improvement_Candidates", index=False)
        T.to_excel(w, sheet_name="Full_Trade_Data", index=False)

    pd.set_option("display.width", 230)
    print("\n" + "=" * 100 + f"\nNIFTY BUTTERFLY+CALENDAR DIAGNOSTIC (n={len(T)} — SMALL SAMPLE)\n" + "=" * 100)
    print("\n--- CUT 1: Exit reason ---"); print(C1.to_string(index=False))
    print(f"\n--- CUT 2: Held-to-expiry ---\n{C2_note}");
    for h in C2_hyp: print(h)
    print("\n--- CUT 3: SL analysis ---"); print(C3.to_string(index=False)); print(C3_detail.to_string(index=False))
    print("\n--- CUT 4: Structure attribution ---"); print(C4.to_string(index=False))
    print("\n--- CUT 5: MFE/MAE ---"); print(C5.to_string(index=False))
    print("\n--- CUT 6: VIX buckets ---"); print(C6.to_string(index=False))
    print("\n--- CUT 7: Seasonality/Direction ---"); print(C7_month.to_string(index=False)); print(C7_dir.to_string(index=False)); print(f"corr(spot_return%, pnl)={corr_dir_pnl}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
