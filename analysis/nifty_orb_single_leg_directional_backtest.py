# -*- coding: utf-8 -*-
"""nifty_orb_single_leg_directional_backtest.py — single-leg, DIRECTIONAL variants of the ORB ATM strategy:
trade only the ONE option that matches the breakout direction, not the full straddle. Since "high breakout
-> buy CE or sell PE" names two DIFFERENT risk profiles (long call vs short put), both are built and
reported side by side rather than picking one silently:

  Variant DIR_LONG:  high breakout -> BUY CE (bullish, long) | low breakout -> BUY PE (bearish, long)
  Variant DIR_SHORT: high breakout -> SELL PE (bullish, short) | low breakout -> SELL CE (bearish, short)

Same trigger/ATM/expiry mechanism as the straddle versions (reused directly, same 481 days, same 5 flagged
data-gap days). SL is the SAME 20%-of-entry-price rule as the straddle versions, direction-appropriate:
DIR_LONG = 20% fall (loss-based, like the long straddle); DIR_SHORT = 20% rise (loss-based for a short,
like the short straddle). EOD square-off at 15:15 close if SL not hit. Single leg per day -- no "both/one/
neither SL" pattern (only one leg exists); reported as SL-hit vs EOD-exit instead.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import nifty_orb_atm_straddle_backtest as short

OUTDIR = rb.RESULTS / "nifty_orb_single_leg_directional"; OUTDIR.mkdir(parents=True, exist_ok=True)
EOD_HM = short.EOD_HM
SL_LONG_FRAC = 0.80   # 20% fall
SL_SHORT_FRAC = 1.20  # 20% rise


def track_leg(leg_day, leg_hm, trig_hm, entry_price, sl_level, is_long):
    path = leg_day[leg_hm > trig_hm]
    path_hm = leg_hm[leg_hm > trig_hm]
    for ts_i, (idx, r) in zip(path_hm, path.iterrows()):
        if ts_i > EOD_HM:
            break
        if is_long:
            if r["open"] <= sl_level:
                return idx, "sl_20pct", float(r["open"])
            if r["close"] <= sl_level:
                return idx, "sl_20pct", float(r["close"])
        else:
            if r["open"] >= sl_level:
                return idx, "sl_20pct", float(r["open"])
            if r["close"] >= sl_level:
                return idx, "sl_20pct", float(r["close"])
    eod_rows = leg_day[leg_hm == EOD_HM]
    if len(eod_rows):
        return eod_rows.index[0], "eod_1515", float(eod_rows["close"].iloc[0])
    after = leg_day[leg_hm >= EOD_HM]
    if len(after):
        return after.index[0], "eod_1515_approx", float(after["close"].iloc[0])
    return leg_day.index[-1], "eod_data_ended_early", float(leg_day["close"].iloc[-1])


def run_variant(TRIG, variant, leg_cache_mem):
    """variant: 'DIR_LONG' or 'DIR_SHORT'."""
    is_long = variant == "DIR_LONG"
    trades, no_entry_days, data_issues = [], [], []

    def get_leg(exp, atm, otype):
        key = (exp, atm, otype)
        if key not in leg_cache_mem:
            leg_cache_mem[key] = short.load_leg(exp, atm, otype)
        return leg_cache_mem[key]

    for _, row in TRIG.iterrows():
        d = row["date"]
        if pd.isna(row["trigger_side"]) or pd.isna(row["expiry"]):
            no_entry_days.append({"date": d, "reason": "no_or_touch" if pd.isna(row["trigger_side"]) else "no_current_expiry"})
            continue
        exp = row["expiry"]; atm = row["atm"]; trig_hm = int(row["trigger_hm"]); side = row["trigger_side"]

        if is_long:
            otype = "CE" if side == "high" else "PE"
        else:
            otype = "PE" if side == "high" else "CE"

        leg = get_leg(exp, atm, otype)
        if leg is None:
            data_issues.append({"date": d, "expiry": exp, "atm": atm, "otype": otype, "reason": "leg_fetch_failed"}); continue
        leg_ts = pd.to_datetime(leg["timestamp"])
        leg_day = leg[leg_ts.dt.date == d].set_index(leg_ts[leg_ts.dt.date == d]).sort_index()
        if leg_day.empty:
            data_issues.append({"date": d, "expiry": exp, "atm": atm, "otype": otype, "reason": "no_candles_on_entry_day"}); continue
        leg_hm = leg_day.index.hour * 60 + leg_day.index.minute
        entry_rows = leg_day[leg_hm == trig_hm]
        if entry_rows.empty:
            data_issues.append({"date": d, "expiry": exp, "atm": atm, "otype": otype, "reason": "no_trigger_minute_candle"}); continue
        entry_px = float(entry_rows["close"].iloc[0])
        sl_level = entry_px * (SL_LONG_FRAC if is_long else SL_SHORT_FRAC)

        exit_ts, exit_reason, exit_px = track_leg(leg_day, leg_hm, trig_hm, entry_px, sl_level, is_long)
        pnl = (exit_px - entry_px) if is_long else (entry_px - exit_px)

        trades.append({
            "date": d, "or_high": row["or_high"], "or_low": row["or_low"], "trigger_side": side,
            "trigger_time": f"{trig_hm//60:02d}:{trig_hm%60:02d}", "expiry": exp, "atm_strike": int(atm),
            "option_type": otype, "action": "BUY" if is_long else "SELL", "entry_price": entry_px,
            "exit_time": f"{exit_ts.hour:02d}:{exit_ts.minute:02d}", "exit_price": exit_px, "exit_reason": exit_reason,
            "pnl": round(pnl, 2),
        })

    T = pd.DataFrame(trades); NE = pd.DataFrame(no_entry_days); DI = pd.DataFrame(data_issues)
    win = T["pnl"] > 0
    summary = {
        "variant": variant, "days_with_entry": len(T), "days_no_entry": len(NE), "data_issues": len(DI),
        "win_rate_pct": round(win.mean() * 100, 2), "total_pnl": round(T["pnl"].sum(), 2), "avg_pnl": round(T["pnl"].mean(), 2),
        "pct_sl_hit": round((T["exit_reason"] == "sl_20pct").mean() * 100, 2),
        "avg_pnl_sl_hit": round(T.loc[T["exit_reason"] == "sl_20pct", "pnl"].mean(), 2),
        "pct_eod_exit": round((T["exit_reason"].isin(["eod_1515", "eod_1515_approx"])).mean() * 100, 2),
        "avg_pnl_eod_exit": round(T.loc[T["exit_reason"].isin(["eod_1515", "eod_1515_approx"]), "pnl"].mean(), 2),
        "high_breakout_days": int((T["trigger_side"] == "high").sum()), "high_breakout_avg_pnl": round(T.loc[T["trigger_side"] == "high", "pnl"].mean(), 2),
        "low_breakout_days": int((T["trigger_side"] == "low").sum()), "low_breakout_avg_pnl": round(T.loc[T["trigger_side"] == "low", "pnl"].mean(), 2),
    }
    return T, NE, DI, summary


def main():
    print("Loading NIFTY spot 1-min data and building expiry calendar (shared)...", flush=True)
    sp = short.load_spot()
    expiries = short.build_expiry_calendar()
    TRIG = short.detect_triggers(sp)
    TRIG["expiry"] = TRIG["date"].apply(lambda d: short.current_expiry(d, expiries))
    TRIG["atm"] = (TRIG["trigger_level"] / short.STRIKE_STEP).round() * short.STRIKE_STEP
    print(f"trading days: {len(TRIG)} | triggers: {TRIG['trigger_side'].notna().sum()}", flush=True)

    leg_cache_mem = {}
    results = {}
    for variant in ["DIR_LONG", "DIR_SHORT"]:
        print(f"\nRunning {variant}...", flush=True)
        T, NE, DI, summary = run_variant(TRIG, variant, leg_cache_mem)
        results[variant] = (T, NE, DI, summary)
        pd.set_option("display.width", 200)
        print(f"trades={len(T)} no_entry={len(NE)} issues={len(DI)}")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    # ---- 4-way comparison: DIR_LONG, DIR_SHORT, and the two straddle variants if available ----
    comp_rows = [results["DIR_LONG"][3], results["DIR_SHORT"][3]]
    for name, path in [("SHORT_STRADDLE", rb.RESULTS / "nifty_orb_atm_straddle" / "nifty_orb_atm_straddle.xlsx"),
                        ("LONG_STRADDLE", rb.RESULTS / "nifty_orb_atm_long_straddle" / "nifty_orb_atm_long_straddle.xlsx")]:
        if path.exists():
            try:
                s = pd.read_excel(path, sheet_name="Summary").iloc[0].to_dict()
                comp_rows.append({"variant": name, "days_with_entry": s["days_with_entry"], "win_rate_pct": s["win_rate_pct"],
                                  "total_pnl": s["total_pnl"], "avg_pnl": s["avg_pnl"]})
            except Exception:
                pass
    COMP = pd.DataFrame(comp_rows)
    print("\n=== 4-WAY COMPARISON ===")
    print(COMP[[c for c in ["variant", "days_with_entry", "win_rate_pct", "total_pnl", "avg_pnl"] if c in COMP.columns]].to_string(index=False))

    out_fn = OUTDIR / "orb_single_leg_directional.xlsx"
    try:
        out_fn.touch(exist_ok=True)
    except PermissionError:
        out_fn = OUTDIR / "orb_single_leg_directional_v2.xlsx"
    with pd.ExcelWriter(out_fn, engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "Two single-leg directional variants of the ORB ATM strategy: DIR_LONG (buy CE on high "
                      "breakout, buy PE on low breakout) and DIR_SHORT (sell PE on high breakout, sell CE on "
                      "low breakout) -- built as separate variants since 'buy CE' and 'sell PE' on the same "
                      "breakout are different risk profiles, not interchangeable choices."},
            {"note": "Same trigger/ATM/expiry mechanism, same 481 trading days, same flagged data gaps as the "
                      "two straddle backtests. SL = 20% of entry price, direction-appropriate (fall for long, "
                      "rise for short); EOD square-off at 15:15 close if SL not hit."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        COMP.to_excel(w, sheet_name="4Way_Comparison", index=False)
        for variant in ["DIR_LONG", "DIR_SHORT"]:
            T, NE, DI, summary = results[variant]
            pd.DataFrame([summary]).to_excel(w, sheet_name=f"{variant}_Summary", index=False)
            T.to_excel(w, sheet_name=f"{variant}_Trades", index=False)
            if len(NE):
                NE.to_excel(w, sheet_name=f"{variant}_No_Entry", index=False)
            if len(DI):
                DI.to_excel(w, sheet_name=f"{variant}_Data_Issues", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 26)
    print(f"\nSaved -> {out_fn}")


if __name__ == "__main__":
    main()
