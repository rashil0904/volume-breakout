# -*- coding: utf-8 -*-
"""nifty_orb_atm_long_straddle_backtest.py — "ORB ATM Long Straddle" variant: identical trigger mechanism,
data, ATM selection, and expiry calendar as nifty_orb_atm_straddle_backtest.py (the SHORT version), but
BUYING both legs instead of selling. Reuses that module's load_spot/build_expiry_calendar/current_expiry/
detect_triggers/fetch_and_cache/load_leg directly -- no new API calls, same 5 flagged data-gap days.

DIRECTION FLIP (confirmed per the spec's own request):
- Entry: BUY CE + BUY PE at the triggering candle's CLOSE (same fill convention as the short version, for
  direct comparability).
- SL: LOSS-based, per leg, independent -- CE_sl_level = ce_buy * 0.80 (a 20% FALL), triggered when that
  leg's price drops TO OR BELOW its own SL level (the mirror image of the short version's "rises to 120%").
  Checked touch-based, OPEN-then-CLOSE every minute (a gap-DOWN triggers at the true gap price).
- P&L: pnl = exit_price - buy_price per leg (profit if the leg's price rose from entry, the correct sign
  for a long position -- opposite of the short version's buy_price - exit_price).
- EOD square-off: 15:15 CLOSE, same convention as the short version.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import nifty_orb_atm_straddle_backtest as short

OUTDIR = rb.RESULTS / "nifty_orb_atm_long_straddle"; OUTDIR.mkdir(parents=True, exist_ok=True)
SL_FRAC_LONG = 0.80   # 20% FALL from buy price
EOD_HM = short.EOD_HM


def main():
    print("Loading NIFTY spot 1-min data and building expiry calendar (shared with SHORT version)...", flush=True)
    sp = short.load_spot()
    expiries = short.build_expiry_calendar()
    TRIG = short.detect_triggers(sp)
    TRIG["expiry"] = TRIG["date"].apply(lambda d: short.current_expiry(d, expiries))
    TRIG["atm"] = (TRIG["trigger_level"] / short.STRIKE_STEP).round() * short.STRIKE_STEP
    print(f"trading days: {len(TRIG)} | triggers: {TRIG['trigger_side'].notna().sum()}", flush=True)

    trades, no_entry_days, data_issues = [], [], []
    leg_cache_mem = {}

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
        exp = row["expiry"]; atm = row["atm"]; trig_hm = int(row["trigger_hm"])

        ce = get_leg(exp, atm, "CE"); pe = get_leg(exp, atm, "PE")
        if ce is None or pe is None:
            data_issues.append({"date": d, "expiry": exp, "atm": atm, "reason": "leg_fetch_failed"}); continue

        ce_ts = pd.to_datetime(ce["timestamp"]); pe_ts = pd.to_datetime(pe["timestamp"])
        ce_day = ce[ce_ts.dt.date == d].set_index(ce_ts[ce_ts.dt.date == d]).sort_index()
        pe_day = pe[pe_ts.dt.date == d].set_index(pe_ts[pe_ts.dt.date == d]).sort_index()
        if ce_day.empty or pe_day.empty:
            data_issues.append({"date": d, "expiry": exp, "atm": atm, "reason": "no_candles_on_entry_day"}); continue

        ce_hm = ce_day.index.hour * 60 + ce_day.index.minute
        pe_hm = pe_day.index.hour * 60 + pe_day.index.minute

        entry_ce_rows = ce_day[ce_hm == trig_hm]; entry_pe_rows = pe_day[pe_hm == trig_hm]
        if entry_ce_rows.empty or entry_pe_rows.empty:
            data_issues.append({"date": d, "expiry": exp, "atm": atm, "reason": "no_trigger_minute_candle"}); continue
        ce_buy = float(entry_ce_rows["close"].iloc[0]); pe_buy = float(entry_pe_rows["close"].iloc[0])
        ce_sl_level = ce_buy * SL_FRAC_LONG; pe_sl_level = pe_buy * SL_FRAC_LONG

        def track_leg(leg_day, leg_hm, sl_level):
            path = leg_day[leg_hm > trig_hm]
            path_hm = leg_hm[leg_hm > trig_hm]
            for ts_i, (idx, r) in zip(path_hm, path.iterrows()):
                if ts_i > EOD_HM:
                    break
                if r["open"] <= sl_level:
                    return idx, "sl_20pct", float(r["open"])
                if r["close"] <= sl_level:
                    return idx, "sl_20pct", float(r["close"])
            eod_rows = leg_day[leg_hm == EOD_HM]
            if len(eod_rows):
                return eod_rows.index[0], "eod_1515", float(eod_rows["close"].iloc[0])
            after_or_eq = leg_day[leg_hm >= EOD_HM]
            if len(after_or_eq):
                return after_or_eq.index[0], "eod_1515_approx", float(after_or_eq["close"].iloc[0])
            return leg_day.index[-1], "eod_data_ended_early", float(leg_day["close"].iloc[-1])

        ce_exit_ts, ce_reason, ce_exit_px = track_leg(ce_day, ce_hm, ce_sl_level)
        pe_exit_ts, pe_reason, pe_exit_px = track_leg(pe_day, pe_hm, pe_sl_level)

        # LONG position: pnl = exit - entry (profit if price rose), opposite sign of the short version
        ce_pnl = ce_exit_px - ce_buy; pe_pnl = pe_exit_px - pe_buy
        combined_pnl = ce_pnl + pe_pnl
        both_sl = ce_reason.startswith("sl") and pe_reason.startswith("sl")
        one_sl = (ce_reason.startswith("sl")) != (pe_reason.startswith("sl"))
        pattern = "both_sl" if both_sl else ("one_sl" if one_sl else "neither_sl")

        trades.append({
            "date": d, "or_high": row["or_high"], "or_low": row["or_low"], "trigger_side": row["trigger_side"],
            "trigger_time": f"{trig_hm//60:02d}:{trig_hm%60:02d}", "expiry": exp, "atm_strike": int(atm),
            "ce_buy": ce_buy, "pe_buy": pe_buy,
            "ce_exit_time": f"{ce_exit_ts.hour:02d}:{ce_exit_ts.minute:02d}", "ce_exit_price": ce_exit_px, "ce_exit_reason": ce_reason,
            "pe_exit_time": f"{pe_exit_ts.hour:02d}:{pe_exit_ts.minute:02d}", "pe_exit_price": pe_exit_px, "pe_exit_reason": pe_reason,
            "ce_pnl": round(ce_pnl, 2), "pe_pnl": round(pe_pnl, 2), "combined_pnl": round(combined_pnl, 2),
            "exit_pattern": pattern,
        })

    T = pd.DataFrame(trades); NE = pd.DataFrame(no_entry_days); DI = pd.DataFrame(data_issues)
    print(f"\nTRADES: {len(T)} | NO-ENTRY days: {len(NE)} | DATA ISSUES (flagged, excluded): {len(DI)}", flush=True)

    win = T["combined_pnl"] > 0
    summary = {
        "total_days_in_window": len(TRIG), "days_with_entry": len(T), "days_no_entry": len(NE),
        "win_rate_pct": round(win.mean() * 100, 2), "total_pnl": round(T["combined_pnl"].sum(), 2),
        "avg_pnl": round(T["combined_pnl"].mean(), 2),
    }
    for pat in ["both_sl", "one_sl", "neither_sl"]:
        sub = T[T["exit_pattern"] == pat]
        summary[f"pct_{pat}"] = round(len(sub) / len(T) * 100, 2) if len(T) else 0.0
        summary[f"avg_pnl_{pat}"] = round(sub["combined_pnl"].mean(), 2) if len(sub) else np.nan

    pd.set_option("display.width", 240)
    print("\n=== SUMMARY (LONG straddle) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # ---- comparison vs the SHORT version, if its results exist ----
    short_fn = rb.RESULTS / "nifty_orb_atm_straddle" / "nifty_orb_atm_straddle.xlsx"
    comparison = None
    if short_fn.exists():
        try:
            short_summary = pd.read_excel(short_fn, sheet_name="Summary").iloc[0].to_dict()
            comparison = pd.DataFrame([
                {"metric": "days_with_entry", "SHORT": short_summary["days_with_entry"], "LONG": summary["days_with_entry"]},
                {"metric": "win_rate_pct", "SHORT": short_summary["win_rate_pct"], "LONG": summary["win_rate_pct"]},
                {"metric": "total_pnl", "SHORT": short_summary["total_pnl"], "LONG": summary["total_pnl"]},
                {"metric": "avg_pnl", "SHORT": short_summary["avg_pnl"], "LONG": summary["avg_pnl"]},
                {"metric": "pct_both_sl", "SHORT": short_summary["pct_both_sl"], "LONG": summary["pct_both_sl"]},
                {"metric": "avg_pnl_both_sl", "SHORT": short_summary["avg_pnl_both_sl"], "LONG": summary["avg_pnl_both_sl"]},
                {"metric": "pct_one_sl", "SHORT": short_summary["pct_one_sl"], "LONG": summary["pct_one_sl"]},
                {"metric": "avg_pnl_one_sl", "SHORT": short_summary["avg_pnl_one_sl"], "LONG": summary["avg_pnl_one_sl"]},
                {"metric": "pct_neither_sl", "SHORT": short_summary["pct_neither_sl"], "LONG": summary["pct_neither_sl"]},
                {"metric": "avg_pnl_neither_sl", "SHORT": short_summary["avg_pnl_neither_sl"], "LONG": summary["avg_pnl_neither_sl"]},
            ])
            print("\n=== SHORT vs LONG comparison ===")
            print(comparison.to_string(index=False))
        except Exception as e:
            print(f"\n(comparison skipped: {e})")

    out_fn = OUTDIR / "nifty_orb_atm_long_straddle.xlsx"
    try:
        out_fn.touch(exist_ok=True)
    except PermissionError:
        out_fn = OUTDIR / "nifty_orb_atm_long_straddle_v2.xlsx"
    with pd.ExcelWriter(out_fn, engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "Same trigger/ATM/expiry mechanism as the SHORT ORB ATM straddle (identical trading days, "
                      "identical strikes, identical 5 flagged data-gap days) -- only direction is flipped: BUY "
                      "CE+PE at the trigger candle's close, SL = 20% FALL from buy price (loss-based), P&L = "
                      "exit-entry per leg (long-position sign convention)."},
            {"note": "Entry/EOD fill price uses CLOSE, same convention as the short version, for direct "
                      "comparability -- flagged, as it was not specified in the spec."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        pd.DataFrame([summary]).to_excel(w, sheet_name="Summary", index=False)
        if comparison is not None:
            comparison.to_excel(w, sheet_name="SHORT_vs_LONG", index=False)
        T.to_excel(w, sheet_name="Trades", index=False)
        if len(NE):
            NE.to_excel(w, sheet_name="No_Entry_Days", index=False)
        if len(DI):
            DI.to_excel(w, sheet_name="Data_Issues", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 26)
    print(f"\nSaved -> {out_fn}")


if __name__ == "__main__":
    main()
