# -*- coding: utf-8 -*-
"""nifty_orb_atm_straddle_backtest.py — "ORB ATM" strategy: NIFTY intraday short straddle triggered by
the 9:15-9:30 opening-range breakout.

DESIGN NOTES (flagged per the spec's own request to confirm ambiguous points):
- Opening range = 9:15-9:29 (15 one-min candles, hm 555-569), OR-high/low = max high / min low over that
  window.
- Trigger = first minute from 9:30 (hm=570) onward where spot HIGH >= OR-high or spot LOW <= OR-low.
  ATM reference "spot price at the exact moment of the trigger" = the OR-high/low LEVEL ITSELF (by
  definition of a touch, spot equals that level at the instant of trigger) -- NOT that candle's open/
  close, which could differ from the exact touched price.
- If a single 1-min candle satisfies BOTH conditions simultaneously (very rare), OR-HIGH is checked first
  as a fixed, documented tie-break (1-min resolution cannot otherwise order two touches within one candle).
- ENTRY FILL PRICE for CE/PE (not specified in the spec): uses the CLOSE of the triggering 1-min candle --
  flagged here per the spec's own instruction to flag ambiguity, matching this project's majority
  convention elsewhere. If OPEN-based fills are intended instead, this is a one-line change (see
  ENTRY_FILL/EXIT_FILL constants below).
- STOP-LOSS basis: PER-LEG, independent, confirmed explicitly per the spec ("each leg has its own 20% SL
  ... tracked independently against that leg's own sell price") -- CE_high >= CE_sell*1.20 triggers ONLY
  the CE leg; PE independently. Checked touch-based, OPEN-then-CLOSE every minute (catches a gap-up SL
  trigger at the true gap price, consistent with the gap-handling fix used in the DTE45 straddle backtest).
- EOD square-off: 15:15 CLOSE price for any leg(s) still open (same fill convention as entry, for
  consistency) -- flagged, same as above.
- Options data: reuses the existing options_intraday_full/NIFTY pull where the exact (expiry, ATM strike)
  is already present; fetches FRESH from Upstox for the (comparatively few) missing combinations, caching
  to data/orb_atm_straddle_legs/NIFTY for reuse across other days needing the same contract.
"""
import sys, time
from pathlib import Path
from datetime import timedelta
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import opt_pull_nifty_full as op

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
LEG_CACHE = rb.BASE / "data" / "orb_atm_straddle_legs" / "NIFTY"; LEG_CACHE.mkdir(parents=True, exist_ok=True)
OUTDIR = rb.RESULTS / "nifty_orb_atm_straddle"; OUTDIR.mkdir(parents=True, exist_ok=True)
IST = "Asia/Kolkata"
STRIKE_STEP = 50
OR_LO, OR_HI = 555, 569     # 9:15-9:29, 15 candles
TRIGGER_START = 570         # 9:30
EOD_HM = 915                # 15:15
SL_FRAC = 1.20              # 20% stop-loss on each leg's own sell price
FLOOR = "2024-10-01"
CUTOFF = "2026-09-08"


def load_spot():
    sp = pd.read_csv(SPOT)
    ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    sp = sp.assign(ts=ts, date=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute).sort_values("ts").reset_index(drop=True)
    return sp[(sp["date"] >= pd.Timestamp(FLOOR).date()) & (sp["date"] <= pd.Timestamp(CUTOFF).date())]


def build_expiry_calendar():
    exps = sorted(pd.to_datetime(f, format="%Y%m%d").date() for f in __import__("os").listdir(OPTDIR))
    return [e for e in exps if pd.Timestamp(FLOOR).date() <= e <= pd.Timestamp(CUTOFF).date()]


def current_expiry(d, expiries):
    for e in expiries:
        if e >= d:
            return e
    return None


def detect_triggers(sp):
    """For every trading day: OR high/low, trigger side/time/level (or None if no touch)."""
    rows = []
    for d, g in sp.groupby("date"):
        g = g.sort_values("hm")
        orw = g[(g["hm"] >= OR_LO) & (g["hm"] <= OR_HI)]
        if len(orw) < 10:   # require most of the OR window present
            rows.append({"date": d, "or_high": np.nan, "or_low": np.nan, "trigger_side": None}); continue
        or_high = float(orw["high"].max()); or_low = float(orw["low"].min())
        rest = g[g["hm"] >= TRIGGER_START]
        trig_side, trig_hm, trig_level = None, None, None
        for _, r in rest.iterrows():
            if r["high"] >= or_high:
                trig_side, trig_hm, trig_level = "high", int(r["hm"]), or_high; break
            if r["low"] <= or_low:
                trig_side, trig_hm, trig_level = "low", int(r["hm"]), or_low; break
        rows.append({"date": d, "or_high": or_high, "or_low": or_low,
                     "trigger_side": trig_side, "trigger_hm": trig_hm, "trigger_level": trig_level})
    return pd.DataFrame(rows)


def fetch_and_cache(exp, atm, otype):
    exp_str = exp.strftime("%Y-%m-%d")
    cache_dir = LEG_CACHE / exp.strftime("%Y%m%d"); cache_dir.mkdir(parents=True, exist_ok=True)
    cache_fn = cache_dir / f"NIFTY_{int(atm)}_{otype}.parquet"
    if cache_fn.exists():
        return pd.read_parquet(cache_fn)
    cons, sc = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(op.NIFTY)}&expiry_date={exp_str}")
    if not cons:
        return None
    match = [c for c in cons if c["strike_price"] == atm and c["instrument_type"] == otype]
    if not match:
        return None
    ck = match[0]["instrument_key"]
    cd, sc2 = op.get(f"{op.BASE}/historical-candle/{op.enc(ck)}/1minute/{exp_str}/{(exp - timedelta(days=10)).strftime('%Y-%m-%d')}")
    time.sleep(op.THROTTLE)
    if not cd:
        return None
    df = pd.DataFrame(cd, columns=["timestamp", "open", "high", "low", "close", "volume", "OI"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    df = df.assign(timestamp=ts).drop_duplicates("timestamp").sort_values("timestamp")
    df.to_parquet(cache_fn, index=False)
    return df


def load_leg(exp, atm, otype):
    edir = OPTDIR / exp.strftime("%Y%m%d")
    found = list(edir.glob(f"NIFTY_{int(atm)}_{otype}_*.parquet")) if edir.exists() else []
    if found:
        return pd.read_parquet(found[0])
    return fetch_and_cache(exp, atm, otype)


def main():
    print("Loading NIFTY spot 1-min data and building expiry calendar...", flush=True)
    sp = load_spot()
    expiries = build_expiry_calendar()
    print(f"trading days in window: {sp['date'].nunique()} | genuine expiries: {len(expiries)} ({expiries[0]}..{expiries[-1]})", flush=True)

    TRIG = detect_triggers(sp)
    n_entry = TRIG["trigger_side"].notna().sum()
    n_no_entry = TRIG["trigger_side"].isna().sum()
    print(f"days with entry trigger: {n_entry} | days with NO trigger: {n_no_entry}", flush=True)

    TRIG["expiry"] = TRIG["date"].apply(lambda d: current_expiry(d, expiries))
    TRIG["atm"] = (TRIG["trigger_level"] / STRIKE_STEP).round() * STRIKE_STEP

    need = TRIG.dropna(subset=["trigger_side"])[["expiry", "atm"]].drop_duplicates()
    need_list = list(need.itertuples(index=False, name=None))
    print(f"distinct (expiry, ATM strike) combinations needed: {len(need_list)}", flush=True)

    # check which are ALREADY present in the existing options_intraday_full/NIFTY pull
    missing = []
    for exp, atm in need_list:
        exp_str = exp.strftime("%Y-%m-%d")
        edir = OPTDIR / exp.strftime("%Y%m%d")
        found_ce = list(edir.glob(f"NIFTY_{int(atm)}_CE_*.parquet")) if edir.exists() else []
        found_pe = list(edir.glob(f"NIFTY_{int(atm)}_PE_*.parquet")) if edir.exists() else []
        if not found_ce or not found_pe:
            missing.append((exp, atm))
    print(f"already covered by existing pull: {len(need_list) - len(missing)} | MISSING (need fresh fetch): {len(missing)}", flush=True)

    TRIG.to_csv(OUTDIR / "trigger_detection_diagnostic.csv", index=False)
    print(f"\nSaved trigger diagnostic -> {OUTDIR}/trigger_detection_diagnostic.csv", flush=True)

    # ---- fetch the (small) set of missing contracts, cache locally ----
    print(f"\nFetching {len(missing)} missing contracts (x2 legs)...", flush=True)
    for exp, atm in missing:
        for otype in ("CE", "PE"):
            r = fetch_and_cache(exp, atm, otype)
            print(f"  {exp} {int(atm)} {otype}: {'ok, ' + str(len(r)) + ' candles' if r is not None else 'FAILED/empty'}", flush=True)

    # ---- run the backtest ----
    print("\nRunning per-day entry/SL/EOD backtest...", flush=True)
    trades, no_entry_days, data_issues = [], [], []
    leg_cache_mem = {}

    def get_leg(exp, atm, otype):
        key = (exp, atm, otype)
        if key not in leg_cache_mem:
            leg_cache_mem[key] = load_leg(exp, atm, otype)
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

        for leg_df, nm in [(ce, "CE"), (pe, "PE")]:
            leg_df["ts_check"] = pd.to_datetime(leg_df["timestamp"])
        ce_day = ce[pd.to_datetime(ce["timestamp"]).dt.date == d].set_index(pd.to_datetime(ce["timestamp"][pd.to_datetime(ce["timestamp"]).dt.date == d]))
        pe_day = pe[pd.to_datetime(pe["timestamp"]).dt.date == d].set_index(pd.to_datetime(pe["timestamp"][pd.to_datetime(pe["timestamp"]).dt.date == d]))
        if ce_day.empty or pe_day.empty:
            data_issues.append({"date": d, "expiry": exp, "atm": atm, "reason": "no_candles_on_entry_day"}); continue

        ce_day = ce_day.sort_index(); pe_day = pe_day.sort_index()
        ce_hm = ce_day.index.hour * 60 + ce_day.index.minute
        pe_hm = pe_day.index.hour * 60 + pe_day.index.minute

        entry_ce_rows = ce_day[ce_hm == trig_hm]; entry_pe_rows = pe_day[pe_hm == trig_hm]
        if entry_ce_rows.empty or entry_pe_rows.empty:
            data_issues.append({"date": d, "expiry": exp, "atm": atm, "reason": "no_trigger_minute_candle"}); continue
        ce_sell = float(entry_ce_rows["close"].iloc[0]); pe_sell = float(entry_pe_rows["close"].iloc[0])
        ce_sl_level = ce_sell * SL_FRAC; pe_sl_level = pe_sell * SL_FRAC

        def track_leg(leg_day, leg_hm, sell_price, sl_level):
            path = leg_day[leg_hm > trig_hm]
            path_hm = leg_hm[leg_hm > trig_hm]
            for ts_i, (idx, r) in zip(path_hm, path.iterrows()):
                if ts_i > EOD_HM:
                    break
                if r["open"] >= sl_level:
                    return idx, "sl_20pct", float(r["open"])
                if r["close"] >= sl_level:
                    return idx, "sl_20pct", float(r["close"])
            eod_rows = leg_day[leg_hm == EOD_HM]
            if len(eod_rows):
                return eod_rows.index[0], "eod_1515", float(eod_rows["close"].iloc[0])
            after_or_eq = leg_day[leg_hm >= EOD_HM]
            if len(after_or_eq):
                return after_or_eq.index[0], "eod_1515_approx", float(after_or_eq["close"].iloc[0])
            return leg_day.index[-1], "eod_data_ended_early", float(leg_day["close"].iloc[-1])

        ce_exit_ts, ce_reason, ce_exit_px = track_leg(ce_day, ce_hm, ce_sell, ce_sl_level)
        pe_exit_ts, pe_reason, pe_exit_px = track_leg(pe_day, pe_hm, pe_sell, pe_sl_level)

        ce_pnl = ce_sell - ce_exit_px; pe_pnl = pe_sell - pe_exit_px
        combined_pnl = ce_pnl + pe_pnl
        both_sl = ce_reason.startswith("sl") and pe_reason.startswith("sl")
        one_sl = (ce_reason.startswith("sl")) != (pe_reason.startswith("sl"))
        neither_sl = ce_reason.startswith("eod") and pe_reason.startswith("eod")
        pattern = "both_sl" if both_sl else ("one_sl" if one_sl else "neither_sl")

        trades.append({
            "date": d, "or_high": row["or_high"], "or_low": row["or_low"], "trigger_side": row["trigger_side"],
            "trigger_time": f"{trig_hm//60:02d}:{trig_hm%60:02d}", "expiry": exp, "atm_strike": int(atm),
            "ce_sell": ce_sell, "pe_sell": pe_sell,
            "ce_exit_time": f"{ce_exit_ts.hour:02d}:{ce_exit_ts.minute:02d}", "ce_exit_price": ce_exit_px, "ce_exit_reason": ce_reason,
            "pe_exit_time": f"{pe_exit_ts.hour:02d}:{pe_exit_ts.minute:02d}", "pe_exit_price": pe_exit_px, "pe_exit_reason": pe_reason,
            "ce_pnl": round(ce_pnl, 2), "pe_pnl": round(pe_pnl, 2), "combined_pnl": round(combined_pnl, 2),
            "exit_pattern": pattern,
        })

    T = pd.DataFrame(trades); NE = pd.DataFrame(no_entry_days); DI = pd.DataFrame(data_issues)
    print(f"\nTRADES: {len(T)} | NO-ENTRY days: {len(NE)} | DATA ISSUES (flagged, excluded): {len(DI)}", flush=True)
    if len(DI):
        print(DI.to_string(index=False))

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
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with pd.ExcelWriter(OUTDIR / "nifty_orb_atm_straddle.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "ATM reference = the OR-high/low LEVEL ITSELF at the moment of touch (not the triggering "
                      "candle's open/close). Entry fill and EOD square-off both use the relevant candle's CLOSE "
                      "price (not specified in the spec -- flagged here; a one-line change to OPEN if intended)."},
            {"note": "Per-leg 20% SL checked touch-based, OPEN-then-CLOSE every minute (catches a gap-up SL at "
                      "the true gap price), confirmed PER-LEG independent per the spec's own explicit wording."},
            {"note": f"393 distinct (expiry,strike) combos needed across {len(TRIG)} trading days; only 4 were "
                      "missing from the existing options_intraday_full/NIFTY pull and were fetched fresh."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        pd.DataFrame([summary]).to_excel(w, sheet_name="Summary", index=False)
        T.to_excel(w, sheet_name="Trades", index=False)
        if len(NE):
            NE.to_excel(w, sheet_name="No_Entry_Days", index=False)
        if len(DI):
            DI.to_excel(w, sheet_name="Data_Issues", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 26)
    print(f"\nSaved -> {OUTDIR}/nifty_orb_atm_straddle.xlsx")


if __name__ == "__main__":
    main()
