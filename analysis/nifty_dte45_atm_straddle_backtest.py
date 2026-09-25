# -*- coding: utf-8 -*-
"""nifty_dte45_atm_straddle_backtest.py — "Nifty Monthly DTE 45 ATM Short Straddle" backtest.

DESIGN NOTES (per explicit spec):
- Universe: NIFTY MONTHLY options only. Monthly expiry calendar derived from the just-pulled REAL
  data/futures_intraday_full/NIFTY manifest (22 genuine monthly futures contracts, Oct-2024..Aug-2026;
  Dec-2024 excluded -- confirmed genuine Upstox data gap, not substituted).
- Entry: nearest trading day to (expiry - 45 calendar days), 15:15 OPEN price (changed from close per
  explicit follow-up instruction -- both the futures ATM reference and the CE/PE entry premium use the
  15:15 candle's OPEN, not its close).
- ATM strike = nearest 50-point strike to the MONTHLY FUTURES 15:15 OPEN on the entry day (NOT spot,
  NOT synthetic) -- this is why the futures pull was needed first.
- Structure: sell 1 lot ATM CE + 1 lot ATM PE.
- Exit priority, touch-based every 1 min from entry: (1) combined premium <= 50% of entry credit -> target;
  (2) combined premium >= 200% of entry credit -> stop; (3) neither by DTE=21 -> time exit at 15:15 OPEN.
  "Combined premium" per minute checks OPEN before CLOSE (CE+PE of each), so a gap-through is filled at
  the true gap-open price rather than that candle's close -- NOT low+low/high+high (that would double-
  count an impossible simultaneous favorable/adverse touch on both legs at different instants).
- OPTIONS DATA: fetched FRESH per cycle directly from Upstox for the EXACT futures-derived ATM strike
  (NOT reused from the pre-existing options_intraday_full/NIFTY pull, which selected its +-15 strike band
  around SPOT-AT-EXPIRY, not futures-at-DTE45 -- confirmed by inspection that the two ATM references can
  diverge by more than the pulled band over a 45-day window). Any cycle whose fetched premium data does
  NOT reach back to the actual entry date is EXCLUDED and flagged -- not substituted with spot/synthetic
  pricing or a truncated start, per the explicit "stop and flag" instruction in the spec.
"""
import sys, time
from pathlib import Path
from datetime import timedelta
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import opt_pull_nifty_full as op

NIFTY = "NSE_INDEX|Nifty 50"
FUT_MANIFEST = rb.BASE / "data" / "futures_intraday_full" / "manifest_nifty_fut.csv"
FUT_DIR = rb.BASE / "data" / "futures_intraday_full" / "NIFTY"
LEG_CACHE = rb.BASE / "data" / "dte45_straddle_legs" / "NIFTY"; LEG_CACHE.mkdir(parents=True, exist_ok=True)
OUTDIR = rb.RESULTS / "nifty_dte45_atm_straddle"; OUTDIR.mkdir(parents=True, exist_ok=True)
STRIKE_STEP = 50
TARGET_FRAC = 0.50   # exit when straddle value decays to 50% of entry credit (profit = 50% of credit)
STOPLOSS_FRAC = 2.00  # exit when straddle value reaches 200% of entry credit (e.g. sold for 600 -> stop
                       # at 1200; the LOSS at that point equals the original credit, per spec's own example)
ENTRY_DTE, EXIT_DTE = 45, 21
CHUNK, LOOKBACK = 30, 60   # 60-day lookback safely covers the 24-day DTE45->DTE21 window + buffer


def nearest_trading_day(target_date, trading_days_sorted):
    """trading_days_sorted: sorted list of date objects. Returns the nearest one (ties -> earlier)."""
    best, best_diff = None, None
    for d in trading_days_sorted:
        diff = abs((d - target_date).days)
        if best_diff is None or diff < best_diff or (diff == best_diff and d < best):
            best, best_diff = d, diff
    return best


def fetch_leg(cons, exp, strike, otype, from_date, to_date):
    match = [c for c in cons if c["strike_price"] == strike and c["instrument_type"] == otype]
    if not match:
        return None, "no_contract_at_strike"
    c = match[0]; ck = c["instrument_key"]; sym = c["trading_symbol"]
    fn = LEG_CACHE / exp.replace("-", "") / (sym.replace(" ", "_").replace("/", "-") + ".parquet")
    fn.parent.mkdir(parents=True, exist_ok=True)
    if fn.exists():
        return pd.read_parquet(fn), None
    frames = []; hi = to_date; got = False; consec = 0
    floor = from_date - timedelta(days=10)   # small buffer past the needed start
    for _ in range(LOOKBACK // CHUNK + 3):
        lo = hi - timedelta(days=CHUNK)
        if hi < floor:
            break
        cd, sc = op.get(f"{op.BASE}/historical-candle/{op.enc(ck)}/1minute/{hi.isoformat()}/{lo.isoformat()}")
        time.sleep(op.THROTTLE)
        if cd:
            frames.append(cd); got = True; consec = 0
        else:
            consec += 1
            if got or consec >= 2:
                break
        hi = lo
    rows = [r for f in frames for r in f]
    if not rows:
        return None, "empty_no_candles"
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "OI"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.assign(timestamp=ts, symbol=sym, strike=strike, option_type=otype, expiry_date=exp).drop_duplicates("timestamp").sort_values("timestamp")
    df.to_parquet(fn, index=False)
    return df, None


def main():
    man = pd.read_csv(FUT_MANIFEST)
    man["expiry"] = pd.to_datetime(man["expiry"]).dt.date
    man = man.sort_values("expiry").reset_index(drop=True)
    print(f"monthly futures expiries available: {len(man)} ({man['expiry'].min()} .. {man['expiry'].max()})", flush=True)

    trades, skipped = [], []
    for _, row in man.iterrows():
        exp = row["expiry"]; exp_str = exp.strftime("%Y-%m-%d")
        fut_fn = FUT_DIR / exp.strftime("%Y%m%d") / (row["symbol"].replace(" ", "_").replace("/", "-") + ".parquet")
        fut = pd.read_parquet(fut_fn)
        fut["timestamp"] = pd.to_datetime(fut["timestamp"])
        fut["date"] = fut["timestamp"].dt.date
        trading_days = sorted(fut["date"].unique())

        entry_target = exp - timedelta(days=ENTRY_DTE)
        exit_target = exp - timedelta(days=EXIT_DTE)
        entry_date = nearest_trading_day(entry_target, trading_days)
        exit_date_fallback = nearest_trading_day(exit_target, trading_days)
        entry_dte_actual = (exp - entry_date).days
        exit_dte_actual = (exp - exit_date_fallback).days
        direction_flag = "same_day" if entry_date == entry_target else ("later" if entry_date > entry_target else "earlier")

        f1515 = fut[(fut["date"] == entry_date) & (fut["timestamp"].dt.strftime("%H:%M") == "15:15")]
        if f1515.empty:
            skipped.append({"expiry": exp_str, "reason": "no_1515_futures_candle_on_entry_day", "entry_date": entry_date}); continue
        fut_px = float(f1515["open"].iloc[0])
        atm = round(fut_px / STRIKE_STEP) * STRIKE_STEP

        cons, sc = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(NIFTY)}&expiry_date={exp_str}")
        if not cons:
            skipped.append({"expiry": exp_str, "reason": f"no_option_contracts_http{sc}", "entry_date": entry_date}); continue

        ce, err_ce = fetch_leg(cons, exp_str, atm, "CE", entry_date, exp)
        pe, err_pe = fetch_leg(cons, exp_str, atm, "PE", entry_date, exp)
        if ce is None or pe is None:
            skipped.append({"expiry": exp_str, "reason": f"leg_fetch_failed CE={err_ce} PE={err_pe}", "entry_date": entry_date, "atm": atm}); continue

        ce_min_date = ce["timestamp"].min().date(); pe_min_date = pe["timestamp"].min().date()
        if ce_min_date > entry_date or pe_min_date > entry_date:
            skipped.append({"expiry": exp_str, "reason": f"DATA GAP: premium history starts AFTER entry date "
                            f"(CE from {ce_min_date}, PE from {pe_min_date}, need {entry_date}) -- EXCLUDED, not substituted",
                            "entry_date": entry_date, "atm": atm}); continue

        ces = ce.set_index("timestamp")["close"]; pes = pe.set_index("timestamp")["close"]
        ceo = ce.set_index("timestamp")["open"]; peo = pe.set_index("timestamp")["open"]
        common_ts = ces.index.intersection(pes.index).sort_values()
        common_ts = common_ts[common_ts.normalize() >= pd.Timestamp(entry_date)]
        if common_ts.empty:
            skipped.append({"expiry": exp_str, "reason": "no_common_timestamps_from_entry", "entry_date": entry_date, "atm": atm}); continue

        entry_ts_candidates = common_ts[(common_ts.normalize() == pd.Timestamp(entry_date)) & (common_ts.strftime("%H:%M") == "15:15")]
        if entry_ts_candidates.empty:
            skipped.append({"expiry": exp_str, "reason": "no_1515_option_candle_on_entry_day", "entry_date": entry_date, "atm": atm}); continue
        entry_ts = entry_ts_candidates[0]
        entry_ce = float(ceo.loc[entry_ts]); entry_pe = float(peo.loc[entry_ts]); entry_credit = entry_ce + entry_pe
        target_level = entry_credit * TARGET_FRAC
        stop_level = entry_credit * STOPLOSS_FRAC

        # HARD CEILING at DTE=21: target/SL are only checked from entry THROUGH the DTE=21 time-exit boundary
        # -- the position is forcibly closed at DTE=21 regardless, so the touch-search window must never
        # extend past that point (a prior bug let it search all the way to expiry, producing "target" hits
        # reported at DTE<21, which should never happen since the position wouldn't still be open by then).
        boundary_candidates = common_ts[common_ts.normalize() <= pd.Timestamp(exit_date_fallback)]
        boundary_ts = boundary_candidates.max() if len(boundary_candidates) else entry_ts
        path_ts = common_ts[(common_ts > entry_ts) & (common_ts <= boundary_ts)]
        combined_close = (ces.loc[path_ts] + pes.loc[path_ts])
        combined_open = (ceo.loc[path_ts] + peo.loc[path_ts])
        # check OPEN before CLOSE at every minute -- catches a level touched right at a gap-open (the true
        # tradable price when the market reopens) instead of only the close of that first post-gap candle,
        # and also catches a touch-then-revert within a single minute if it shows up as an open-vs-close gap.
        # (True intra-minute high/low touches that revert with NO visible open/close divergence still can't
        # be recovered from 1-min OHLC without tick data -- flagged as a residual limitation, not silently
        # assumed away.)
        exit_ts, exit_reason, exit_val, exit_via = None, None, None, None
        for t in path_ts:
            o = combined_open.loc[t]; c = combined_close.loc[t]
            if o <= target_level:
                exit_ts, exit_reason, exit_val, exit_via = t, "target_50pct", float(o), "gap_open"; break
            if o >= stop_level:
                exit_ts, exit_reason, exit_val, exit_via = t, "stoploss_200pct_of_credit", float(o), "gap_open"; break
            if c <= target_level:
                exit_ts, exit_reason, exit_val, exit_via = t, "target_50pct", float(c), "intraminute_close"; break
            if c >= stop_level:
                exit_ts, exit_reason, exit_val, exit_via = t, "stoploss_200pct_of_credit", float(c), "intraminute_close"; break
        if exit_ts is None:
            time_exit_candidates = path_ts[(path_ts.normalize() == pd.Timestamp(exit_date_fallback)) & (path_ts.strftime("%H:%M") == "15:15")]
            if len(time_exit_candidates):
                exit_ts = time_exit_candidates[0]; exit_reason = "time_exit_dte21"; exit_val = float(combined_open.loc[exit_ts]); exit_via = "scheduled_open"
            elif len(path_ts):
                # DTE21 day itself had no 15:15 candle (data gap on that specific day) -- use the last
                # available candle AT OR BEFORE the DTE21 boundary (never past it), flagged distinctly.
                exit_ts = path_ts[-1]; exit_reason = "time_exit_dte21_no_1515_candle"; exit_val = float(combined_open.loc[exit_ts]); exit_via = "scheduled_open"
            else:
                # no candles at all between entry and the DTE21 boundary (shouldn't happen given the futures-
                # derived boundary date, but flagged rather than silently producing an empty trade)
                exit_ts = entry_ts; exit_reason = "no_data_between_entry_and_dte21"; exit_val = entry_credit; exit_via = "none"

        exit_date = exit_ts.date(); exit_dte = (exp - exit_date).days
        pnl = entry_credit - exit_val
        trades.append({
            "expiry": exp_str, "entry_date": entry_date, "entry_dte": entry_dte_actual, "entry_date_direction": direction_flag,
            "atm_strike": atm, "futures_px_at_entry": fut_px, "entry_ce": entry_ce, "entry_pe": entry_pe, "entry_credit": entry_credit,
            "exit_date": exit_date, "exit_dte": exit_dte, "exit_reason": exit_reason, "exit_value": exit_val, "exit_via": exit_via,
            "pnl_points": round(pnl, 2), "days_held": (exit_date - entry_date).days,
        })
        print(f"[{exp_str}] entry {entry_date} (DTE{entry_dte_actual}) ATM={atm} credit={entry_credit:.2f} "
              f"-> exit {exit_date} (DTE{exit_dte}) {exit_reason} via={exit_via} val={exit_val:.2f} pnl={pnl:+.2f}", flush=True)

    T = pd.DataFrame(trades); S = pd.DataFrame(skipped)
    print(f"\n{'='*90}\nTRADES: {len(T)} | SKIPPED CYCLES: {len(S)}", flush=True)
    if len(S):
        print("\n--- SKIPPED (flagged, not substituted) ---")
        print(S.to_string(index=False))

    if len(T):
        win = T["pnl_points"] > 0
        summary = {
            "n_trades": len(T), "win_rate_pct": round(win.mean() * 100, 2), "total_pnl_points": round(T["pnl_points"].sum(), 2),
            "avg_pnl_points": round(T["pnl_points"].mean(), 2), "avg_days_held": round(T["days_held"].mean(), 2),
            "max_profit": round(T["pnl_points"].max(), 2), "max_loss": round(T["pnl_points"].min(), 2),
            "pct_target": round((T["exit_reason"] == "target_50pct").mean() * 100, 2),
            "pct_stoploss": round((T["exit_reason"] == "stoploss_200pct_of_credit").mean() * 100, 2),
            "pct_time_exit": round((T["exit_reason"] == "time_exit_dte21").mean() * 100, 2),
            "pct_dte21_no_1515_candle": round((T["exit_reason"] == "time_exit_dte21_no_1515_candle").mean() * 100, 2),
            "n_target_or_stop_via_gap_open": int((T["exit_via"] == "gap_open").sum()),
            "n_skipped_cycles": len(S),
        }
        pd.set_option("display.width", 220)
        print("\n=== SUMMARY ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("\n=== PER-TRADE TABLE ===")
        print(T.to_string(index=False))

        out_fn = OUTDIR / "nifty_dte45_atm_straddle.xlsx"
        try:
            out_fn.touch(exist_ok=True)
        except PermissionError:
            out_fn = OUTDIR / "nifty_dte45_atm_straddle_v2.xlsx"
            print(f"NOTE: primary output file is locked (likely open in Excel) -- saving to {out_fn.name} instead", flush=True)
        with pd.ExcelWriter(out_fn, engine="openpyxl") as w:
            note = pd.DataFrame([
                {"note": "Entry: nearest trading day to (expiry-45 calendar days), 15:15 OPEN price of NIFTY "
                          "MONTHLY FUTURES for ATM strike selection (round to nearest 50), then 15:15 OPEN of "
                          "the ATM CE/PE for entry credit -- open-price convention (changed from close per "
                          "explicit follow-up instruction), used for entry AND the scheduled DTE=21 time exit."},
                {"note": "Exit priority: 50%-of-credit target (value decays to 0.5x entry credit), then stoploss "
                          "at 200%-of-credit (value rises to 2x entry credit -- a loss equal to the original "
                          "credit received, per the spec's own worked example: sold for 600, stop at 1200). Both "
                          "checked touch-based on a per-minute COMBINED (CE+PE) series checking OPEN before CLOSE "
                          "at every minute -- a gap-through fills at the true gap-open price, not the close of "
                          "that first post-gap candle (not low+low/high+high, which would double-count an "
                          "impossible simultaneous move on both legs); else DTE=21 time exit at 15:15 OPEN."},
                {"note": "Options data fetched FRESH per cycle for the EXACT futures-derived ATM strike (not reused "
                          "from the pre-existing +-15-around-spot-at-expiry pull, confirmed to sometimes miss the "
                          "correct DTE45 strike). Cycles whose fetched premium history starts AFTER the required "
                          "entry date are EXCLUDED and flagged in the Skipped_Cycles sheet -- not substituted."},
                {"note": "Dec-2024 monthly expiry excluded entirely: confirmed genuine Upstox data gap in the "
                          "underlying futures contract itself (see full_pull_nifty_futures.py), so ATM could not "
                          "be determined from futures for that cycle -- not substituted with spot/synthetic."},
            ])
            note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
            pd.DataFrame([summary]).to_excel(w, sheet_name="Summary", index=False)
            T.to_excel(w, sheet_name="Trades", index=False)
            if len(S):
                S.to_excel(w, sheet_name="Skipped_Cycles", index=False)
            for sh in w.sheets.values():
                for c in sh.columns:
                    width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                    sh.column_dimensions[c[0].column_letter].width = min(width + 2, 34)
        print(f"\nSaved -> {out_fn}")
    else:
        print("\nNO TRADES PRODUCED.")


if __name__ == "__main__":
    main()
