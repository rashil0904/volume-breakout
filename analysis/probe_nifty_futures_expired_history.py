# -*- coding: utf-8 -*-
"""probe_nifty_futures_expired_history.py — test Upstox's expired-instruments coverage for NIFTY FUTURES
(monthly contracts), analogous to probe_expired_history.py (NIFTY options) and full_pull_banknifty_futures.py
(BankNifty futures, which found a floor of 2024-09-30 contract-start / 2024-10-30 expiry). Reuses the exact
get()/enc()/BASE helpers from opt_pull_nifty_full.py (same auth, same expired-instruments base URL).

STEPS (per the task):
  1. Get Expiries for NIFTY (NSE_INDEX|Nifty 50) via /expired-instruments/expiries.
  2. Walk expired expiries OLDEST-FIRST, calling /future/contract at each to find the FIRST one that is a
     genuine (non-empty) FUTURES expiry (NIFTY's list from this endpoint is dominated by weekly OPTIONS
     expiries; futures are monthly, so most candidates return empty and must be skipped).
  3. For that earliest genuine futures expiry, resolve its expired_instrument_key via /future/contract.
  4. Pull 1-min historical candles for that contract's full life (contract start ~= previous month's expiry
     + 1 day, through its own expiry) in 30-day chunks, exactly like full_pull_banknifty_futures.py's
     pull_contract(), and check for genuine continuous coverage (not just a partial tail).
  5. Compare the resulting floor to the already-established NIFTY OPTIONS floor (2024-10-01, FLOOR constant
     in opt_pull_nifty_full.py) and BankNifty FUTURES floor (2024-09-30 contract start, per
     full_pull_banknifty_futures.py's docstring).
"""
import sys, time
from pathlib import Path
from datetime import timedelta
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import opt_pull_nifty_full as op

NIFTY = "NSE_INDEX|Nifty 50"
KNOWN_NIFTY_OPTIONS_FLOOR = "2024-10-01"
KNOWN_BANKNIFTY_FUTURES_CONTRACT_START = "2024-09-30"


def main():
    exps_raw, sc = op.get(f"{op.BASE}/expiries?instrument_key={op.enc(NIFTY)}")
    if not exps_raw:
        print(f"expiries call FAILED (http {sc}). instrument_key={NIFTY}"); return
    today = time.strftime("%Y-%m-%d")
    exps = sorted(set(e for e in exps_raw if e < today))
    print(f"Get Expiries: instrument_key={NIFTY} | {len(exps)} expired expiries returned | "
          f"range {exps[0]} .. {exps[-1]}", flush=True)

    # ---- STEP 2/3: walk oldest-first, find first genuine FUTURES expiry ----
    print("\nSTEP 2/3: scanning oldest-first for the first genuine NIFTY FUTURES expiry (via /future/contract)...", flush=True)
    first_fut_expiry, first_fut_contract = None, None
    checked = 0
    for e in exps:
        cons, csc = op.get(f"{op.BASE}/future/contract?instrument_key={op.enc(NIFTY)}&expiry_date={e}")
        checked += 1
        tag = f"{len(cons)} contract(s)" if cons else f"none (http {csc})"
        print(f"  [{checked}] {e}: {tag}", flush=True)
        time.sleep(0.15)
        if cons:
            first_fut_expiry, first_fut_contract = e, cons[0]
            break
    if first_fut_expiry is None:
        print("\nNo genuine futures expiry found at all in the expired list -- stopping."); return

    ck = first_fut_contract["instrument_key"]; sym = first_fut_contract.get("trading_symbol", "?")
    lot = first_fut_contract.get("lot_size")
    print(f"\n==> EARLIEST NIFTY FUTURES EXPIRY: {first_fut_expiry} | symbol={sym} | instrument_key={ck} | lot_size={lot}", flush=True)

    # ---- STEP 4: pull full contract life, chunked, newest->oldest ----
    print(f"\nSTEP 4: pulling 1-min historical candles for {sym} ({ck}) across its full life...", flush=True)
    exp_ts = pd.Timestamp(first_fut_expiry)
    to_date = exp_ts.date()
    frames = []
    hi = to_date
    LOOKBACK, CHUNK = 90, 30   # NIFTY futures contracts run ~1-3 months; 90-day lookback is a safe margin
    floor_date = (exp_ts - pd.Timedelta(days=LOOKBACK)).date()
    consec_empty = 0; got_any = False
    for _ in range(LOOKBACK // CHUNK + 2):
        lo = hi - timedelta(days=CHUNK)
        if hi < floor_date:
            break
        cd, csc2 = op.get(f"{op.BASE}/historical-candle/{op.enc(ck)}/1minute/{hi.isoformat()}/{lo.isoformat()}")
        print(f"  chunk {lo}..{hi}: {len(cd) if cd else 0} candles (http {csc2})", flush=True)
        time.sleep(op.THROTTLE)
        if cd:
            frames.append(cd); got_any = True; consec_empty = 0
        else:
            consec_empty += 1
            if got_any or consec_empty >= 2:
                break
        hi = lo

    rows = [r for f in frames for r in f]
    if not rows:
        print(f"\n==> NO 1-MIN DATA returned for {sym} at all (contract resolves but historical-candle is empty).")
        return

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "OI"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.assign(timestamp=ts).drop_duplicates("timestamp").sort_values("timestamp")
    trading_days = sorted(df["timestamp"].dt.date.unique())
    n_days = len(trading_days)
    zero_vol = int((df["volume"] == 0).sum())
    has_oi = df["OI"].notna().any() and (df["OI"] != 0).any()

    print(f"\n==> DATA CHECK for {sym} ({ck}):")
    print(f"    total 1-min candles: {len(df):,}")
    print(f"    date range: {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}")
    print(f"    trading days covered: {n_days} (first={trading_days[0]}, last={trading_days[-1]})")
    print(f"    zero-volume candles: {zero_vol:,} ({zero_vol/len(df)*100:.2f}%)")
    print(f"    OI field populated (non-zero anywhere): {has_oi}")

    # gap check: are there missing trading days inside the covered range (partial vs continuous)?
    full_range_days = pd.bdate_range(trading_days[0], trading_days[-1])
    missing_bdays = [d.date() for d in full_range_days if d.date() not in set(trading_days)]
    print(f"    business days in range with NO candles at all (potential gaps, holidays included): {len(missing_bdays)}")
    if missing_bdays:
        print(f"      sample: {missing_bdays[:10]}")

    print(f"\n==> COMPARISON TO ESTABLISHED FLOORS:")
    print(f"    NIFTY FUTURES earliest data found:      {trading_days[0]}")
    print(f"    NIFTY OPTIONS known floor:               {KNOWN_NIFTY_OPTIONS_FLOOR}")
    print(f"    BankNifty FUTURES known contract start:  {KNOWN_BANKNIFTY_FUTURES_CONTRACT_START}")
    if str(trading_days[0]) == KNOWN_NIFTY_OPTIONS_FLOOR:
        print("    -> MATCHES the NIFTY options floor exactly.")
    elif str(trading_days[0]) < KNOWN_NIFTY_OPTIONS_FLOOR:
        print("    -> EARLIER than the NIFTY options floor -- NIFTY futures has a longer history available.")
    else:
        print("    -> LATER than the NIFTY options floor -- NIFTY futures is MORE restricted than options.")

    df.to_csv(Path(__file__).resolve().parent.parent / "results" / f"probe_nifty_futures_{sym.replace(' ', '_')}.csv", index=False)
    print(f"\nSaved raw probe candles -> results/probe_nifty_futures_{sym.replace(' ', '_')}.csv")


if __name__ == "__main__":
    main()
