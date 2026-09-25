# -*- coding: utf-8 -*-
"""
entry_slippage_tradebook.py — actual tradebook fill vs the backtest's 15:15 (3:15pm) reference.

STEP1 tradebook.xlsx sheet 'trade_book' -> filter entry date <= 2026-07-31 (20 trades).
STEP2 reference = the 15:15 one-min candle OPEN (flag a: the backtest's entry convention) from
      master_data/<SYMBOL>.parquet (NSE 1-min). Symbol match flagged (flag b).
STEP3 slippage_pct = (actual_entry - ref_1515_open)/ref_1515_open*100  (+ = filled HIGHER = adverse long).
STEP4 signed avg (key, flag c) + median + mean|.| + adverse/favorable counts + min/max/pctiles.
STEP5 per-trade table + summary -> Excel/CSV, print headline.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

TB = Path.home() / "Downloads" / "tradebook.xlsx"
MASTER = rb.BASE / "master_data"
OUTDIR = rb.RESULTS / "entry_slippage"
IST = "Asia/Kolkata"
CUTOFF = pd.Timestamp("2026-07-31").date()

# manual ticker overrides for tradebook name -> 1-min symbol (filled after alignment check)
OVERRIDE = {}
# symbols NOT in the local 1-min cache -> resolved NSE_EQ instrument key, fetched live from Upstox
INSTRUMENT = {"ASHIKAG": "NSE_EQ|INE094B01013"}   # Ashika Global Sec Ltd (on NSE, not in master_data cache)


def _extract(ts, opens, lows, vols, d):
    """From same-day 1-min arrays return (open_1515, low_1515, total_day_vol, n_candles) over 09:15-15:29."""
    hm = ts.dt.hour * 60 + ts.dt.minute
    day = (ts.dt.date == d) & (hm >= 555) & (hm < 930)          # (a) full session 09:15-15:29
    if not day.any():
        return None, None, None, 0
    m15 = day & (hm == 915)
    open_1515 = float(opens[m15].iloc[0]) if m15.any() else None
    low_1515 = float(lows[m15].iloc[0]) if m15.any() else None
    win = day & (hm >= 900) & (hm <= 922)        # 15:00 .. 15:22 inclusive (23 candles normally)
    avg_vol_win = float(vols[win].mean()) if win.any() else np.nan
    n_win = int(win.sum())
    return open_1515, low_1515, float(vols[day].sum()), int(day.sum()), avg_vol_win, n_win


def _fetch_day(inst_key, d):
    """Live Upstox 1-min for one day -> (ts Series, open Series, vol Series)."""
    import requests, data_loading as dl
    H = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}
    ds = d.isoformat()
    url = f"https://api.upstox.com/v3/historical-candle/{inst_key}/minutes/1/{ds}/{ds}"
    r = requests.get(url, headers=H, timeout=40)
    if r.status_code != 200:
        return None
    c = r.json().get("data", {}).get("candles", [])
    if not c:
        return None
    df = pd.DataFrame(c, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
    return ts, df["open"].astype(float), df["low"].astype(float), df["volume"].astype(float)


def ref_and_volume(sym, d):
    """Return (open_1515, low_1515, total_day_vol, n_candles, avg_vol_1500_1522, n_win, status)."""
    p = MASTER / f"{sym}.parquet"
    if p.exists():
        df = pd.read_parquet(p, columns=["timestamp", "open", "low", "volume"])
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
        o1515, l1515, tv, nc, aw, nw = _extract(ts, df["open"].astype(float), df["low"].astype(float), df["volume"].astype(float), d)
        if o1515 is None:
            return None, None, None, nc, None, 0, "no_1515_candle_on_date"
        return o1515, l1515, tv, nc, aw, nw, "ok"
    if sym in INSTRUMENT:                  # live-fetch fallback for symbols absent from the cache
        got = _fetch_day(INSTRUMENT[sym], d)
        if got is None:
            return None, None, None, 0, None, 0, "fetch_failed"
        o1515, l1515, tv, nc, aw, nw = _extract(got[0], got[1], got[2], got[3], d)
        return (o1515, l1515, tv, nc, aw, nw, "ok_fetched") if o1515 is not None else (None, None, None, nc, None, 0, "fetch_failed")
    return None, None, None, 0, None, 0, "no_1min_symbol"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    d = pd.read_excel(TB, sheet_name="trade_book")
    d["ed"] = pd.to_datetime(d["Position entry date"]).dt.date
    sub = d[d["ed"] <= CUTOFF][["Stock Name", "ed", "Entry Price"]].reset_index(drop=True)

    rows, unmatched = [], []
    for r in sub.itertuples():
        name = str(r._1).strip().upper()
        sym = OVERRIDE.get(name, name)
        ref, low1515, tot_vol, n_candles, avg_vol_win, n_win, status = ref_and_volume(sym, r.ed)
        if status not in ("ok", "ok_fetched"):
            unmatched.append({"stock": name, "entry_date": str(r.ed),
                              "actual_entry": round(float(r._3), 4), "reason": status})
            continue
        actual = float(r._3)
        slip = (actual - ref) / ref * 100.0
        avg_vol = tot_vol / n_candles if n_candles else np.nan     # (b) divide by actual n_candles present
        rows.append({"stock": name, "symbol_used": sym, "entry_date": str(r.ed), "ref_source": status,
                     "actual_entry": round(actual, 4), "reference_1515_open": round(ref, 4),
                     "slippage_pct": round(slip, 4), "direction": "adverse" if slip > 0 else ("favorable" if slip < 0 else "flat"),
                     "total_entry_day_volume": int(tot_vol), "n_candles": n_candles,
                     "avg_1min_candle_volume": round(avg_vol, 1),
                     "entry_candle_low_1515": round(low1515, 4),
                     "avgvol_x_low": round(avg_vol * low1515, 1),
                     "avg_1min_vol_1500_1522": round(avg_vol_win, 1), "n_candles_1500_1522": n_win})
    R = pd.DataFrame(rows)

    s = R["slippage_pct"].values
    summary = {
        "n_trades_total_le_cutoff": len(sub),
        "n_matched": len(R), "n_unmatched": len(unmatched),
        "AVG_slippage_pct_signed": round(float(np.mean(s)), 4),
        "median_slippage_pct": round(float(np.median(s)), 4),
        "mean_ABS_slippage_pct": round(float(np.mean(np.abs(s))), 4),
        "n_filled_higher_adverse": int((s > 0).sum()),
        "n_filled_lower_favorable": int((s < 0).sum()),
        "n_flat": int((s == 0).sum()),
        "min_slippage_pct": round(float(s.min()), 4), "max_slippage_pct": round(float(s.max()), 4),
        "p25": round(float(np.percentile(s, 25)), 4), "p75": round(float(np.percentile(s, 75)), 4),
    }
    S = pd.DataFrame([summary])
    U = pd.DataFrame(unmatched)

    with pd.ExcelWriter(OUTDIR / "entry_slippage.xlsx", engine="openpyxl") as w:
        R.to_excel(w, sheet_name="per_trade", index=False)
        S.to_excel(w, sheet_name="summary", index=False)
        if not U.empty:
            U.to_excel(w, sheet_name="unmatched", index=False)
    R.to_csv(OUTDIR / "entry_slippage_per_trade.csv", index=False)

    pd.set_option("display.width", 200)
    print("=" * 92)
    print("ENTRY SLIPPAGE — actual tradebook fill vs backtest 15:15 (3:15pm) candle OPEN")
    print("=" * 92)
    print(f"tradebook trades (entry <= {CUTOFF}): {len(sub)} | matched to 1-min: {len(R)} | unmatched: {len(unmatched)}")
    print("\n--- PER-TRADE (with entry-day volume) ---")
    print(R[["stock", "entry_date", "total_entry_day_volume", "avg_1min_candle_volume",
             "avg_1min_vol_1500_1522", "n_candles_1500_1522", "avgvol_x_low"]].to_string(index=False))
    partial = R[R["n_candles"] != 375]
    if not partial.empty:
        print(f"\n  NOTE (flag b) — {len(partial)} day(s) with n_candles != 375 (half-day/partial data), avg divided by ACTUAL n:")
        print(partial[["stock", "entry_date", "n_candles", "total_entry_day_volume", "avg_1min_candle_volume"]].to_string(index=False))
    else:
        print("\n  all 20 entry-days have the full 375 candles (avg = total/375).")
    if unmatched:
        print("\n--- UNMATCHED (excluded) ---")
        print(U.to_string(index=False))
    print("\n--- SUMMARY ---")
    for k, val in summary.items():
        print(f"  {k:28s}: {val}")
    print(f"\n  HEADLINE — avg signed slippage (directional cost, +=adverse): {summary['AVG_slippage_pct_signed']:+.4f}%")
    print(f"            mean absolute slippage (magnitude)               : {summary['mean_ABS_slippage_pct']:.4f}%")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
