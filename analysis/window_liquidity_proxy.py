# -*- coding: utf-8 -*-
"""
window_liquidity_proxy.py — for EVERY backtest trade (baseline_final all_trades, ~3,436 trades),
compute a liquidity proxy on the 15:00-15:22 window of the entry day, then average per symbol.

WINDOW (flag a): 15:00..15:22 inclusive = 23 one-min candles (hm 900..922).
  window_low (flag b)          = MIN of 1-min lows in the window.
  window_total_volume          = SUM of 1-min volumes in the window.
  n_candles_in_window          = count present (should be 23; fewer if data missing -> flagged).
  avg_volume_per_candle (c)    = window_total_volume / actual n_candles present (NOT fixed 23).
  product (flag d)             = window_low * avg_volume_per_candle.
PER-SYMBOL (flag e): avg_product = simple mean of per-trade products over that symbol's trades.
TRADE SET (flag f): results/baseline_and_cross_final/baseline_final_performance.xlsx :: all_trades.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

TRADES = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
MASTER = rb.BASE / "master_data"
OUTDIR = rb.RESULTS / "window_liquidity"
IST = "Asia/Kolkata"
W_LO, W_HI = 900, 922            # 15:00 .. 15:22 inclusive


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tr = pd.read_excel(TRADES, sheet_name="all_trades", usecols=["symbol", "entry_date"])
    tr["entry_date"] = pd.to_datetime(tr["entry_date"]).dt.date
    print(f"backtest trades: {len(tr):,} | distinct symbols: {tr['symbol'].nunique():,}")

    rows, missing = [], []
    for sym, g in tr.groupby("symbol", sort=False):
        p = MASTER / f"{sym}.parquet"
        want_dates = list(g["entry_date"])
        if not p.exists():
            for d in want_dates:
                missing.append({"symbol": sym, "entry_date": str(d), "reason": "no_1min_symbol"})
            continue
        df = pd.read_parquet(p, columns=["timestamp", "low", "volume"])
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
        hm = ts.dt.hour * 60 + ts.dt.minute
        win = df[(hm >= W_LO) & (hm <= W_HI)].assign(d=ts[(hm >= W_LO) & (hm <= W_HI)].dt.date)
        agg = win.groupby("d").agg(window_low=("low", "min"),
                                   window_total_volume=("volume", "sum"),
                                   n=("volume", "size"))
        adict = agg.to_dict("index")
        for d in want_dates:
            rec = adict.get(d)
            if rec is None:
                missing.append({"symbol": sym, "entry_date": str(d), "reason": "no_1500_1522_candles_on_date"})
                continue
            nc = int(rec["n"]); wl = float(rec["window_low"]); wv = float(rec["window_total_volume"])
            avgv = wv / nc if nc else np.nan
            rows.append({"symbol": sym, "entry_date": str(d), "window_low": round(wl, 4),
                         "window_total_volume": int(wv), "n_candles_in_window": nc,
                         "avg_volume_per_candle": round(avgv, 2), "product": round(wl * avgv, 2)})
    R = pd.DataFrame(rows).sort_values(["symbol", "entry_date"]).reset_index(drop=True)

    # per-symbol: simple mean of per-trade products (flag e)
    sym_g = R.groupby("symbol")["product"]
    PS = pd.DataFrame({"n_trades": sym_g.size(), "avg_product": sym_g.mean().round(2),
                       "min_product": sym_g.min().round(2), "max_product": sym_g.max().round(2)}) \
        .reset_index().sort_values("avg_product", ascending=False).reset_index(drop=True)

    MISS = pd.DataFrame(missing)
    partial = R[R["n_candles_in_window"] != 23]

    R.to_parquet(OUTDIR / "per_trade_window_liquidity.parquet", index=False)
    R.to_csv(OUTDIR / "per_trade_window_liquidity.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "window_liquidity.xlsx", engine="openpyxl") as w:
        R.to_excel(w, sheet_name="per_trade", index=False)
        PS.to_excel(w, sheet_name="per_symbol_avg", index=False)
        if not MISS.empty:
            MISS.to_excel(w, sheet_name="missing", index=False)
        if not partial.empty:
            partial.to_excel(w, sheet_name="partial_lt23_candles", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96)
    print("WINDOW LIQUIDITY PROXY — 15:00-15:22 (23 one-min candles) product = window_low x avg_vol/candle")
    print("=" * 96)
    print(f"per-trade rows computed: {len(R):,} | symbols: {R['symbol'].nunique():,} | "
          f"missing (excluded): {len(MISS)} | days with <23 candles: {len(partial)}")
    print("\n--- PER-TRADE (first 12) ---")
    print(R.head(12).to_string(index=False))
    print("\n--- PER-SYMBOL AVG (top 15 by avg_product) ---")
    print(PS.head(15).to_string(index=False))
    print("\n--- PER-SYMBOL AVG (bottom 10 by avg_product) ---")
    print(PS.tail(10).to_string(index=False))
    print(f"\n  repeat-traded symbols (n_trades>1): {(PS.n_trades > 1).sum():,} of {len(PS):,}")
    if len(partial):
        print(f"\n  NOTE (flag c) — {len(partial)} trade-day(s) with <23 candles (avg divided by ACTUAL n present):")
        print(partial[["symbol", "entry_date", "n_candles_in_window", "window_total_volume", "avg_volume_per_candle"]].head(15).to_string(index=False))
    if len(MISS):
        print(f"\n  MISSING 1-min in window (excluded): {len(MISS)} — reasons: {MISS['reason'].value_counts().to_dict()}")
        print(MISS.head(10).to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
