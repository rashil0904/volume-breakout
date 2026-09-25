# -*- coding: utf-8 -*-
"""
rolling_5d_oi_price.py
======================
Rolling 5-TRADING-day combined-OI change vs 5-day futures price change.
Reuses data/nifty_futures_oi_price_daily.xlsx (combined_oi_all_expiries, near-month futures
close, same-contract close_change_pct, rollover_day=post_expiry, spot_close) — no OI recompute.

For each day T with >=5 prior trading days (T-5 = 5 TRADING days back, by position):
  oi_change_5d_num = combined_oi[T] - combined_oi[T-5]
  oi_change_5d_pct = (combined_oi[T] - combined_oi[T-5]) / combined_oi[T-5] * 100
  price_change_5d_pct_raw  = near-month futures_close[T] vs [T-5]      (spans contracts if window crosses a roll)
  price_change_5d_pct_cont = BACK-ADJUSTED CONTINUOUS close (roll-free, from same-contract daily returns)
  price_change_5d_pct_spot = spot Nifty close (continuous cross-check)
Flags: endpoint_post_expiry (T or T-5 is post_expiry -> OI understated), window_spans_rollover
(a contract switch inside T-4..T -> raw price has a contract-switch artifact; use _cont instead).
Regime classification uses the CLEAN continuous 5-day price.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "nifty_futures_oi_price_daily.xlsx"
OUT = REPO / "data" / "rolling_5d_oi_price.xlsx"
LAG = 5
OI_THR, PX_THR = 3.0, 1.0


def main():
    df = pd.read_excel(SRC, sheet_name="futures_oi_price_daily").sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df.loc[0, "rollover_day"] = False                       # first row isn't a real switch
    n = len(df)
    oi = df["combined_oi_all_expiries"].values.astype(float)
    fc = df["futures_close"].values.astype(float)
    spot = df["spot_close"].values.astype(float)
    cc = df["close_change_pct"].values.astype(float)        # same-contract daily return %
    roll = df["rollover_day"].astype(bool).values

    # back-adjusted continuous futures close (roll-free): chain same-contract daily returns
    cont = np.empty(n); cont[0] = fc[0]
    for i in range(1, n):
        cont[i] = cont[i-1] * (1 + (cc[i] / 100.0 if not np.isnan(cc[i]) else 0.0))

    def lag(a): return np.r_[np.full(LAG, np.nan), a[:-LAG]]
    oi_l, fc_l, spot_l, cont_l = lag(oi), lag(fc), lag(spot), lag(cont)

    df["oi_change_5d_num"] = oi - oi_l
    df["oi_change_5d_pct"] = (oi - oi_l) / oi_l * 100
    df["price_change_5d_pct_raw"] = (fc - fc_l) / fc_l * 100
    df["price_change_5d_pct_cont"] = (cont - cont_l) / cont_l * 100
    df["price_change_5d_pct_spot"] = (spot - spot_l) / spot_l * 100
    # flags
    df["endpoint_post_expiry"] = roll | np.r_[np.full(LAG, False), roll[:-LAG]]
    win_roll = np.zeros(n, bool)                            # any rollover in T-4..T (window spans a switch)
    for k in range(0, LAG):                                 # positions T, T-1, ..., T-4
        win_roll |= np.r_[np.full(k, False), roll[:n-k]] if k else roll
    df["window_spans_rollover"] = win_roll

    valid = df.iloc[LAG:].copy()                            # rows with a full 5-day lookback

    # ── regime classification on CLEAN continuous 5-day price ──
    o = valid["oi_change_5d_pct"]; p = valid["price_change_5d_pct_cont"]
    def regime(oi_c, px_c):
        if abs(oi_c) < OI_THR or abs(px_c) < PX_THR:
            return "flat"
        if oi_c >= OI_THR and px_c >= PX_THR:  return "OI up + price up"
        if oi_c >= OI_THR and px_c <= -PX_THR: return "OI up + price down"
        if oi_c <= -OI_THR and px_c >= PX_THR: return "OI down + price up"
        return "OI down + price down"
    valid["regime_5d"] = [regime(a, b) for a, b in zip(o, p)]

    reg_counts = (valid["regime_5d"].value_counts()
                  .rename_axis("regime_5d").reset_index(name="days"))
    reg_counts["pct"] = (reg_counts["days"] / len(valid) * 100).round(1)

    # ── summary stats ──
    def stat(col):
        s = valid[col].dropna()
        return {"metric": col, "avg": round(s.mean(), 3), "min": round(s.min(), 3), "max": round(s.max(), 3)}
    stats = pd.DataFrame([stat("oi_change_5d_pct"), stat("price_change_5d_pct_raw"),
                          stat("price_change_5d_pct_cont"), stat("price_change_5d_pct_spot")])
    same = int((np.sign(valid["oi_change_5d_pct"]) == np.sign(valid["price_change_5d_pct_cont"])).sum())
    opp = len(valid) - same

    out_cols = ["date", "combined_oi_all_expiries", "oi_change_5d_num", "oi_change_5d_pct",
                "futures_close", "price_change_5d_pct_raw", "price_change_5d_pct_cont",
                "price_change_5d_pct_spot", "regime_5d", "endpoint_post_expiry", "window_spans_rollover"]
    series = valid[out_cols].copy()
    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        series.to_excel(w, sheet_name="rolling_5d_series", index=False)
        reg_counts.to_excel(w, sheet_name="regime_counts", index=False)
        stats.to_excel(w, sheet_name="summary_stats", index=False)
    series.to_csv(REPO / "data" / "rolling_5d_oi_price.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"Rolling 5-trading-day OI vs price — {len(valid)} valid days (of {n})")
    print(f"  windows spanning a rollover: {int(valid['window_spans_rollover'].sum())} "
          f"| windows with a post-expiry endpoint: {int(valid['endpoint_post_expiry'].sum())}")
    print("\n--- SUMMARY STATS ---")
    print(stats.to_string(index=False))
    print(f"\n  5d OI & 5d price SAME direction: {same} days ({same/len(valid)*100:.1f}%) | "
          f"OPPOSITE: {opp} ({opp/len(valid)*100:.1f}%)")
    print("\n--- 5-DAY REGIME COUNTS (thresholds |OI|>=3%, |price|>=1%, price=continuous) ---")
    print(reg_counts.to_string(index=False))
    print("\n--- SAMPLE (first 3 & last 3 valid rows) ---")
    print(pd.concat([series.head(3), series.tail(3)]).to_string(index=False))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
