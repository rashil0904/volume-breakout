# -*- coding: utf-8 -*-
"""
rolling_5d_oi_forward.py
========================
Extend the rolling 5-day OI/price analysis with FORWARD 5-day futures returns, to test whether
the trailing 5-day OI/price setup predicts the next 5 days. Reuses
data/nifty_futures_oi_price_daily.xlsx (combined_oi, near-month close, same-contract
close_change_pct, rollover_day=post_expiry) — no OI recompute.

Forward returns are cumulative from T's close on the BACK-ADJUSTED CONTINUOUS futures price
(roll-free, same treatment as the trailing side), long-perspective (+ve = price rose):
  fwd_ret_T+N = (cont[T+N] - cont[T]) / cont[T] * 100
Trailing regime (|5d OI|>=3%, |5d price|>=1%, continuous price):
  OI up+price up / OI up+price down / OI down+price up / OI down+price down / flat.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "nifty_futures_oi_price_daily.xlsx"
OUT = REPO / "data" / "rolling_5d_oi_forward.xlsx"
LAG, FWD = 5, 5
OI_THR, PX_THR = 3.0, 1.0
H = [1, 2, 3, 4, 5]


def regime(oi_c, px_c):
    if abs(oi_c) < OI_THR or abs(px_c) < PX_THR:
        return "flat"
    if oi_c >= OI_THR and px_c >= PX_THR:  return "OI up + price up"
    if oi_c >= OI_THR and px_c <= -PX_THR: return "OI up + price down"
    if oi_c <= -OI_THR and px_c >= PX_THR: return "OI down + price up"
    return "OI down + price down"


def agg_block(sub):
    rows = []
    for N in H:
        r = sub[f"fwd_ret_T{N}"].dropna()
        rows.append({"horizon": f"T+{N}", "n": len(r),
                     "avg_fwd_pct": round(float(r.mean()), 4) if len(r) else np.nan,
                     "median_fwd_pct": round(float(r.median()), 4) if len(r) else np.nan,
                     "win_pct": round(float((r > 0).mean() * 100), 1) if len(r) else np.nan})
    return pd.DataFrame(rows)


def main():
    df = pd.read_excel(SRC, sheet_name="futures_oi_price_daily").sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df.loc[0, "rollover_day"] = False
    n = len(df)
    oi = df["combined_oi_all_expiries"].values.astype(float)
    cc = df["close_change_pct"].values.astype(float)
    roll = df["rollover_day"].astype(bool).values

    cont = np.empty(n); cont[0] = df["futures_close"].iloc[0]
    for i in range(1, n):
        cont[i] = cont[i-1] * (1 + (cc[i] / 100.0 if not np.isnan(cc[i]) else 0.0))

    # trailing (T vs T-5)
    oi_l = np.r_[np.full(LAG, np.nan), oi[:-LAG]]
    cont_l = np.r_[np.full(LAG, np.nan), cont[:-LAG]]
    df["oi_change_5d_num"] = oi - oi_l
    df["oi_change_5d_pct"] = (oi - oi_l) / oi_l * 100
    df["price_change_5d_pct"] = (cont - cont_l) / cont_l * 100          # continuous trailing

    # forward (T -> T+N) on continuous
    for N in H:
        fwd = np.r_[cont[N:], np.full(N, np.nan)]
        df[f"fwd_ret_T{N}"] = (fwd - cont) / cont * 100
    fwd_cross = np.zeros(n, bool)                                       # any rollover in T+1..T+5
    for k in range(1, FWD + 1):
        fwd_cross |= np.r_[roll[k:], np.zeros(k, bool)]
    df["fwd_window_spans_rollover"] = fwd_cross
    df["trail_window_spans_rollover"] = pd.Series(roll).rolling(LAG, min_periods=1).max().astype(bool).values

    df["regime_5d"] = [regime(a, b) for a, b in zip(df["oi_change_5d_pct"], df["price_change_5d_pct"])]

    # valid = full trailing lookback AND full forward window
    valid = df.iloc[LAG:n - FWD].copy()
    dropped_end = FWD
    print(f"Valid days (trailing 5d + forward 5d both available): {len(valid)} of {n} "
          f"(dropped {LAG} head for lookback, {dropped_end} tail for forward)")

    # 2. by trailing regime
    reg_tables = {}
    for reg in ["OI up + price up", "OI up + price down", "OI down + price up", "OI down + price down", "flat"]:
        sub = valid[valid["regime_5d"] == reg]
        reg_tables[reg] = agg_block(sub).assign(regime=reg, n_days=len(sub))

    # 3. by trailing OI-change magnitude bucket
    MAG = [("OI up 3-6%", lambda x: (x >= 3) & (x < 6)), ("OI up 6-9%", lambda x: (x >= 6) & (x < 9)),
           ("OI up 9-12%", lambda x: (x >= 9) & (x < 12)), ("OI up 12%+", lambda x: x >= 12),
           ("OI down 3-6%", lambda x: (x <= -3) & (x > -6)), ("OI down 6-9%", lambda x: (x <= -6) & (x > -9)),
           ("OI down 9-12%", lambda x: (x <= -9) & (x > -12)), ("OI down 12%+", lambda x: x <= -12)]
    mag_tables = {}
    for label, cond in MAG:
        sub = valid[cond(valid["oi_change_5d_pct"])]
        mag_tables[label] = agg_block(sub).assign(bucket=label, n_days=len(sub))

    out_cols = ["date", "combined_oi_all_expiries", "oi_change_5d_pct", "oi_change_5d_num",
                "price_change_5d_pct", "regime_5d"] + [f"fwd_ret_T{N}" for N in H] + \
               ["fwd_window_spans_rollover", "trail_window_spans_rollover"]
    series = valid[out_cols]

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        series.to_excel(w, sheet_name="rolling_5d_fwd_series", index=False)
        pd.concat(reg_tables.values()).to_excel(w, sheet_name="forward_by_trailing_regime", index=False)
        pd.concat(mag_tables.values()).to_excel(w, sheet_name="forward_by_oi_magnitude", index=False)
    series.to_csv(REPO / "data" / "rolling_5d_oi_forward.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"  forward windows spanning a rollover: {int(valid['fwd_window_spans_rollover'].sum())} "
          f"(continuous price already roll-adjusts; flagged for transparency)\n")
    print("=" * 90 + "\n2. FORWARD 5-day return BY TRAILING REGIME (continuous, long-perspective +ve=price rose)\n" + "=" * 90)
    for reg, t in reg_tables.items():
        print(f"\n--- {reg}  (n={t['n_days'].iloc[0]}) ---")
        print(t[["horizon", "avg_fwd_pct", "median_fwd_pct", "win_pct", "n"]].to_string(index=False))
    print("\n" + "=" * 90 + "\n3. FORWARD 5-day return BY TRAILING OI-CHANGE MAGNITUDE\n" + "=" * 90)
    comp = []
    for label, t in mag_tables.items():
        row = {"bucket": label, "n_days": t["n_days"].iloc[0]}
        for N in H:
            hv = t[t["horizon"] == f"T+{N}"]
            row[f"avg_T{N}"] = hv["avg_fwd_pct"].iloc[0]
            row[f"win_T{N}"] = hv["win_pct"].iloc[0]
        comp.append(row)
    compdf = pd.DataFrame(comp)
    print(compdf[["bucket", "n_days"] + [f"avg_T{N}" for N in H] + [f"win_T{N}" for N in H]].to_string(index=False))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
