# -*- coding: utf-8 -*-
"""
oi_up_price_up_forward.py
=========================
ONE condition: OI increased AND price increased on the same day -> forward Nifty futures
CLOSE returns for the next 5 trading days (cumulative from the signal day's close).

Filter (cleaned series, post_expiry signal days excluded):
  combined_oi_change_pct >= +3   AND   close_change_pct >= +1
Forward (long-side, near-month futures close):
  ret_T{N} = (close[D+N] - close[D]) / close[D] * 100     (cumulative from signal close)
Rollover: a forward window crossing a near-month contract switch spans two contracts; those
horizons are flagged (expiry_within_T{N}) and reported separately as a CLEAN aggregate.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "nifty_futures_combined_oi_price_daily.xlsx"
OUT = REPO / "data" / "oi_up_price_up_forward.xlsx"
OI_FLOOR, PRICE_FLOOR, H = 3.0, 1.0, [1, 2, 3, 4, 5]


def main():
    df = pd.read_excel(SRC, sheet_name="combined_oi_price_daily").sort_values("date").reset_index(drop=True)
    n = len(df)
    close = df["futures_close"].values.astype(float)
    pe = df["post_expiry_day"].astype(bool).values

    for N in H:
        fwd = np.r_[close[N:], np.full(N, np.nan)]
        df[f"ret_T{N}"] = np.round((fwd - close) / close * 100, 4)
        cross = np.zeros(n, bool)
        for k in range(1, N + 1):
            cross |= np.r_[pe[k:], np.zeros(k, bool)]
        df[f"expiry_within_T{N}"] = cross

    sig = df[(~df["post_expiry_day"]) & (df["combined_oi_change_pct"] >= OI_FLOOR)
             & (df["close_change_pct"] >= PRICE_FLOOR)].copy().reset_index(drop=True)
    nq = len(sig)
    print(f"Qualifying signal days (OI change >= +{OI_FLOOR}% AND close change >= +{PRICE_FLOOR}%): {nq}")

    detail = sig[["date", "combined_oi_change_pct", "combined_oi_change_num", "close_change_pct"]
                 + [f"ret_T{N}" for N in H]].copy()
    detail["rollover_in_5d_window"] = sig["expiry_within_T5"].values

    # aggregate — all rows and clean (rows whose horizon doesn't cross a rollover)
    agg = []
    for N in H:
        col = f"ret_T{N}"
        a = sig[col].dropna()
        cl = sig[~sig[f"expiry_within_T{N}"]][col].dropna()
        agg.append({
            "horizon": f"T+{N}",
            "avg_return_pct": round(float(a.mean()), 4) if len(a) else np.nan,
            "median_return_pct": round(float(a.median()), 4) if len(a) else np.nan,
            "win_rate_pct": round(float((a > 0).mean() * 100), 1) if len(a) else np.nan,
            "n_days": len(a),
            "avg_return_pct_clean": round(float(cl.mean()), 4) if len(cl) else np.nan,
            "median_return_pct_clean": round(float(cl.median()), 4) if len(cl) else np.nan,
            "win_rate_pct_clean": round(float((cl > 0).mean() * 100), 1) if len(cl) else np.nan,
            "n_days_clean": len(cl),
        })
    summary = pd.DataFrame(agg)

    # detail sheet with aggregate rows appended
    agg_rows = []
    for stat, fn in [("AVG", lambda s: s.mean()), ("MEDIAN", lambda s: s.median()),
                     ("WIN_RATE_%", lambda s: (s > 0).mean() * 100)]:
        r = {"date": stat, "combined_oi_change_pct": "", "combined_oi_change_num": "", "close_change_pct": ""}
        for N in H:
            r[f"ret_T{N}"] = round(float(fn(sig[f"ret_T{N}"].dropna())), 4)
        r["rollover_in_5d_window"] = ""
        agg_rows.append(r)
    detail_out = pd.concat([detail, pd.DataFrame(agg_rows)], ignore_index=True)

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        detail_out.to_excel(w, sheet_name="signal_days_detail", index=False)
        summary.to_excel(w, sheet_name="aggregate_summary", index=False)

    n_expiry_sig = int(sig["is_expiry_day"].sum()) if "is_expiry_day" in sig else int(sig["expiry_within_T1"].sum())
    pd.set_option("display.width", 220)
    print("\n=== DETAIL (per qualifying signal day; cumulative futures-close return from signal close) ===")
    print(detail.to_string(index=False))
    print("\n=== AGGREGATE SUMMARY (T+1..T+5) ===")
    print(summary.to_string(index=False))
    print(f"\n  rollover note: forward windows crossing a contract switch are flagged; the 'clean' columns "
          f"exclude them. Signal days whose T+1 is a post-expiry day: {int(sig['expiry_within_T1'].sum())} of {nq}.")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
