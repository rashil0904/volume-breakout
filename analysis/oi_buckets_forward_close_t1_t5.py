# -*- coding: utf-8 -*-
"""
oi_buckets_forward_close_t1_t5.py
=================================
For each OI-change day D (T+0), forward NEAR-MONTH FUTURES CLOSE return at T+1..T+5:
  ret_T{N} = (futures_close[D+N] - futures_close[D]) / futures_close[D] * 100     (cumulative, vs D's close)
Bucketed on combined_oi_change_pct (day D), cleaned series (D not a post_expiry_day).

Rollover caveat: futures_close[D+N] vs futures_close[D] spans contracts if an expiry/rollover
falls within D+1..D+N -> distorted. Flagged per horizon; a CLEAN table (horizons with NO
expiry crossing) is also produced. (Increase buckets are mostly expiry days, so their multi-day
futures-close returns cross a rollover immediately and can't be measured cleanly in futures.)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "nifty_futures_combined_oi_price_daily.xlsx"
OUT = REPO / "data" / "combined_oi_buckets_forward_close_t1_t5.xlsx"
HORIZONS = [1, 2, 3, 4, 5]

BUCKETS = [
    ("+5% to +10%",  lambda x: (x >= 5) & (x < 10)),
    ("+10% to +15%", lambda x: (x >= 10) & (x < 15)),
    ("+15% to +20%", lambda x: (x >= 15) & (x < 20)),
    (">= +20%",      lambda x: x >= 20),
    ("-5% to -10%",  lambda x: (x <= -5) & (x > -10)),
    ("-10% to -15%", lambda x: (x <= -10) & (x > -15)),
    ("-15% to -20%", lambda x: (x <= -15) & (x > -20)),
    ("<= -20%",      lambda x: x <= -20),
    ("within +/-5%", lambda x: (x > -5) & (x < 5)),
]


def main():
    df = pd.read_excel(SRC, sheet_name="combined_oi_price_daily")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    close = df["futures_close"].values.astype(float)
    pe = df["post_expiry_day"].astype(bool).values

    for N in HORIZONS:
        fwd = np.r_[close[N:], np.full(N, np.nan)]                 # close[D+N]
        df[f"ret_T{N}_close_pct"] = np.round((fwd - close) / close * 100, 4)
        cross = np.zeros(n, bool)                                  # any expiry/rollover in D+1..D+N
        for k in range(1, N + 1):
            cross |= np.r_[pe[k:], np.zeros(k, bool)]
        df[f"expiry_within_T{N}"] = cross

    clean = df[~df["post_expiry_day"]]                             # genuine OI-change days
    denom = len(clean)
    oi = clean["combined_oi_change_pct"]

    rows_all, rows_clean = [], []
    for label, cond in BUCKETS:
        sub = clean[cond(oi) & oi.notna()]
        nb = len(sub)
        r_all = {"bucket": label, "count": nb, "pct_of_days": round(nb / denom * 100, 2),
                 "avg_oi_change_pct": round(float(sub["combined_oi_change_pct"].mean()), 4) if nb else np.nan}
        r_cl = {"bucket": label}
        for N in HORIZONS:
            col = f"ret_T{N}_close_pct"
            a = sub[col].dropna()
            r_all[f"avg_ret_T{N}_pct"] = round(float(a.mean()), 4) if len(a) else np.nan
            # clean: only horizons with NO expiry crossing
            cl = sub[~sub[f"expiry_within_T{N}"]][col].dropna()
            r_cl[f"cleanN_T{N}"] = len(cl)
            r_cl[f"avg_ret_T{N}_pct_clean"] = round(float(cl.mean()), 4) if len(cl) else np.nan
        rows_all.append(r_all); rows_clean.append(r_cl)

    tbl_all = pd.DataFrame(rows_all)
    tbl_clean = pd.DataFrame(rows_clean)

    keep = ["date", "contract_expiry", "futures_close", "combined_oi_change_pct",
            "post_expiry_day", "is_expiry_day"] + [f"ret_T{N}_close_pct" for N in HORIZONS] \
           + [f"expiry_within_T{N}" for N in HORIZONS]
    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        tbl_all.to_excel(w, sheet_name="fwd_close_T1_T5_all", index=False)
        tbl_clean.to_excel(w, sheet_name="fwd_close_T1_T5_clean", index=False)
        df[keep].to_excel(w, sheet_name="series_with_forward", index=False)

    pd.set_option("display.width", 230)
    print("FORWARD FUTURES-CLOSE return by OI-change bucket (T+N close vs day-D close), cleaned OI days:")
    print("\n=== ALL observations (multi-day returns may span a rollover; see clean table) ===")
    show = ["bucket", "count", "avg_oi_change_pct"] + [f"avg_ret_T{N}_pct" for N in HORIZONS]
    print(tbl_all[show].to_string(index=False))
    print("\n=== CLEAN only (horizons that do NOT cross an expiry/rollover) — trustworthy read ===")
    showc = ["bucket"] + sum([[f"avg_ret_T{N}_pct_clean", f"cleanN_T{N}"] for N in HORIZONS], [])
    print(tbl_clean[showc].to_string(index=False))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
