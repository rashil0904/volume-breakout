# -*- coding: utf-8 -*-
"""
combined_oi_change_buckets.py
=============================
Bucket the Nifty futures COMBINED OI day-over-day change into 5%-step magnitude buckets,
separately for increases and decreases. Reuses data/nifty_futures_combined_oi_price_daily.xlsx
(combined_oi_change_pct) — does NOT recompute OI.

Buckets (magnitude-inclusive on the lower/magnitude side):
  INCREASES: [+5,+10) [+10,+15) [+15,+20) [+20,∞)
  DECREASES: (-10,-5] (-15,-10] (-20,-15] (-∞,-20]
  within ±5%: -5 < change < 5   (neither bucket)
Per bucket: count, % of all trading days, avg oi_change_pct, avg oi_change_num,
avg futures close_change_pct (OI-move vs price-move context). Rollover days in each
bucket flagged (a rollover can distort the day-over-day combined-OI change).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "nifty_futures_combined_oi_price_daily.xlsx"
OUT = REPO / "data" / "combined_oi_change_buckets.xlsx"

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
    total_days = len(df)
    oi = df["combined_oi_change_pct"]
    n_nan = int(oi.isna().sum())

    rows, roll_detail = [], []
    for label, cond in BUCKETS:
        mask = cond(oi) & oi.notna()
        sub = df[mask]
        n = len(sub)
        n_roll = int(sub["rollover_day"].sum())
        rows.append({
            "bucket": label,
            "count": n,
            "pct_of_all_days": round(n / total_days * 100, 2),
            "avg_oi_change_pct": round(float(sub["combined_oi_change_pct"].mean()), 4) if n else np.nan,
            "avg_oi_change_num": round(float(sub["combined_oi_change_num"].mean()), 0) if n else np.nan,
            "avg_close_change_pct": round(float(sub["close_change_pct"].mean()), 4) if n else np.nan,
            "rollover_days_in_bucket": n_roll,
        })
        if label not in ("within +/-5%",) and n_roll:
            for r in sub[sub["rollover_day"]].itertuples():
                roll_detail.append({"bucket": label, "date": r.date,
                                    "combined_oi_change_pct": r.combined_oi_change_pct,
                                    "combined_oi_change_num": r.combined_oi_change_num,
                                    "close_change_pct": r.close_change_pct,
                                    "contract_expiry": r.contract_expiry})
    table = pd.DataFrame(rows)
    roll = pd.DataFrame(roll_detail)

    inc = table[table["bucket"].str.startswith("+") | table["bucket"].str.startswith(">=")]
    dec = table[table["bucket"].str.startswith("-") | table["bucket"].str.startswith("<=")]
    tot_inc = int(inc["count"].sum()); tot_dec = int(dec["count"].sum())
    within = int(table.loc[table["bucket"] == "within +/-5%", "count"].iloc[0])

    summary = pd.DataFrame([
        {"metric": "total days OI increase >= +5%", "value": tot_inc,
         "pct_of_all_days": round(tot_inc / total_days * 100, 2)},
        {"metric": "total days OI decrease <= -5%", "value": tot_dec,
         "pct_of_all_days": round(tot_dec / total_days * 100, 2)},
        {"metric": "total days within +/-5%", "value": within,
         "pct_of_all_days": round(within / total_days * 100, 2)},
        {"metric": "no-change-data (first day, NaN)", "value": n_nan,
         "pct_of_all_days": round(n_nan / total_days * 100, 2)},
        {"metric": "all trading days", "value": total_days, "pct_of_all_days": 100.0},
    ])

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        table.to_excel(w, sheet_name="buckets", index=False)
        summary.to_excel(w, sheet_name="totals", index=False)
        if len(roll):
            roll.to_excel(w, sheet_name="rollover_in_extreme_buckets", index=False)

    pd.set_option("display.width", 200)
    print("=" * 110)
    print(f"COMBINED OI change buckets — Nifty futures  ({total_days} trading days, "
          f"{n_nan} first-day NaN excluded from buckets)")
    print("=" * 110)
    print(table.to_string(index=False))
    print("\n--- TOTALS ---")
    print(summary.to_string(index=False))
    lean = ("more prone to large INCREASES" if tot_inc > tot_dec else
            "more prone to large DECREASES" if tot_dec > tot_inc else "evenly split")
    print(f"\n  Increase>=5% days: {tot_inc}  |  Decrease<=5% days: {tot_dec}  ->  series is {lean} "
          f"(ratio inc:dec = {tot_inc}:{tot_dec})")
    print("\n--- ROLLOVER days landing in magnitude buckets (|change|>=5%) ---")
    if len(roll):
        print(roll.to_string(index=False))
        print(f"  ({len(roll)} rollover days in magnitude buckets — day-over-day combined-OI change "
              f"may be a rollover artifact, not a genuine surge)")
    else:
        print("  none")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
