# -*- coding: utf-8 -*-
"""
add_intraday_close_from_open.py
===============================
Bucket on combined_oi_change_pct (day D), but pair each bucket with the NEXT DAY's (D+1)
futures price move — a forward/predictive read of "OI change today -> price tomorrow":

  OI change day               = D  (combined_oi_change_pct[D] = OI(D) vs OI(D-1))
  next_day_close_change_pct   = (futures_close[D+1] - futures_close[D]) / futures_close[D] * 100   (30-Jul close from 29-Jul close)
  next_day_close_from_open_pct= (futures_close[D+1] - futures_open[D+1]) / futures_open[D+1] * 100  (30-Jul close from 30-Jul open)

Near-month futures price. Buckets on the CLEANED series (day D not a post_expiry_day).
Note: when D is an expiry day, D+1 is a rollover -> next_day_close_change_pct spans two
contracts (flagged); next_day_close_from_open_pct is intraday on one contract, always clean.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "nifty_futures_combined_oi_price_daily.xlsx"
OUT = REPO / "data" / "combined_oi_change_buckets_nextday_price.xlsx"

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

    # same-day intraday, then shift price forward one day to align D+1 move to OI-change day D
    df["intraday_close_from_open_pct"] = ((df["futures_close"] - df["futures_open"]) / df["futures_open"] * 100).round(4)
    df["next_day_close_change_pct"] = df["close_change_pct"].shift(-1).round(4)          # close(D+1) vs close(D)
    df["next_day_close_from_open_pct"] = df["intraday_close_from_open_pct"].shift(-1).round(4)  # open(D+1)->close(D+1)
    df["next_day_is_rollover"] = df["post_expiry_day"].shift(-1).fillna(False)           # D+1 spans contracts

    clean = df[~df["post_expiry_day"]]                                                   # bucket on genuine OI-change days
    denom = len(clean)
    oi = clean["combined_oi_change_pct"]
    rows = []
    for label, cond in BUCKETS:
        sub = clean[cond(oi) & oi.notna()]
        n = len(sub)
        nd_cc = sub["next_day_close_change_pct"].dropna()
        nd_oc = sub["next_day_close_from_open_pct"].dropna()
        rows.append({
            "bucket": label, "count": n, "pct_of_days": round(n / denom * 100, 2),
            "avg_oi_change_pct": round(float(sub["combined_oi_change_pct"].mean()), 4) if n else np.nan,
            "avg_oi_change_num": round(float(sub["combined_oi_change_num"].mean()), 0) if n else np.nan,
            "avg_nextday_close_change_pct": round(float(nd_cc.mean()), 4) if len(nd_cc) else np.nan,
            "avg_nextday_close_from_open_pct": round(float(nd_oc.mean()), 4) if len(nd_oc) else np.nan,
            "nextday_rollover_days_in_bucket": int(sub["next_day_is_rollover"].sum()),
        })
    table = pd.DataFrame(rows)

    keep = ["date", "contract_expiry", "futures_open", "futures_close", "combined_oi",
            "combined_oi_change_pct", "close_change_pct", "intraday_close_from_open_pct",
            "next_day_close_change_pct", "next_day_close_from_open_pct",
            "post_expiry_day", "is_expiry_day", "next_day_is_rollover"]
    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        table.to_excel(w, sheet_name="buckets_nextday_price", index=False)
        df[keep].to_excel(w, sheet_name="series_with_nextday", index=False)

    pd.set_option("display.width", 210)
    print("ALIGNMENT — OI change on day D paired with NEXT day (D+1) price. Example (consecutive rows):")
    ex = df[["date", "combined_oi_change_pct", "futures_close", "next_day_close_change_pct",
             "next_day_close_from_open_pct"]].tail(4)
    print(ex.to_string(index=False))
    print("\n  Read: on each row's date D, avg_oi_change is D's OI move; the two next_day_* columns are"
          "\n  D+1's price (close-vs-prev-close and open-to-close). Bucketed on OI change (day D), cleaned.\n")
    print(table.to_string(index=False))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
