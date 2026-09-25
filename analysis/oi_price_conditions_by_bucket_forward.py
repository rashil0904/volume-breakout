# -*- coding: utf-8 -*-
"""
oi_price_conditions_by_bucket_forward.py
========================================
Four OI x price conditions, each split by 3% OI-change bucket (3..15 + catch-all), with
forward Nifty-futures directional CLEAN returns T+1..T+5 (cumulative from signal close).
Cleaned series (post_expiry signal days excluded). Near-month futures close.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "nifty_futures_combined_oi_price_daily.xlsx"
OUT = REPO / "data" / "oi_price_conditions_by_bucket_forward.xlsx"
H = [1, 2, 3, 4, 5]

INC = [("+3-6%", lambda x: (x >= 3) & (x < 6)), ("+6-9%", lambda x: (x >= 6) & (x < 9)),
       ("+9-12%", lambda x: (x >= 9) & (x < 12)), ("+12-15%", lambda x: (x >= 12) & (x < 15)),
       (">= +15%", lambda x: x >= 15)]
DEC = [("-3-6%", lambda x: (x <= -3) & (x > -6)), ("-6-9%", lambda x: (x <= -6) & (x > -9)),
       ("-9-12%", lambda x: (x <= -9) & (x > -12)), ("-12-15%", lambda x: (x <= -12) & (x > -15)),
       ("<= -15%", lambda x: x <= -15)]
# condition -> (price_dir, direction, buckets)
CONDS = [
    ("LONG BUILDUP  (OI up & price up -> LONG)",    1,  "LONG",  INC),
    ("SHORT BUILDUP (OI up & price down -> SHORT)", -1, "SHORT", INC),
    ("SHORT COVERING (OI down & price up -> LONG)",  1,  "LONG",  DEC),
    ("LONG UNWINDING (OI down & price down -> SHORT)", -1, "SHORT", DEC),
]


def main():
    df = pd.read_excel(SRC, sheet_name="combined_oi_price_daily").sort_values("date").reset_index(drop=True)
    n = len(df); close = df["futures_close"].values.astype(float); pe = df["post_expiry_day"].astype(bool).values
    for N in H:
        fwd = np.r_[close[N:], np.full(N, np.nan)]
        df[f"long_T{N}"] = (fwd - close) / close * 100
        cross = np.zeros(n, bool)
        for k in range(1, N + 1):
            cross |= np.r_[pe[k:], np.zeros(k, bool)]
        df[f"cross_T{N}"] = cross
    base = df[~df["post_expiry_day"]]

    writer_frames = {}
    pd.set_option("display.width", 240)
    for title, pdir, direction, buckets in CONDS:
        rows = []
        for label, cond in buckets:
            sub = base[cond(base["combined_oi_change_pct"]) &
                       ((base["close_change_pct"] >= 1) if pdir == 1 else (base["close_change_pct"] <= -1))]
            row = {"oi_bucket": label, "n_days": len(sub)}
            for N in H:
                r = sub[f"long_T{N}"].copy()
                if direction == "SHORT":
                    r = -r
                cl = r[~sub[f"cross_T{N}"]].dropna()
                row[f"avg_clean_T{N}"] = round(float(cl.mean()), 3) if len(cl) else np.nan
                row[f"win_clean_T{N}"] = round(float((cl > 0).mean() * 100), 0) if len(cl) else np.nan
                row[f"n_clean_T{N}"] = len(cl)
            rows.append(row)
        t = pd.DataFrame(rows)
        writer_frames[title] = t
        show = ["oi_bucket", "n_days"] + [f"avg_clean_T{N}" for N in H] + [f"n_clean_T{N}" for N in H]
        print(f"\n=== {title} ===  (avg_clean = directional forward return %, +ve = direction paid off)")
        print(t[show].to_string(index=False))

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        for title, t in writer_frames.items():
            t.to_excel(w, sheet_name=title.split("(")[0].strip()[:28], index=False)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
