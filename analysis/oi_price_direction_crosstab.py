# -*- coding: utf-8 -*-
"""
oi_price_direction_crosstab.py
==============================
Cross-tab each combined-OI-change bucket against same-day futures PRICE direction (±1% on
close_change_pct), on the CLEANED series (post_expiry_day already excluded). Reuses
data/nifty_futures_combined_oi_price_daily.xlsx — does NOT recompute OI or price.

price direction (same day, near-month futures close_change_pct):
  price_up   = close_change_pct >= +1
  price_down = close_change_pct <= -1
  price_flat = -1 < close_change_pct < +1

positioning read:
  OI up  + price up   = long buildup      | OI up  + price down = short buildup
  OI down + price up  = short covering     | OI down + price down = long unwinding
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "nifty_futures_combined_oi_price_daily.xlsx"
OUT = REPO / "data" / "oi_price_direction_crosstab.xlsx"

BUCKETS = [
    ("+5% to +10%",  "OI up",   lambda x: (x >= 5) & (x < 10)),
    ("+10% to +15%", "OI up",   lambda x: (x >= 10) & (x < 15)),
    ("+15% to +20%", "OI up",   lambda x: (x >= 15) & (x < 20)),
    (">= +20%",      "OI up",   lambda x: x >= 20),
    ("-5% to -10%",  "OI down", lambda x: (x <= -5) & (x > -10)),
    ("-10% to -15%", "OI down", lambda x: (x <= -10) & (x > -15)),
    ("-15% to -20%", "OI down", lambda x: (x <= -15) & (x > -20)),
    ("<= -20%",      "OI down", lambda x: x <= -20),
    ("within +/-5%", "flat OI", lambda x: (x > -5) & (x < 5)),
]
LABELS = {("OI up", "up"): "long buildup", ("OI up", "down"): "short buildup",
          ("OI down", "up"): "short covering", ("OI down", "down"): "long unwinding",
          ("flat OI", "up"): "-", ("flat OI", "down"): "-"}


def main():
    df = pd.read_excel(SRC, sheet_name="combined_oi_price_daily")
    df = df[~df["post_expiry_day"]].copy()                    # cleaned series
    cc = df["close_change_pct"]
    df["price_dir"] = np.where(cc >= 1, "up", np.where(cc <= -1, "down", "flat"))
    df = df[df["combined_oi_change_pct"].notna()]

    rows = []
    for label, grp, cond in BUCKETS:
        sub = df[cond(df["combined_oi_change_pct"])]
        tot = len(sub)
        nu = int((sub["price_dir"] == "up").sum())
        nd = int((sub["price_dir"] == "down").sum())
        nf = int((sub["price_dir"] == "flat").sum())
        p = lambda x: round(x / tot * 100, 1) if tot else np.nan
        rows.append({
            "oi_bucket": label, "oi_group": grp, "total_days": tot,
            "n_price_up": nu, "pct_up": p(nu), "n_price_down": nd, "pct_down": p(nd),
            "n_price_flat": nf, "pct_flat": p(nf),
            "up_interpretation": LABELS[(grp, "up")] if grp != "flat OI" else "-",
            "down_interpretation": LABELS[(grp, "down")] if grp != "flat OI" else "-",
            "avg_oi_change_num": round(float(sub["combined_oi_change_num"].mean()), 0) if tot else np.nan,
            "avg_close_change_pct": round(float(sub["close_change_pct"].mean()), 4) if tot else np.nan,
        })
    tab = pd.DataFrame(rows)

    # ── four-quadrant positioning totals (OI move >=5%, price move >=1%) ──
    oi = df["combined_oi_change_pct"]; pdr = df["price_dir"]
    q = {
        "long buildup (OI up>=5 & price up>=1)":     int(((oi >= 5) & (pdr == "up")).sum()),
        "short buildup (OI up>=5 & price down>=1)":  int(((oi >= 5) & (pdr == "down")).sum()),
        "OI up>=5 & price flat":                     int(((oi >= 5) & (pdr == "flat")).sum()),
        "short covering (OI down>=5 & price up>=1)": int(((oi <= -5) & (pdr == "up")).sum()),
        "long unwinding (OI down>=5 & price down>=1)": int(((oi <= -5) & (pdr == "down")).sum()),
        "OI down>=5 & price flat":                   int(((oi <= -5) & (pdr == "flat")).sum()),
        "within +/-5% OI (no strong OI signal)":     int(((oi > -5) & (oi < 5)).sum()),
    }
    quad = pd.DataFrame([{"positioning": k, "days": v,
                          "pct_of_cleaned": round(v / len(df) * 100, 2)} for k, v in q.items()])

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        tab.to_excel(w, sheet_name="oi_price_crosstab", index=False)
        quad.to_excel(w, sheet_name="four_quadrant_totals", index=False)

    pd.set_option("display.width", 250)
    print(f"OI x PRICE-DIRECTION cross-tab — cleaned series ({len(df)} days, post-expiry excluded)")
    print("price: up>=+1%, down<=-1%, flat between\n")
    show = ["oi_bucket", "oi_group", "total_days", "n_price_up", "pct_up", "n_price_down", "pct_down",
            "n_price_flat", "pct_flat", "up_interpretation", "down_interpretation",
            "avg_oi_change_num", "avg_close_change_pct"]
    print(tab[show].to_string(index=False))

    print("\n--- LARGE OI-move buckets (>=+15% and <=-15%): directional vs flat skew ---")
    big = tab[tab["oi_bucket"].isin(["+15% to +20%", ">= +20%", "-15% to -20%", "<= -20%"])]
    for _, r in big.iterrows():
        if r["total_days"] == 0:
            print(f"  {r['oi_bucket']:13s}: 0 days on cleaned series (were post-expiry artifacts, removed)")
        else:
            dirn = r["pct_up"] + r["pct_down"]
            print(f"  {r['oi_bucket']:13s}: {int(r['total_days'])} days | up {r['pct_up']}% ({r['up_interpretation']}) "
                  f"| down {r['pct_down']}% ({r['down_interpretation']}) | flat {r['pct_flat']}%  "
                  f"-> {'DIRECTIONAL' if dirn >= 60 else 'mostly FLAT'} ({dirn:.0f}% directional)")

    print("\n--- FOUR-QUADRANT positioning totals (cleaned series) ---")
    print(quad.to_string(index=False))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
